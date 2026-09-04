"""First-run provisioning: HuggingFace token + ML model download (BR-8..BR-11).

Model downloads run in a background thread with coarse per-repo progress. The
persisted ``provisioning_completed`` flag is the source of truth for whether
models are present — a partial/failed download (EC-4) never sets it, so it can be
safely retried when connectivity returns.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import config
from backend.schemas import DownloadState, ProvisioningStatus
from backend.services import align_models, service_config

logger = logging.getLogger(__name__)

# Bump when ``required_repos()`` gains (or drops) a model so already-provisioned
# installs re-provision instead of lazily downloading the addition mid-run.
#   v1: whisper (+ segmentation-3.0, speaker-diarization-3.1 when a token is set)
#   v2: also pre-fetch the wespeaker embedding model that speaker-diarization-3.1
#       loads at the diarization step — previously downloaded at runtime (70%),
#       stalling the job (see debug session macos-transcribe-stuck-70).
#   v3: also pre-fetch the HuggingFace alignment models for the configured
#       ``align_languages`` (default Thai) — previously downloaded lazily at the
#       align step (50%), looking like a hang (#141). Torch-native align models
#       (en/fr/de/es/it) are unaffected.
PROVISIONING_VERSION = 3

_lock = threading.Lock()
_state: dict = {"state": DownloadState.IDLE, "progress": 0, "error": None}
_thread: threading.Thread | None = None


def _set_state(**kwargs) -> None:
    with _lock:
        _state.update(kwargs)


def _get_state() -> dict:
    with _lock:
        return dict(_state)


def _whisper_repo(model: str) -> str:
    """WhisperX loads faster-whisper CT2 weights from the Systran mirror."""
    return f"Systran/faster-whisper-{model}"


def required_repos() -> list[str]:
    """Repos needed for provisioning to be complete.

    The whisper model is always required. Diarization repos are only required
    when a token is configured; without one, diarization is disabled (BR-10) and
    provisioning completes with the transcription model alone. The alignment repos
    for the configured ``align_languages`` are always required (token-independent):
    pre-fetching them keeps the align step from downloading lazily mid-transcription
    (#141).
    """
    cfg = service_config.load()
    repos = [_whisper_repo(cfg.whisper_model)]
    if cfg.hf_token:
        repos += [
            "pyannote/segmentation-3.0",
            "pyannote/speaker-diarization-3.1",
            # Embedding model referenced by speaker-diarization-3.1's config and
            # loaded when the pipeline is instantiated. Not fetching it here made
            # the bundled service download it lazily at the diarization step (70%),
            # over plain HTTP into the isolated bundle cache, with no timeout —
            # hanging the job. Pre-fetch it so diarization runs entirely offline.
            "pyannote/wespeaker-voxceleb-resnet34-LM",
        ]
    # HF alignment models for the configured languages (torch-native and unknown
    # codes contribute nothing). Deduped against the list already assembled so a
    # repo is never counted twice in the download progress.
    for repo in align_models.align_repos_for(cfg.align_languages):
        if repo not in repos:
            repos.append(repo)
    return repos


def _snapshot_downloader() -> Callable[[str], None]:
    """Return a ``(repo_id) -> None`` downloader. Isolated so tests can patch it
    without importing huggingface_hub (absent in the lightweight CI env)."""
    from huggingface_hub import snapshot_download

    token = config.current_hf_token() or None

    def _download(repo_id: str) -> None:
        snapshot_download(repo_id=repo_id, token=token)

    return _download


def _run_download() -> None:
    try:
        _set_state(state=DownloadState.DOWNLOADING, progress=0, error=None)
        repos = required_repos()
        download = _snapshot_downloader()
        for index, repo in enumerate(repos):
            logger.info("Provisioning: downloading %s", repo)
            download(repo)
            _set_state(progress=int((index + 1) / len(repos) * 100))

        cfg = service_config.load()
        cfg.provisioning_completed = True
        cfg.provisioning_version = PROVISIONING_VERSION
        service_config.save(cfg)
        _set_state(state=DownloadState.COMPLETED, progress=100)
        logger.info("Provisioning complete")
    except Exception as exc:  # noqa: BLE001 - surfaced to the client via status
        logger.exception("Model download failed")
        _set_state(state=DownloadState.FAILED, error=str(exc))


def start_download() -> ProvisioningStatus:
    """Kick off a background model download if one is not already running."""
    global _thread
    with _lock:
        already_running = _thread is not None and _thread.is_alive()
    if not already_running:
        thread = threading.Thread(target=_run_download, name="model-download", daemon=True)
        with _lock:
            _thread = thread
        thread.start()
    return status()


def models_ready() -> bool:
    """Whether required models are present.

    A completed provisioning at an older schema version (before a required repo
    was added) reads as NOT ready, so already-provisioned installs re-provision
    to fetch the additions rather than downloading them lazily mid-transcription.
    """
    cfg = service_config.load()
    return cfg.provisioning_completed and cfg.provisioning_version >= PROVISIONING_VERSION


def set_token(token: str) -> ProvisioningStatus:
    """Store the HuggingFace token locally (BR-8). Empty disables diarization."""
    cfg = service_config.load()
    cfg.hf_token = token.strip()
    service_config.save(cfg)
    return status()


def status() -> ProvisioningStatus:
    cfg = service_config.load()
    st = _get_state()
    # Report readiness via models_ready() (version-gated), not the raw flag, so
    # /api/provisioning agrees with /api/health and the upload gate. Otherwise an
    # install completed at an older schema version would read as completed here —
    # routing the client past the setup wizard — while every upload 503s.
    ready = models_ready()
    return ProvisioningStatus(
        provisioning_completed=ready,
        models_present=ready,
        whisper_model=cfg.whisper_model,
        diarization_available=bool(cfg.hf_token),
        download_state=st["state"],
        download_progress=st["progress"],
        download_error=st["error"],
    )
