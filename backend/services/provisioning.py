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
from backend.services import service_config

logger = logging.getLogger(__name__)

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
    provisioning completes with the transcription model alone.
    """
    cfg = service_config.load()
    repos = [_whisper_repo(cfg.whisper_model)]
    if cfg.hf_token:
        repos += ["pyannote/segmentation-3.0", "pyannote/speaker-diarization-3.1"]
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
    """Whether required models are present (source of truth: persisted flag)."""
    return service_config.load().provisioning_completed


def set_token(token: str) -> ProvisioningStatus:
    """Store the HuggingFace token locally (BR-8). Empty disables diarization."""
    cfg = service_config.load()
    cfg.hf_token = token.strip()
    service_config.save(cfg)
    return status()


def status() -> ProvisioningStatus:
    cfg = service_config.load()
    st = _get_state()
    return ProvisioningStatus(
        provisioning_completed=cfg.provisioning_completed,
        models_present=cfg.provisioning_completed,
        whisper_model=cfg.whisper_model,
        diarization_available=bool(cfg.hf_token),
        download_state=st["state"],
        download_progress=st["progress"],
        download_error=st["error"],
    )
