"""Language → alignment-model resolution, shared by provisioning and the transcriber.

Kept deliberately import-light (no ``whisperx``/``torch``/``huggingface_hub``) so
``provisioning.required_repos()`` and its tests can resolve repos without pulling
in the heavy ML stack, which is absent in the lightweight CI env (mirrors the lazy
isolation in ``provisioning._snapshot_downloader``).

``HF_ALIGN_REPOS`` mirrors WhisperX's ``DEFAULT_ALIGN_MODELS_HF`` (plus this app's
``th`` override) so the repo provisioning pre-fetches is exactly the one the
transcriber loads at the align stage. A drift guard in ``tests/unit/test_align_models.py``
asserts the copy stays in sync with the installed WhisperX where it is importable.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Torch-native alignment languages: bundled by torchaudio into the shared
# ``~/.cache/torch`` (not HuggingFace), so they are never part of provisioning's
# HF pre-fetch and the transcriber lets WhisperX resolve them (``model_name=None``).
TORCH_ALIGN_LANGUAGES = frozenset({"en", "fr", "de", "es", "it"})

# HuggingFace wav2vec2 alignment repos keyed by language code. Copied from
# WhisperX ``DEFAULT_ALIGN_MODELS_HF`` so provisioning downloads exactly what the
# transcriber loads, with one app-specific override:
#   ``th`` — WhisperX ships no Thai default; this app supplies one.
HF_ALIGN_REPOS: dict[str, str] = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "id": "cahya/wav2vec2-large-xlsr-indonesian",
    # App-specific override (WhisperX has no Thai default).
    "th": "airesearch/wav2vec2-large-xlsr-53-th",
}


def align_repos_for(languages: Iterable[str]) -> list[str]:
    """HF alignment repos to pre-fetch for ``languages``.

    Torch-native and unknown codes contribute nothing. The result is deduplicated
    and order-stable (first occurrence wins) so provisioning's per-repo progress is
    deterministic.
    """
    repos: list[str] = []
    for code in languages:
        repo = HF_ALIGN_REPOS.get(code)
        if repo and repo not in repos:
            repos.append(repo)
    return repos


def align_model_cached(repo_id: str) -> bool:
    """Whether ``repo_id``'s alignment model is already in the local HF cache.

    A ``config.json`` hit is the presence proxy — WhisperX loads the model via
    ``Wav2Vec2ForCTC.from_pretrained``, which always needs it. Used to decide
    whether loading the model will trigger a mid-transcription download that
    should be surfaced as a distinct job stage rather than a frozen "Aligning…"
    (#145).

    ``huggingface_hub`` is imported lazily (kept out of this module's top-level
    imports; mirrors ``provisioning._snapshot_downloader``) so align resolution
    stays usable in the lightweight CI env without the ML stack. The probe uses
    the default cache dir, which ``huggingface_hub`` derives from ``HF_HOME`` at
    import time — the same env the bundled service sets before first import — so
    the probe and the subsequent load share cache resolution by construction.

    Degrades safe: any failure (including ``huggingface_hub`` being absent)
    returns ``True`` ("assume present"), which merely omits the download
    indicator and never blocks alignment — the ``ALIGNMENT_TIMEOUT_SEC`` watchdog
    remains the safety net.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    except Exception:  # noqa: BLE001 - safe degrade; keep the load path unaffected
        logger.debug("align cache probe failed for %s; assuming present", repo_id, exc_info=True)
        return True
