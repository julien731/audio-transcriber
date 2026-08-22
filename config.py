from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from backend.services import app_paths, service_config

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"


def is_bundled() -> bool:
    """True when running as the bundled, self-contained service."""
    return app_paths.is_bundled()


if is_bundled():
    # Bundled service: the Application Support config is the single source of
    # truth. Ambient environment variables and any stray .env from a prior
    # Terminal checkout are ignored (BR-19, BR-21).
    _cfg = service_config.load()
    DATA_DIR = Path(_cfg.data_dir)
    MODELS_DIR = Path(_cfg.models_dir)
    HF_TOKEN = _cfg.hf_token
    WHISPER_MODEL = _cfg.whisper_model
    WHISPER_DEVICE = "auto"
    WHISPER_BATCH_SIZE = 16
    MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
else:
    # Developer checkout: preserve the historical env/.env behavior so the
    # Makefile, Conductor, and the existing test suite keep working (BR-18).
    load_dotenv(override=False)
    DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    MODELS_DIR = Path(os.getenv("MODELS_DIR", str(DATA_DIR / "models")))
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
    WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(500 * 1024 * 1024)))  # 500MB

MEETINGS_DIR = DATA_DIR / "meetings"
MEETINGS_DIR.mkdir(parents=True, exist_ok=True)


def current_hf_token() -> str:
    """Live HuggingFace token.

    In bundled mode this re-reads the Application Support config so a token set
    at runtime via first-run provisioning takes effect without a restart (BR-8).
    An empty token disables diarization rather than failing (BR-10).
    """
    if is_bundled():
        return service_config.load().hf_token
    return HF_TOKEN
