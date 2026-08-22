"""Load and persist the local service configuration (BR-8, BR-11, BR-12).

The config file is the single source of truth for a bundled service; ambient
environment variables and stray ``.env`` files are ignored (BR-21). This module
must not import ``config`` (import-cycle constraint).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend.schemas import ServiceConfig
from backend.services import app_paths

CONFIG_FILENAME = "config.json"


def config_path() -> Path:
    return app_paths.app_support_dir() / CONFIG_FILENAME


def _resolve_defaults(cfg: ServiceConfig) -> ServiceConfig:
    base = app_paths.app_support_dir()
    if not cfg.data_dir:
        cfg.data_dir = str(base / "data")
    if not cfg.models_dir:
        cfg.models_dir = str(base / "models")
    return cfg


def load() -> ServiceConfig:
    """Read the persisted config, filling in default paths. Never raises for a
    missing file — a fresh install returns defaults (BR-15)."""
    path = config_path()
    if path.exists():
        cfg = ServiceConfig(**json.loads(path.read_text(encoding="utf-8")))
    else:
        cfg = ServiceConfig()
    return _resolve_defaults(cfg)


def save(cfg: ServiceConfig) -> None:
    """Persist config atomically with owner-only permissions (the file may hold
    the HuggingFace token — OQ-3)."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)
