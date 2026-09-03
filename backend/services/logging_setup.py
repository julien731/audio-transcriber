"""Rotating file logging for the bundled service (story #137).

The bundled service otherwise relies on uvicorn's default logging to stderr,
which the macOS app can no longer count on draining forever — and stderr output
is lost the moment the app exits. This module installs a size-bounded rotating
file handler on the *root* logger so both application (``logger.*``) and uvicorn
(``uvicorn``/``uvicorn.access``) records land in a file the user can share.

Only the bundled entrypoint (``service_main.py``) calls this; the dev entrypoint
(``run.py``) is intentionally untouched. Nothing here writes to stdout — the
readiness handshake owns stdout (see ``service_main.py``), and a stray log line
there would corrupt the JSON the client parses.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from backend.services import app_paths

# 5 MB per file, 3 rotated backups → ~20 MB worst case (BR: size-bounded logs).
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Marks our handler on the root logger so re-invocation is idempotent (the
# transcription pipeline spawns workers that re-exec this binary).
_HANDLER_MARKER = "_mt_service_log_handler"


def _already_installed(root: logging.Logger, log_file: Path) -> bool:
    for handler in root.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return True
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_file:
            return True
    return False


def configure_service_logging(log_dir: Path | None = None) -> Path:
    """Install a rotating file handler on the root logger. Idempotent.

    Returns the path of the active ``service.log``. Raises the root level to
    INFO so uvicorn access logs and ``logger.info`` records are captured —
    ``uvicorn.Config(log_level=...)`` is a no-op once we pass ``log_config=None``.
    """
    directory = log_dir if log_dir is not None else app_paths.logs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    log_file = (directory / "service.log").resolve()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if _already_installed(root, log_file):
        return log_file

    handler = RotatingFileHandler(
        log_file,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    return log_file
