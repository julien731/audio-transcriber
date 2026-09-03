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
import re
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

# Secret-shaped tokens to mask before anything reaches disk (story #139). Our own
# code never logs the HuggingFace token, but a third-party dependency could embed
# it in a traceback (e.g. a 401 body), which would then land in service.log and
# the user-shareable diagnostics zip. Keep every pattern in this one list so more
# credential shapes can be added later. The Swift side mirrors this list in
# macos/Sources/MeetingTranscriberKit/Logging/SecretRedaction.swift — keep them
# in sync (both are exercised with the shared `hf_TESTTOKEN...` test vector).
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"hf_[A-Za-z0-9]+"), "hf_***"),
]


def redact_secrets(text: str) -> str:
    """Mask every known secret shape in ``text``. Safe on arbitrary strings."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Mask secrets in a record's message, traceback, and stack info.

    Installed on the file handler (not the root logger): a filter added to a
    logger only runs for records logged *directly* on it, so records propagated
    up from child loggers — notably ``uvicorn`` — would bypass it. A handler
    filter runs for every record the handler emits.

    The traceback is pre-rendered here and stashed in ``record.exc_text`` so the
    handler's formatter reuses the redacted text instead of re-rendering the raw
    ``exc_info``. Fails open: any formatting error leaves the record as-is rather
    than raising back at the log call site.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.args:
                try:
                    record.msg = record.getMessage()
                except Exception:  # noqa: BLE001
                    # Malformed format record (e.g. mismatched %-args): keep the
                    # raw template but drop args so the formatter cannot raise.
                    record.msg = str(record.msg)
                record.args = None
            if isinstance(record.msg, str):
                record.msg = redact_secrets(record.msg)
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = redact_secrets(record.exc_text)
            if record.stack_info:
                record.stack_info = redact_secrets(record.stack_info)
        except Exception:  # noqa: BLE001 — redaction must never break logging.
            pass
        return True


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
    handler.addFilter(SecretRedactingFilter())
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    return log_file
