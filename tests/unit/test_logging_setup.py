from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from backend.services import app_paths, logging_setup


@pytest.fixture
def _restore_root_logging():
    """Remove handlers/level added by a test, so root-logger mutations do not
    leak across the pytest session (story #137 architect finding)."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in original_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(original_level)


pytestmark = pytest.mark.usefixtures("_restore_root_logging")


class TestConfigureServiceLogging:
    def test_installs_rotating_handler_and_returns_log_path(self, tmp_path: Path):
        log_file = logging_setup.configure_service_logging(log_dir=tmp_path)

        assert log_file == (tmp_path / "service.log").resolve()
        from logging.handlers import RotatingFileHandler

        handlers = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
        assert any(Path(h.baseFilename) == log_file for h in handlers)

    def test_application_info_record_reaches_file(self, tmp_path: Path):
        log_file = logging_setup.configure_service_logging(log_dir=tmp_path)

        logging.getLogger("backend.services.some_module").info("hello from app")

        assert "hello from app" in log_file.read_text()

    def test_uvicorn_info_record_reaches_file(self, tmp_path: Path):
        # Root level must be INFO for uvicorn access/error logs to propagate,
        # since log_config=None means uvicorn never sets its own levels.
        log_file = logging_setup.configure_service_logging(log_dir=tmp_path)

        logging.getLogger("uvicorn.access").info("127.0.0.1 - GET /api/health 200")

        assert "GET /api/health 200" in log_file.read_text()

    def test_no_stdout_stream_handler_installed(self, tmp_path: Path):
        logging_setup.configure_service_logging(log_dir=tmp_path)

        for handler in logging.getLogger().handlers:
            stream = getattr(handler, "stream", None)
            assert stream is not sys.stdout

    def test_is_idempotent(self, tmp_path: Path):
        from logging.handlers import RotatingFileHandler

        logging_setup.configure_service_logging(log_dir=tmp_path)
        logging_setup.configure_service_logging(log_dir=tmp_path)

        rotating = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
        matching = [h for h in rotating if Path(h.baseFilename) == (tmp_path / "service.log").resolve()]
        assert len(matching) == 1

    def test_rotates_when_size_exceeded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(logging_setup, "_MAX_BYTES", 1024)
        logging_setup.configure_service_logging(log_dir=tmp_path)
        logger = logging.getLogger("backend.services.rotation_probe")

        for i in range(200):
            logger.info("padding line %03d %s", i, "x" * 80)

        assert (tmp_path / "service.log.1").exists()

    def test_defaults_to_app_paths_logs_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", tmp_path)
        log_file = logging_setup.configure_service_logging()

        assert log_file == (tmp_path / "Logs" / "service.log").resolve()


# Shared canonical token; the Swift redaction test uses the same string so a
# pattern divergence between the two runtimes surfaces (story #139).
_SEEDED_TOKEN = "hf_TESTTOKEN0123456789abcdef"


class TestRedactSecrets:
    def test_masks_hf_token(self):
        redacted = logging_setup.redact_secrets(f"auth failed for {_SEEDED_TOKEN} now")

        assert _SEEDED_TOKEN not in redacted
        assert "hf_***" in redacted

    def test_leaves_non_secret_text_untouched(self):
        assert logging_setup.redact_secrets("plain message, no secrets") == ("plain message, no secrets")


class TestSecretRedactingFilterOnDisk:
    def test_message_token_is_redacted_on_disk(self, tmp_path: Path):
        log_file = logging_setup.configure_service_logging(log_dir=tmp_path)

        logging.getLogger("backend.services.redaction_probe").info("received token %s from provider", _SEEDED_TOKEN)

        contents = log_file.read_text()
        assert _SEEDED_TOKEN not in contents
        assert "hf_***" in contents

    def test_exception_traceback_token_is_redacted_on_disk(self, tmp_path: Path):
        log_file = logging_setup.configure_service_logging(log_dir=tmp_path)
        logger = logging.getLogger("backend.services.redaction_probe")

        try:
            raise RuntimeError(f"401 Unauthorized: {_SEEDED_TOKEN}")
        except RuntimeError:
            logger.exception("diarization call failed")

        contents = log_file.read_text()
        assert _SEEDED_TOKEN not in contents
        assert "hf_***" in contents
        # The traceback itself must be present (redaction, not omission).
        assert "Traceback" in contents

    def test_filter_neutralizes_malformed_format_record(self):
        # A mismatched-args record makes getMessage() raise. The filter must not
        # raise and must drop args so a downstream formatter cannot raise either.
        record = logging.LogRecord(
            name="probe",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="value is %d and %d",  # too few args on purpose
            args=(1,),
            exc_info=None,
        )

        assert logging_setup.SecretRedactingFilter().filter(record) is True
        assert record.args is None
        # getMessage() must now be safe (no formatting) — proves no downstream raise.
        assert record.getMessage() == "value is %d and %d"
