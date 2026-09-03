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
