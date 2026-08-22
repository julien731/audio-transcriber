from __future__ import annotations

import importlib
from pathlib import Path

import pytest


class TestConfigDefaults:
    def test_default_whisper_model(self):
        import config

        assert config.WHISPER_MODEL == "large-v3"

    def test_default_whisper_device(self):
        import config

        assert config.WHISPER_DEVICE == "auto"

    def test_default_whisper_batch_size(self):
        import config

        assert config.WHISPER_BATCH_SIZE == 16

    def test_default_max_upload_size(self):
        import config

        assert config.MAX_UPLOAD_SIZE == 500 * 1024 * 1024

    def test_default_data_dir(self):
        import config

        assert config.DATA_DIR == config.BASE_DIR / "data"

    def test_meetings_dir_is_subdir_of_data_dir(self):
        import config

        assert config.MEETINGS_DIR == config.DATA_DIR / "meetings"

    def test_templates_dir(self):
        import config

        assert config.TEMPLATES_DIR == config.BASE_DIR / "templates"

    def test_hf_token_default_empty(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        import config

        importlib.reload(config)
        # When no HF_TOKEN is set, it should default to empty string
        assert isinstance(config.HF_TOKEN, str)


class TestConfigEnvOverrides:
    @pytest.fixture(autouse=True)
    def _restore_config(self):
        """Restore config module to default state after each test."""
        yield
        import config

        importlib.reload(config)

    def test_custom_data_dir(self, monkeypatch, tmp_path: Path):
        custom_dir = tmp_path / "custom_data"
        monkeypatch.setenv("DATA_DIR", str(custom_dir))
        import config

        importlib.reload(config)
        assert config.DATA_DIR == custom_dir
        assert config.MEETINGS_DIR == custom_dir / "meetings"
        # Verify directory was created
        assert config.MEETINGS_DIR.exists()

    def test_custom_whisper_model(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL", "base")
        import config

        importlib.reload(config)
        assert config.WHISPER_MODEL == "base"

    def test_custom_whisper_device(self, monkeypatch):
        monkeypatch.setenv("WHISPER_DEVICE", "cpu")
        import config

        importlib.reload(config)
        assert config.WHISPER_DEVICE == "cpu"

    def test_custom_whisper_batch_size(self, monkeypatch):
        monkeypatch.setenv("WHISPER_BATCH_SIZE", "8")
        import config

        importlib.reload(config)
        assert config.WHISPER_BATCH_SIZE == 8

    def test_custom_max_upload_size(self, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_SIZE", "1048576")
        import config

        importlib.reload(config)
        assert config.MAX_UPLOAD_SIZE == 1048576

    def test_custom_hf_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
        import config

        importlib.reload(config)
        assert config.HF_TOKEN == "hf_test_token_123"


class TestBundledModeIsolation:
    """In bundled mode the Application Support config is the sole source of
    truth; ambient env vars and .env are ignored (BR-19, BR-21)."""

    @pytest.fixture(autouse=True)
    def _restore_config(self):
        yield
        import config

        importlib.reload(config)

    def test_bundled_reads_service_config_and_ignores_env(self, monkeypatch, tmp_path: Path):
        from backend.schemas import ServiceConfig
        from backend.services import app_paths, service_config

        base = tmp_path / "AppSupport"
        monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", base)
        monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)

        # Ambient values that MUST be ignored.
        monkeypatch.setenv("HF_TOKEN", "env_token")
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "env_data"))
        monkeypatch.setenv("WHISPER_MODEL", "base")

        service_config.save(
            ServiceConfig(
                hf_token="cfg_token",
                whisper_model="large-v3",
                data_dir=str(base / "data"),
                models_dir=str(base / "models"),
            )
        )

        import config

        importlib.reload(config)

        assert config.is_bundled() is True
        assert config.HF_TOKEN == "cfg_token"
        assert config.current_hf_token() == "cfg_token"
        assert config.WHISPER_MODEL == "large-v3"
        assert config.DATA_DIR == base / "data"
        assert config.MODELS_DIR == base / "models"

    def test_current_hf_token_reflects_runtime_update(self, monkeypatch, tmp_path: Path):
        from backend.schemas import ServiceConfig
        from backend.services import app_paths, service_config

        base = tmp_path / "AppSupport"
        monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", base)
        monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)
        service_config.save(ServiceConfig(hf_token="", data_dir=str(base / "data"), models_dir=str(base / "models")))

        import config

        importlib.reload(config)
        assert config.current_hf_token() == ""

        # Provisioning writes a token at runtime — no restart required (BR-8).
        service_config.save(
            ServiceConfig(hf_token="new_token", data_dir=str(base / "data"), models_dir=str(base / "models"))
        )
        assert config.current_hf_token() == "new_token"
