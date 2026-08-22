from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from backend.schemas import ServiceConfig
from backend.services import app_paths, service_config


@pytest.fixture
def app_support(tmp_path: Path, monkeypatch):
    base = tmp_path / "AppSupport"
    monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", base)
    return base


class TestLoad:
    def test_missing_file_returns_defaults(self, app_support):
        cfg = service_config.load()
        assert cfg.whisper_model == "large-v3"
        assert cfg.hf_token == ""
        assert cfg.provisioning_completed is False
        assert cfg.imported_from is None

    def test_default_paths_resolved_under_app_support(self, app_support):
        cfg = service_config.load()
        assert cfg.data_dir == str(app_support / "data")
        assert cfg.models_dir == str(app_support / "models")

    def test_roundtrip_preserves_values(self, app_support):
        original = ServiceConfig(
            hf_token="hf_secret",
            whisper_model="large-v3",
            provisioning_completed=True,
            imported_from="/old/data",
        )
        service_config.save(original)
        loaded = service_config.load()
        assert loaded.hf_token == "hf_secret"
        assert loaded.provisioning_completed is True
        assert loaded.imported_from == "/old/data"


class TestSave:
    def test_writes_owner_only_permissions(self, app_support):
        service_config.save(ServiceConfig(hf_token="hf_secret"))
        mode = stat.S_IMODE(os.stat(service_config.config_path()).st_mode)
        assert mode == 0o600

    def test_atomic_no_tmp_left_behind(self, app_support):
        service_config.save(ServiceConfig())
        leftovers = list(app_support.glob("*.tmp"))
        assert leftovers == []
