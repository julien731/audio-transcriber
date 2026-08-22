from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.schemas import ServiceConfig
from backend.services import app_paths, service_config, service_runtime


@pytest.fixture
def app_support(tmp_path: Path, monkeypatch):
    base = tmp_path / "AppSupport"
    monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", base)
    return base


class TestBootstrap:
    def test_noop_in_dev_mode(self, app_support, monkeypatch):
        monkeypatch.delattr(app_paths.sys, "frozen", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        service_runtime.bootstrap()
        assert "HF_HOME" not in os.environ

    def test_bundled_sets_hf_home_to_models_dir(self, app_support, monkeypatch):
        monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)
        service_config.save(ServiceConfig(models_dir=str(app_support / "models"), data_dir=str(app_support / "data")))
        service_runtime.bootstrap()
        assert os.environ["HF_HOME"] == str(app_support / "models")

    def test_bundled_prepends_bundled_bin_to_path(self, app_support, monkeypatch, tmp_path):
        bin_dir = tmp_path / "bundle" / "bin"
        bin_dir.mkdir(parents=True)
        monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)
        monkeypatch.setattr(app_paths.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
        monkeypatch.setenv("PATH", "/usr/bin")
        service_config.save(ServiceConfig(models_dir=str(app_support / "models"), data_dir=str(app_support / "data")))
        service_runtime.bootstrap()
        assert os.environ["PATH"].startswith(f"{bin_dir}{os.pathsep}")
