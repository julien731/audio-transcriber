from __future__ import annotations

from pathlib import Path

import pytest

from backend.services import app_paths


@pytest.fixture
def app_support(tmp_path: Path, monkeypatch):
    base = tmp_path / "AppSupport"
    monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", base)
    return base


class TestBundleDetection:
    def test_not_bundled_by_default(self, monkeypatch):
        monkeypatch.delattr(app_paths.sys, "frozen", raising=False)
        assert app_paths.is_bundled() is False

    def test_bundled_when_frozen(self, monkeypatch):
        monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)
        assert app_paths.is_bundled() is True

    def test_bundle_dir_none_in_dev(self, monkeypatch):
        monkeypatch.delattr(app_paths.sys, "_MEIPASS", raising=False)
        assert app_paths.bundle_dir() is None

    def test_bundle_dir_uses_meipass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app_paths.sys, "_MEIPASS", str(tmp_path), raising=False)
        assert app_paths.bundle_dir() == tmp_path
        assert app_paths.bundled_bin_dir() == tmp_path / "bin"


class TestAppSupportDir:
    def test_override_is_created(self, app_support):
        result = app_paths.app_support_dir()
        assert result == app_support
        assert result.exists()


class TestFfmpegPath:
    def test_prefers_bundled_binary(self, monkeypatch, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "ffmpeg").write_text("#!/bin/sh\n")
        monkeypatch.setattr(app_paths.sys, "_MEIPASS", str(tmp_path), raising=False)
        assert app_paths.ffmpeg_path() == str(bin_dir / "ffmpeg")

    def test_falls_back_to_path_in_dev(self, monkeypatch):
        monkeypatch.delattr(app_paths.sys, "_MEIPASS", raising=False)
        monkeypatch.setattr(app_paths.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        assert app_paths.ffmpeg_path() == "/usr/bin/ffmpeg"

    def test_returns_none_when_missing(self, monkeypatch):
        monkeypatch.delattr(app_paths.sys, "_MEIPASS", raising=False)
        monkeypatch.setattr(app_paths.shutil, "which", lambda name: None)
        assert app_paths.ffmpeg_path() is None
