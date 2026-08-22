from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services import app_paths, data_import, service_config


def _write_meeting(data_dir: Path, meeting_id: str, title: str = "Old meeting") -> None:
    mdir = data_dir / "meetings" / meeting_id
    mdir.mkdir(parents=True)
    (mdir / "metadata.json").write_text(json.dumps({"id": meeting_id, "title": title}))
    (mdir / "transcript.json").write_text(json.dumps({"segments": [], "language": "en"}))
    (mdir / "audio.wav").write_bytes(b"RIFF")


class TestFindLegacyDataDir:
    def test_returns_dir_with_meetings(self, tmp_path: Path):
        legacy = tmp_path / "old" / "data"
        _write_meeting(legacy, "m1")
        assert data_import.find_legacy_data_dir([legacy]) == legacy

    def test_returns_none_when_no_meetings(self, tmp_path: Path):
        empty = tmp_path / "old" / "data"
        (empty / "meetings").mkdir(parents=True)
        assert data_import.find_legacy_data_dir([empty]) is None

    def test_excludes_destination(self, tmp_path: Path):
        dest = tmp_path / "new" / "data"
        _write_meeting(dest, "m1")
        assert data_import.find_legacy_data_dir([dest], exclude=dest) is None


class TestImportMeetings:
    def test_copies_meetings_non_destructively(self, tmp_path: Path):
        src = tmp_path / "old" / "data"
        dest = tmp_path / "new" / "data"
        _write_meeting(src, "m1")
        _write_meeting(src, "m2")

        count = data_import.import_meetings(src, dest)

        assert count == 2
        assert (dest / "meetings" / "m1" / "metadata.json").exists()
        # Original left intact (BR-14)
        assert (src / "meetings" / "m1" / "metadata.json").exists()

    def test_does_not_overwrite_existing(self, tmp_path: Path):
        src = tmp_path / "old" / "data"
        dest = tmp_path / "new" / "data"
        _write_meeting(src, "m1", title="from source")
        _write_meeting(dest, "m1", title="existing")

        count = data_import.import_meetings(src, dest)

        assert count == 0
        existing = json.loads((dest / "meetings" / "m1" / "metadata.json").read_text())
        assert existing["title"] == "existing"


class TestRunFirstRunImport:
    @pytest.fixture
    def bundled(self, tmp_path: Path, monkeypatch):
        base = tmp_path / "AppSupport"
        monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", base)
        monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)
        return base

    def test_noop_in_dev_mode(self, tmp_path: Path, monkeypatch):
        base = tmp_path / "AppSupport"
        monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", base)
        monkeypatch.delattr(app_paths.sys, "frozen", raising=False)
        # Should not raise or write config in dev mode.
        data_import.run_first_run_import()
        assert not service_config.config_path().exists()

    def test_imports_and_records_source(self, bundled: Path, tmp_path: Path, monkeypatch):
        legacy = tmp_path / "old" / "data"
        _write_meeting(legacy, "m1")
        monkeypatch.setattr(data_import, "legacy_candidates", lambda: [legacy])

        data_import.run_first_run_import()

        cfg = service_config.load()
        assert cfg.imported_from == str(legacy)
        assert (Path(cfg.data_dir) / "meetings" / "m1" / "metadata.json").exists()

    def test_empty_first_run_no_error(self, bundled: Path, monkeypatch):
        monkeypatch.setattr(data_import, "legacy_candidates", lambda: [])
        data_import.run_first_run_import()  # BR-15 / EC-7: no error
        cfg = service_config.load()
        assert cfg.imported_from is None

    def test_idempotent_skips_second_run(self, bundled: Path, tmp_path: Path, monkeypatch):
        legacy = tmp_path / "old" / "data"
        _write_meeting(legacy, "m1")
        monkeypatch.setattr(data_import, "legacy_candidates", lambda: [legacy])

        data_import.run_first_run_import()
        # Add a new meeting to source; a second run must NOT re-import.
        _write_meeting(legacy, "m2")
        data_import.run_first_run_import()

        cfg = service_config.load()
        assert not (Path(cfg.data_dir) / "meetings" / "m2").exists()
