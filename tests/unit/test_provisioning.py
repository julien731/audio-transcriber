from __future__ import annotations

import pytest

from backend.schemas import DownloadState
from backend.services import provisioning, service_config


@pytest.fixture(autouse=True)
def _reset_state():
    provisioning._set_state(state=DownloadState.IDLE, progress=0, error=None)
    yield
    provisioning._set_state(state=DownloadState.IDLE, progress=0, error=None)


class TestSetToken:
    def test_stores_token_and_reports_diarization_available(self):
        status = provisioning.set_token("hf_secret")
        assert status.diarization_available is True
        assert service_config.load().hf_token == "hf_secret"

    def test_empty_token_disables_diarization(self):
        status = provisioning.set_token("")
        assert status.diarization_available is False


class TestRequiredRepos:
    def test_without_token_only_whisper(self):
        provisioning.set_token("")
        repos = provisioning.required_repos()
        assert repos == ["Systran/faster-whisper-large-v3"]

    def test_with_token_includes_diarization(self):
        provisioning.set_token("hf_secret")
        repos = provisioning.required_repos()
        assert "pyannote/speaker-diarization-3.1" in repos
        assert "pyannote/segmentation-3.0" in repos

    def test_with_token_includes_embedding_model(self):
        """speaker-diarization-3.1 loads this embedding model at runtime; it must
        be pre-fetched so diarization never downloads mid-transcription."""
        provisioning.set_token("hf_secret")
        assert "pyannote/wespeaker-voxceleb-resnet34-LM" in provisioning.required_repos()


class TestDownload:
    def test_success_marks_provisioning_completed(self, monkeypatch):
        provisioning.set_token("")
        calls = []
        monkeypatch.setattr(provisioning, "_snapshot_downloader", lambda: calls.append)

        provisioning._run_download()

        assert calls == ["Systran/faster-whisper-large-v3"]
        status = provisioning.status()
        assert status.download_state == DownloadState.COMPLETED
        assert status.download_progress == 100
        assert status.provisioning_completed is True
        assert provisioning.models_ready() is True

    def test_failure_does_not_mark_completed(self, monkeypatch):
        """Offline/partial download (EC-4) must not read as complete."""
        provisioning.set_token("")

        def _boom():
            def _dl(repo):
                raise ConnectionError("offline")

            return _dl

        monkeypatch.setattr(provisioning, "_snapshot_downloader", _boom)

        provisioning._run_download()

        status = provisioning.status()
        assert status.download_state == DownloadState.FAILED
        assert "offline" in status.download_error
        assert status.provisioning_completed is False
        assert provisioning.models_ready() is False

    def test_start_download_runs_in_background(self, monkeypatch):
        provisioning.set_token("")
        monkeypatch.setattr(provisioning, "_snapshot_downloader", lambda: lambda repo: None)

        provisioning.start_download()
        thread = provisioning._thread
        assert thread is not None
        thread.join(timeout=5)

        assert provisioning.models_ready() is True


class TestModelsReadyDefault:
    def test_false_before_provisioning(self):
        assert provisioning.models_ready() is False


class TestProvisioningVersionGate:
    def test_completed_at_old_version_reads_not_ready(self):
        """An install provisioned before a required repo was added must re-provision."""
        cfg = service_config.load()
        cfg.provisioning_completed = True
        cfg.provisioning_version = provisioning.PROVISIONING_VERSION - 1
        service_config.save(cfg)

        assert provisioning.models_ready() is False

    def test_run_download_stamps_current_version(self, monkeypatch):
        provisioning.set_token("")
        monkeypatch.setattr(provisioning, "_snapshot_downloader", lambda: lambda repo: None)

        provisioning._run_download()

        cfg = service_config.load()
        assert cfg.provisioning_version == provisioning.PROVISIONING_VERSION
        assert provisioning.models_ready() is True

    def test_status_reports_not_completed_at_old_version(self):
        """/api/provisioning must agree with the upload gate: an install completed
        at an older schema version reads as not completed so the client re-runs setup."""
        cfg = service_config.load()
        cfg.provisioning_completed = True
        cfg.provisioning_version = provisioning.PROVISIONING_VERSION - 1
        service_config.save(cfg)

        st = provisioning.status()
        assert st.provisioning_completed is False
        assert st.models_present is False
