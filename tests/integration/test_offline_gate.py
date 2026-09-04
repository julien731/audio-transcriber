from __future__ import annotations

from pathlib import Path

import pytest

from backend.services import provisioning, service_config


def _upload(sample_audio: Path):
    return {"file": ("sample.wav", sample_audio.read_bytes(), "audio/wav")}


class TestOfflineGateDevMode:
    """In dev mode the gate is a no-op; existing upload/retry behavior stands
    (BR-18). start_transcription is patched so no real ML runs."""

    async def test_create_succeeds_without_models(self, client, sample_audio, monkeypatch):
        monkeypatch.setattr("backend.routers.meetings.start_transcription", lambda *a, **k: None)
        res = await client.post("/api/meetings", files=_upload(sample_audio))
        assert res.status_code == 200
        assert "meeting_id" in res.json()


class TestOfflineGateBundled:
    """In bundled mode, new transcriptions are rejected until models are present
    (BR-24, BR-25, EC-5). Existing meetings remain readable (BR-23)."""

    @pytest.fixture(autouse=True)
    def _bundled(self, monkeypatch):
        import config

        monkeypatch.setattr(config, "is_bundled", lambda: True)
        monkeypatch.setattr("backend.routers.meetings.start_transcription", lambda *a, **k: None)

    async def test_create_rejected_when_models_missing(self, client, sample_audio):
        assert provisioning.models_ready() is False
        res = await client.post("/api/meetings", files=_upload(sample_audio))
        assert res.status_code == 503
        assert "internet connection" in res.json()["detail"]

    async def test_retry_rejected_when_models_missing(self, client, populated_meeting):
        res = await client.post(f"/api/meetings/{populated_meeting}/retry")
        assert res.status_code == 503

    async def test_reading_existing_meeting_still_works(self, client, populated_meeting):
        res = await client.get(f"/api/meetings/{populated_meeting}")
        assert res.status_code == 200

    async def test_create_allowed_once_provisioned(self, client, sample_audio):
        cfg = service_config.load()
        cfg.provisioning_completed = True
        cfg.provisioning_version = provisioning.PROVISIONING_VERSION
        service_config.save(cfg)

        res = await client.post("/api/meetings", files=_upload(sample_audio))
        assert res.status_code == 200

    async def test_create_rejected_when_provisioned_at_old_version(self, client, sample_audio):
        """An install completed before a required repo was added must re-provision."""
        cfg = service_config.load()
        cfg.provisioning_completed = True
        cfg.provisioning_version = provisioning.PROVISIONING_VERSION - 1
        service_config.save(cfg)

        res = await client.post("/api/meetings", files=_upload(sample_audio))
        assert res.status_code == 503
