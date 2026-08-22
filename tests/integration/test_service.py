from __future__ import annotations

from backend.services import provisioning, service_config


class TestHealth:
    async def test_health_reports_ready(self, client, monkeypatch):
        import config

        monkeypatch.setattr(config, "current_hf_token", lambda: "")
        res = await client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ready"
        assert body["provisioning_completed"] is False
        assert body["diarization_available"] is False


class TestProvisioningEndpoints:
    async def test_get_provisioning_status(self, client):
        res = await client.get("/api/provisioning")
        assert res.status_code == 200
        body = res.json()
        assert body["provisioning_completed"] is False
        assert body["whisper_model"] == "large-v3"
        assert body["download_state"] == "idle"

    async def test_post_token_persists_and_enables_diarization(self, client):
        res = await client.post("/api/provisioning/token", json={"hf_token": "hf_secret"})
        assert res.status_code == 200
        assert res.json()["diarization_available"] is True
        assert service_config.load().hf_token == "hf_secret"

    async def test_post_empty_token_disables_diarization(self, client):
        res = await client.post("/api/provisioning/token", json={"hf_token": ""})
        assert res.status_code == 200
        assert res.json()["diarization_available"] is False

    async def test_post_models_starts_download(self, client, monkeypatch):
        monkeypatch.setattr(provisioning, "_snapshot_downloader", lambda: lambda repo: None)

        res = await client.post("/api/provisioning/models")
        assert res.status_code == 200

        if provisioning._thread is not None:
            provisioning._thread.join(timeout=5)

        health = await client.get("/api/health")
        assert health.json()["provisioning_completed"] is True
