from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def base_metadata() -> dict:
    return json.loads((FIXTURES_DIR / "metadata.json").read_text())


@pytest.fixture
def base_transcript() -> dict:
    return json.loads((FIXTURES_DIR / "transcript.json").read_text())


def _write_meeting(meetings_dir, sample_audio, metadata, transcript=None) -> str:
    meeting_id = metadata["id"]
    meeting_dir = meetings_dir / meeting_id
    meeting_dir.mkdir()
    (meeting_dir / "metadata.json").write_text(json.dumps(metadata))
    if transcript is not None:
        (meeting_dir / "transcript.json").write_text(json.dumps(transcript))
    shutil.copy(sample_audio, meeting_dir / metadata["audio_filename"])
    return meeting_id


class TestGetAnalysisPrompt:
    async def test_returns_assembled_prompt(self, client, meetings_dir, sample_audio, base_metadata, base_transcript):
        meeting_id = _write_meeting(meetings_dir, sample_audio, base_metadata, base_transcript)

        res = await client.get(f"/api/meetings/{meeting_id}/analysis-prompt", params={"template_type": "interview"})
        assert res.status_code == 200
        prompt = res.json()["prompt"]
        # Transcript substituted with speaker display names from fixture metadata.
        assert "[PASTE TRANSCRIPT HERE]" not in prompt
        assert "Alice:" in prompt or "Bob:" in prompt

    async def test_409_when_transcript_not_ready(self, client, meetings_dir, sample_audio, base_metadata):
        # No transcript.json written (EC-11).
        meeting_id = _write_meeting(meetings_dir, sample_audio, base_metadata, transcript=None)
        res = await client.get(f"/api/meetings/{meeting_id}/analysis-prompt", params={"template_type": "interview"})
        assert res.status_code == 409

    async def test_404_for_missing_meeting(self, client):
        res = await client.get("/api/meetings/missing/analysis-prompt", params={"template_type": "interview"})
        assert res.status_code == 404

    async def test_404_for_unknown_template(self, client, meetings_dir, sample_audio, base_metadata, base_transcript):
        meeting_id = _write_meeting(meetings_dir, sample_audio, base_metadata, base_transcript)
        res = await client.get(f"/api/meetings/{meeting_id}/analysis-prompt", params={"template_type": "nope"})
        assert res.status_code == 404

    async def test_meeting_context_override_is_used(
        self, client, meetings_dir, sample_audio, base_metadata, base_transcript
    ):
        meeting_id = _write_meeting(meetings_dir, sample_audio, base_metadata, base_transcript)
        res = await client.get(
            f"/api/meetings/{meeting_id}/analysis-prompt",
            params={"template_type": "interview", "meeting_context": "Live notes from client"},
        )
        assert res.status_code == 200
        prompt = res.json()["prompt"]
        assert "## Meeting Context" in prompt
        assert "Live notes from client" in prompt

    async def test_matches_client_substitution_semantics(
        self, client, meetings_dir, sample_audio, base_metadata, base_transcript
    ):
        """Server output equals the legacy client assembly for the same inputs
        (BR-18): raw template with placeholders replaced once, transcript lines
        formatted as [HH:MM:SS] Name: text."""
        from backend.schemas import Transcript
        from backend.services import analysis_prompt
        from config import TEMPLATES_DIR

        meeting_id = _write_meeting(meetings_dir, sample_audio, base_metadata, base_transcript)
        res = await client.get(f"/api/meetings/{meeting_id}/analysis-prompt", params={"template_type": "other"})
        server_prompt = res.json()["prompt"]

        template = (TEMPLATES_DIR / "other.md").read_text(encoding="utf-8")
        transcript_text = analysis_prompt.build_transcript_text(
            Transcript(**base_transcript), base_metadata["speakers"]
        )
        context = (base_metadata.get("context") or "").strip()
        expected = analysis_prompt.assemble_prompt(template, "", context, transcript_text)
        assert server_prompt == expected
