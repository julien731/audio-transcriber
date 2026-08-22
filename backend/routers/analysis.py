from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    AnalysisPromptResponse,
    AudioAnalysis,
    AudioAnalysisStatus,
    MeetingMetadata,
    Transcript,
)
from backend.services import analysis_context, analysis_prompt
from config import MEETINGS_DIR, TEMPLATES_DIR

router = APIRouter()

TEMPLATE_FILES = {
    "interview": "interview_analysis.md",
    "sales": "sales_meeting_analysis.md",
    "client": "client_meeting_analysis.md",
    "other": "other.md",
    "prototype": "prototype_scope.md",
}


@router.get("/templates/{template_type}")
async def get_template(template_type: str):
    filename = TEMPLATE_FILES.get(template_type)
    if not filename:
        raise HTTPException(status_code=404, detail="Template not found")

    template_path = TEMPLATES_DIR / filename
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template file not found")

    return {"template": template_path.read_text(encoding="utf-8")}


def _load_transcript(meeting_dir: Path) -> Transcript | None:
    transcript_path = meeting_dir / "transcript.json"
    if transcript_path.exists():
        return Transcript(**json.loads(transcript_path.read_text(encoding="utf-8")))
    return None


def _render_audio_context(meeting_dir: Path, metadata: MeetingMetadata) -> str:
    """Render the Audio Analysis Context markdown for a meeting.

    Empty string when the meeting is opted out of audio analysis, so the prompt
    stays byte-identical to the pre-feature output (BR-4.4); a brief
    unavailability note when analysis was attempted but produced nothing (BR-4.5).
    """
    if not metadata.audio_analysis_enabled:
        return ""

    audio_analysis_path = meeting_dir / "audio_analysis.json"
    if audio_analysis_path.exists():
        audio_analysis = AudioAnalysis(**json.loads(audio_analysis_path.read_text(encoding="utf-8")))
    else:
        audio_analysis = AudioAnalysis(
            status=AudioAnalysisStatus.UNAVAILABLE,
            reason="audio analysis has not produced output for this meeting",
        )

    transcript = _load_transcript(meeting_dir)
    return analysis_context.render(audio_analysis, transcript, speakers=metadata.speakers)


@router.get("/meetings/{meeting_id}/analysis-context")
async def get_analysis_context(meeting_id: str):
    """Return the rendered Audio Analysis Context markdown for a meeting."""
    meeting_dir = MEETINGS_DIR / meeting_id
    metadata_path = meeting_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Meeting not found")

    metadata = MeetingMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
    return {"context": _render_audio_context(meeting_dir, metadata)}


@router.get("/meetings/{meeting_id}/analysis-prompt", response_model=AnalysisPromptResponse)
async def get_analysis_prompt(meeting_id: str, template_type: str, meeting_context: str | None = None):
    """Return a fully-assembled analysis prompt for a meeting (BR-16, BR-17).

    Template selection, audio-analysis context, meeting context, and the
    transcript are substituted server-side. ``meeting_context`` overrides the
    saved meeting context when provided (the web client passes its live textarea
    value). Responds 409 when the transcript is not ready yet (EC-11).
    """
    filename = TEMPLATE_FILES.get(template_type)
    if not filename:
        raise HTTPException(status_code=404, detail="Template not found")
    template_path = TEMPLATES_DIR / filename
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template file not found")

    meeting_dir = MEETINGS_DIR / meeting_id
    metadata_path = meeting_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Meeting not found")
    metadata = MeetingMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))

    transcript = _load_transcript(meeting_dir)
    if transcript is None:
        raise HTTPException(status_code=409, detail="Transcript is not ready yet")

    template = template_path.read_text(encoding="utf-8")
    audio_context = _render_audio_context(meeting_dir, metadata)
    context = (meeting_context if meeting_context is not None else metadata.context or "").strip()
    transcript_text = analysis_prompt.build_transcript_text(transcript, metadata.speakers)

    prompt = analysis_prompt.assemble_prompt(template, audio_context, context, transcript_text)
    return AnalysisPromptResponse(prompt=prompt)
