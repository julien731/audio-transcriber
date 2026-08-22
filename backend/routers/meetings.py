from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import config
from backend.schemas import (
    AudioAnalysis,
    JobStatus,
    MeetingDetail,
    MeetingMetadata,
    MeetingStatus,
    MeetingSummary,
    MeetingType,
    MeetingUpdate,
    SegmentSpeakerUpdate,
    Transcript,
)
from backend.services import provisioning
from backend.services.job_queue import job_queue
from backend.services.transcriber import start_transcription
from config import MAX_UPLOAD_SIZE, MEETINGS_DIR

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".webm"}

MODELS_MISSING_DETAIL = (
    "Transcription models are not installed yet. First-time model download "
    "requires an internet connection — complete setup, then try again."
)


def _require_models_present() -> None:
    """Reject new transcriptions when required models are absent (BR-24, BR-25).

    Bundled-only: a dev checkout resolves its models lazily via WhisperX as
    before, so existing behavior and tests are unaffected (BR-18). Browsing and
    reading existing meetings is never gated (BR-23).
    """
    if not config.is_bundled():
        return
    if not provisioning.models_ready():
        raise HTTPException(status_code=503, detail=MODELS_MISSING_DETAIL)


# Language codes the uploader can select. Used to reject unknown/garbage codes
# from the multi-select before they reach the transcription pipeline.
SUPPORTED_LANGUAGES = {
    "en",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "nl",
    "ja",
    "zh",
    "ko",
    "ru",
    "th",
    "ar",
    "hi",
    "tr",
    "pl",
    "vi",
    "id",
}


def _sanitize_languages(codes: list[str]) -> list[str]:
    """Normalize, drop unknown codes, and de-duplicate while preserving order."""
    cleaned: list[str] = []
    for code in codes:
        normalized = code.strip().lower()
        if normalized in SUPPORTED_LANGUAGES and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _load_metadata(meeting_id: str) -> MeetingMetadata:
    meta_path = MEETINGS_DIR / meeting_id / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Meeting not found")
    with open(meta_path) as f:
        return MeetingMetadata(**json.load(f))


def _save_metadata(metadata: MeetingMetadata):
    meta_path = MEETINGS_DIR / metadata.id / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)


@router.get("/meetings", response_model=list[MeetingSummary])
async def list_meetings():
    meetings = []
    if not MEETINGS_DIR.exists():
        return meetings

    for meeting_dir in MEETINGS_DIR.iterdir():
        if not meeting_dir.is_dir():
            continue
        meta_path = meeting_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = MeetingMetadata(**json.load(f))
            meetings.append(
                MeetingSummary(
                    id=meta.id,
                    title=meta.title,
                    type=meta.type,
                    created_at=meta.created_at,
                    duration_seconds=meta.duration_seconds,
                    status=meta.status,
                )
            )
        except Exception:
            continue

    meetings.sort(key=lambda m: m.created_at, reverse=True)
    return meetings


@router.post("/meetings")
async def create_meeting(
    file: UploadFile = File(...),
    title: str = Form(""),
    meeting_type: str = Form("other"),
    expected_languages: list[str] = Form(default=[]),
    num_speakers: str = Form("auto"),
    preprocess_audio: str = Form("true"),
    audio_analysis_enabled: str = Form("false"),
    context: str = Form(""),
):
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 500MB)")

    # Gate before writing anything so a rejected upload leaves no partial meeting.
    _require_models_present()

    meeting_id = str(uuid.uuid4())
    meeting_dir = MEETINGS_DIR / meeting_id
    meeting_dir.mkdir(parents=True)

    # Save audio file with original extension
    audio_filename = f"audio{ext}"
    audio_path = meeting_dir / audio_filename
    audio_path.write_bytes(content)

    # Create job
    job = job_queue.create_job(meeting_id)

    # Create metadata
    effective_title = title.strip() or Path(file.filename).stem
    try:
        mt = MeetingType(meeting_type)
    except ValueError:
        mt = MeetingType.OTHER

    # Expected-language set drives single- vs multilingual routing (BR-2, BR-3).
    # Derive the single-language input: 0 selected → auto-detect; exactly 1 → that
    # language; 2+ → multilingual path (single-language input unused, kept as auto).
    sanitized_languages = _sanitize_languages(expected_languages)
    effective_language = sanitized_languages[0] if len(sanitized_languages) == 1 else "auto"
    effective_num_speakers = None
    if num_speakers.strip() != "auto":
        try:
            effective_num_speakers = int(num_speakers)
        except ValueError:
            pass

    effective_preprocess = preprocess_audio.lower() not in ("false", "0", "no")
    effective_audio_analysis = audio_analysis_enabled.lower() in ("true", "1", "yes", "on")

    metadata = MeetingMetadata(
        id=meeting_id,
        title=effective_title,
        type=mt,
        audio_filename=audio_filename,
        language=effective_language,
        expected_languages=sanitized_languages,
        num_speakers=effective_num_speakers,
        preprocess_audio=effective_preprocess,
        audio_analysis_enabled=effective_audio_analysis,
        status=MeetingStatus.PROCESSING,
        job_id=job.id,
        context=context.strip(),
    )
    _save_metadata(metadata)

    # Start transcription
    start_transcription(meeting_id, job.id)

    return {"meeting_id": meeting_id, "job_id": job.id}


@router.post("/meetings/{meeting_id}/cancel")
async def cancel_transcription(meeting_id: str):
    metadata = _load_metadata(meeting_id)

    if metadata.status != MeetingStatus.PROCESSING:
        raise HTTPException(status_code=409, detail="Meeting is not currently processing")

    error_message = "Transcription cancelled by user"

    metadata.status = MeetingStatus.ERROR
    metadata.error = error_message
    _save_metadata(metadata)

    if metadata.job_id:
        job_queue.update_job(metadata.job_id, status=JobStatus.FAILED, error=error_message)

    return {"ok": True}


@router.post("/meetings/{meeting_id}/retry")
async def retry_transcription(meeting_id: str):
    metadata = _load_metadata(meeting_id)

    _require_models_present()

    job = job_queue.create_job(meeting_id)
    metadata.status = MeetingStatus.PROCESSING
    metadata.error = None
    metadata.audio_analysis_status = None
    metadata.job_id = job.id
    _save_metadata(metadata)

    start_transcription(meeting_id, job.id)

    return {"meeting_id": meeting_id, "job_id": job.id}


@router.get("/meetings/{meeting_id}", response_model=MeetingDetail)
async def get_meeting(meeting_id: str):
    metadata = _load_metadata(meeting_id)

    transcript = None
    transcript_path = MEETINGS_DIR / meeting_id / "transcript.json"
    if transcript_path.exists():
        with open(transcript_path) as f:
            transcript = Transcript(**json.load(f))

    audio_analysis = None
    audio_analysis_path = MEETINGS_DIR / meeting_id / "audio_analysis.json"
    if audio_analysis_path.exists():
        with open(audio_analysis_path) as f:
            audio_analysis = AudioAnalysis(**json.load(f))

    return MeetingDetail(metadata=metadata, transcript=transcript, audio_analysis=audio_analysis)


@router.patch("/meetings/{meeting_id}", response_model=MeetingMetadata)
async def update_meeting(meeting_id: str, update: MeetingUpdate):
    metadata = _load_metadata(meeting_id)

    if update.title is not None:
        metadata.title = update.title
    if update.type is not None:
        metadata.type = update.type
    if update.speakers is not None:
        metadata.speakers = update.speakers
    if update.context is not None:
        metadata.context = update.context.strip()

    _save_metadata(metadata)
    return metadata


@router.patch("/meetings/{meeting_id}/segments/speaker")
async def update_segment_speaker(meeting_id: str, update: SegmentSpeakerUpdate):
    """Rename the speaker for a single transcript segment."""
    metadata = _load_metadata(meeting_id)

    transcript_path = MEETINGS_DIR / meeting_id / "transcript.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    transcript = Transcript(**json.loads(transcript_path.read_text()))

    # Find the segment
    segment = next((s for s in transcript.segments if s.id == update.segment_id), None)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    normalized = update.speaker_name.strip().casefold()
    existing_id = next(
        (sid for sid, name in metadata.speakers.items() if name.strip().casefold() == normalized),
        None,
    )

    if existing_id is not None:
        segment.speaker = existing_id
        transcript_path.write_text(json.dumps(transcript.model_dump(), ensure_ascii=False, indent=2))
        return {"ok": True}

    new_speaker_id = f"{segment.speaker}_seg_{update.segment_id}"
    segment.speaker = new_speaker_id
    transcript_path.write_text(json.dumps(transcript.model_dump(), ensure_ascii=False, indent=2))

    metadata.speakers[new_speaker_id] = update.speaker_name
    _save_metadata(metadata)

    return {"ok": True}


@router.delete("/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    meeting_dir = MEETINGS_DIR / meeting_id
    if not meeting_dir.exists():
        raise HTTPException(status_code=404, detail="Meeting not found")
    shutil.rmtree(meeting_dir)
    return {"ok": True}


@router.get("/meetings/{meeting_id}/audio")
async def stream_audio(meeting_id: str):
    metadata = _load_metadata(meeting_id)
    audio_path = MEETINGS_DIR / meeting_id / metadata.audio_filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    ext = audio_path.suffix.lower()
    media_types = {
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(str(audio_path), media_type=media_type)
