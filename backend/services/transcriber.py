from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import defaultdict
from pathlib import Path

import config
from backend.schemas import (
    AudioAnalysis,
    AudioAnalysisStatus,
    EmotionUnavailable,
    JobStatus,
    MeetingMetadata,
    MeetingStatus,
    TranscriptSegment,
)
from backend.services.align_models import HF_ALIGN_REPOS
from backend.services.job_queue import job_queue
from backend.services.multilingual_transcriber import transcribe_multilingual
from config import MEETINGS_DIR, WHISPER_BATCH_SIZE, WHISPER_DEVICE, WHISPER_MODEL

logger = logging.getLogger(__name__)

# Wall-clock ceilings for stages that load a model (and may download it) before
# computing. They convert a wedged or stalled stage — an interrupted model
# download, a slow-mirror fetch, or a deadlocked pipeline — into a graceful
# degrade instead of an indefinite hang (debug session macos-transcribe-stuck-70).
DIARIZATION_TIMEOUT_SEC = 30 * 60
ALIGNMENT_TIMEOUT_SEC = 30 * 60


def _call_with_timeout(fn, timeout_sec, label):
    """Run ``fn()`` under a wall-clock watchdog in a daemon thread.

    Returns ``("ok", value)``, ``("timeout", None)``, or ``("error", exc)``. On
    timeout the worker is abandoned rather than interrupted — a running native
    call (torch/pyannote) cannot be cancelled cleanly — so the caller degrades
    and moves on instead of blocking the meeting forever.
    """
    box: dict = {}

    def _worker() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - surfaced via return value
            box["error"] = exc

    worker = threading.Thread(target=_worker, name=label, daemon=True)
    worker.start()
    worker.join(timeout=timeout_sec)

    if worker.is_alive():
        return "timeout", None
    if "error" in box:
        return "error", box["error"]
    return "ok", box.get("value")


# Languages with a wav2vec2 alignment model available in WhisperX (single-language path).
ALIGNMENT_LANGUAGES = {
    "en",
    "fr",
    "de",
    "es",
    "it",
    "ja",
    "zh",
    "nl",
    "uk",
    "pt",
    "ar",
    "cs",
    "ru",
    "pl",
    "hu",
    "fi",
    "fa",
    "el",
    "tr",
    "da",
    "he",
    "vi",
    "ko",
    "ur",
    "te",
    "hi",
    "ta",
    "id",
    "ms",
    "th",
}

# Alignment repos are resolved from the shared align_models.HF_ALIGN_REPOS mapping
# (imported above) so the repo provisioning pre-fetches is exactly the one loaded
# here. Torch-native and unlisted languages resolve to None via ``.get()``, letting
# WhisperX pick its own default (still watchdogged at the align stage).


def _diarize_and_assign(
    job_id: str,
    audio,
    num_speakers: int | None,
    device: str,
    result: dict,
    *,
    progress: int = 70,
) -> tuple[dict, list[tuple[float, float, str]] | None]:
    """Run PyAnnote diarization and assign speakers to the transcript result.

    Shared by the single-language and multilingual pipelines. No-op (returns the
    result unchanged with no turns) when HF_TOKEN is unset. Returns
    ``(result_with_speakers, diarize_turns)`` where ``diarize_turns`` are the raw
    PyAnnote turns (with overlaps) required by interaction analysis (BR-3.1).
    """
    hf_token = config.current_hf_token()
    if not hf_token:
        return result, None

    import torch
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    job_queue.update_job(job_id, stage="diarizing", progress=progress)

    def _run_diarization():
        diarize_model = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-3.1",
            token=hf_token,
            device=device,
        )
        diarize_kwargs = {}
        if num_speakers:
            diarize_kwargs["num_speakers"] = num_speakers
        return diarize_model(audio, **diarize_kwargs)

    status, value = _call_with_timeout(_run_diarization, DIARIZATION_TIMEOUT_SEC, "diarization")
    if status == "timeout":
        logger.error("Diarization exceeded %ss; continuing without speaker labels", DIARIZATION_TIMEOUT_SEC)
        return result, None
    if status == "error":
        # A token accepted at provisioning time may later be revoked or be
        # unverifiable (e.g. offline). Degrade to no-diarization rather than
        # failing the whole meeting (BR-10, EC-3).
        logger.error("Diarization failed; continuing without speaker labels", exc_info=value)
        return result, None

    diarize_segments = value

    # Capture raw PyAnnote turns (with overlaps) before assign_word_speakers
    # collapses them into per-speaker non-overlapping transcript segments.
    # Required by interaction analysis (BR-3.1).
    try:
        diarize_turns = [
            (float(row["start"]), float(row["end"]), str(row["speaker"])) for _, row in diarize_segments.iterrows()
        ]
    except Exception:
        logger.exception("Failed to capture raw diarization turns; interaction analysis will be unavailable")
        diarize_turns = None
    result = assign_word_speakers(diarize_segments, result)

    if device == "cuda":
        torch.cuda.empty_cache()

    return result, diarize_turns


def _get_device() -> str:
    if WHISPER_DEVICE != "auto":
        return WHISPER_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _is_cancelled(metadata_path: Path) -> bool:
    """Check if the meeting has been cancelled (status is no longer PROCESSING)."""
    try:
        with open(metadata_path) as f:
            meta = json.load(f)
        return meta.get("status") != MeetingStatus.PROCESSING.value
    except Exception:
        return False


def _run_emotion_analysis(
    job_id: str,
    audio,
    segment_models: list[TranscriptSegment],
    detected_language: str,
) -> tuple[AudioAnalysisStatus, str | None, list, list[EmotionUnavailable]]:
    """Run SER per segment, gated on each segment's language (BR-10).

    English segments are analyzed; segments in a language the emotion model does
    not support are marked unavailable. The meeting is no longer skipped wholesale
    when it merely contains some non-English speech (EC-6). A segment with no
    per-segment language (single-language path, pre-feature transcripts) falls
    back to the meeting's detected language (EC-8).

    Returns (status, reason, annotations, unavailable_markers).
    """
    from backend.services.emotion_analyzer import SUPPORTED_LANGUAGES, analyze_segments

    supported: list[TranscriptSegment] = []
    unavailable: list[EmotionUnavailable] = []
    for segment in segment_models:
        language = segment.language or detected_language
        if language in SUPPORTED_LANGUAGES:
            supported.append(segment)
        else:
            unavailable.append(EmotionUnavailable(segment_id=segment.id, reason=f"language_not_supported:{language}"))

    if not supported:
        languages = ",".join(sorted({m.reason.split(":", 1)[1] for m in unavailable}))
        logger.info("Skipping emotion analysis: no emotion-supported segments (languages: %s)", languages)
        return AudioAnalysisStatus.UNAVAILABLE, f"language_not_supported:{languages}", [], unavailable

    try:
        job_queue.update_job(job_id, stage="emotion_analysis", progress=92)

        annotations = analyze_segments(audio_path=None, segments=supported, audio_array=audio)
        return AudioAnalysisStatus.COMPLETED, None, annotations, unavailable
    except Exception as e:
        logger.exception("Emotion analysis failed")
        return AudioAnalysisStatus.FAILED, str(e), [], unavailable


def _run_prosody_analysis(
    job_id: str,
    audio,
    segment_models: list[TranscriptSegment],
) -> tuple[AudioAnalysisStatus, str | None, list, list]:
    """Run prosody extraction (language-agnostic, BR-2.1).

    Returns (status, reason, annotations, unavailable_markers).
    """
    try:
        job_queue.update_job(job_id, stage="prosody_extraction", progress=95)

        from backend.services.prosody_analyzer import analyze_segments as analyze_prosody

        annotations, unavailable = analyze_prosody(audio_array=audio, segments=segment_models)
        return AudioAnalysisStatus.COMPLETED, None, annotations, unavailable
    except Exception as e:
        logger.exception("Prosody extraction failed")
        return AudioAnalysisStatus.FAILED, str(e), [], []


def _run_interaction_analysis(
    job_id: str,
    segment_models: list[TranscriptSegment],
    diarize_turns: list[tuple[float, float, str]] | None,
) -> tuple[AudioAnalysisStatus, str | None, list, list, bool]:
    """Run interaction pattern detection (language-agnostic).

    Returns (status, reason, events, segment_interactions, dominant_speaker_limitation).
    Without raw PyAnnote turns (no diarization), the stage is unavailable.
    """
    if not diarize_turns:
        logger.info("Skipping interaction analysis: no diarization turns available")
        return AudioAnalysisStatus.UNAVAILABLE, "diarization_unavailable", [], [], False

    try:
        job_queue.update_job(job_id, stage="interaction_analysis", progress=97)

        from backend.services.interaction_analyzer import analyze as analyze_interactions

        result = analyze_interactions(transcript_segments=segment_models, diarize_turns=diarize_turns)
        return (
            AudioAnalysisStatus.COMPLETED,
            None,
            result.events,
            result.segment_interactions,
            result.dominant_speaker_limitation,
        )
    except Exception as e:
        logger.exception("Interaction analysis failed")
        return AudioAnalysisStatus.FAILED, str(e), [], [], False


def _roll_up_status(*statuses: AudioAnalysisStatus) -> AudioAnalysisStatus:
    """Combine per-stage statuses into the pipeline's overall status.

    COMPLETED if any stage produced output. FAILED only when every applicable
    stage failed. UNAVAILABLE only when every stage was skipped.
    """
    if any(s == AudioAnalysisStatus.COMPLETED for s in statuses):
        return AudioAnalysisStatus.COMPLETED
    if any(s == AudioAnalysisStatus.FAILED for s in statuses):
        return AudioAnalysisStatus.FAILED
    return AudioAnalysisStatus.UNAVAILABLE


def _run_audio_analysis(
    job_id: str,
    audio,
    segments: list[dict],
    detected_language: str,
    diarize_turns: list[tuple[float, float, str]] | None = None,
) -> AudioAnalysis:
    """Run audio analysis stages (SER + prosody + interactions). Each stage is
    best-effort and reports its own status; the rolled-up status reflects
    whether the pipeline produced any usable output.
    """
    segment_models = [TranscriptSegment(**s) for s in segments]

    emotion_status, emotion_reason, emotions, emotion_unavailable = _run_emotion_analysis(
        job_id, audio, segment_models, detected_language
    )
    prosody_status, prosody_reason, prosody, prosody_unavailable = _run_prosody_analysis(job_id, audio, segment_models)
    (
        interaction_status,
        interaction_reason,
        interactions,
        segment_interactions,
        dominant_speaker_limitation,
    ) = _run_interaction_analysis(job_id, segment_models, diarize_turns)

    overall = _roll_up_status(emotion_status, prosody_status, interaction_status)
    overall_reason: str | None = None
    if overall != AudioAnalysisStatus.COMPLETED:
        overall_reason = emotion_reason or prosody_reason or interaction_reason

    return AudioAnalysis(
        status=overall,
        reason=overall_reason,
        emotion_status=emotion_status,
        emotion_reason=emotion_reason,
        emotions=emotions,
        emotion_unavailable=emotion_unavailable,
        prosody_status=prosody_status,
        prosody_reason=prosody_reason,
        prosody=prosody,
        prosody_unavailable=prosody_unavailable,
        interaction_status=interaction_status,
        interaction_reason=interaction_reason,
        interactions=interactions,
        segment_interactions=segment_interactions,
        dominant_speaker_limitation=dominant_speaker_limitation,
    )


def _run_single_language_transcription(
    job_id: str,
    audio,
    metadata: MeetingMetadata,
    device: str,
    compute_type: str,
    batch_size: int,
) -> tuple[list[dict], str, list[tuple[float, float, str]] | None]:
    """Existing single-language pipeline: transcribe -> align -> diarize -> build.

    Unchanged behavior for the 0/1-expected-language case (BR-2, EC-7).
    Returns (segments, detected_language, diarize_turns).
    """
    import torch
    import whisperx

    # Transcribe
    job_queue.update_job(job_id, stage="transcribing", progress=20)
    lang = metadata.language if metadata.language and metadata.language != "auto" else None
    model = whisperx.load_model(WHISPER_MODEL, device, compute_type=compute_type, language=lang)

    # Estimate audio duration for time-based progress during transcription
    audio_duration_sec = len(audio) / 16000  # whisperx uses 16kHz sample rate
    # Rough estimate: ~10s of audio per second on CPU, ~60s on GPU
    est_transcribe_sec = max(audio_duration_sec / (60 if device == "cuda" else 10), 5)

    transcribe_opts = {"batch_size": batch_size}
    if metadata.language and metadata.language != "auto":
        transcribe_opts["language"] = metadata.language

    # Smooth progress updates during transcription (20% -> 49%)
    _transcribe_done = threading.Event()

    def _update_transcribe_progress():
        start = time.monotonic()
        while not _transcribe_done.wait(timeout=2):
            elapsed = time.monotonic() - start
            # Asymptotic curve: approaches 49% but never reaches it
            fraction = elapsed / (elapsed + est_transcribe_sec)
            pct = 20 + int(fraction * 29)
            job_queue.update_job(job_id, stage="transcribing", progress=min(pct, 49))

    progress_thread = threading.Thread(target=_update_transcribe_progress, daemon=True)
    progress_thread.start()

    try:
        result = model.transcribe(audio, **transcribe_opts)
    finally:
        _transcribe_done.set()
        progress_thread.join(timeout=5)

    if not result or not result.get("segments"):
        raise RuntimeError("No speech detected in audio")

    detected_language = result.get("language", "unknown")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Align
    if detected_language in ALIGNMENT_LANGUAGES:
        job_queue.update_job(job_id, stage="aligning", progress=50)
        align_model_name = HF_ALIGN_REPOS.get(detected_language)
        # Load (and, on first use, download) the alignment model under a watchdog
        # so a stalled fetch degrades to segment-level timestamps instead of
        # hanging at the 50% stage (debug session macos-transcribe-stuck-70).
        status, loaded = _call_with_timeout(
            lambda: whisperx.load_align_model(
                language_code=detected_language, device=device, model_name=align_model_name
            ),
            ALIGNMENT_TIMEOUT_SEC,
            "align-load",
        )
        if status != "ok":
            logger.error(
                "Alignment model load %s for %s; keeping segment-level timestamps",
                status,
                detected_language,
                exc_info=loaded if status == "error" else None,
            )
        else:
            model_a, align_metadata = loaded
            result = whisperx.align(
                result["segments"],
                model_a,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            del model_a
            if device == "cuda":
                torch.cuda.empty_cache()

    # Diarize + assign speakers (shared with the multilingual path)
    result, diarize_turns = _diarize_and_assign(job_id, audio, metadata.num_speakers, device, result)

    # Build transcript segments
    segments = []
    for i, seg in enumerate(result["segments"]):
        segments.append(
            {
                "id": f"seg_{i:04d}",
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg["text"].strip(),
            }
        )

    return segments, detected_language, diarize_turns


def _finite(value, fallback: float) -> float:
    """Coerce ``value`` to a finite float, falling back when it is missing/NaN/inf."""
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return fallback
    return coerced if math.isfinite(coerced) else fallback


def _segment_level(group: list[dict], language: str) -> list[dict]:
    """Segment-level (chunk) timestamps with text untouched and no word data.

    Used when a language has no alignment model (BR-9, EC-3) or when alignment
    fails (Truth-2: degrade timing only, never the transcript text).
    """
    return [{"start": s["start"], "end": s["end"], "text": s["text"], "language": language} for s in group]


def _finalize_aligned(segments: list[dict], group: list[dict], language: str) -> list[dict]:
    """Stamp the (single) language on each aligned segment and guarantee usable timing.

    ``whisperx.align`` sentence-splits its input, so the output count differs from
    the input — segments are never positionally zipped back. Every segment from a
    single ``align`` call shares one language, so ``language`` is re-attached as a
    constant (BR-8). Non-finite timestamps are repaired to keep the player's
    seek/highlight (AC3) deterministic.
    """
    if not segments:
        return _segment_level(group, language)

    finalized: list[dict] = []
    last_end = group[0]["start"] if group else 0.0
    for seg in segments:
        start = _finite(seg.get("start"), last_end)
        end = _finite(seg.get("end"), start)
        if end < start:
            end = start
        seg["start"], seg["end"], seg["language"] = start, end, language
        finalized.append(seg)
        last_end = end
    return finalized


def _align_multilingual_segments(ml_segments: list[dict], audio, device: str) -> list[dict]:
    """Group segments by detected language and align each group with its own model.

    Segments are grouped by their ``language`` field and each group is aligned with
    that language's alignment model (BR-8), using the repo resolved from
    ``align_models.HF_ALIGN_REPOS`` where one is listed. Languages without an alignment
    model, or whose alignment raises, retain segment-level timestamps with text
    unchanged (BR-9, EC-3, Truth-2). Returns a flat list sorted by start time.
    """
    import torch
    import whisperx

    by_language: dict[str, list[dict]] = defaultdict(list)
    for seg in ml_segments:
        by_language[seg["language"]].append(seg)

    aligned: list[dict] = []
    for language in sorted(by_language):
        group = by_language[language]
        if language not in ALIGNMENT_LANGUAGES:
            aligned.extend(_segment_level(group, language))
            continue
        try:
            align_model_name = HF_ALIGN_REPOS.get(language)
            # Watchdog the load/download so a stalled fetch degrades to
            # segment-level timestamps instead of hanging (see single-language path).
            status, loaded = _call_with_timeout(
                lambda: whisperx.load_align_model(language_code=language, device=device, model_name=align_model_name),
                ALIGNMENT_TIMEOUT_SEC,
                "align-load",
            )
            if status == "timeout":
                raise TimeoutError(f"alignment model load exceeded {ALIGNMENT_TIMEOUT_SEC}s")
            if status == "error":
                raise loaded
            model_a, align_metadata = loaded
            try:
                result = whisperx.align(
                    [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in group],
                    model_a,
                    align_metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
            finally:
                del model_a
                if device == "cuda":
                    torch.cuda.empty_cache()
            aligned.extend(_finalize_aligned(result["segments"], group, language))
        except Exception:
            logger.exception("Alignment failed for language %s; retaining segment-level timestamps", language)
            aligned.extend(_segment_level(group, language))

    aligned.sort(key=lambda s: s["start"])
    return aligned


def _run_multilingual_transcription(
    job_id: str,
    audio,
    selected_languages: set[str],
    num_speakers: int | None,
    device: str,
    compute_type: str,
) -> tuple[list[dict], str, list[tuple[float, float, str]] | None]:
    """Per-chunk multilingual pipeline (BR-4 through BR-9).

    Triggered for 2+ expected languages. Each VAD speech chunk is detected
    (constrained to the selected set) and transcribed in its own language, then
    segments are grouped by detected language and word-aligned per language (BR-8,
    BR-9) before the shared diarization / speaker-assignment stage.
    Returns (segments, detected_language, diarize_turns) where detected_language is
    the dominant.
    """
    import torch
    import whisperx

    job_queue.update_job(job_id, stage="transcribing", progress=20)
    # No fixed language: per-chunk detection/decode picks the language.
    model = whisperx.load_model(WHISPER_MODEL, device, compute_type=compute_type)

    def _progress(pct: int) -> None:
        # Cap below the alignment/diarization stages so they have progress headroom.
        job_queue.update_job(job_id, stage="transcribing", progress=min(pct, 80))

    ml_segments, detected_language = transcribe_multilingual(audio, selected_languages, model, progress_cb=_progress)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    if not ml_segments:
        raise RuntimeError("No speech detected in audio")

    # Per-language word alignment (BR-8, BR-9, EC-3).
    job_queue.update_job(job_id, stage="aligning", progress=82)
    aligned = _align_multilingual_segments(ml_segments, audio, device)

    # Diarize + assign speakers (shared with the single-language path).
    result, diarize_turns = _diarize_and_assign(job_id, audio, num_speakers, device, {"segments": aligned}, progress=85)

    segments = []
    for i, seg in enumerate(result["segments"]):
        segments.append(
            {
                "id": f"seg_{i:04d}",
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "speaker": seg.get("speaker", "UNKNOWN"),
                "text": seg["text"].strip(),
                "language": seg["language"],
            }
        )

    return segments, detected_language, diarize_turns


def _run_transcription(meeting_id: str, job_id: str):
    """Run transcription in a background thread."""
    meeting_dir = MEETINGS_DIR / meeting_id
    metadata_path = meeting_dir / "metadata.json"
    preprocessed_path = None

    try:
        with open(metadata_path) as f:
            metadata = MeetingMetadata(**json.load(f))

        audio_path = meeting_dir / metadata.audio_filename

        if metadata.preprocess_audio:
            job_queue.update_job(job_id, status=JobStatus.PROCESSING, stage="preprocessing", progress=5)
            from backend.services.audio_preprocessor import preprocess_audio

            preprocessed_path = preprocess_audio(audio_path)
            audio_path = preprocessed_path

        job_queue.update_job(job_id, status=JobStatus.PROCESSING, stage="transcribing", progress=10)

        # Import heavy deps only when needed
        import warnings

        warnings.filterwarnings("ignore", message="Model was trained with")
        warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends")

        import functools

        import torch

        # PyTorch 2.6+ compat patch
        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load

        import whisperx

        device = _get_device()
        compute_type = "float16" if device == "cuda" else "float32"
        batch_size = WHISPER_BATCH_SIZE if device == "cuda" else 4

        # Load audio
        audio = whisperx.load_audio(str(audio_path))

        # Route on the expected-language set (BR-2, BR-3): 0/1 languages use the
        # existing single-language pipeline; 2+ trigger per-chunk multilingual
        # detection and transcription.
        diarize_turns: list[tuple[float, float, str]] | None = None
        if len(metadata.expected_languages) >= 2:
            segments, detected_language, diarize_turns = _run_multilingual_transcription(
                job_id, audio, set(metadata.expected_languages), metadata.num_speakers, device, compute_type
            )
        else:
            segments, detected_language, diarize_turns = _run_single_language_transcription(
                job_id, audio, metadata, device, compute_type, batch_size
            )

        job_queue.update_job(job_id, progress=90, stage="saving")

        transcript = {
            "segments": segments,
            "language": detected_language,
        }

        # Audio analysis (best-effort, must not fail the meeting — AC8/BR-1.4)
        audio_analysis: AudioAnalysis | None = None
        if metadata.audio_analysis_enabled:
            try:
                audio_analysis = _run_audio_analysis(
                    job_id, audio, segments, detected_language, diarize_turns=diarize_turns
                )
            except Exception as e:
                logger.exception("Audio analysis raised unexpectedly; continuing with meeting")
                audio_analysis = AudioAnalysis(status=AudioAnalysisStatus.FAILED, reason=str(e))

        # Check if cancelled before saving results
        if _is_cancelled(metadata_path):
            logger.info("Transcription for meeting %s was cancelled, discarding results", meeting_id)
            return

        # Save transcript
        transcript_path = meeting_dir / "transcript.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        # Save audio analysis (if produced) and record its status on metadata
        if audio_analysis is not None:
            audio_analysis_path = meeting_dir / "audio_analysis.json"
            with open(audio_analysis_path, "w", encoding="utf-8") as f:
                json.dump(audio_analysis.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            metadata.audio_analysis_status = audio_analysis.status

        # Update metadata
        duration = segments[-1]["end"] if segments else 0
        unique_speakers = sorted(set(s["speaker"] for s in segments))
        speaker_map = {spk: spk for spk in unique_speakers}

        metadata.status = MeetingStatus.READY
        metadata.duration_seconds = duration
        metadata.speakers = speaker_map

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)

        job_queue.update_job(job_id, status=JobStatus.COMPLETED, progress=100, stage="done")

    except Exception as e:
        logger.exception("Transcription failed for meeting %s", meeting_id)
        job_queue.update_job(job_id, status=JobStatus.FAILED, error=str(e))

        # Update meeting status to error (only if not already cancelled)
        if not _is_cancelled(metadata_path):
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                meta["status"] = "error"
                meta["error"] = str(e)
                with open(metadata_path, "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    finally:
        if preprocessed_path and preprocessed_path.exists():
            preprocessed_path.unlink()


def start_transcription(meeting_id: str, job_id: str):
    """Start transcription in a background thread."""
    thread = threading.Thread(
        target=_run_transcription,
        args=(meeting_id, job_id),
        daemon=True,
    )
    thread.start()
