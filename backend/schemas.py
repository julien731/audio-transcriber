from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MeetingType(str, Enum):
    INTERVIEW = "interview"
    SALES = "sales"
    CLIENT = "client"
    OTHER = "other"


class MeetingStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(str, Enum):
    UPLOADING = "uploading"
    PREPROCESSING = "preprocessing"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    DIARIZING = "diarizing"
    EMOTION_ANALYSIS = "emotion_analysis"
    PROSODY_EXTRACTION = "prosody_extraction"
    INTERACTION_ANALYSIS = "interaction_analysis"


class EmotionCategory(str, Enum):
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    FRUSTRATED = "frustrated"
    UNCERTAIN = "uncertain"
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"


class AudioAnalysisStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class EmotionAnnotation(BaseModel):
    segment_id: str
    speaker: str
    start: float
    end: float
    primary_emotion: EmotionCategory
    confidence: float
    emotion_scores: dict[str, float]
    low_confidence: bool


class EmotionUnavailable(BaseModel):
    segment_id: str
    reason: str


class ProsodyAnnotation(BaseModel):
    segment_id: str
    speaker: str
    start: float
    end: float
    volume_mean: float
    volume_variance: float
    pitch_mean: float
    pitch_variance: float
    speaking_rate: float
    pause_ratio: float


class ProsodyUnavailable(BaseModel):
    segment_id: str
    reason: str


class InteractionEventType(str, Enum):
    INTERRUPTION = "interruption"
    OVERLAP = "overlap"
    LONG_PAUSE = "long_pause"
    HESITATION = "hesitation"


class InteractionEvent(BaseModel):
    event_type: InteractionEventType
    timestamp: float
    speaker_a: str
    speaker_b: str
    duration: float
    context: str = ""


class SegmentInteraction(BaseModel):
    segment_id: str
    preceded_by_interruption: bool = False
    followed_by_interruption: bool = False
    hesitation_before: float = 0.0


class AudioAnalysis(BaseModel):
    status: AudioAnalysisStatus
    reason: str | None = None
    emotion_status: AudioAnalysisStatus | None = None
    emotion_reason: str | None = None
    emotions: list[EmotionAnnotation] = Field(default_factory=list)
    # Per-segment markers for segments whose language the emotion (SER) model does
    # not support (BR-10). Data-only, mirroring prosody_unavailable.
    emotion_unavailable: list[EmotionUnavailable] = Field(default_factory=list)
    prosody_status: AudioAnalysisStatus | None = None
    prosody_reason: str | None = None
    prosody: list[ProsodyAnnotation] = Field(default_factory=list)
    prosody_unavailable: list[ProsodyUnavailable] = Field(default_factory=list)
    interaction_status: AudioAnalysisStatus | None = None
    interaction_reason: str | None = None
    interactions: list[InteractionEvent] = Field(default_factory=list)
    segment_interactions: list[SegmentInteraction] = Field(default_factory=list)
    dominant_speaker_limitation: bool = False


class TranscriptSegment(BaseModel):
    id: str
    start: float
    end: float
    speaker: str
    text: str
    # Detected language for this segment, set only on the multilingual path.
    # None ⇒ fall back to Transcript.language (EC-8 backward-compat: pre-feature
    # transcripts and single-language meetings carry no per-segment language).
    language: str | None = None


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str = "en"


class MeetingMetadata(BaseModel):
    id: str
    title: str
    type: MeetingType = MeetingType.OTHER
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: float | None = None
    audio_filename: str = ""
    status: MeetingStatus = MeetingStatus.PROCESSING
    language: str = "auto"
    # Expected-language set chosen at upload. Size 0/1 ⇒ single-language pipeline
    # (language above is the input); size ≥ 2 ⇒ per-chunk multilingual pipeline
    # (BR-2, BR-3).
    expected_languages: list[str] = Field(default_factory=list)
    num_speakers: int | None = None
    preprocess_audio: bool = True
    audio_analysis_enabled: bool = False
    audio_analysis_status: AudioAnalysisStatus | None = None
    job_id: str | None = None
    speakers: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    context: str = ""


class MeetingUpdate(BaseModel):
    title: str | None = None
    type: MeetingType | None = None
    speakers: dict[str, str] | None = None
    context: str | None = None


class MeetingSummary(BaseModel):
    id: str
    title: str
    type: MeetingType
    created_at: datetime
    duration_seconds: float | None
    status: MeetingStatus


class MeetingDetail(BaseModel):
    metadata: MeetingMetadata
    transcript: Transcript | None = None
    audio_analysis: AudioAnalysis | None = None


class JobInfo(BaseModel):
    id: str
    meeting_id: str
    status: JobStatus
    progress: int = 0
    stage: str = ""
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class SegmentSpeakerUpdate(BaseModel):
    segment_id: str
    speaker_name: str


class ServiceConfig(BaseModel):
    """Local service configuration persisted under Application Support.

    Empty ``data_dir``/``models_dir`` are resolved to defaults under the
    Application Support directory by ``service_config.load()``.
    """

    hf_token: str = ""
    whisper_model: str = "large-v3"
    data_dir: str = ""
    models_dir: str = ""
    provisioning_completed: bool = False
    # Schema version of the model set last provisioned. When the required-repo
    # set changes (e.g. a new dependency model is added), bumping
    # ``provisioning.PROVISIONING_VERSION`` makes already-provisioned installs
    # re-provision to fetch the additions instead of downloading them lazily
    # mid-transcription (BR-24, EC-4).
    provisioning_version: int = 0
    imported_from: str | None = None


class DownloadState(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthResponse(BaseModel):
    status: str  # "ready" | "starting"
    provisioning_completed: bool
    diarization_available: bool


class ProvisioningStatus(BaseModel):
    provisioning_completed: bool
    models_present: bool
    whisper_model: str
    diarization_available: bool
    download_state: DownloadState = DownloadState.IDLE
    download_progress: int = 0
    download_error: str | None = None


class TokenUpdate(BaseModel):
    hf_token: str = ""


class AnalysisPromptResponse(BaseModel):
    prompt: str
