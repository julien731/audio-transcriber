# Candidate Terms

_No `docs/glossary.md` exists — all terms below are new candidates for triage._

## Core Entities

| Term | Source | Inferred description |
|------|--------|----------------------|
| Meeting | schemas.py (MeetingMetadata/Summary/Detail), routers/meetings.py | A recorded/uploaded audio session that gets transcribed and analyzed |
| Transcript | schemas.py (Transcript) | Full set of transcribed speech segments plus detected language |
| Transcript Segment | schemas.py (TranscriptSegment) | One timed utterance: start/end, speaker, text, optional per-segment language |
| Speaker | schemas.py, speaker-editor.js | A diarized voice in a meeting; renamable, color-coded in UI |
| Meeting Type | schemas.py (MeetingType) | Category of meeting: interview, sales, client, other |
| Context | schemas.py (MeetingMetadata.context) | Free-text user-supplied background fed into analysis prompts |

## Job / Pipeline

| Term | Source | Inferred description |
|------|--------|----------------------|
| Job | schemas.py (JobInfo), routers/jobs.py, services/job_queue.py | In-memory unit tracking one transcription run's progress |
| Job Stage | schemas.py (JobStage) | Pipeline phase: uploading, preprocessing, transcribing, aligning, diarizing, emotion_analysis, prosody_extraction, interaction_analysis |
| Diarization | services/transcriber.py, provisioning.py | PyAnnote step assigning speech turns to speakers |
| Alignment | services/transcriber.py | WhisperX wav2vec2 step producing word-level timestamps |
| Preprocessing | services/audio_preprocessor.py | High-pass filter + noise reduction + loudness normalization (-23 LUFS) |
| Multilingual / Per-chunk | services/multilingual_transcriber.py | Pipeline detecting & transcribing each VAD chunk in its own language |
| Dominant Language | services/multilingual_transcriber.py | Duration-weighted top language across confidently-classified chunks |
| VAD Chunk | services/multilingual_transcriber.py | Voice-activity-detected speech segment used as classification unit |

## Audio Analysis

| Term | Source | Inferred description |
|------|--------|----------------------|
| Audio Analysis | schemas.py (AudioAnalysis) | Optional layer bundling emotion, prosody, interaction results |
| Emotion Annotation | schemas.py (EmotionAnnotation), services/emotion_analyzer.py | Per-segment emotion with confidence + score distribution |
| Emotion Category | schemas.py (EmotionCategory) | neutral, confident, frustrated, uncertain, engaged, disengaged |
| SER Model | services/emotion_analyzer.py | Speech Emotion Recognition model (English-only) |
| Prosody Annotation | schemas.py (ProsodyAnnotation), services/prosody_analyzer.py | Per-segment volume, pitch, speaking rate, pause ratio (Praat pitch) |
| Interaction Event | schemas.py (InteractionEvent), services/interaction_analyzer.py | Cross-speaker event: interruption, overlap, long_pause, hesitation |
| Segment Interaction | schemas.py (SegmentInteraction) | Per-segment interruption flags + hesitation-before gap |
| Dominant Speaker Limitation | schemas.py (AudioAnalysis) | Flag noting reduced analysis fidelity for one-speaker dominance |

## Analysis Prompts / Templates

| Term | Source | Inferred description |
|------|--------|----------------------|
| Analysis Template | templates/, routers/analysis.py | Markdown prompt scaffold selected by meeting type |
| Interview Analysis | templates/interview_analysis.md | Template producing structured interview notes |
| Sales Meeting Analysis | templates/sales_meeting_analysis.md | Template extracting sales intelligence + prospect follow-up |
| Client Meeting Analysis | templates/client_meeting_analysis.md | Template extracting decisions, action items, client summary |
| Prototype Scope | templates/prototype_scope.md | Template generating a prototype scope doc for AI coding tools |
| Analysis Prompt | schemas.py (AnalysisPromptResponse), services/analysis_prompt.py | Fully-rendered prompt (template + transcript + context) |

## Provisioning / Config

| Term | Source | Inferred description |
|------|--------|----------------------|
| Provisioning | schemas.py (ProvisioningStatus), services/provisioning.py | First-run download/setup of Whisper + diarization models |
| Service Config | schemas.py (ServiceConfig) | Locally persisted settings (HF token, model, data/models dirs) |
| HF Token | schemas.py, services/provisioning.py | HuggingFace token; empty disables diarization |
| Whisper Model | schemas.py, config | Whisper model size (default large-v3) |
| Download State | schemas.py (DownloadState) | Model-download lifecycle: idle, downloading, completed, failed |
