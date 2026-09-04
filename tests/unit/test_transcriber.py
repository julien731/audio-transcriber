from __future__ import annotations

import json
import struct
import sys
import threading
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import backend.services.transcriber as transcriber_module
from backend.schemas import AudioAnalysis, AudioAnalysisStatus, JobStatus, MeetingStatus
from backend.services.job_queue import JobQueue
from backend.services.transcriber import (
    _align_multilingual_segments,
    _diarize_and_assign,
    _get_device,
    _is_cancelled,
    _run_audio_analysis,
)


def _create_test_audio(path: Path) -> None:
    """Create a minimal valid WAV file for testing."""
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(struct.pack("<" + "h" * 16, *([0] * 16)))


def _create_meeting_on_disk(
    meetings_dir: Path,
    meeting_id: str = "test-meeting",
    status: str = "processing",
    audio_filename: str = "audio.wav",
    language: str = "auto",
    num_speakers: int | None = None,
    expected_languages: list[str] | None = None,
    audio_analysis_enabled: bool = False,
) -> Path:
    """Create a meeting directory with metadata and audio file."""
    meeting_dir = meetings_dir / meeting_id
    meeting_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": meeting_id,
        "title": "Test Meeting",
        "type": "other",
        "status": status,
        "audio_filename": audio_filename,
        "language": language,
        "expected_languages": expected_languages or [],
        "num_speakers": num_speakers,
        "preprocess_audio": False,
        "audio_analysis_enabled": audio_analysis_enabled,
        "created_at": "2026-03-14T00:00:00",
        "speakers": {},
    }
    (meeting_dir / "metadata.json").write_text(json.dumps(metadata))
    _create_test_audio(meeting_dir / audio_filename)

    return meeting_dir


class TestIsCancelled:
    def test_returns_false_when_processing(self, tmp_path: Path):
        meta = {"id": "m1", "title": "Test", "status": "processing", "audio_filename": "a.wav"}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(meta))

        assert _is_cancelled(meta_path) is False

    def test_returns_true_when_error(self, tmp_path: Path):
        meta = {"id": "m1", "title": "Test", "status": "error", "audio_filename": "a.wav"}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(meta))

        assert _is_cancelled(meta_path) is True

    def test_returns_true_when_ready(self, tmp_path: Path):
        meta = {"id": "m1", "title": "Test", "status": "ready", "audio_filename": "a.wav"}
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text(json.dumps(meta))

        assert _is_cancelled(meta_path) is True

    def test_returns_false_on_read_error(self, tmp_path: Path):
        meta_path = tmp_path / "nonexistent.json"
        assert _is_cancelled(meta_path) is False


class TestGetDevice:
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cuda")
    def test_returns_configured_device(self):
        assert _get_device() == "cuda"

    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    def test_returns_cpu_when_configured(self):
        assert _get_device() == "cpu"

    @patch("backend.services.transcriber.WHISPER_DEVICE", "auto")
    def test_auto_falls_back_to_cpu_without_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert _get_device() == "cpu"

    @patch("backend.services.transcriber.WHISPER_DEVICE", "auto")
    def test_auto_detects_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _get_device() == "cuda"

    @patch("backend.services.transcriber.WHISPER_DEVICE", "auto")
    def test_auto_returns_cpu_when_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _get_device() == "cpu"


class TestRunTranscription:
    """Tests for _run_transcription with all ML dependencies mocked."""

    def _build_mocks(self):
        """Create mock objects for whisperx and torch."""
        mock_whisperx = MagicMock()
        mock_whisperx.load_audio.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": " Hello there.", "speaker": "SPEAKER_00"},
                {"start": 1.5, "end": 3.0, "text": " How are you?", "speaker": "SPEAKER_01"},
            ],
            "language": "en",
        }
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": " Hello there.", "speaker": "SPEAKER_00"},
                {"start": 1.5, "end": 3.0, "text": " How are you?", "speaker": "SPEAKER_01"},
            ],
        }

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_diarize_pipeline = MagicMock()
        mock_assign_speakers = MagicMock(
            return_value={
                "segments": [
                    {"start": 0.0, "end": 1.5, "text": " Hello there.", "speaker": "SPEAKER_00"},
                    {"start": 1.5, "end": 3.0, "text": " How are you?", "speaker": "SPEAKER_01"},
                ],
            }
        )

        return mock_whisperx, mock_torch, mock_diarize_pipeline, mock_assign_speakers

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "test-token")
    def test_happy_path_writes_transcript_and_updates_metadata(self, tmp_path: Path):
        mock_whisperx, mock_torch, mock_diarize_cls, mock_assign = self._build_mocks()
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        meeting_dir = _create_meeting_on_disk(meetings_dir)
        job = queue.create_job("test-meeting")

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(
                        DiarizationPipeline=mock_diarize_cls,
                        assign_word_speakers=mock_assign,
                    ),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        # Verify transcript written
        transcript_path = meeting_dir / "transcript.json"
        assert transcript_path.exists()
        transcript = json.loads(transcript_path.read_text())
        assert len(transcript["segments"]) == 2
        assert transcript["segments"][0]["text"] == "Hello there."
        assert transcript["segments"][0]["speaker"] == "SPEAKER_00"
        assert transcript["language"] == "en"

        # Verify metadata updated to READY
        metadata = json.loads((meeting_dir / "metadata.json").read_text())
        assert metadata["status"] == MeetingStatus.READY.value
        assert metadata["duration_seconds"] == 3.0
        assert "SPEAKER_00" in metadata["speakers"]
        assert "SPEAKER_01" in metadata["speakers"]

        # Verify job completed
        updated_job = queue.get_job(job.id)
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.progress == 100

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "test-token")
    def test_degrades_when_diarization_raises_at_runtime(self, tmp_path: Path):
        """A token accepted at provisioning but rejected at runtime must not fail
        the meeting — it completes without speaker labels (BR-10, EC-3)."""
        mock_whisperx, mock_torch, mock_diarize_cls, mock_assign = self._build_mocks()
        # Simulate a revoked/unverifiable token: DiarizationPipeline blows up.
        mock_diarize_cls.side_effect = RuntimeError("401 Unauthorized")
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        meeting_dir = _create_meeting_on_disk(meetings_dir)
        job = queue.create_job("test-meeting")

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(
                        DiarizationPipeline=mock_diarize_cls,
                        assign_word_speakers=mock_assign,
                    ),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        # Transcript still written; meeting READY; diarization was skipped.
        assert (meeting_dir / "transcript.json").exists()
        metadata = json.loads((meeting_dir / "metadata.json").read_text())
        assert metadata["status"] == MeetingStatus.READY.value
        mock_assign.assert_not_called()
        assert queue.get_job(job.id).status == JobStatus.COMPLETED

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_skips_diarization_without_hf_token(self, tmp_path: Path):
        mock_whisperx, mock_torch, mock_diarize_cls, mock_assign = self._build_mocks()
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        _create_meeting_on_disk(meetings_dir)
        job = queue.create_job("test-meeting")

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(
                        DiarizationPipeline=mock_diarize_cls,
                        assign_word_speakers=mock_assign,
                    ),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        # Diarization pipeline should not be called
        mock_diarize_cls.assert_not_called()

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_error_sets_job_failed_and_updates_metadata(self, tmp_path: Path):
        mock_whisperx = MagicMock()
        mock_whisperx.load_audio.side_effect = RuntimeError("Audio file corrupt")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        meeting_dir = _create_meeting_on_disk(meetings_dir)
        job = queue.create_job("test-meeting")

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        # Job should be FAILED
        updated_job = queue.get_job(job.id)
        assert updated_job.status == JobStatus.FAILED
        assert "Audio file corrupt" in updated_job.error

        # Metadata should have error status
        metadata = json.loads((meeting_dir / "metadata.json").read_text())
        assert metadata["status"] == "error"
        assert "Audio file corrupt" in metadata["error"]

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_discards_results_when_cancelled(self, tmp_path: Path):
        mock_whisperx, mock_torch, mock_diarize_cls, mock_assign = self._build_mocks()
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        meeting_dir = _create_meeting_on_disk(meetings_dir)
        job = queue.create_job("test-meeting")

        # Change status to "error" before saving results to simulate cancellation
        def cancel_before_save(*args, **kwargs):
            meta = json.loads((meeting_dir / "metadata.json").read_text())
            meta["status"] = "error"
            (meeting_dir / "metadata.json").write_text(json.dumps(meta))
            return mock_whisperx.align.return_value

        # Make align trigger the cancellation
        mock_whisperx.align.side_effect = cancel_before_save

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(
                        DiarizationPipeline=mock_diarize_cls,
                        assign_word_speakers=mock_assign,
                    ),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        # Transcript should NOT be written
        assert not (meeting_dir / "transcript.json").exists()

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_no_speech_detected_raises_error(self, tmp_path: Path):
        mock_whisperx = MagicMock()
        mock_whisperx.load_audio.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": [], "language": "en"}
        mock_whisperx.load_model.return_value = mock_model
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        _create_meeting_on_disk(meetings_dir)
        job = queue.create_job("test-meeting")

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        updated_job = queue.get_job(job.id)
        assert updated_job.status == JobStatus.FAILED
        assert "No speech detected" in updated_job.error

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_thai_alignment_uses_custom_model(self, tmp_path: Path):
        mock_whisperx, mock_torch, mock_diarize_cls, mock_assign = self._build_mocks()
        mock_model = mock_whisperx.load_model.return_value
        mock_model.transcribe.return_value["language"] = "th"

        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        _create_meeting_on_disk(meetings_dir, language="auto")
        job = queue.create_job("test-meeting")

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(
                        DiarizationPipeline=mock_diarize_cls,
                        assign_word_speakers=mock_assign,
                    ),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        mock_whisperx.load_align_model.assert_called_once_with(
            language_code="th",
            device="cpu",
            model_name="airesearch/wav2vec2-large-xlsr-53-th",
        )

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_unsupported_language_skips_alignment(self, tmp_path: Path):
        mock_whisperx, mock_torch, mock_diarize_cls, mock_assign = self._build_mocks()
        mock_model = mock_whisperx.load_model.return_value
        mock_model.transcribe.return_value["language"] = "xx"

        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        _create_meeting_on_disk(meetings_dir, language="auto")
        job = queue.create_job("test-meeting")

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(
                        DiarizationPipeline=mock_diarize_cls,
                        assign_word_speakers=mock_assign,
                    ),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        mock_whisperx.load_align_model.assert_not_called()

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_english_alignment_uses_default_model(self, tmp_path: Path):
        mock_whisperx, mock_torch, mock_diarize_cls, mock_assign = self._build_mocks()

        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        _create_meeting_on_disk(meetings_dir, language="auto")
        job = queue.create_job("test-meeting")

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch.dict(
                "sys.modules",
                {
                    "whisperx": mock_whisperx,
                    "torch": mock_torch,
                    "whisperx.diarize": MagicMock(
                        DiarizationPipeline=mock_diarize_cls,
                        assign_word_speakers=mock_assign,
                    ),
                    "functools": __import__("functools"),
                    "warnings": __import__("warnings"),
                },
            ),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        mock_whisperx.load_align_model.assert_called_once_with(
            language_code="en",
            device="cpu",
            model_name=None,
        )


class TestStartTranscription:
    def test_spawns_daemon_thread(self):
        with patch("backend.services.transcriber._run_transcription") as mock_run:
            from backend.services.transcriber import start_transcription

            start_transcription("m1", "j1")

            # Give the thread a moment to start
            import time

            time.sleep(0.05)

            mock_run.assert_called_once_with("m1", "j1")


class TestRunAudioAnalysis:
    """Covers the inline call site that the analyzer unit tests don't exercise."""

    def _segments(self):
        return [
            {"id": "seg_0000", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hi"},
        ]

    def _diarize_turns(self):
        return [(0.0, 1.0, "SPEAKER_00")]

    def test_unsupported_language_skips_emotion_but_runs_prosody(self):
        with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])) as mock_prosody:
            with patch("backend.services.transcriber.job_queue.update_job"):
                result = _run_audio_analysis(
                    job_id="j1",
                    audio=MagicMock(),
                    segments=self._segments(),
                    detected_language="fr",
                    diarize_turns=self._diarize_turns(),
                )
        # Emotions skipped (English-only), prosody ran (language-agnostic, BR-2.1)
        assert result.emotion_status == AudioAnalysisStatus.UNAVAILABLE
        assert result.emotion_reason == "language_not_supported:fr"
        assert result.prosody_status == AudioAnalysisStatus.COMPLETED
        # Interaction is also language-agnostic
        assert result.interaction_status == AudioAnalysisStatus.COMPLETED
        # Pipeline status rolls up — completes when at least one stage produces output
        assert result.status == AudioAnalysisStatus.COMPLETED
        mock_prosody.assert_called_once()

    def test_english_runs_both_stages_and_returns_completed(self):
        with patch("backend.services.emotion_analyzer.analyze_segments", return_value=[]) as mock_emotion:
            with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])) as mock_prosody:
                with patch("backend.services.transcriber.job_queue.update_job"):
                    result = _run_audio_analysis(
                        job_id="j1",
                        audio=MagicMock(),
                        segments=self._segments(),
                        detected_language="en",
                        diarize_turns=self._diarize_turns(),
                    )
        assert result.status == AudioAnalysisStatus.COMPLETED
        assert result.emotion_status == AudioAnalysisStatus.COMPLETED
        assert result.prosody_status == AudioAnalysisStatus.COMPLETED
        assert result.interaction_status == AudioAnalysisStatus.COMPLETED
        mock_emotion.assert_called_once()
        mock_prosody.assert_called_once()

    def test_emotion_failure_does_not_block_prosody(self):
        with patch(
            "backend.services.emotion_analyzer.analyze_segments",
            side_effect=RuntimeError("model crashed"),
        ):
            with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])):
                with patch("backend.services.transcriber.job_queue.update_job"):
                    result = _run_audio_analysis(
                        job_id="j1",
                        audio=MagicMock(),
                        segments=self._segments(),
                        detected_language="en",
                        diarize_turns=self._diarize_turns(),
                    )
        # Emotion failed but prosody completed → overall COMPLETED
        assert result.emotion_status == AudioAnalysisStatus.FAILED
        assert "model crashed" in result.emotion_reason
        assert result.prosody_status == AudioAnalysisStatus.COMPLETED
        assert result.status == AudioAnalysisStatus.COMPLETED

    def test_prosody_failure_does_not_block_emotion(self):
        with patch("backend.services.emotion_analyzer.analyze_segments", return_value=[]):
            with patch(
                "backend.services.prosody_analyzer.analyze_segments",
                side_effect=RuntimeError("praat crashed"),
            ):
                with patch("backend.services.transcriber.job_queue.update_job"):
                    result = _run_audio_analysis(
                        job_id="j1",
                        audio=MagicMock(),
                        segments=self._segments(),
                        detected_language="en",
                        diarize_turns=self._diarize_turns(),
                    )
        assert result.prosody_status == AudioAnalysisStatus.FAILED
        assert "praat crashed" in result.prosody_reason
        assert result.emotion_status == AudioAnalysisStatus.COMPLETED
        assert result.status == AudioAnalysisStatus.COMPLETED

    def test_all_stages_fail_rolls_up_to_failed(self):
        with patch(
            "backend.services.emotion_analyzer.analyze_segments",
            side_effect=RuntimeError("emotion boom"),
        ):
            with patch(
                "backend.services.prosody_analyzer.analyze_segments",
                side_effect=RuntimeError("prosody boom"),
            ):
                with patch(
                    "backend.services.interaction_analyzer.analyze",
                    side_effect=RuntimeError("interaction boom"),
                ):
                    with patch("backend.services.transcriber.job_queue.update_job"):
                        result = _run_audio_analysis(
                            job_id="j1",
                            audio=MagicMock(),
                            segments=self._segments(),
                            detected_language="en",
                            diarize_turns=self._diarize_turns(),
                        )
        assert result.status == AudioAnalysisStatus.FAILED
        assert result.emotion_status == AudioAnalysisStatus.FAILED
        assert result.prosody_status == AudioAnalysisStatus.FAILED
        assert result.interaction_status == AudioAnalysisStatus.FAILED

    def test_prosody_unavailable_markers_propagated(self):
        from backend.schemas import ProsodyUnavailable

        with patch("backend.services.emotion_analyzer.analyze_segments", return_value=[]):
            with patch(
                "backend.services.prosody_analyzer.analyze_segments",
                return_value=([], [ProsodyUnavailable(segment_id="seg_0000", reason="non_speech")]),
            ):
                with patch("backend.services.transcriber.job_queue.update_job"):
                    result = _run_audio_analysis(
                        job_id="j1",
                        audio=MagicMock(),
                        segments=self._segments(),
                        detected_language="en",
                        diarize_turns=self._diarize_turns(),
                    )
        assert len(result.prosody_unavailable) == 1
        assert result.prosody_unavailable[0].reason == "non_speech"

    def test_no_diarize_turns_marks_interaction_unavailable(self):
        with patch("backend.services.emotion_analyzer.analyze_segments", return_value=[]):
            with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])):
                with patch("backend.services.transcriber.job_queue.update_job"):
                    result = _run_audio_analysis(
                        job_id="j1",
                        audio=MagicMock(),
                        segments=self._segments(),
                        detected_language="en",
                        diarize_turns=None,
                    )
        assert result.interaction_status == AudioAnalysisStatus.UNAVAILABLE
        assert result.interaction_reason == "diarization_unavailable"
        # Emotion + prosody still produced output → overall COMPLETED
        assert result.status == AudioAnalysisStatus.COMPLETED

    def test_interaction_failure_does_not_block_other_stages(self):
        with patch("backend.services.emotion_analyzer.analyze_segments", return_value=[]):
            with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])):
                with patch(
                    "backend.services.interaction_analyzer.analyze",
                    side_effect=RuntimeError("interaction boom"),
                ):
                    with patch("backend.services.transcriber.job_queue.update_job"):
                        result = _run_audio_analysis(
                            job_id="j1",
                            audio=MagicMock(),
                            segments=self._segments(),
                            detected_language="en",
                            diarize_turns=self._diarize_turns(),
                        )
        assert result.interaction_status == AudioAnalysisStatus.FAILED
        assert "interaction boom" in result.interaction_reason
        assert result.status == AudioAnalysisStatus.COMPLETED

    def test_interaction_results_propagated(self):
        from backend.schemas import InteractionEvent, InteractionEventType, SegmentInteraction
        from backend.services.interaction_analyzer import InteractionAnalysisResult

        analyzer_result = InteractionAnalysisResult(
            events=[
                InteractionEvent(
                    event_type=InteractionEventType.INTERRUPTION,
                    timestamp=2.0,
                    speaker_a="SPEAKER_00",
                    speaker_b="SPEAKER_01",
                    duration=0.5,
                    context="hi",
                )
            ],
            segment_interactions=[SegmentInteraction(segment_id="seg_0000", followed_by_interruption=True)],
            dominant_speaker_limitation=True,
        )
        with patch("backend.services.emotion_analyzer.analyze_segments", return_value=[]):
            with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])):
                with patch("backend.services.interaction_analyzer.analyze", return_value=analyzer_result):
                    with patch("backend.services.transcriber.job_queue.update_job"):
                        result = _run_audio_analysis(
                            job_id="j1",
                            audio=MagicMock(),
                            segments=self._segments(),
                            detected_language="en",
                            diarize_turns=self._diarize_turns(),
                        )
        assert len(result.interactions) == 1
        assert result.interactions[0].event_type == InteractionEventType.INTERRUPTION
        assert len(result.segment_interactions) == 1
        assert result.segment_interactions[0].followed_by_interruption is True
        assert result.dominant_speaker_limitation is True

    def _mixed_segments(self):
        return [
            {"id": "seg_0000", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello", "language": "en"},
            {"id": "seg_0001", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "สวัสดี", "language": "th"},
        ]

    def test_mixed_meeting_analyzes_english_and_marks_others_unavailable(self):
        # AC1 / EC-6 / BR-10: English segment analyzed, Thai segment marked unavailable,
        # meeting not skipped wholesale.
        captured = {}

        def fake_analyze(audio_path, segments, audio_array=None):
            captured["segments"] = list(segments)
            return []

        with patch("backend.services.emotion_analyzer.analyze_segments", side_effect=fake_analyze):
            with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])):
                with patch("backend.services.transcriber.job_queue.update_job"):
                    result = _run_audio_analysis(
                        job_id="j1",
                        audio=MagicMock(),
                        segments=self._mixed_segments(),
                        detected_language="en",
                        diarize_turns=self._diarize_turns(),
                    )
        # Only the English segment is sent to the SER model.
        assert [s.id for s in captured["segments"]] == ["seg_0000"]
        assert result.emotion_status == AudioAnalysisStatus.COMPLETED
        # The Thai segment is marked unavailable, not dropped silently.
        assert len(result.emotion_unavailable) == 1
        assert result.emotion_unavailable[0].segment_id == "seg_0001"
        assert result.emotion_unavailable[0].reason == "language_not_supported:th"
        # Prosody + interaction unaffected (AC3 / BR-11).
        assert result.prosody_status == AudioAnalysisStatus.COMPLETED
        assert result.interaction_status == AudioAnalysisStatus.COMPLETED
        assert result.status == AudioAnalysisStatus.COMPLETED

    def test_no_english_segments_reports_unavailable_not_failed(self):
        # AC2: a meeting with no English segments reports emotion unavailable while
        # prosody and interaction still run.
        segments = [
            {"id": "seg_0000", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "สวัสดี", "language": "th"},
        ]
        with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])) as mock_prosody:
            with patch("backend.services.transcriber.job_queue.update_job"):
                result = _run_audio_analysis(
                    job_id="j1",
                    audio=MagicMock(),
                    segments=segments,
                    detected_language="th",
                    diarize_turns=self._diarize_turns(),
                )
        assert result.emotion_status == AudioAnalysisStatus.UNAVAILABLE
        assert result.emotion_reason == "language_not_supported:th"
        assert len(result.emotion_unavailable) == 1
        # Prosody + interaction still run (AC3 / BR-11).
        assert result.prosody_status == AudioAnalysisStatus.COMPLETED
        assert result.interaction_status == AudioAnalysisStatus.COMPLETED
        assert result.status == AudioAnalysisStatus.COMPLETED
        mock_prosody.assert_called_once()

    def test_no_english_aggregate_reason_lists_languages_sorted(self):
        # The aggregate reason pins the multi-language format for any future consumer.
        segments = [
            {"id": "seg_0000", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "x", "language": "th"},
            {"id": "seg_0001", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "y", "language": "fr"},
        ]
        with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])):
            with patch("backend.services.transcriber.job_queue.update_job"):
                result = _run_audio_analysis(
                    job_id="j1",
                    audio=MagicMock(),
                    segments=segments,
                    detected_language="th",
                    diarize_turns=self._diarize_turns(),
                )
        assert result.emotion_reason == "language_not_supported:fr,th"
        assert len(result.emotion_unavailable) == 2

    def test_emotion_failure_still_propagates_unavailable_markers(self):
        # Even when the SER run raises, markers partitioned before the call survive.
        with patch(
            "backend.services.emotion_analyzer.analyze_segments",
            side_effect=RuntimeError("model crashed"),
        ):
            with patch("backend.services.prosody_analyzer.analyze_segments", return_value=([], [])):
                with patch("backend.services.transcriber.job_queue.update_job"):
                    result = _run_audio_analysis(
                        job_id="j1",
                        audio=MagicMock(),
                        segments=self._mixed_segments(),
                        detected_language="en",
                        diarize_turns=self._diarize_turns(),
                    )
        assert result.emotion_status == AudioAnalysisStatus.FAILED
        assert "model crashed" in result.emotion_reason
        assert [m.segment_id for m in result.emotion_unavailable] == ["seg_0001"]


class TestTranscriptionRouting:
    """Routing between single-language and multilingual pipelines (BR-2, BR-3, EC-7)."""

    def _mocks(self):
        mock_whisperx = MagicMock()
        mock_whisperx.load_audio.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.5, "text": " Hello.", "speaker": "SPEAKER_00"}],
            "language": "en",
        }
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {
            "segments": [{"start": 0.0, "end": 1.5, "text": " Hello.", "speaker": "SPEAKER_00"}],
        }
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_diarize_cls = MagicMock()
        mock_assign = MagicMock(
            return_value={"segments": [{"start": 0.0, "end": 1.5, "text": " Hello.", "speaker": "SPEAKER_00"}]}
        )
        return mock_whisperx, mock_torch, mock_diarize_cls, mock_assign

    @staticmethod
    def _sys_modules(mock_whisperx, mock_torch, mock_diarize_cls, mock_assign):
        return {
            "whisperx": mock_whisperx,
            "torch": mock_torch,
            "whisperx.diarize": MagicMock(
                DiarizationPipeline=mock_diarize_cls,
                assign_word_speakers=mock_assign,
            ),
            "functools": __import__("functools"),
            "warnings": __import__("warnings"),
        }

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_single_language_runs_unchanged_single_path(self, tmp_path: Path):
        """Exactly one expected language → existing single path, forcing that language (EC-7, AC2)."""
        mw, mt, mdc, ma = self._mocks()
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        _create_meeting_on_disk(meetings_dir, language="en", expected_languages=["en"])
        job = queue.create_job("test-meeting")
        ml = MagicMock()

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch("backend.services.transcriber.transcribe_multilingual", ml),
            patch.dict("sys.modules", self._sys_modules(mw, mt, mdc, ma)),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        ml.assert_not_called()
        # Single path forces the selected language into load_model AND transcribe.
        assert mw.load_model.call_args.kwargs["language"] == "en"
        assert mw.load_model.return_value.transcribe.call_args.kwargs["language"] == "en"
        # Alignment still runs for the single path.
        assert mw.align.called

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_no_expected_languages_auto_detects_single(self, tmp_path: Path):
        """Zero expected languages → single path with auto-detect (language=None)."""
        mw, mt, mdc, ma = self._mocks()
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        _create_meeting_on_disk(meetings_dir, language="auto", expected_languages=[])
        job = queue.create_job("test-meeting")
        ml = MagicMock()

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch("backend.services.transcriber.transcribe_multilingual", ml),
            patch.dict("sys.modules", self._sys_modules(mw, mt, mdc, ma)),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        ml.assert_not_called()
        assert mw.load_model.call_args.kwargs.get("language") is None
        assert "language" not in mw.load_model.return_value.transcribe.call_args.kwargs

    @staticmethod
    def _ml_mocks():
        """Mocks for the multilingual path: align echoes its input segments and
        assign_word_speakers stamps a speaker while preserving the language tag."""
        mock_whisperx = MagicMock()
        mock_whisperx.load_audio.return_value = MagicMock()
        mock_whisperx.load_model.return_value = MagicMock()
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        # align sentence-splits in reality; here it echoes each input segment so the
        # per-language grouping (one call per language) stays observable.
        mock_whisperx.align.side_effect = lambda segs, *a, **k: {"segments": [dict(s) for s in segs]}

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        diarize_df = MagicMock()
        diarize_df.iterrows.return_value = [(0, {"start": 0.0, "end": 9.0, "speaker": "SPEAKER_00"})]
        diarize_df.__len__.return_value = 1
        # The pipeline instance is callable; calling it (diarize_model(audio)) yields the dataframe.
        diarize_instance = MagicMock(return_value=diarize_df)
        mock_diarize_cls = MagicMock(return_value=diarize_instance)

        def _assign(_diarize_df, result):
            for seg in result["segments"]:
                seg["speaker"] = "SPEAKER_00"
            return result

        return mock_whisperx, mock_torch, mock_diarize_cls, MagicMock(side_effect=_assign)

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "test-token")
    def test_two_languages_align_diarize_and_preserve_language(self, tmp_path: Path):
        """Two+ expected languages → multilingual path now aligns, diarizes, assigns
        speakers, and preserves each segment's detected language (BR-8, AC1)."""
        mw, mt, mdc, ma = self._ml_mocks()
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        meeting_dir = _create_meeting_on_disk(meetings_dir, language="auto", expected_languages=["en", "th"])
        job = queue.create_job("test-meeting")
        ml = MagicMock(
            return_value=(
                [
                    {"start": 0.0, "end": 2.0, "text": "Hello", "language": "en"},
                    {"start": 2.0, "end": 4.0, "text": "สวัสดี", "language": "th"},
                ],
                "en",
            )
        )

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch("backend.services.transcriber.transcribe_multilingual", ml),
            patch.dict("sys.modules", self._sys_modules(mw, mt, mdc, ma)),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        ml.assert_called_once()
        # Multilingual loads the model without a fixed language.
        assert "language" not in mw.load_model.call_args.kwargs
        # Alignment runs once per language group (en, th) with each language's own model (BR-8).
        align_languages = sorted(c.kwargs["language_code"] for c in mw.load_align_model.call_args_list)
        assert align_languages == ["en", "th"]
        # Diarization now runs on the multilingual path.
        mdc.assert_called_once()

        transcript = json.loads((meeting_dir / "transcript.json").read_text())
        assert {s["language"] for s in transcript["segments"]} == {"en", "th"}
        assert all(s["speaker"] == "SPEAKER_00" for s in transcript["segments"])
        assert transcript["language"] == "en"  # dominant

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "test-token")
    def test_multilingual_audio_analysis_gets_dominant_and_turns(self, tmp_path: Path):
        """Audio analysis on the multilingual path receives the dominant language and
        the diarization turns captured during speaker assignment."""
        mw, mt, mdc, ma = self._ml_mocks()
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        _create_meeting_on_disk(
            meetings_dir, language="auto", expected_languages=["en", "th"], audio_analysis_enabled=True
        )
        job = queue.create_job("test-meeting")
        ml = MagicMock(
            return_value=([{"start": 0.0, "end": 2.0, "text": "Hello", "language": "en"}], "en"),
        )
        maa = MagicMock(return_value=AudioAnalysis(status=AudioAnalysisStatus.UNAVAILABLE))

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch("backend.services.transcriber.transcribe_multilingual", ml),
            patch("backend.services.transcriber._run_audio_analysis", maa),
            patch.dict("sys.modules", self._sys_modules(mw, mt, mdc, ma)),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        maa.assert_called_once()
        # detected_language is the 4th positional arg; diarize_turns is a kwarg.
        assert maa.call_args.args[3] == "en"
        assert maa.call_args.kwargs["diarize_turns"] == [(0.0, 9.0, "SPEAKER_00")]

    @patch("backend.services.transcriber.WHISPER_BATCH_SIZE", 16)
    @patch("backend.services.transcriber.WHISPER_MODEL", "large-v3")
    @patch("backend.services.transcriber.WHISPER_DEVICE", "cpu")
    @patch("config.current_hf_token", new=lambda: "")
    def test_multilingual_without_hf_token_completes_unknown_speakers(self, tmp_path: Path):
        """Multilingual path with no HF_TOKEN still aligns and completes; diarization
        is skipped so segments keep UNKNOWN speakers (no crash)."""
        mw, mt, mdc, ma = self._ml_mocks()
        queue = JobQueue()
        meetings_dir = tmp_path / "meetings"
        meeting_dir = _create_meeting_on_disk(meetings_dir, language="auto", expected_languages=["en", "th"])
        job = queue.create_job("test-meeting")
        ml = MagicMock(
            return_value=(
                [
                    {"start": 0.0, "end": 2.0, "text": "Hello", "language": "en"},
                    {"start": 2.0, "end": 4.0, "text": "สวัสดี", "language": "th"},
                ],
                "en",
            )
        )

        with (
            patch("backend.services.transcriber.MEETINGS_DIR", meetings_dir),
            patch("backend.services.transcriber.job_queue", queue),
            patch("backend.services.transcriber.transcribe_multilingual", ml),
            patch.dict("sys.modules", self._sys_modules(mw, mt, mdc, ma)),
        ):
            from backend.services.transcriber import _run_transcription

            _run_transcription("test-meeting", job.id)

        mdc.assert_not_called()
        transcript = json.loads((meeting_dir / "transcript.json").read_text())
        assert all(s["speaker"] == "UNKNOWN" for s in transcript["segments"])
        assert {s["language"] for s in transcript["segments"]} == {"en", "th"}


def _fake_whisperx(align_side_effect):
    """Build a mock whisperx module for direct _align_multilingual_segments tests."""
    mw = MagicMock()
    mw.load_align_model.return_value = (MagicMock(), MagicMock())
    mw.align.side_effect = align_side_effect
    return mw


def _echo_align(segs, *args, **kwargs):
    """Stand in for whisperx.align: echo each input segment unchanged."""
    return {"segments": [dict(s) for s in segs]}


class TestAlignMultilingualSegments:
    """Per-language alignment grouping, fallback, and timestamp repair (BR-8, BR-9, EC-3)."""

    @staticmethod
    def _run(ml_segments, align_side_effect):
        mw = _fake_whisperx(align_side_effect)
        with patch.dict("sys.modules", {"whisperx": mw, "torch": MagicMock()}):
            result = _align_multilingual_segments("job-1", ml_segments, MagicMock(), "cpu")
        return result, mw

    def test_each_language_aligned_with_its_own_model(self):
        """BR-8: one align model per detected language; Thai uses the custom model."""
        ml = [
            {"start": 0.0, "end": 2.0, "text": "Hello", "language": "en"},
            {"start": 2.0, "end": 4.0, "text": "สวัสดี", "language": "th"},
        ]
        _result, mw = self._run(ml, _echo_align)

        calls = {c.kwargs["language_code"]: c.kwargs["model_name"] for c in mw.load_align_model.call_args_list}
        assert calls["en"] is None
        assert calls["th"] == "airesearch/wav2vec2-large-xlsr-53-th"

    def test_segments_grouped_by_language(self):
        """Segments of the same language are aligned together in a single call."""
        ml = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "language": "en"},
            {"start": 1.0, "end": 2.0, "text": "สวัสดี", "language": "th"},
            {"start": 2.0, "end": 3.0, "text": "Bye", "language": "en"},
        ]
        _result, mw = self._run(ml, _echo_align)

        # Two align calls (one per language), the English call carrying both English segments.
        assert mw.align.call_count == 2
        en_call = next(c for c in mw.align.call_args_list if len(c.args[0]) == 2)
        assert [s["text"] for s in en_call.args[0]] == ["Hello", "Bye"]

    def test_sentence_split_retains_all_segments(self):
        """align may return MORE segments than it was given (sentence-splitting); all are kept and tagged."""

        def split(segs, *args, **kwargs):
            return {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hello"},
                    {"start": 1.0, "end": 2.0, "text": "there"},
                ]
            }

        ml = [{"start": 0.0, "end": 2.0, "text": "Hello there", "language": "en"}]
        result, _mw = self._run(ml, split)

        assert len(result) == 2
        assert all(s["language"] == "en" for s in result)

    def test_align_failure_falls_back_to_segment_level(self):
        """Truth-2: a configured model that fails to align degrades timing only — text is preserved."""

        def boom(segs, *args, **kwargs):
            raise RuntimeError("alignment exploded")

        ml = [{"start": 1.0, "end": 3.0, "text": "Hello", "language": "en"}]
        result, _mw = self._run(ml, boom)

        assert len(result) == 1
        assert result[0]["text"] == "Hello"
        assert result[0]["start"] == 1.0
        assert result[0]["end"] == 3.0
        assert "words" not in result[0]

    def test_non_finite_timestamps_repaired(self):
        """NaN/inf timestamps from alignment are repaired so click-to-seek stays deterministic (AC3)."""

        def nan_align(segs, *args, **kwargs):
            return {"segments": [{"start": float("nan"), "end": float("inf"), "text": "Hello", "words": []}]}

        ml = [{"start": 5.0, "end": 7.0, "text": "Hello", "language": "en"}]
        result, _mw = self._run(ml, nan_align)

        # Non-finite start falls back to the group's first start; end clamps to >= start.
        assert result[0]["start"] == 5.0
        assert result[0]["end"] == 5.0

    def test_unsupported_language_skips_alignment_and_keeps_segment_level(self):
        """AC2/EC-3/BR-9: a language with no alignment model is never sent to load_align_model;
        its segment retains chunk-level timestamps, unchanged text, and no word data."""
        ml = [
            {"start": 0.0, "end": 2.0, "text": "Hello", "language": "en"},
            {"start": 2.0, "end": 4.0, "text": "xin chào", "language": "xx"},
        ]
        result, mw = self._run(ml, _echo_align)

        aligned_languages = [c.kwargs["language_code"] for c in mw.load_align_model.call_args_list]
        assert "xx" not in aligned_languages

        xx_seg = next(s for s in result if s["language"] == "xx")
        assert xx_seg["start"] == 2.0
        assert xx_seg["end"] == 4.0
        assert xx_seg["text"] == "xin chào"
        assert "words" not in xx_seg  # words-less segment still survives downstream diarization

    def test_output_sorted_by_start(self):
        """The merged cross-language result is ordered by start time (AC3)."""
        ml = [
            {"start": 4.0, "end": 6.0, "text": "later", "language": "th"},
            {"start": 0.0, "end": 2.0, "text": "first", "language": "en"},
        ]
        result, _mw = self._run(ml, _echo_align)

        starts = [s["start"] for s in result]
        assert starts == sorted(starts)

    def test_align_load_timeout_keeps_segment_level(self, monkeypatch):
        """A stalled alignment-model load degrades to segment-level timestamps
        instead of hanging (debug session macos-transcribe-stuck-70)."""
        monkeypatch.setattr(transcriber_module, "ALIGNMENT_TIMEOUT_SEC", 0.2)
        release = threading.Event()

        def _blocking_load(*a, **k):
            release.wait(30)
            return (MagicMock(), MagicMock())

        mw = MagicMock()
        mw.load_align_model.side_effect = _blocking_load
        mw.align.side_effect = _echo_align

        ml = [{"start": 1.0, "end": 3.0, "text": "Bonjour", "language": "fr"}]
        with patch.dict("sys.modules", {"whisperx": mw, "torch": MagicMock()}):
            result = _align_multilingual_segments("job-1", ml, MagicMock(), "cpu")
        release.set()

        seg = result[0]
        assert seg["start"] == 1.0
        assert seg["end"] == 3.0
        assert seg["text"] == "Bonjour"
        assert "words" not in seg  # never aligned; word data absent


class TestLoadAlignModelWatchdogged:
    """The align-model load surfaces a distinct download stage when an HF-backed
    model is not yet cached, so a mid-transcription fetch is not mistaken for a
    hang (#145). The queue keeps only the latest state, so stages are recorded
    via a spy on update_job."""

    @staticmethod
    def _record_stages(queue: JobQueue, job_id: str) -> list[str]:
        stages: list[str] = []
        original = queue.update_job

        def _spy(jid, **kwargs):
            if kwargs.get("stage") is not None:
                stages.append(kwargs["stage"])
            return original(jid, **kwargs)

        queue.update_job = _spy  # type: ignore[method-assign]
        return stages

    def _run(self, *, align_model_name, cached):
        queue = JobQueue()
        job = queue.create_job("m1")
        stages = self._record_stages(queue, job.id)
        with (
            patch.object(transcriber_module, "job_queue", queue),
            patch.object(transcriber_module, "align_model_cached", return_value=cached) as probe,
        ):
            status, loaded = transcriber_module._load_align_model_watchdogged(
                job.id,
                lambda: ("model", "metadata"),
                align_model_name=align_model_name,
                align_progress=50,
            )
        assert status == "ok"
        assert loaded == ("model", "metadata")
        return stages, probe

    def test_hf_backed_uncached_surfaces_download_then_aligning(self):
        stages, probe = self._run(align_model_name="jonatasgrosman/wav2vec2-x", cached=False)
        assert stages == ["downloading_align_model", "aligning"]
        probe.assert_called_once_with("jonatasgrosman/wav2vec2-x")

    def test_hf_backed_cached_does_not_flip_stage(self):
        stages, _probe = self._run(align_model_name="jonatasgrosman/wav2vec2-x", cached=True)
        assert "downloading_align_model" not in stages

    def test_torch_native_skips_probe_and_stage(self):
        # model_name=None (en/fr/de/es/it): shared torch cache, never a download stage.
        stages, probe = self._run(align_model_name=None, cached=False)
        assert "downloading_align_model" not in stages
        probe.assert_not_called()


class TestDiarizationWatchdog:
    """The diarization step must degrade to no-diarization instead of hanging
    forever when the pipeline stalls (debug session macos-transcribe-stuck-70)."""

    def _patch_diarize_module(self, pipeline_cls, assign=None):
        return patch.dict(
            sys.modules,
            {
                "torch": MagicMock(),
                "whisperx.diarize": MagicMock(
                    DiarizationPipeline=pipeline_cls,
                    assign_word_speakers=assign or MagicMock(),
                ),
            },
        )

    def test_times_out_and_degrades(self, monkeypatch):
        monkeypatch.setattr("config.current_hf_token", lambda: "hf_secret")
        monkeypatch.setattr(transcriber_module, "DIARIZATION_TIMEOUT_SEC", 0.2)

        release = threading.Event()

        def _never_returns(*a, **k):
            # Simulate a wedged pipeline construction (e.g. a stalled download).
            release.wait(30)
            return MagicMock()

        pipeline_cls = MagicMock(side_effect=_never_returns)
        result = {"segments": []}
        with self._patch_diarize_module(pipeline_cls):
            out, turns = _diarize_and_assign("job1", object(), None, "cpu", result)

        release.set()
        assert out is result  # unchanged transcript
        assert turns is None  # diarization skipped

    def test_completes_within_deadline(self, monkeypatch):
        monkeypatch.setattr("config.current_hf_token", lambda: "hf_secret")
        monkeypatch.setattr(transcriber_module, "DIARIZATION_TIMEOUT_SEC", 5)

        seg = MagicMock()
        seg.iterrows.return_value = [(0, {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"})]
        diarize_model = MagicMock(return_value=seg)
        pipeline_cls = MagicMock(return_value=diarize_model)
        with self._patch_diarize_module(pipeline_cls):
            out, turns = _diarize_and_assign("job1", object(), None, "cpu", {"segments": []})

        assert turns == [(0.0, 1.0, "SPEAKER_00")]
