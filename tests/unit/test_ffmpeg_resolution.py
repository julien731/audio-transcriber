from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.services import app_paths
from backend.services.audio_preprocessor import _convert_to_wav


class TestConvertToWavFfmpegResolution:
    def test_uses_resolved_ffmpeg_binary(self, tmp_path: Path):
        audio = tmp_path / "input.m4a"
        audio.write_bytes(b"fake")

        with (
            patch.object(app_paths, "ffmpeg_path", return_value="/bundle/bin/ffmpeg"),
            patch("backend.services.audio_preprocessor.subprocess.run") as mock_run,
        ):
            result = _convert_to_wav(audio)

        assert result == tmp_path / "audio_converted.wav"
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "/bundle/bin/ffmpeg"

    def test_falls_back_to_literal_ffmpeg_when_unresolved(self, tmp_path: Path):
        audio = tmp_path / "input.m4a"
        audio.write_bytes(b"fake")

        with (
            patch.object(app_paths, "ffmpeg_path", return_value=None),
            patch("backend.services.audio_preprocessor.subprocess.run", MagicMock()) as mock_run,
        ):
            _convert_to_wav(audio)

        assert mock_run.call_args.args[0][0] == "ffmpeg"
