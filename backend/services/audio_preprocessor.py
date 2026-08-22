from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from backend.services import app_paths

logger = logging.getLogger(__name__)

HIGHPASS_CUTOFF_HZ = 80
NOISE_PROP_DECREASE = 0.75
TARGET_LUFS = -23.0

SOUNDFILE_FORMATS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def _convert_to_wav(audio_path: Path) -> Path:
    """Convert non-WAV audio to WAV using ffmpeg. Returns path to the converted file."""
    wav_path = audio_path.parent / "audio_converted.wav"
    # Prefer the bundled, arm64-native ffmpeg over any system/Homebrew binary
    # (BR-3, BR-20, EC-10). Falls back to the literal "ffmpeg" only in a dev
    # checkout with nothing on PATH, preserving the previous behavior.
    ffmpeg = app_paths.ffmpeg_path() or "ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        check=True,
        capture_output=True,
    )
    return wav_path


def preprocess_audio(audio_path: Path) -> Path:
    """Apply audio preprocessing: high-pass filter, noise reduction, loudness normalization.

    Returns the path to the preprocessed WAV file (saved alongside the original).
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import butter, sosfilt

    logger.info("Preprocessing audio: %s", audio_path.name)

    converted_path = None
    if audio_path.suffix.lower() not in SOUNDFILE_FORMATS:
        logger.info("Converting %s to WAV via ffmpeg", audio_path.suffix)
        converted_path = _convert_to_wav(audio_path)
        read_path = converted_path
    else:
        read_path = audio_path

    data, sample_rate = sf.read(read_path, dtype="float64")

    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 1. High-pass filter (80 Hz, 4th-order Butterworth)
    sos = butter(4, HIGHPASS_CUTOFF_HZ, btype="high", fs=sample_rate, output="sos")
    data = sosfilt(sos, data)

    # 2. Noise reduction (conservative)
    import noisereduce as nr

    data = nr.reduce_noise(
        y=data,
        sr=sample_rate,
        prop_decrease=NOISE_PROP_DECREASE,
        stationary=False,
    )

    # 3. Loudness normalization to -23 LUFS
    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(data)

    if not np.isinf(loudness):
        data = pyln.normalize.loudness(data, loudness, TARGET_LUFS)

    # Save preprocessed copy
    output_path = audio_path.parent / "audio_preprocessed.wav"
    sf.write(str(output_path), data, sample_rate)

    if converted_path and converted_path.exists():
        converted_path.unlink()

    logger.info("Preprocessed audio saved: %s", output_path.name)
    return output_path
