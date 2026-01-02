#!/usr/bin/env python3
import warnings
import logging
import os
import sys

# Suppress noisy warnings from dependencies (must be done before imports)
warnings.filterwarnings("ignore", message="Model was trained with")
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends")
warnings.filterwarnings("ignore", message="resource_tracker:")
os.environ["PYTHONWARNINGS"] = "ignore"

# Filter out noisy messages from stdout/stderr
_SUPPRESSED_MESSAGES = [
    "Lightning automatically upgraded",
    "No language specified",
    "Performing voice activity detection",
]

class _FilteredStream:
    def __init__(self, original):
        self.original = original
    def write(self, msg):
        if not any(s in msg for s in _SUPPRESSED_MESSAGES):
            self.original.write(msg)
    def flush(self):
        self.original.flush()
    def __getattr__(self, name):
        return getattr(self.original, name)

sys.stdout = _FilteredStream(sys.stdout)
sys.stderr = _FilteredStream(sys.stderr)

# Fix for PyTorch 2.6+ which changed weights_only default to True
# pyannote/whisperx models aren't yet compatible with weights_only=True
import torch
import functools

_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False to support older model formats
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

"""
Audio Transcriber with Speaker Diarization
Uses WhisperX for transcription and pyannote-audio for speaker labeling.

Requirements:
    pip install whisperx torch torchaudio
    ffmpeg must be installed on your system (brew install ffmpeg / apt install ffmpeg)

Setup:
    1. Get a HuggingFace token from https://huggingface.co/settings/tokens
    2. Accept the pyannote model terms:
       - https://huggingface.co/pyannote/speaker-diarization-3.1
       - https://huggingface.co/pyannote/segmentation-3.0
    3. Set your token as an environment variable:
       export HF_TOKEN="your_token_here"

Usage:
    python transcriber.py audio.mp3
    python transcriber.py video.mp4 --model medium --output transcript.txt
    python transcriber.py recording.wav --speakers "Alice,Bob,Carol"
    python transcriber.py audio.mp3 --language en --quiet
"""

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

CACHE_VERSION = 1


def get_cache_path(audio_path: str) -> Path:
    """Get the cache file path for an audio file."""
    return Path(audio_path).with_suffix(".transcription_cache.json")


def load_cache(audio_path: str, model: str, language: str = None) -> dict | None:
    """Load cached transcription if valid. Returns None if cache is invalid/missing."""
    cache_path = get_cache_path(audio_path)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        # Validate cache version
        if cache.get("cache_version") != CACHE_VERSION:
            return None

        # Validate audio file hasn't changed
        audio_mtime = Path(audio_path).stat().st_mtime
        if cache.get("audio_mtime") != audio_mtime:
            return None

        # Validate model matches
        if cache.get("model") != model:
            return None

        # Validate language matches (None matches None)
        if cache.get("language") != language:
            return None

        return cache.get("result")
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def save_cache(audio_path: str, result: dict, model: str, language: str = None):
    """Save transcription result to cache."""
    cache_path = get_cache_path(audio_path)
    audio_mtime = Path(audio_path).stat().st_mtime

    cache = {
        "cache_version": CACHE_VERSION,
        "audio_mtime": audio_mtime,
        "model": model,
        "language": language,
        "result": result,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    # Note: MPS (Apple Silicon) is not supported by ctranslate2/faster_whisper
    return "cpu"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


def log(message: str, quiet: bool = False):
    """Print message unless quiet mode is enabled."""
    if not quiet:
        print(message)


def transcribe_meeting(
    audio_path: str,
    model_name: str = "large-v3",
    device: str = None,
    batch_size: int = None,
    hf_token: str = None,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
    language: str = None,
    quiet: bool = False,
) -> dict:
    """
    Transcribe an audio file with speaker diarization.

    Args:
        audio_path: Path to the audio/video file
        model_name: Whisper model size (tiny, base, small, medium, large-v3)
        device: Device to use (cuda, cpu). Auto-detected if None.
        batch_size: Batch size for transcription. Auto-adjusted based on device if None.
        hf_token: HuggingFace token for diarization models
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        language: Language code (e.g., 'en', 'fr'). Auto-detected if None.
        quiet: Suppress progress messages if True.

    Returns:
        Dictionary with segments containing speaker labels and timestamps

    Raises:
        TranscriptionError: If transcription fails or returns invalid results
    """
    device = device or get_device()
    compute_type = "float16" if device == "cuda" else "float32"

    # Auto-adjust batch size based on device
    if batch_size is None:
        batch_size = 16 if device == "cuda" else 4

    log(f"Using device: {device}", quiet)
    log(f"Loading audio: {audio_path}", quiet)

    # Load audio
    try:
        audio = whisperx.load_audio(audio_path)
    except Exception as e:
        raise TranscriptionError(f"Failed to load audio file: {e}")

    # Transcribe
    log(f"Transcribing with {model_name} model...", quiet)
    try:
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        transcribe_options = {
            "batch_size": batch_size,
            "print_progress": not quiet,
        }
        if language:
            transcribe_options["language"] = language
        result = model.transcribe(audio, **transcribe_options)
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")

    # Validate result
    if not result or "segments" not in result:
        raise TranscriptionError("Transcription returned no results")

    if not result["segments"]:
        raise TranscriptionError("No speech detected in audio")

    detected_language = result.get("language", language or "unknown")
    log(f"Detected language: {detected_language}", quiet)

    # Free up GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Align for word-level timestamps
    log("Aligning timestamps...", quiet)
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
            print_progress=not quiet,
        )
    except Exception as e:
        raise TranscriptionError(f"Timestamp alignment failed: {e}")

    # Validate alignment result
    if not result or "segments" not in result:
        raise TranscriptionError("Alignment returned no results")

    # Free up GPU memory
    del model_a
    if device == "cuda":
        torch.cuda.empty_cache()

    # Speaker diarization
    if hf_token:
        log("Running speaker diarization...", quiet)
        try:
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )

            diarize_kwargs = {}
            if num_speakers:
                diarize_kwargs["num_speakers"] = num_speakers
            if min_speakers:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers:
                diarize_kwargs["max_speakers"] = max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = assign_word_speakers(diarize_segments, result)

            unique_speakers = set(
                s.get("speaker", "UNKNOWN") for s in result["segments"]
            )
            log(f"Identified speakers: {len(unique_speakers)}", quiet)
        except Exception as e:
            raise TranscriptionError(f"Speaker diarization failed: {e}")
    else:
        log("Skipping diarization (no HF_TOKEN provided)", quiet)

    return result


def format_transcript_txt(result: dict, speaker_names: dict = None) -> str:
    """Format transcript as readable text with timestamps and speaker labels."""
    lines = []
    speaker_names = speaker_names or {}
    
    for segment in result["segments"]:
        timestamp = format_timestamp(segment["start"])
        speaker = segment.get("speaker", "UNKNOWN")
        speaker_display = speaker_names.get(speaker, speaker)
        text = segment["text"].strip()
        
        lines.append(f"[{timestamp}] {speaker_display}: {text}")
    
    return "\n".join(lines)


def format_transcript_srt(result: dict, speaker_names: dict = None) -> str:
    """Format transcript as SRT subtitles with speaker labels."""
    lines = []
    speaker_names = speaker_names or {}
    
    for i, segment in enumerate(result["segments"], 1):
        start = format_srt_timestamp(segment["start"])
        end = format_srt_timestamp(segment["end"])
        speaker = segment.get("speaker", "UNKNOWN")
        speaker_display = speaker_names.get(speaker, speaker)
        text = segment["text"].strip()
        
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(f"[{speaker_display}] {text}")
        lines.append("")
    
    return "\n".join(lines)


def format_transcript_json(result: dict, speaker_names: dict = None) -> str:
    """Format transcript as JSON."""
    speaker_names = speaker_names or {}
    
    output = {
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "speaker": speaker_names.get(
                    segment.get("speaker", "UNKNOWN"), 
                    segment.get("speaker", "UNKNOWN")
                ),
                "text": segment["text"].strip()
            }
            for segment in result["segments"]
        ]
    }
    
    return json.dumps(output, indent=2, ensure_ascii=False)


def identify_speakers(result: dict) -> dict:
    """Show first utterance from each speaker for identification."""
    speakers = {}
    max_preview_len = 80
    for segment in result["segments"]:
        speaker = segment.get("speaker", "UNKNOWN")
        if speaker not in speakers:
            timestamp = format_timestamp(segment["start"])
            text = segment["text"].strip()
            if len(text) > max_preview_len:
                text = text[:max_preview_len] + "..."
            speakers[speaker] = f"[{timestamp}] \"{text}\""
    return speakers


def prompt_speaker_names(result: dict) -> dict:
    """Interactively prompt user to name each speaker."""
    speakers_info = identify_speakers(result)
    speaker_names = {}

    print("\n" + "=" * 60)
    print("SPEAKER IDENTIFICATION")
    print("=" * 60)
    print("Enter a name for each speaker (or press Enter to keep ID):\n")

    for speaker_id, sample in sorted(speakers_info.items()):
        print(f"{speaker_id}:")
        print(f"  {sample}")
        try:
            name = input(f"  Name for {speaker_id}: ").strip()
            if name:
                speaker_names[speaker_id] = name
        except EOFError:
            break

    print("=" * 60 + "\n")
    return speaker_names


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video files with speaker diarization"
    )
    parser.add_argument("audio_file", help="Path to audio/video file")
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: same name as input with .txt extension)"
    )
    parser.add_argument(
        "--format", "-f",
        default="txt",
        choices=["txt", "srt", "json"],
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for transcription (default: 16 for CUDA, 4 for CPU)"
    )
    parser.add_argument(
        "--language",
        help="Language code (e.g., 'en', 'fr'). Auto-detected if not specified."
    )
    parser.add_argument(
        "--speakers",
        help="Comma-separated speaker names in order (e.g., 'Alice,Bob,Carol')"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers (helps diarization accuracy)"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers"
    )
    parser.add_argument(
        "--identify-speakers",
        action="store_true",
        help="Show first utterance from each speaker and exit (for mapping names)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactively prompt for speaker names after transcription"
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Skip speaker diarization"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages (only show errors and final output path)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached transcription and re-transcribe"
    )

    args = parser.parse_args()

    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Get HuggingFace token
    hf_token = None if args.no_diarization else os.environ.get("HF_TOKEN")
    if not hf_token and not args.no_diarization and not args.quiet:
        print("Warning: HF_TOKEN not set. Speaker diarization will be skipped.")
        print("Set it with: export HF_TOKEN='your_huggingface_token'")
        print()

    # Try to load from cache first (unless --no-cache is specified)
    result = None
    if not args.no_cache:
        result = load_cache(str(audio_path), args.model, args.language)
    if result:
        if not args.quiet:
            print("Using cached transcription")
    else:
        # Transcribe with keyboard interrupt handling
        try:
            result = transcribe_meeting(
                str(audio_path),
                model_name=args.model,
                device=args.device,
                batch_size=args.batch_size,
                hf_token=hf_token,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                language=args.language,
                quiet=args.quiet,
            )
            # Save to cache for future runs
            save_cache(str(audio_path), result, args.model, args.language)
        except KeyboardInterrupt:
            print("\nTranscription cancelled by user.", file=sys.stderr)
            sys.exit(130)
        except TranscriptionError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # If identify-speakers mode, show speakers and exit
    if args.identify_speakers:
        print("\n" + "=" * 60)
        print("SPEAKER IDENTIFICATION")
        print("=" * 60)
        speakers = identify_speakers(result)
        for speaker, sample in sorted(speakers.items()):
            print(f"\n{speaker}:")
            print(f"  {sample}")
        print("\n" + "=" * 60)
        print("Re-run with --speakers 'Name1,Name2,...' (cached, instant)")
        print("=" * 60)
        return

    # Build speaker name mapping
    speaker_names = {}
    unique_speakers = sorted(set(
        s.get("speaker", "UNKNOWN")
        for s in result["segments"]
    ))

    if args.interactive and hf_token:
        # Interactive mode: prompt for each speaker name
        try:
            speaker_names = prompt_speaker_names(result)
        except KeyboardInterrupt:
            print("\nSpeaker naming cancelled.", file=sys.stderr)
    elif args.speakers:
        # Use provided speaker names
        names = [n.strip() for n in args.speakers.split(",")]

        # Warn about mismatch
        if len(names) != len(unique_speakers) and not args.quiet:
            print(f"Warning: {len(names)} names provided but "
                  f"{len(unique_speakers)} speakers detected.")
            if len(names) < len(unique_speakers):
                print("Some speakers will keep their default IDs.")
            else:
                print("Extra names will be ignored.")
            print()

        for i, speaker_id in enumerate(unique_speakers):
            if i < len(names):
                speaker_names[speaker_id] = names[i]

    # Format output
    formatters = {
        "txt": format_transcript_txt,
        "srt": format_transcript_srt,
        "json": format_transcript_json,
    }
    formatted = formatters[args.format](result, speaker_names)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = audio_path.with_suffix(f".{args.format}")

    # Write output
    output_path.write_text(formatted, encoding="utf-8")
    print(f"\nTranscript saved to: {output_path}")

    # Print summary
    if not args.quiet:
        duration = result["segments"][-1]["end"] if result["segments"] else 0
        print(f"Duration: {format_timestamp(duration)}")
        print(f"Segments: {len(result['segments'])}")
        if hf_token:
            print(f"Speakers: {len(unique_speakers)}")


if __name__ == "__main__":
    main()
