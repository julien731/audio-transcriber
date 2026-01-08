# Audio Transcriber

Local audio/video transcription with speaker diarization using WhisperX.

## Setup

1. **Install ffmpeg** (required for audio processing):
   ```bash
   brew install ffmpeg        # macOS
   # apt install ffmpeg       # Ubuntu/Debian
   ```

2. **Install Python dependencies:**
   ```bash
   pip install whisperx torch torchaudio
   ```

3. **Get HuggingFace token** (required for speaker diarization):
   - Create account at https://huggingface.co
   - Get token from https://huggingface.co/settings/tokens
   - Accept terms for these models:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0

4. **Set token** (choose one method):

   **Option A: Using a `.env` file (recommended)**

   Copy the example file and add your token:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and replace `your_huggingface_token_here` with your actual token:
   ```
   HF_TOKEN=hf_your_actual_token
   ```
   The `.env` file is gitignored, so your token stays private.

   **Option B: Using an environment variable**
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
   Note: This only persists for the current terminal session.

## Usage

**Basic transcription:**
```bash
python transcriber.py meeting.mp3
```

Supports mp3, mp4, wav, m4a, and other ffmpeg-compatible formats. Output is saved alongside the input file (e.g., `meeting.mp3` → `meeting.txt`).

**With speaker names:**
```bash
# First, identify who's who
python transcriber.py meeting.mp3 --identify-speakers

# Then run with names mapped
python transcriber.py meeting.mp3 --speakers "Julien,Alice,Bob"

# Or use interactive mode to name speakers as you go
python transcriber.py meeting.mp3 --interactive
```

**Options:**
```bash
# Specify language (faster — skips auto-detection)
python transcriber.py meeting.mp3 --language en

# Use smaller model (faster, less accurate)
python transcriber.py meeting.mp3 --model medium

# Output as SRT subtitles
python transcriber.py meeting.mp3 --format srt

# Output as JSON
python transcriber.py meeting.mp3 --format json -o transcript.json

# Help with diarization accuracy
python transcriber.py meeting.mp3 --num-speakers 3

# Lower batch size if you get out-of-memory errors
python transcriber.py meeting.mp3 --batch-size 4

# Skip diarization entirely
python transcriber.py meeting.mp3 --no-diarization

# Quiet mode (suppress progress messages)
python transcriber.py meeting.mp3 --quiet
```

## Output Example

```
[00:00:15] Julien: So let's start with the Q4 roadmap discussion.
[00:00:42] Alice: I think we should prioritize the API work first.
[00:01:03] Julien: Makes sense. What's the timeline looking like?
[00:01:15] Bob: I'd estimate about three weeks for the core functionality.
```

## Tips

- **Language:** Use `--language en` (or `fr`, `de`, etc.) to skip auto-detection and speed up transcription.
- **Model choice:** `large-v3` is best quality but slower. `medium` is a good balance. `small` or `base` for quick drafts.
- **GPU memory:** If you run out, reduce `--batch-size` to 8 or 4.
- **Known speaker count:** Use `--num-speakers` if you know exactly how many people are in the meeting — improves accuracy.
- **Long meetings:** 1-1.5h meetings typically take 15-25 min to process with `large-v3` on a decent GPU.
