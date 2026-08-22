# Audio Transcriber

Local audio/video transcription with speaker diarization, speaker labeling, and AI-powered meeting analysis. Built with WhisperX and Claude.

Works in two ways:
- **Web app** -- upload recordings, view synced transcripts, rename speakers, and generate AI analyses from your browser
- **CLI** -- transcribe files directly from the terminal

## Getting Started

### Step 0: Open Terminal

All commands below are run in the **Terminal** app.

- **macOS:** Open Finder → Applications → Utilities → Terminal (or press `Cmd + Space`, type "Terminal", and hit Enter)

### Step 1: Install Homebrew (macOS only)

Homebrew is a package manager that makes it easy to install developer tools on macOS. Skip this if you already have it (run `brew --version` to check).

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. When it finishes, **close and reopen Terminal** so the `brew` command is available.

### Step 2: Install Python 3.12 and ffmpeg

This project requires **Python 3.12 specifically** (not 3.11, not 3.13 -- some dependencies only work with 3.12).

```bash
# macOS
brew install python@3.12 ffmpeg ffmpeg@7
# Symlink ffmpeg@7's versioned dylibs so torchcodec (a transitive
# dependency of pyannote.audio) can find them at runtime:
for lib in libavcodec.61 libavdevice.61 libavfilter.10 libavformat.61 \
           libavutil.59 libpostproc.58 libswresample.5 libswscale.8; do
  ln -sf "/opt/homebrew/opt/ffmpeg@7/lib/${lib}.dylib" \
         "/opt/homebrew/lib/${lib}.dylib"
done
```

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3.12 python3.12-venv ffmpeg
```

Verify both are installed:

```bash
python3.12 --version   # should print Python 3.12.x
ffmpeg -version         # should print version info
```

### Step 3: Download and install the project

```bash
cd ~/Desktop
git clone <this-repo-url>
cd audio-transcription
make setup
```

> **Don't have `git`?** Run `brew install git` (macOS) or `sudo apt install git` (Linux) first, or download the project as a ZIP from GitHub and unzip it.

`make setup` creates an isolated Python environment and installs all dependencies. This may take a few minutes (PyTorch is a large download).

### Step 4: Get a HuggingFace token (free)

Speaker identification requires a free HuggingFace account and token.

1. Create an account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens and create a token (choose "Read" access)
3. Accept the license agreement on each of these model pages (click "Agree and access repository" on each):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1

### Step 5: Configure your `.env` file

```bash
cp .env.example .env
```

Open `.env` in any text editor (TextEdit on macOS, or `nano .env` in Terminal) and paste your token:

```
HF_TOKEN=hf_your_token_here
```

The `.env` file is gitignored, so your keys stay private.

### Step 6: Run the app

```bash
make run
```

Open http://localhost:8000 in your browser. You can now upload audio/video files and start transcribing.

> **Use `localhost`, not `0.0.0.0`.** Some browser features like desktop notifications require a secure context. Chrome treats `localhost` as secure, but not `0.0.0.0`.

## Web App

The web app lets you upload recordings, track transcription progress, view transcripts synchronized with audio playback, and generate LLM-ready prompts for analysis.

### Features

- **Upload** audio/video files (mp3, mp4, m4a, wav, webm) up to 500 MB
- **Real-time progress** tracking as files are transcribed
- **Audio player** synced with the transcript -- click any line to jump to that moment
- **Playback speed** control (0.5x to 2x)
- **Speaker renaming** -- click a speaker label to assign a real name; recent names are remembered
- **LLM-ready analysis prompts** -- pick a template (interview, sales, client, general), and the app combines it with your transcript into a prompt you can paste into any LLM
- **Desktop notifications** -- get a browser notification when transcription finishes (or fails), even if the tab is in the background
- **Retry** failed transcriptions

### Upload Options

When uploading, you can optionally specify:

| Option | Default | What it does |
|--------|---------|-------------|
| Title | Filename | Display name for the meeting |
| Meeting type | Other | Determines which analysis template is used |
| Language | Auto-detect | Set explicitly for faster transcription |
| Number of speakers | Auto-detect | Set explicitly for better speaker identification |

## Bundled Local Service

The backend can be packaged as a self-contained, Apple-Silicon service that runs
with no pre-installed Python, Homebrew, ffmpeg, or environment variables. It
provisions its models on first run, stores data under Application Support, and
exposes the full HTTP API for a thin native client to drive. See
[docs/packaging.md](docs/packaging.md) for how to build and run it.

## CLI

The CLI is useful for batch processing or scripting.

**Basic transcription:**
```bash
python transcriber.py meeting.mp3
```

Supports mp3, mp4, wav, m4a, and other ffmpeg-compatible formats. Output is saved alongside the input file (e.g., `meeting.mp3` -> `meeting.txt`).

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
# Specify language (faster -- skips auto-detection)
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

## Configuration

All settings are configured via environment variables (in your `.env` file or exported in your shell).

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | | HuggingFace token for speaker diarization |
| `WHISPER_MODEL` | `large-v3` | Whisper model size (`large-v3`, `medium`, `small`, `base`) |
| `WHISPER_DEVICE` | `auto` | Compute device (`auto`, `cuda`, `cpu`) |
| `WHISPER_BATCH_SIZE` | `16` | Batch size for transcription (lower if out of memory) |
| `DATA_DIR` | `./data` | Where meeting data is stored |
| `MAX_UPLOAD_SIZE` | `500MB` | Maximum upload file size |

## Tips

- **Language:** Use `--language en` (or `fr`, `de`, etc.) to skip auto-detection and speed up transcription.
- **Model choice:** `large-v3` is best quality but slower. `medium` is a good balance. `small` or `base` for quick drafts.
- **GPU memory:** If you run out, reduce `--batch-size` (CLI) or set `WHISPER_BATCH_SIZE` in `.env` (web app) to 8 or 4.
- **Known speaker count:** Specifying the number of speakers improves diarization accuracy.
- **No GPU?** It still works on CPU, just slower. Set `WHISPER_DEVICE=cpu` if auto-detection has issues.
