# Packaging the bundled service

The backend can run as a self-contained, Apple-Silicon service that needs no
pre-installed Python, Homebrew, ffmpeg, or environment variables (spec:
`docs/specs/local-transcription-service.md`, BR-1/2/3). This document covers
producing that bundle. The native macOS app that consumes the service is a
separate concern (`docs/specs/native-macos-app.md`).

> Status: the runtime service behavior (ports, provisioning, isolation, data
> migration, offline handling, API) is implemented and tested. Producing and
> verifying the frozen artifact happens **on an Apple-Silicon build machine** —
> it is not built in CI (ubuntu, lightweight deps) because it pulls the multi-GB
> torch/whisperx/pyannote tree.

## Prerequisites

- An Apple-Silicon (arm64) Mac
- Python 3.12 and a clean virtualenv with `requirements.txt` installed
- `pip install pyinstaller`

## Build steps

```bash
# 1. Vendor the static, arm64-native ffmpeg + ffprobe into vendor/bin
./scripts/vendor_ffmpeg.sh

# 2. Build the self-contained service
pyinstaller MeetingTranscriber.spec

# 3. The bundle is written to dist/MeetingTranscriber/
```

## Running the bundle

The bundled executable (`dist/MeetingTranscriber/MeetingTranscriber`) is the
entrypoint defined in `service_main.py`. On launch it:

1. Binds an ephemeral `localhost` port (BR-4).
2. Announces the port two ways (OQ-2): a `service.json` file in Application
   Support and a stdout line `{"event":"ready","port":<N>}`.
3. Serves the API with auto-reload disabled (BR-6).

A client discovers the port from either signal, then polls
`GET /api/health` until `status: "ready"` before issuing requests (BR-5).

## Runtime behavior (bundled mode)

Bundled mode is detected via `sys.frozen` (`config.is_bundled()`). In this mode,
distinct from a developer checkout:

- Configuration is read **only** from `~/Library/Application Support/
  MeetingTranscriber/config.json`; ambient env vars and stray `.env` files are
  ignored (BR-19, BR-21).
- Writable data (meetings, config, models) lives under Application Support, never
  inside the read-only bundle (BR-12).
- The model cache (`HF_HOME`) and ffmpeg lookup are redirected into the bundle /
  Application Support for this process only — the user's global PATH and shell
  environment are untouched (BR-20).
- On first run, a prior Terminal-installation data directory (if found) is
  imported non-destructively (BR-13, BR-14); otherwise the library starts empty
  (BR-15).

## First-run provisioning

Models are not bundled (they are multi-GB). On first run the client:

1. `POST /api/provisioning/token` with the user's HuggingFace token (optional —
   an empty token disables diarization, BR-10).
2. `POST /api/provisioning/models` to download the transcription and diarization
   models into Application Support, polling `GET /api/provisioning` for progress
   (BR-9, BR-11).

Until models are present, new-transcription requests are rejected with a clear
reason (a connection is required for the first-time download — BR-24, BR-25).

## Notes / open items

- `OQ-1` (PyInstaller vs py2app): this spec ships a PyInstaller spec. py2app is
  the native app's territory (`.app` shell, signing, distribution).
- `OQ-3` (Keychain): the token is currently stored in a `0600` `config.json`.
  Moving it to the macOS Keychain is a future refinement.
- `OQ-8` (model cache): the cache is isolated in Application Support. Seeding it
  from an existing `~/.cache/huggingface` is a possible future optimization.
