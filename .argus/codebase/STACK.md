# Tech Stack

## Overview

Three build targets from one repo:
```
backend/ + config.py + transcriber.py   → Python FastAPI transcription service
frontend/                               → vanilla JS SPA served by FastAPI
macos/                                  → native SwiftUI app wrapping the service
+ Node tooling (root package.json)      → semantic-release automation only
```

## Languages / Runtimes

- **Python 3.12** — backend, transcription, CLI (`.tool-versions` pins none for Python; `target-version = py312`)
- **Swift 6** (tools 6.0, language mode v5) — native macOS app, `macos/`
- **JavaScript (ES6)** — frontend SPA (no framework, no bundler)
- **Node.js >= 24** (`.nvmrc` = 24, `.tool-versions` = 24.0.1) — release tooling only, not app runtime
- **HTML/CSS** — frontend

## Backend (Python)

Deps in `requirements.txt`:
- **FastAPI** >= 0.115.0, **Uvicorn[standard]** >= 0.34.0 (ASGI), **Pydantic** (via FastAPI)
- **python-multipart** >= 0.0.20, **python-dotenv** >= 1.0.0
- **WhisperX** (`git+https://github.com/m-bain/whisperX.git`, unpinned) — STT + alignment
- **torch**, **torchaudio** (unpinned); **torchcodec == 0.7.0** (pinned to torch 2.8 ABI; needs FFmpeg 4-7)
- **huggingface_hub** >= 0.24.0 — model provisioning via `snapshot_download`
- **noisereduce** >= 3.0.0, **pyloudnorm** >= 0.1.1, **soundfile** >= 0.12.0, **praat-parselmouth** >= 0.4.7 — audio preprocessing
- PyAnnote (`pyannote/speaker-diarization-3.1`) pulled transitively via WhisperX

Dev deps (`requirements-dev.txt` + tail of `requirements.txt`): pytest >= 8, pytest-asyncio >= 0.24, httpx >= 0.27, pytest-cov >= 6, **ruff** >= 0.9.

## Frontend

- Vanilla JS, no build step. Global `<script>` tags in `frontend/index.html`. CSS custom-property theming.

## macOS App (`macos/`)

- **Swift Package Manager** (`Package.swift`, tools 6.0), platform `.macOS(.v13)`, deploy target 13.0
- **SwiftUI** `@main` shell; logic split into `MeetingTranscriberKit` (testable lib) + `MeetingTranscriberApp` (UI)
- **Sparkle 2.9.6** (pinned exact, `Package.resolved`) — auto-updater
- Tests are `swift run` executables, not `swift test` (CLT-only toolchain): `MeetingTranscriberKitTests`, `MeetingTranscriberIntegrationTests`
- App is a thin front end; all transcription/workflow logic stays in the Python service over localhost HTTP
- Not sandboxed; library validation disabled (`macos/Resources/MeetingTranscriber.entitlements`)

## Bundled Service Packaging

- **PyInstaller** spec `MeetingTranscriber.spec` — self-contained arm64 service embedding Python + torch/whisperx/pyannote + vendored ffmpeg
- **FFmpeg 7.1** static arm64 binaries vendored via `scripts/vendor_ffmpeg.sh` → `vendor/bin`
- Service entrypoint `service_main.py` (ephemeral port + stdout/`service.json` handshake); dev entrypoint `run.py` (Uvicorn :8000, reload)
- Pinned service artifact tracked in `macos/service-manifest.json` (version/commit/sha256/arch)

## Package Managers

- **pip** + venv (`.venv`), driven by `Makefile` (`make setup`/`run`/`app`)
- **npm** (`package-lock.json`) — release tooling only; `semantic-release` 25.0.3, `@semantic-release/{exec,github}`, `@bobvanderlinden/semantic-release-pull-request-analyzer`
- **SwiftPM** for macos

## Tooling / CI

- **Ruff** lint + format (`pyproject.toml`: line-length 120, select E/F/I/W, double quotes)
- **pytest** (`asyncio_mode = auto`; markers unit/integration/e2e), **coverage** `fail_under = 80` over `backend`,`config`
- CI `.github/workflows/ci.yml`: Lint & Format + Test & Coverage on ubuntu, Python 3.12
- Release `.github/workflows/release.yml` + `release-macos.yml` (macos-26 / Tahoe SDK), config `.releaserc.js`
- Conductor scripts `.conductor/settings.toml`: `make setup`, run `web` (default) + `app`
- Entry points: `run.py` (dev), `service_main.py` (bundled), `transcriber.py` (standalone CLI), `macos/scripts/build_app.sh`

## Storage / Build Tools

- File-based storage, no database. Dev: `DATA_DIR` (default `./data`); bundled: Application Support.
- No Dockerfile.
