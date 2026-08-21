# Spec: Local Transcription Service (Bundled Backend)

## Overview

### Problem Statement

The transcription pipeline (FastAPI + WhisperX + PyAnnote) works, but today it is only reachable by first surviving a Terminal ordeal — install Homebrew, clone the repo, create a virtualenv, install dependencies, set environment variables, and launch a server. That barrier is being removed so the team can eventually run a real native Mac app (see `native-macos-app.md`). Before that native app can exist, the backend has to become a **self-contained local service**: a bundled process that starts with no pre-installed Python, Homebrew, ffmpeg, or environment variables, provisions its own models, stores data in a user-writable location, and exposes a complete HTTP API that a thin client can drive without reimplementing business logic. This spec covers turning the backend into that bundled service. It does not build any UI — the native app is a separate spec that consumes this service.

### Goals

- Make the backend a self-contained local service that runs with zero pre-installed developer tooling (no system Python, Homebrew, ffmpeg, or env vars).
- Bundle the Python interpreter, all libraries, and required native binaries so the service runs on a clean Apple Silicon machine.
- Provision the HuggingFace token and ML models on first run, into a user-writable location.
- Make the HTTP API the single source of truth — move business logic that currently lives in the web client (notably analysis-prompt assembly) into the API so any client stays thin.
- Store all writable data outside the bundle and non-destructively import existing Terminal-installation data.
- Guarantee the service never conflicts with an existing Terminal installation on the same machine.

### Scope

This spec covers:

- Packaging the backend as a bundled, self-contained service for Apple Silicon (interpreter + libraries + native binaries such as ffmpeg).
- Service lifecycle: start on a local ephemeral port, expose a health signal, run transcription jobs in the background.
- First-run provisioning of the HuggingFace token and ML models into a user-writable location.
- Relocating all writable data (meetings, config, models) to a user-writable location and non-destructively importing existing data.
- API completeness: moving client-side business logic (analysis-prompt assembly) server-side so the API is the single source of truth.
- Isolation from any existing Terminal installation on the same machine.
- Offline behavior of the service.

Out of scope:

- Any user interface — the native SwiftUI app that consumes this service is specified in `native-macos-app.md`.
- App-shell concerns owned by the native app: window/menu/dock behavior, auto-update, code-signing, Gatekeeper, and distribution (see `native-macos-app.md`).
- Running transcription against a remote/on-premise server. This spec keeps the service local-only; remote is a future spec. See "Future Considerations" in `.argus/project.md`.
- Changes to transcription quality, diarization accuracy, or analysis content — tracked in `transcription-quality-improvements.md`, `multilingual-transcription.md`, and related specs.
- Authentication (the service binds to localhost only and remains unauthenticated, consistent with current behavior).

## User Stories

### Client Application (the native app, or any future client)

- As a **Client**, I can start the bundled service and learn when it is ready to accept requests, so that I can present a working UI without race conditions.
- As a **Client**, I can drive the entire transcription workflow (upload, poll job progress, read transcripts, rename speakers, generate analysis prompts) through the HTTP API without reimplementing any business logic locally, so that the client stays thin.
- As a **Client**, I can request a fully-assembled analysis prompt for a meeting from the API, so that I do not have to replicate template substitution.
- As a **Client**, I can rely on the service running with no pre-installed tooling on a clean machine, so that I do not have to manage dependencies.

### Maintainer (service publisher)

- As a **Maintainer**, I can build a self-contained service bundle that includes the interpreter, libraries, and native binaries, so that it runs anywhere on Apple Silicon.
- As a **Maintainer**, I can rely on the service isolating itself from any existing Terminal installation, so that early adopters can run both during transition.

## Business Rules

### Service Packaging & Runtime

| # | Rule | Rationale |
|---|------|-----------|
| BR-1 | The service is delivered as a self-contained bundle targeting Apple Silicon. | Target machines are Apple Silicon (per project constraints). |
| BR-2 | The bundle embeds the Python interpreter and all Python libraries (transcription, diarization, web server) so nothing needs to be installed by the user. | The current Terminal-based dependency install is the exact barrier being removed. |
| BR-3 | The bundle includes all required native binaries — notably ffmpeg for audio decoding — rather than relying on system-installed versions. | ffmpeg is a native executable (not a Python wheel) that WhisperX shells out to; without it bundled, transcription fails on a clean machine. |
| BR-4 | The service binds to `localhost` on an ephemeral (auto-selected) free port and never assumes a fixed port. | Avoids collisions with other software and with an old Terminal server; localhost preserves the private, unauthenticated model. |
| BR-5 | The service exposes a health/readiness signal that a client can poll to know when it is ready to accept requests. | A client must not present UI before the service is up. |
| BR-6 | The service runs without auto-reload and runs transcription jobs in background threads as it does today. | Reload is a dev-only concern; the background-job model is unchanged. |
| BR-7 | The service runs fully locally; no audio, transcript, or meeting data is transmitted to any external service. | Privacy is the core reason the product exists (per project.md). |

### First-Run Provisioning

| # | Rule | Rationale |
|---|------|-----------|
| BR-8 | On first run, the service accepts a HuggingFace token provided by the client and stores it locally; the token is never embedded in the distributed build. | Diarization requires the token; embedding a shared token in every build is a credential-leak risk. |
| BR-9 | The service downloads the required transcription and diarization models into a user-writable location and exposes download progress to the client. | Models are multi-GB; bundling them bloats the artifact, so they are fetched once on first run, and the client needs progress to display it. |
| BR-10 | If the token is empty or rejected, the service runs transcription with diarization disabled rather than failing outright, and reports that diarization is unavailable. | Mirrors the existing graceful-degradation behavior (diarization skipped when token absent). |
| BR-11 | The service exposes whether first-run provisioning is complete (required models present); this state is persisted so provisioning does not repeat. | Lets the client decide whether to show setup, and avoids re-downloading. |

### Data Storage & Migration

| # | Rule | Rationale |
|---|------|-----------|
| BR-12 | All writable data (meetings, configuration, downloaded models) is stored under a user-writable Application Support location, not inside the bundle. | Bundles are read-only; writing inside them fails and breaks on update. |
| BR-13 | On first run, the service detects an existing Terminal-installation data directory and imports its meetings into the new location. | Existing users must not lose their transcription history. |
| BR-14 | Import copies rather than moves data, leaving the original directory intact. | Non-destructive migration lets users fall back to the old install. |
| BR-15 | If no prior data directory is found, first run proceeds with an empty library without error. | New users are the common case and must not see migration errors. |

### API Completeness (single source of truth)

| # | Rule | Rationale |
|---|------|-----------|
| BR-16 | Business logic currently performed by the web client is moved into the HTTP API so that any client can stay thin. The API is the single source of truth for workflow behavior. | A native client must not reimplement logic that lives in JavaScript today; divergence between clients is a maintenance and correctness risk. |
| BR-17 | The API exposes an endpoint that returns a fully-assembled analysis prompt for a meeting — with template selection, meeting context, audio-analysis context, and transcript already substituted — rather than returning only a raw template. | Prompt assembly (placeholder substitution) currently happens in `analysis-viewer.js`; centralizing it means clients request a ready-to-use prompt. |
| BR-18 | Existing REST endpoints and their JSON contracts remain backward compatible for the current web client during the transition. | The web client must keep working until the native app replaces it. |

### Isolation from Existing Installations

| # | Rule | Rationale |
|---|------|-----------|
| BR-19 | The service runs entirely from its bundled interpreter and libraries; it never invokes the user's system Python, Homebrew packages, or any existing virtualenv. | A pre-existing (possibly mismatched) install must not alter its behavior. |
| BR-20 | The service resolves ffmpeg and other bundled binaries by a bundle-scoped path that takes precedence within its own process; it does not modify the user's global `PATH` or shell environment. | A different system/Homebrew ffmpeg must never be picked up, and the service must not leak environment changes. |
| BR-21 | The service ignores ambient environment variables and stray config files from an existing installation — including `HF_TOKEN`, `DATA_DIR`, `WHISPER_MODEL`, `WHISPER_DEVICE`, `PORT`, `PYTHONPATH`, `PYTHONHOME`, and any `.env` in a prior checkout — reading configuration only from its own Application Support config. | A pre-set env var or leftover `.env` pointing at the old data dir or a different model would silently break or cross-wire the service. |
| BR-22 | Running the service alongside an existing Terminal installation leaves the old installation's files, data directory, and running processes untouched; interaction is limited to the one-time non-destructive import (BR-13, BR-14). | Users must be able to fall back to the old install; the service must not mutate or lock its state. |

### Offline Behavior

| # | Rule | Rationale |
|---|------|-----------|
| BR-23 | When required models are already present, the service operates fully offline for browsing, reading, managing, and transcribing meetings. | Users' data and models are local; connectivity should not block core use. |
| BR-24 | If required models are not yet downloaded and the machine is offline, the service still serves existing meetings but rejects new-transcription requests with a clear reason (first-time model download requires a connection). | Partial-use offline behavior chosen over hard-blocking the whole service. |
| BR-25 | Starting a new transcription requires the required models to be present locally. | Transcription cannot run without its models. |

## Data Requirements

### Service Configuration

Stored as local configuration in the Application Support location; created/updated during first-run provisioning.

| Field | Required | Notes |
|-------|----------|-------|
| `hf_token` | No | HuggingFace token for diarization. Empty is allowed (diarization disabled). Stored locally only; never embedded in the build. *Technical note from engineering: consider the macOS Keychain rather than a plaintext config file for the token.* |
| `whisper_model` | Yes | Transcription model identifier. Defaults to `large-v3` (accuracy is the stated priority). *Technical note from engineering: keep configurable to allow a faster model later without a code change.* |
| `data_dir` | Yes | Resolved user-writable path for meetings and transcripts. Defaults to the Application Support meetings location. |
| `models_dir` | Yes | User-writable location where downloaded models are cached. |
| `provisioning_completed` | Yes | Whether first-run provisioning (required models present) has finished. |
| `imported_from` | No | Path of a prior Terminal-installation data directory that was imported on first run, if any. |

### Existing Entities (unchanged)

The `MeetingMetadata`, `Transcript`, and `JobInfo` models and the per-meeting on-disk layout (`metadata.json`, `transcript.json`, audio file) are unchanged by this spec; only their storage location moves (see BR-12).

### Entity Relationships

| Entity | Relationship | Entity |
|--------|-------------|--------|
| Service Configuration | points to | Data Directory (meetings location) |
| Service Configuration | points to | Models Directory |
| Data Directory | has many | Meetings |
| Meeting | has one | Transcript |
| Meeting | has one | Audio File |

## Edge Cases

| # | Scenario | Expected Behavior |
|---|----------|--------------------|
| EC-1 | The chosen ephemeral port is unavailable or the service fails to start. | The service retries with another free port; if it still cannot start, it exits with a clear, machine-readable error the client can surface. |
| EC-2 | Client requests the health signal before the service is ready. | The health signal reports not-ready; it flips to ready only once the service can accept requests (BR-5). |
| EC-3 | An invalid or empty HuggingFace token is provided. | The service runs with diarization disabled and reports diarization unavailable (BR-10). |
| EC-4 | Machine goes offline mid model-download during first-run provisioning. | Download fails gracefully; existing meetings remain served and the download can be retried when connectivity returns (BR-24). |
| EC-5 | A new-transcription request arrives while models are missing and the machine is offline. | The request is rejected with a clear reason that a connection is required for first-time model download (BR-24). |
| EC-6 | A prior Terminal-installation data directory exists at first run. | Its meetings are copied into the new location, the original is left intact, and the client sees the imported history (BR-13, BR-14). |
| EC-7 | No prior data directory exists at first run. | The service starts with an empty library and no migration error (BR-15). |
| EC-8 | The old Terminal-version server is already running (e.g., on port 8000) when the service starts. | The service starts on a different free ephemeral port and runs normally; the old server is unaffected (BR-4, BR-22). |
| EC-9 | `HF_TOKEN`, `DATA_DIR`, `WHISPER_MODEL`, or similar variables are set in the environment or a prior `.env`. | The service ignores them and uses only its own Application Support config (BR-21). |
| EC-10 | A different ffmpeg version is installed via Homebrew on `PATH`. | The service uses its own bundled ffmpeg and does not alter the global `PATH` (BR-20). |
| EC-11 | Client requests an assembled analysis prompt for a meeting whose transcript is not yet ready. | The service responds with a clear not-ready error rather than an incomplete prompt (BR-17). |

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| OQ-1 | Which packaging toolchain (e.g., PyInstaller vs py2app) best handles the WhisperX/PyAnnote/torch dependency tree for a bundled service? | Engineering | Open |
| OQ-2 | How does the native app discover the service's ephemeral port and readiness — stdout handshake, a written port file, or a fixed-port fallback? | Engineering | Open — spans this spec and `native-macos-app.md` |
| OQ-3 | Should the HuggingFace token live in the macOS Keychain rather than a plaintext config file? | Engineering | Open |
| OQ-4 | Is the current data location a per-user `./data` next to the checkout, and does auto-import need to search multiple candidate locations? | Engineering | Open |
| OQ-5 | Beyond analysis-prompt assembly, what other logic currently lives in the web client and should move server-side (e.g., speaker-color assignment, formatters)? Which are genuinely business logic vs. presentation the native client should own? | Engineering | Open |
| OQ-6 | How is the bundled ffmpeg sourced — static binary, `imageio-ffmpeg` wheel, or copied Homebrew binary? | Engineering | Resolved — vendor a pinned, static, arm64-native ffmpeg binary in the bundle, resolved by a bundle-scoped path. Chosen over `imageio-ffmpeg` (ffmpeg-only, no ffprobe, extra abstraction) and the Homebrew binary (dynamic linking to dylibs absent on a clean machine). |
| OQ-7 | Model default stays `large-v3`; should a faster model be selectable later? | Product | Resolved — keep `large-v3` default; configurability retained via `whisper_model`, no selection UI in scope. |
| OQ-8 | Should the model cache be fully isolated in Application Support, or reuse an existing HuggingFace cache (`~/.cache/huggingface`) if present? | Engineering | Open — isolation is safest for the no-conflict guarantee (BR-19); reusing an existing cache would avoid a multi-GB re-download. Leaning: isolated cache, optionally seeded by a one-time copy from `~/.cache/huggingface` if present. |
