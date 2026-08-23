# Architecture

## Pattern

**Local service + two clients.** A single FastAPI process (the "service") serves
the REST API and the vanilla-JS SPA. Two front ends consume it:

1. **Web SPA** — served by FastAPI, run in dev via `run.py` (uvicorn reload).
2. **Native macOS app** (`macos/`) — SwiftUI shell that *spawns and supervises*
   its own embedded copy of the service (frozen by PyInstaller), then drives it
   over the same HTTP API. No transcription/workflow logic lives in Swift.

No database, no microservices, no reverse proxy. Storage is the filesystem; job
state is an in-memory threaded queue.

## Process Topology

```
DEV (web):   run.py ──► backend.main:app (uvicorn, reload, fixed port 8000)
                          └─ serves SPA + /api

BUNDLED:     macOS app (MeetingTranscriberApp)
               │ spawns child process, passes MT_SERVICE_NONCE
               ▼
             service_main.py (PyInstaller binary)
               ├─ binds ephemeral 127.0.0.1 port (holds socket, no TOCTOU)
               ├─ writes ~/…/Application Support/MeetingTranscriber/service.json
               ├─ emits stdout handshake {"event":"ready","port",...}
               └─ runs backend.main:app (reload=False)
               ▲
               │ ServiceSupervisor/Discovery/Handshake read port+nonce,
               │ then poll GET /api/health until ready, then show UI
             MeetingTranscriberKit (Swift, HTTP client)
```

## Transcription Data Flow

```
Upload (multipart POST /api/meetings)
  → save audio + metadata.json (status=PROCESSING)
  → create in-memory JobInfo, spawn daemon thread
      → (opt) audio_preprocessor: high-pass, denoise, loudness
      → WhisperX transcribe + align   (multilingual_transcriber for ≥2 langs)
      → PyAnnote diarize speakers
      → (opt) prosody / emotion / interaction analyzers → audio insights
      → write transcript.json, update metadata (status=READY)
  → return job_id
Frontend polls GET /api/jobs/{jobId} every 3s → progress bar → auto-navigate
```

Multiprocessing note: transcription uses `spawn` workers that re-exec the frozen
binary; `service_main.py` calls `multiprocessing.freeze_support()` so a worker
never boots a second service (which would trip recovery and kill the live job).

## Concurrency & State

- Daemon `threading.Thread` per upload; no concurrency cap.
- `JobQueue` (`backend/services/job_queue.py`) — thread-locked in-memory dict.
- No job persistence; on restart `recovery.recover_stuck_meetings()` marks any
  `PROCESSING` meeting as `ERROR`.
- Server state: JSON files on disk + in-memory job dict.
- Client state: globals + localStorage (theme, recent speaker names).

## Startup (lifespan in `backend/main.py`)

`service_runtime.bootstrap()` (bundled-only: redirect HF_HOME + ffmpeg into the
bundle) → `run_first_run_import()` (copy prior Terminal-install meetings) →
`recover_stuck_meetings()`.

## Bundle-awareness layer (bundled vs dev checkout)

- `app_paths.py` — single source of "where things live"; `is_bundled()` via
  `sys.frozen`. Must NOT import `config` (cycle).
- `service_config.py` — persisted `config.json` in Application Support is the
  source of truth; ambient env / stray `.env` ignored in bundled mode (BR-21).
- `provisioning.py` — first-run HF token + ML model download (bg thread,
  coarse progress); `provisioning_completed` flag gates readiness.

## Key Abstractions

- **Pydantic schemas** (`backend/schemas.py`) — `MeetingMetadata`, `Transcript`,
  `JobInfo`, `HealthResponse`, `ProvisioningStatus`, `ServiceConfig`, enums.
- **Routers** (`backend/routers/`) — `meetings`, `jobs`, `analysis`, `service`
  (health + provisioning), all mounted under `/api`.
- **Analysis prompt assembly** — server-side (`analysis_prompt.py`,
  `analysis_context.py`) injecting audio-insight context into LLM templates.
- **SwiftUI Kit** (`MeetingTranscriberKit`) — API client, models, presentation
  formatters, and service Discovery/Handshake/Supervisor; UI-independent, tested.

## API Design

RESTful JSON under `/api/`. `/api/health` is the readiness probe. Catch-all
route serves `index.html` for SPA routing; static assets under `/static`.
