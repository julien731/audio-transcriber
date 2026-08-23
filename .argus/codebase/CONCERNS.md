# Concerns & Technical Debt

Scope note: app is now a **local-first, single-user macOS service** (bundled via PyInstaller, driven by a native Swift app). Many "no auth" risks are lower severity because the API binds locally, but they remain if the port is exposed.

## Security

- **Secrets on disk in plaintext** — HF_TOKEN persisted to `~/Library/Application Support/MeetingTranscriber/config.json`. Mitigated by `os.chmod(0o600)` + atomic write (`backend/services/service_config.py`), but **not** stored in macOS Keychain. Token now user-editable via app Settings.
- **Path traversal** — `meeting_id` from URL used directly as `MEETINGS_DIR / meeting_id` in read/delete/audio routes (`_load_metadata`, `backend/routers/meetings.py`). Creation uses UUIDs, but GET/DELETE/`/audio` accept arbitrary IDs without sanitization.
- **`torch.load(weights_only=False)` monkeypatch** — arbitrary-pickle deserialization of model weights, patched globally in **both** `backend/services/transcriber.py:587` and root `transcriber.py:53`.
- **Uploads fully buffered in RAM** — `content = await file.read()` holds up to 500MB (`MAX_UPLOAD_SIZE`) in memory before size check (`backend/routers/meetings.py:152`).
- **No authentication / CORS / rate limiting** on the API.
- **`disable-library-validation` entitlement + ad-hoc self-signing** — required for Sparkle/XPC but weakens code-integrity guarantees (`macos/Resources/MeetingTranscriber.entitlements`, `macos/scripts/build_app.sh`).

## macOS Distribution

- **Not notarized** — app is ad-hoc / self-signed, so every download is Gatekeeper-blocked; users must manually Settings → Open Anyway on first launch (`docs/macos-app.md`). Real fix noted as paid Apple notarization.
- **Sparkle EdDSA enclosure signature is the sole update trust anchor** (`macos/.../UpdaterController.swift`) — no cert-chain fallback.
- **Fragile native version coupling** — `torchcodec==0.7.0` pinned to torch 2.8 ABI; requires ffmpeg 4–7 at runtime; vendored ffmpeg binary in bundle (`requirements.txt`, `backend/services/app_paths.py:ffmpeg_path`).

## Scalability & Reliability

- **In-memory job queue** — `job_queue` dict lost on restart; no persistence/cleanup (`backend/services/job_queue.py`). Partially mitigated: `recovery.recover_stuck_meetings()` flips orphaned PROCESSING → ERROR at startup, but in-flight work is discarded, not resumed.
- **No concurrency limit** — every upload spawns an unbounded daemon thread (`start_transcription`, `transcriber.py:688`). Concurrent jobs contend for GPU/CPU/RAM → OOM risk.
- **Model reloaded per job** — `whisperx.load_model(...)` called fresh each transcription (`backend/services/transcriber.py:324,517`); no model cache/warm pool.
- **O(n) directory scan** — meeting list iterates every `MEETINGS_DIR` subdir per request (`backend/routers/meetings.py:106`).
- **File-based storage, no DB** — per-meeting `metadata.json` + `transcript.json`; no indexing, transactions, or query layer.

## Data Integrity

- **Cooperative cancellation via status polling** — `_is_cancelled()` re-reads metadata status; race windows between cancel/delete and the writing thread (`backend/services/transcriber.py:138,637`).
- **No file locking** between API endpoints and background threads.
- **Non-atomic metadata writes** — `recovery.py` and meeting saves write JSON in place (only `service_config.save` is atomic); partial writes possible on crash.
- **Delete during transcription** — removing a meeting mid-job can error the running thread.

## Technical Debt

- **Duplicate transcriber logic** — root `transcriber.py` (601 lines, standalone CLI) vs `backend/services/transcriber.py` (695 lines); divergent `torch.load` patches and duplicated pipeline code.
- **Large files** — `backend/services/transcriber.py` (695), root `transcriber.py` (601), `frontend/js/components/transcript-viewer.js` (662).
- **Broad `except Exception` swallowing** — many silent catches, esp. in `transcriber.py` (100,114,144,487,680) and analyzers.
- **Unpinned ML deps** — `whisperx @ git+HEAD`, `torch`/`torchaudio` unpinned (`requirements.txt`); breaking upstream changes possible.
- **Global frontend state** — vanilla JS, no modules/bundler; shared window globals.

## Improved Since Last Mapping

- **Tests now exist** — Python `tests/{unit,integration,e2e}` + Swift `macos/.../*Tests`; CI gates on Lint & Test/Coverage (`.github/workflows/ci.yml`). Prior "zero tests" concern resolved.
- **Startup recovery** — orphaned PROCESSING meetings auto-transitioned to ERROR (`backend/services/recovery.py`).
- **Config isolation** — bundled service ignores ambient env/`.env`; single config source of truth (`service_config.py`, BR-21).

## Key Files

- `backend/routers/meetings.py` — upload, CRUD, path handling
- `backend/services/transcriber.py` / root `transcriber.py` — duplicated pipelines
- `backend/services/job_queue.py` — in-memory jobs
- `backend/services/service_config.py` — HF_TOKEN persistence
- `backend/services/app_paths.py` — bundle/paths/ffmpeg resolution
- `backend/services/recovery.py` — stuck-meeting recovery
- `macos/scripts/build_app.sh`, `macos/Resources/*.entitlements`, `docs/macos-app.md` — signing/Gatekeeper
- `requirements.txt` — ML dependency pinning
