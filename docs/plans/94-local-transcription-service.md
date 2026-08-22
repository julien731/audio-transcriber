# Plan: Local Transcription Service (Bundled Backend)

**Spec:** docs/specs/local-transcription-service.md
**Issue:** #94 (Native macOS app with bundled local transcription service — this is the backend/service half)
**Branch:** feature/94-local-transcription-service
**Base branch:** main
**Date:** 2026-08-22
**Architect review:** iteration 1 → 1 Major + 4 advisory (all folded in); iteration 2 → pass, no findings.
**Mode:** Standard, test-first for pure-logic units (config isolation, data import, prompt assembly, ffmpeg/path resolution). Heavy ML/download paths covered with mocked deps (CI has no torch/whisperx/huggingface_hub).

## Scope decision (packaging artifact)

The spec's packaging rules (BR-1/2/3) require producing a multi-GB, Apple-Silicon-only, self-contained bundle (interpreter + torch/whisperx/pyannote + static ffmpeg). OQ-1 (PyInstaller vs py2app) is **Open**. That artifact cannot be built or verified in this environment or in CI (ubuntu, lightweight deps). Decision: deliver the **runtime service behavior in full, with tests** (BR-4..BR-25 — ~90% of the spec's verifiable value), plus **packaging scaffolding** (PyInstaller spec, ffmpeg vendoring script, build docs, bundle-mode detection) that a maintainer runs on an Apple Silicon machine. This is flagged for the user at the approval gate.

## Existing code surveyed

Queries run (grep/Glob, dev mode — Serena not enabled in .argus/config.yml):
1. `HF_TOKEN|DATA_DIR|MEETINGS_DIR|WHISPER_MODEL` usage → `config.py` (env-var + `.env` driven, module-level constants), consumed by `backend/services/transcriber.py`, `recovery.py`, `routers/meetings.py`, `routers/analysis.py`. Constants imported **by value** widely.
2. `ffmpeg|subprocess|PATH` → `backend/services/audio_preprocessor.py:19` calls bare `["ffmpeg", ...]` (relies on PATH); WhisperX also shells out to ffmpeg internally (`whisperx.load_audio`). `transcriber.py` (root CLI) documents `brew install ffmpeg`.
3. analysis prompt assembly → `frontend/js/components/analysis-viewer.js` does client-side substitution of `[AUDIO ANALYSIS CONTEXT]`, `[MEETING CONTEXT]`, `[PASTE TRANSCRIPT HERE]` into the template (BR-17 target). Placeholders confirmed present in `templates/*.md` (prototype_scope.md has no AUDIO placeholder). Transcript line format = `[HH:MM:SS] SpeakerName: text` (`formatTimestamp` HH:MM:SS + `.segment-time`). JS `String.replace(str,str)` replaces **first** occurrence only → server must use `str.replace(old,new,1)` for byte-identical output (BR-18).
4. model loading → `whisperx.load_model(WHISPER_MODEL,...)`, `load_align_model`, `DiarizationPipeline(token=HF_TOKEN)`. Downloads go to HF cache (HF_HOME). Diarization is skipped (no-op) when HF_TOKEN empty (`transcriber.py:81`) — matches BR-10.
5. tests/CI → mature: `tests/{unit,integration,e2e}`, `conftest.py` has async `client` fixture patching `config.DATA_DIR`/`MEETINGS_DIR` on config + router modules; ruff + pytest, coverage fail_under=80. CI installs **only lightweight deps** (no torch/whisperx/huggingface_hub) → all new modules must import heavy/ML/hub deps lazily inside functions.

Conclusion: config.py is the central seam. Isolation (BR-19/21) and relocation (BR-12) both flow through it. Token is read by-value into transcriber → needs a live accessor. Prompt assembly must exactly mirror JS substitution. No existing provisioning/health/app-paths code — all net-new.

## Key technical decisions

- **TD-1 Bundle-mode detection.** `getattr(sys, "frozen", False)` (PyInstaller sets it). Bundled mode = source-of-truth is the Application Support ServiceConfig, ambient env/.env ignored (BR-19/21). Dev mode = current env/.env behavior preserved (keeps Makefile/Conductor/tests working, BR-18). Mode resolved via `config.is_bundled()`, monkeypatchable in tests so isolation is testable without a real bundle.
- **TD-2 App paths.** New `backend/services/app_paths.py`: `app_support_dir()` = `~/Library/Application Support/MeetingTranscriber` on macOS, platform fallback elsewhere. Base overridable via a module global (tests/conftest patch it, mirroring the existing DATA_DIR patch pattern).
- **TD-3 ServiceConfig.** `<app_support>/config.json`: `hf_token, whisper_model, data_dir, models_dir, provisioning_completed, imported_from`. Pydantic `ServiceConfig` in schemas.py. `service_config.load()/save()` — atomic write, `chmod 0600` (token at rest, addresses OQ-3 pragmatically; Keychain deferred/noted). Token stays local, never embedded (BR-8).
- **TD-4 Live token.** transcriber switches `from config import HF_TOKEN` → `config.current_hf_token()` (live: ServiceConfig in bundled, env in dev) so a token set via provisioning at runtime takes effect without restart (BR-8/10).
- **TD-5 Model cache isolation (OQ-8: isolated).** At startup set process `os.environ["HF_HOME"] = models_dir` (our own process only — allowed by BR-20). Optional one-time seed from `~/.cache/huggingface` deferred (note only).
- **TD-6 ffmpeg (OQ-6 resolved: vendored static arm64).** `app_paths.ffmpeg_path()`: bundled → `<_MEIPASS>/bin/ffmpeg`, dev → `shutil.which("ffmpeg")`. At startup prepend bundle `bin/` to **process** PATH so WhisperX's own ffmpeg subprocess also resolves the bundled binary and it takes precedence (BR-20, EC-10); global/shell PATH untouched. `scripts/vendor_ffmpeg.sh` downloads pinned static arm64 ffmpeg+ffprobe into `vendor/` for the bundle (documented; not runnable here).
- **TD-7 Ephemeral port + handshake (BR-4/5, EC-1/8, OQ-2).** New `service_main.py` bundled entrypoint: bind uvicorn to **port 0** (OS-assigned free port) and read back the actual bound port for the handshake — avoids the probe-then-bind TOCTOU race; retry / clear machine-readable error on repeated bind failure (EC-1); run uvicorn `reload=False` (BR-6); on startup write `<app_support>/service.json` `{port,pid}` **and** print a stdout JSON handshake `{"event":"ready","port":N}` (support both — OQ-2). Old Terminal server on 8000 is unaffected (different ephemeral port — EC-8). `run.py` (dev) unchanged (PORT env + reload).
- **TD-0 Bundled-mode gating of side effects (BR-18).** All new *runtime* side effects are **bundled-only**: the offline model gate (`require_models_present()` short-circuits to allow when `is_bundled()` is False), first-run relocation/import, `HF_HOME` redirect, and ffmpeg-PATH prepend all run **only in bundled mode**. Dev/CI (`is_bundled()==False`) behavior is byte-for-byte the current behavior → existing integration tests (which POST `/api/meetings` with no ServiceConfig/models) stay green. Tests: dev-mode create succeeds with no config/models; bundled-mode create returns 503. **Import-cycle constraint:** `app_paths.py` and `service_config.py` must not import `config.py`.
- **TD-8 Provisioning (BR-8..BR-11).** `backend/services/provisioning.py` + `backend/routers/service.py`:
  - `GET /api/health` → `{status: ready|starting, provisioning_completed, diarization_available}` (BR-5, EC-2).
  - `GET /api/provisioning` → `{provisioning_completed, models_present, whisper_model, diarization_available, download: {state, progress, error}}`.
  - `POST /api/provisioning/token` → store HF token (BR-8); empty/invalid ⇒ diarization disabled, reported (BR-10, EC-3).
  - `POST /api/provisioning/models` → background-thread download of whisper (`Systran/faster-whisper-<model>`) + `pyannote/speaker-diarization-3.1` (needs token) into HF_HOME via `huggingface_hub.snapshot_download` (imported lazily); coarse per-model progress; persists `provisioning_completed` (BR-9/11). Offline mid-download fails gracefully + retryable (EC-4).
  - `models_present()`/provisioning-complete is gated on the **persisted `provisioning_completed` flag** (written only after all `snapshot_download` calls return) as source of truth; HF-cache presence is a secondary check only, so a partial download (EC-4) never reads as complete. The flag is not set on partial/failed download.
  - **Runtime-invalid token degradation (BR-10, EC-3):** wrap the `DiarizationPipeline` call in `transcriber.py:_diarize_and_assign` in try/except that logs and returns `(result, None)`, so a token that was accepted at submission but is later rejected/unverifiable degrades to no-diarization instead of failing the meeting. Test with a mocked pipeline that raises.
  - `GET /api/health` semantics: EC-2's genuine not-ready window (pre-HTTP) is signalled by the stdout handshake / `service.json`; once HTTP serves, `status` is `ready`. `status:"starting"` is reported only while first-run recovery/import is still running in the lifespan; otherwise `ready`.
- **TD-9 Data relocation + import (BR-12..BR-15, EC-6/7).** `backend/services/data_import.py`: on first run set `data_dir=<app_support>/data`, `models_dir=<app_support>/models`; `find_legacy_data_dir()` scans candidates (repo `./data`, other known paths — OQ-4) for `meetings/*/metadata.json`; `shutil.copytree` non-destructive copy, record `imported_from` (BR-13/14); none found ⇒ empty library, no error (BR-15). Idempotent (guarded by provisioning/import flag).
- **TD-10 Offline gate (BR-23/24/25, EC-5).** `create_meeting` + `retry_transcription` call `require_models_present()` → if models absent raise `HTTPException(503, "...first-time model download requires a connection")`. Browsing/reading existing meetings always works offline (BR-23).
- **TD-11 Server-side prompt assembly (BR-16/17/18, EC-11, OQ-5).** New `GET /api/meetings/{id}/analysis-prompt?template_type=&meeting_context=` (meeting_context optional override; defaults to `metadata.context`). Server loads template, renders audio-analysis context (reuse `analysis_context.render`), builds plain-text transcript `[HH:MM:SS] name: text` from transcript.json + `metadata.speakers`, applies the **exact** JS substitution semantics (first-occurrence replace, same placeholder-strip rules). Transcript missing/not-ready ⇒ `409` clear error (EC-11). Response `{prompt}`. Update `analysis-viewer.js` to call it and render the result (client thin); keep `/templates/{type}` + `/analysis-context` endpoints for backward compat (BR-18). OQ-5 (speaker colors/formatters) — kept client-side as presentation; noted, not moved.

## Files to create / modify

Create:
- `backend/services/app_paths.py` — app support dir, ffmpeg path, bundle detection helpers
- `backend/services/service_config.py` — ServiceConfig load/save (atomic, 0600)
- `backend/services/provisioning.py` — token + model download + status
- `backend/services/data_import.py` — legacy data discovery + non-destructive copy
- `backend/routers/service.py` — /api/health + /api/provisioning*
- `service_main.py` — bundled entrypoint (ephemeral port + handshake + no reload)
- `MeetingTranscriber.spec` (PyInstaller) + `scripts/vendor_ffmpeg.sh` + `docs/packaging.md`
- Tests: `tests/unit/test_app_paths.py`, `test_service_config.py`, `test_provisioning.py`, `test_data_import.py`, `test_analysis_prompt.py`; `tests/integration/test_service.py`, `test_analysis_prompt.py`, `test_offline_gate.py`

Modify:
- `config.py` — bundle-mode branch (ServiceConfig source of truth + env-ignore), `current_hf_token()`, keep dev behavior
- `schemas.py` — `ServiceConfig`, `HealthResponse`, `ProvisioningStatus`, `AnalysisPromptResponse`
- `backend/main.py` — mount service router; startup: set HF_HOME, prepend ffmpeg to process PATH, run first-run import
- `backend/routers/meetings.py` — offline model gate on create/retry
- `backend/routers/analysis.py` — analysis-prompt endpoint
- `backend/services/transcriber.py` — `current_hf_token()`, bundled ffmpeg via process PATH
- `backend/services/audio_preprocessor.py` — resolve ffmpeg via `app_paths.ffmpeg_path()`
- `frontend/js/components/analysis-viewer.js` + `frontend/js/api.js` — call new endpoint, drop client substitution
- `tests/conftest.py` — fixtures for app_support/ServiceConfig isolation
- `requirements.txt` — add explicit `huggingface_hub`; `README.md` / `Makefile` — build target + docs

## Commit sequence

1. app_paths + service_config + schemas + tests (foundation, isolation)
2. config.py bundle-mode + env isolation + current_hf_token + tests (BR-19/21)
3. ffmpeg resolution (app_paths + audio_preprocessor + startup PATH) + tests (BR-3/20, EC-10)
4. data_import + first-run wiring in main.py + tests (BR-12..15, EC-6/7)
5. provisioning service + router + /api/health + tests (BR-5/8..11, EC-2/3/4)
6. offline gate on create/retry + tests (BR-23..25, EC-5)
7. server-side analysis-prompt endpoint + frontend switch + tests (BR-16/17/18, EC-11)
8. service_main ephemeral port + handshake + tests (BR-4/6, EC-1/8)
9. packaging scaffolding: PyInstaller spec, vendor_ffmpeg.sh, docs/packaging.md, README (BR-1/2/3 — scaffolding)

## Risks / trade-offs

- **Packaging not verifiable here** (see scope decision). Biggest honesty flag.
- **config.py bundle-mode branch** is the highest-risk change (widely imported). Mitigated by preserving dev-mode behavior identically and keeping existing config tests green.
- **CI lacks ML/hub deps** → provisioning/model code must be lazy-import + mockable; tests mock `huggingface_hub`.
- **Prompt byte-fidelity** (BR-18): first-occurrence replace + exact strip rules; covered by a test asserting server output == current client output for each template.
- **Size:** L (9 commits, ~15 files + tests). Optionally split PR after creation.

## Deviations from spec
- OQ-3: token stored in 0600 config.json now; macOS Keychain deferred (noted).
- OQ-8: isolated model cache; ~/.cache/huggingface seeding deferred (noted).
- BR-1/2/3: build tooling scaffolded, artifact produced by maintainer on Apple Silicon (not in this PR's CI).

## Deviations from plan

- Runtime-invalid-token degradation (BR-10/EC-3) landed in commit 2 (with the token accessor) rather than commit 5, since it edits the same `_diarize_and_assign` site.
- Added an autouse `_isolate_app_support` fixture in `tests/conftest.py` so no test reads/writes the real Application Support dir (not anticipated in the plan; required for safe isolation).
- `service_main.main()` hands a pre-bound socket to `uvicorn` (implements TD-7's "port 0, read back" without any probe/rebind gap).
- Added `MODELS_DIR` to `config.py` in both modes (dev default `<data>/models`).
- Added explicit `huggingface_hub>=0.24.0` to `requirements.txt` (lazy-imported by provisioning; mocked in CI tests).
- Coverage: 93.4% overall; `provisioning._snapshot_downloader` (real huggingface_hub path) is intentionally uncovered — CI has no ML/hub deps.
