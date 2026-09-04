# Plan: Surface a model-download indicator when an alignment model is fetched mid-transcription

**Story**: #145
**Spec**: N/A (follow-up to #141; no spec document)
**Branch**: feature/145-align-model-download-indicator
**Date**: 2026-09-04
**Mode**: Standard — mocking-heavy ML pipeline; tests written alongside each change.

## Technical Decisions

### TD-1: Distinct job stage `downloading_align_model`
- **Context**: A lazy align-model download inside the align stage leaves the job frozen at `stage=aligning, progress=50`, indistinguishable from a hang.
- **Decision**: Add `DOWNLOADING_ALIGN_MODEL = "downloading_align_model"` to `JobStage` and set it while the model is being fetched (issue option 1).
- **Alternatives considered**: Reuse provisioning's `DownloadState` (rejected — separate lifecycle, provisioning phase vs job stage); gate/announce the download up front (rejected — the detected language is unknown until transcription runs).

### TD-2: Cache-presence proxy via `huggingface_hub.try_to_load_from_cache`
- **Context**: We must decide whether the align model will download before entering the load call.
- **Decision**: New `align_model_cached(repo_id)` in `align_models.py` probes `try_to_load_from_cache(repo_id, "config.json")`; True iff a cached path string is returned. huggingface_hub imported lazily inside the function (preserves the module's import-light contract). Any exception (incl. `ModuleNotFoundError` when hf_hub is absent in the lightweight CI env) → return True = "assume present, skip the indicator" = today's behavior. Logged at debug so the degrade is observable.
- **Alternatives considered**: Full `scan_cache_dir` (overkill); checking model weights file (filename varies: `pytorch_model.bin` vs `model.safetensors`). `config.json` is a sufficient proxy since whisperx loads via `Wav2Vec2ForCTC.from_pretrained`, which always needs it. The `ALIGNMENT_TIMEOUT_SEC` watchdog covers a false "present".

### TD-3: Only HF-backed models get the indicator
- **Context**: Torch-native align models (en/fr/de/es/it, `model_name=None`) live in the shared `~/.cache/torch` and are unaffected (per the issue).
- **Decision**: Skip the probe entirely when `align_model_name is None`.

## Files to Create or Modify

- `backend/schemas.py` — add `DOWNLOADING_ALIGN_MODEL` to `JobStage`.
- `backend/services/align_models.py` — add `align_model_cached(repo_id)` (lazy hf_hub import; debug-logged safe degrade).
- `backend/services/transcriber.py` — import `align_model_cached`; add `_load_align_model_watchdogged(...)` that sets the download stage when the HF-backed model is not cached, runs the watchdogged load, resets the stage to `aligning`, and returns `(status, loaded)`. The caller passes the whisperx-closed load callable so the helper stays ML-import-free. Wire both align sites (single-language progress=50, multilingual progress=82).
- `frontend/js/components/transcript-viewer.js` — add `downloading_align_model: 'Downloading alignment model...'` to `stageLabels`.
- `macos/Sources/MeetingTranscriberKit/Presentation/JobPresentation.swift` — add `"downloading_align_model": "Downloading alignment model"`.
- `tests/unit/test_align_models.py` — `align_model_cached`: cached→True, not-cached (None)→False, known-missing sentinel→False, hf_hub absent/raises→True.
- `tests/unit/test_transcriber.py` — HF-backed (Thai) run with probe False → `update_job` receives a `stage="downloading_align_model"` call; probe True → no such call; English (`model_name=None`) → no such call. Captured via a recording mock on `update_job` (queue keeps only latest state).
- `macos/Sources/MeetingTranscriberKitTests/UploadValidationTests.swift` — `label(for: "downloading_align_model") == "Downloading alignment model"`.

## Approach per AC

### AC: When a non-pre-provisioned HF align model must download, the job reports a distinct downloading state
Probe cache before the load; set `stage=downloading_align_model` (progress unchanged) while `whisperx.load_align_model` fetches; reset to `aligning` once loaded. Both align sites route through `_load_align_model_watchdogged`.

### AC: Torch-native and already-cached models keep the current behavior
Probe skipped when `model_name is None`; when cached, no stage flip.

### AC: The downloading state is surfaced in both UIs
Web `stageLabels` and Swift `JobStagePresentation` map the new stage to "Downloading alignment model".

### AC: The watchdog remains the safety net
`_call_with_timeout(..., ALIGNMENT_TIMEOUT_SEC, "align-load")` is unchanged; each site's divergent outcome handling (single-language logs-and-degrades; multilingual raises) is preserved.

## Commit Sequence

1. `[#145]` schema enum + `align_models.align_model_cached` + unit tests
2. `[#145]` transcriber helper wiring both align sites + transcriber tests
3. `[#145]` frontend + swift download-stage labels + swift test

## Risks and Trade-offs

- `config.json` presence is a heuristic: cached config but missing weights → false "present" (no indicator), but the watchdog still covers it.
- Very fast downloads → brief flash of the downloading label. Acceptable.
- The probe uses the default HF cache_dir, which huggingface_hub derives from `HF_HOME` at import time — the same env the bundled service sets before first import (mirrors #141 provisioning), so probe and load share cache resolution by construction. Recorded as a code comment.

## Deviations from Plan

- `_align_multilingual_segments` gained a leading `job_id: str` parameter (not called out in the plan) so it can pass the job id into `_load_align_model_watchdogged`. Both the caller in `_run_multilingual_transcription` and the two direct test call sites were updated accordingly.
- No other deviations. The helper takes the whisperx-closed load callable (architect Nit) and the debug-logged safe degrade + shared-cache comment (architect Minors) as approved.
