# Plan: Pre-provision alignment models so no model downloads mid-transcription

**Story**: #141
**Spec**: N/A (GitHub issue, bug)
**Branch**: fix/141-pre-provision-alignment-models
**Date**: 2026-09-04
**Mode**: Standard — pure `align_repos_for` and the `required_repos()` list get targeted unit tests; the transcriber change is a behavior-preserving refactor covered by the existing suite; the ML load/download itself cannot be unit-run.

## Technical Decisions

### TD-1: Approach — prefetch a configured common set during provisioning (issue option 2)
- **Context**: WhisperX HF wav2vec2 alignment models download lazily at the align stage (50%) into the bundled service's isolated `HF_HOME`, indistinguishable from a hang. Torch-native languages (en/fr/de/es/it) live in `~/.cache/torch` and are unaffected. The user's real need is Thai + English; English is torch-native (free), Thai is HF-backed (`airesearch/wav2vec2-large-xlsr-53-th`).
- **Decision**: Add a configurable `align_languages` set (default `["th"]`) whose HF align repos are appended to provisioning's `required_repos()`, reusing the existing download + version-gate + progress-UI mechanism. Keep the `ALIGNMENT_TIMEOUT_SEC` watchdog as the net for the residual (auto-detect / non-configured languages).
- **Alternatives considered**: Option 1 (fetch on selection in the upload form — rejected: large cross-surface diff across web + Swift). Option 3 (prefetch all ~31 HF repos, ~15GB — rejected: too heavy). The residual mid-transcription-download UX is tracked separately in #145.

### TD-2: Single source of truth for language → HF align repo
- **Context**: The transcriber must load exactly the repo provisioning downloaded, or the prefetch is wasted and the lazy download recurs.
- **Decision**: New import-light module `backend/services/align_models.py` holding `HF_ALIGN_REPOS` (whisperx's `DEFAULT_ALIGN_MODELS_HF` plus the `th` override) and `TORCH_ALIGN_LANGUAGES`. Both provisioning and the transcriber resolve repos from it. A `pytest.importorskip("whisperx")` drift test asserts the copy matches whisperx on shared keys (dev/ML-env guard, not a CI gate — whisperx is absent in CI; residual drift is bounded because provisioning and transcriber read the same map).
- **Alternatives considered**: Importing whisperx inside `required_repos()` — rejected: provisioning tests must stay import-light (whisperx/huggingface_hub absent in CI).

## Files to Create or Modify

- `backend/services/align_models.py` (new) — `TORCH_ALIGN_LANGUAGES`, `HF_ALIGN_REPOS`, `align_repos_for()`.
- `backend/schemas.py` — add `ServiceConfig.align_languages: list[str] = ["th"]`.
- `backend/services/provisioning.py` — append `align_repos_for(cfg.align_languages)` in `required_repos()`; bump `PROVISIONING_VERSION` 2 → 3.
- `backend/services/transcriber.py` — resolve align model name from `align_models.HF_ALIGN_REPOS`; remove local `CUSTOM_ALIGN_MODELS`.
- `tests/unit/test_align_models.py` (new) — resolver behavior + drift guard.
- `tests/unit/test_provisioning.py` — fix `test_without_token_only_whisper` for the new default; add `align_languages=[]` case; v3 gate.
- `CHANGELOG.md` — `[Unreleased] > Added`.

## Approach per AC

### AC1: Required alignment models are present before transcription (for the configured set)
Provisioning's `required_repos()` appends the HF align repos for `cfg.align_languages` (default `["th"]`). `PROVISIONING_VERSION` bump makes existing installs re-provision automatically, fetching the Thai model via the existing progress UI (whisper/pyannote already cached → fast no-op).

### AC2: The transcriber loads exactly the pre-provisioned repo
Both align paths resolve `align_model_name` from `HF_ALIGN_REPOS.get(lang)` (single source of truth). Behavior-preserving: torch-native langs still resolve to `None` (whisperx torchaudio default); HF values equal whisperx defaults; `th` unchanged.

### AC3: The set is configurable
`ServiceConfig.align_languages` (persisted config; bundled source of truth, env ignored per BR-21) drives the prefetch set.

### AC4: The watchdog remains the safety net
`ALIGNMENT_TIMEOUT_SEC` and `ALIGNMENT_LANGUAGES` are untouched; the residual mid-transcription-download UX is #145.

## Commit Sequence

1. `[#141]` docs: implementation plan.
2. `[#141]` add `align_models` module + unit tests.
3. `[#141]` transcriber resolves align repos from the shared mapping.
4. `[#141]` prefetch configured align languages during provisioning (+ ServiceConfig field, version bump, tests).
5. `[#141]` update CHANGELOG.

## Risks and Trade-offs

- Copying whisperx's HF map risks drift — mitigated by the importorskip drift test (dev/ML env) and bounded because provisioning and the transcriber read the same map (worst case: a stale-but-valid repo, still watchdogged).
- Default `["th"]` imposes a ~1GB Thai download on re-provision for all installs — accepted (solo project, stated Thai need, configurable knob).
- Auto-detect / non-configured HF languages still lazy-download at the align stage (watchdogged) — accepted; UX tracked in #145.

## Deviations from Plan

- Added a `test_torch_native_languages_match_whisperx` drift guard (asserting `TORCH_ALIGN_LANGUAGES` equals whisperx's `DEFAULT_ALIGN_MODELS_TORCH`) in response to a code-review Nit — the constant was otherwise defined but unenforced and could silently drift. No production-code change.
