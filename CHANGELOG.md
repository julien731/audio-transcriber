# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Redact secret-shaped tokens (at minimum HuggingFace `hf_...` tokens) from `service.log`, the raw child `service-stderr.log` tee, and `app.log` before they are written, so the shareable **Export Diagnostics…** bundle cannot leak a credential embedded in third-party output (e.g. a token surfaced in a dependency's traceback).

## [1.2.0] - 2026-05-08

### Added

- Audio-based emotional intelligence pipeline (opt-in per meeting): speech emotion recognition, prosodic feature extraction, and interaction pattern detection (interruptions, hesitations, single-speaker dominance).
- Audio analysis context surfaced to LLM prompts so meeting analysis weighs tone, energy, and interaction dynamics alongside the transcript.
- In-app surfacing of audio insights: inline emotion / prosody / word-tone-mismatch / interaction indicators on each transcript segment, plus a per-meeting Overview tab with an energy trajectory chart and per-speaker interruption summary.
- Audio preprocessing for uploaded recordings (high-pass filter, noise reduction, loudness normalization).
- Per-meeting context field that feeds into the analysis prompt.
- Speakers sidebar listing unidentified speakers for fast renaming.
- Sticky audio player while scrolling the transcript.
- Cancel an in-flight transcription from the UI.
- Recover stuck PROCESSING meetings on app startup.
- Thai language alignment support.
- Prototype Scope analysis template.
- Test framework: pytest with unit, integration, and E2E suites; Ruff linting and formatting; GitHub Actions CI.

### Changed

- Migrated from Gitflow to trunk-based development. `main` is now the single long-lived branch; `develop` has been retired. CI runs on `main` only and squash-merge is the sole allowed merge method. See `CLAUDE.md` for the full workflow.
- Renaming a segment's speaker now reuses an existing speaker id when the name matches, instead of creating duplicates.
- Refined and clarified the meeting analysis templates.
- Switched the development tooling to the Argus Claude plugin.

### Fixed

- Auto-scroll no longer fights manual scrolling during playback.
- Pass language to `whisperx.load_model()` so the tokenizer is initialized correctly.
- Add missing `soundfile` dependency.

## [1.1.0] - 2026-02-26

### Added

- Web UI for transcription and meeting analysis: upload, progress tracking, synced transcripts, speaker renaming, and LLM prompt generation from analysis templates.
- Browser desktop notifications when transcription completes.

### Changed

- Rewrote "Getting Started" in the README for non-technical users.

### Fixed

- Notification permission request not triggering on first use.

## [1.0.0] - 2026-02-26

### Added

- Initial release: command-line transcription with WhisperX and PyAnnote speaker diarization.
- `.env` file support for persistent `HF_TOKEN` configuration via `python-dotenv`.
- Skip alignment for languages without wav2vec2 support.
- README with virtual environment setup instructions.
