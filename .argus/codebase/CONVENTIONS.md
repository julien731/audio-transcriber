# Coding Conventions

## Python (Backend)

- **Style:** PEP 8, snake_case for functions/variables; `ruff format` (double quotes), line-length **120**, target `py312`.
- **Ruff lint:** `select = ["E", "F", "I", "W"]` (I = import sorting). Per-file ignore: `transcriber.py = ["E402"]`.
- **Type hints:** Modern syntax (`list[str]`, `str | None`) with `from __future__ import annotations` at top of every module.
- **Models:** Pydantic `BaseModel` for all data structures with `Field` defaults.
- **Enums:** `str, Enum` pattern for serializable enums.
- **Imports:** Standard lib -> third-party -> local; heavy ML deps imported lazily inside functions.
- **Error handling:** Broad try/except in transcription pipeline, `logger.exception` for errors.
- **File I/O:** `pathlib.Path` throughout, `json.dump/load` for persistence.

## JavaScript (Frontend)

- **Style:** No framework, vanilla ES6, global functions (loaded via `<script>` tags, no bundler/modules).
- **State:** Global variables (`currentAudio`, `pollInterval`, `autoScroll`, `window._speakerEditorState`).
- **DOM:** Template literals for HTML generation, `innerHTML` assignment; `escapeHtml` for user content.
- **API calls:** Centralized in `frontend/js/api.js` fetch wrapper.
- **Naming:** camelCase for functions/variables.
- **Components:** Each component is a self-contained JS file with a render function.
- **No JS linter/formatter/tests** configured.

## Swift (macOS App)

- **Toolchain:** swift-tools-version 6.0, `platforms: .macOS(.v13)`, but `swiftLanguageMode(.v5)` on all targets (avoids strict-concurrency noise in a UI shell).
- **Structure:** `MeetingTranscriberKit` (testable, UI-independent logic) split into feature dirs (API, Service, Provisioning, Models, Insights, Presentation, Settings...); `MeetingTranscriberApp` = thin SwiftUI `@main` shell. No transcription/workflow logic in Swift — all via service HTTP API.
- **Deps:** pinned exactly for security-sensitive ones (Sparkle `exact: "2.9.6"`).
- **Comments:** file/target headers cite spec + plan paths (docs/specs/*, docs/plans/*).

## Project Conventions

- **No build step for web** — no transpilation, bundling, or minification.
- **Linter/Formatter:** Ruff (`ruff check` + `ruff format`) for Python, configured in `pyproject.toml`. `extend-exclude = ["**/*.md"]` so ruff never reformats fenced code in docs.
- **No type checking** — no mypy or TypeScript.
- **Config:** env vars via `config.py` / python-dotenv (`HF_TOKEN`, `WHISPER_MODEL`, `DATA_DIR`, `MAX_UPLOAD_SIZE`...).
- **Storage:** file-based (per-meeting `metadata.json` + `transcript.json` + audio); no DB.

## Git & Release Workflow

- **Trunk-based dev** — `main` is the single long-lived branch. No `develop`. (Overrides global Gitflow instruction.)
- **Feature branches:** `feature/<issue#>-<slug>` off `main`, short-lived.
- **PRs:** every change reaches `main` via PR; direct pushes rejected. Title format `[#NN] <description>`.
- **Merge:** squash-merge only; linear history enforced (merge + rebase merges disabled).
- **CI gate:** `Lint & Format` + `Test & Coverage` must pass; no approvals required (solo, CI is the gate).
- **PR labels drive releases** (not issue labels), via semantic-release (.releaserc.js):
  `breaking` -> major, `feature` -> minor, `bug` -> patch; other/none -> no release. Highest bump wins.
- **Auto-labeling:** `.github/workflows/auto-label.yml` maps branch prefix -> label (`feature/|feat/|story/|enhancement/`->feature, `fix/|bugfix/|hotfix/|bug/`->bug, `chore/|refactor/|test/|ci/`->chore, `docs/`->documentation). Arbitrary branches need a manual label. Apply `breaking` manually.
- **Tags:** bare semver (no `v` prefix), created by semantic-release on push to `main`.
- **Commits:** conventional-style commit messages.
