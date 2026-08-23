# Project Structure

```
.
├── run.py                     # DEV entry: uvicorn backend.main:app (port 8000, reload)
├── service_main.py            # BUNDLED entry: ephemeral port + handshake + service.json
├── config.py                  # Env/runtime config (HF_TOKEN, paths, model settings)
├── transcriber.py             # Standalone CLI transcription script
├── MeetingTranscriber.spec    # PyInstaller spec (freezes the service binary)
├── Makefile                   # setup / run / app / clean
├── requirements*.txt, pyproject.toml, package.json, .releaserc.js
├── backend/
│   ├── main.py                # FastAPI app; lifespan bootstrap; mounts routers + SPA
│   ├── schemas.py             # Pydantic models + enums
│   ├── routers/
│   │   ├── meetings.py        # CRUD, upload, audio streaming, retry, segment speaker
│   │   ├── jobs.py            # GET /api/jobs/{job_id}
│   │   ├── analysis.py        # GET /api/templates/{type}
│   │   └── service.py         # /api/health, /api/provisioning[/token|/models]
│   └── services/
│       ├── transcriber.py            # Background-thread transcription pipeline
│       ├── multilingual_transcriber.py # Per-chunk multi-language transcription
│       ├── audio_preprocessor.py     # High-pass / denoise / loudness normalize
│       ├── prosody_analyzer.py       # Prosodic features
│       ├── emotion_analyzer.py       # Per-segment emotion
│       ├── interaction_analyzer.py   # Interaction/turn-taking patterns
│       ├── analysis_context.py       # Renders audio-insight context for prompts
│       ├── analysis_prompt.py        # Server-side LLM prompt assembly
│       ├── job_queue.py              # In-memory thread-locked JobQueue
│       ├── recovery.py               # Mark stuck PROCESSING meetings as ERROR
│       ├── data_import.py            # First-run import of prior Terminal install
│       ├── provisioning.py           # HF token + ML model download (bg thread)
│       ├── service_config.py         # Persisted config.json (bundled source of truth)
│       ├── service_runtime.py        # Bundled bootstrap: HF_HOME + ffmpeg redirect
│       └── app_paths.py              # Bundle-aware filesystem locations (no config import)
├── frontend/                  # Vanilla-JS SPA (no bundler; script tags)
│   ├── index.html
│   ├── css/styles.css
│   ├── js/
│   │   ├── app.js  api.js  utils.js
│   │   └── components/
│   │       ├── meeting-list.js  upload.js  transcript-viewer.js
│   │       ├── speaker-editor.js  analysis-viewer.js
│   │       ├── overview-viewer.js      # Meeting overview tab
│   │       └── audio-insights.js       # Prosody/emotion insights UI
│   └── vendor/tom-select/     # 3rd-party select widget
├── macos/                     # Native SwiftUI app (SwiftPM, macOS 13+, Swift 6 toolchain)
│   ├── Package.swift, Package.resolved   # Sparkle 2.9.6 pinned (updater)
│   ├── Resources/             # Info.plist, .icns, entitlements
│   ├── service-manifest.json  # Pinned frozen-service artifact (version+sha256)
│   ├── scripts/               # build_app.sh, fetch_service.sh, package_service.sh,
│   │                          #   make_appcast.sh, make_icon.sh, verify_update.sh, stub_service.py
│   └── Sources/
│       ├── MeetingTranscriberApp/          # @main SwiftUI shell (App, AppDelegate, Views/)
│       ├── MeetingTranscriberKit/          # UI-independent logic (tested):
│       │   ├── API/  Models/  Coding/  Presentation/  Insights/
│       │   ├── Service/   # Discovery, Handshake, Supervisor, HealthClient
│       │   ├── Provisioning/  Settings/  Preferences/
│       ├── MeetingTranscriberKitTests/         # unit suite (swift run, not swift test)
│       └── MeetingTranscriberIntegrationTests/ # integration suite (real stub + HTTP)
├── templates/                 # LLM analysis prompt templates (.md)
├── tests/                     # Python: unit/, integration/, e2e/, fixtures/, conftest.py
├── docs/                      # specs/, plans/, macos-app.md, packaging.md, PRD
├── scripts/                   # setup-release-labels.sh, vendor_ffmpeg.sh
├── .conductor/settings.toml   # Conductor run tabs: web (make run) + app (make app)
├── .github/workflows/         # ci.yml (ruff+pytest), release.yml (semantic-release),
│                              #   release-macos.yml (build .app + Sparkle appcast), auto-label.yml
└── data/meetings/{id}/        # Per-meeting: metadata.json, transcript.json, <audio>
```

## Entry Points

- **Dev web server:** `run.py` → `backend/main.py`
- **Bundled service:** `service_main.py` → `backend/main.py` (PyInstaller)
- **Native app:** `macos/Sources/MeetingTranscriberApp/App.swift` (spawns service)
- **Web frontend:** `frontend/index.html`
- **CLI:** `transcriber.py`
- **Conductor:** `make run` (web) / `make app` (native, local-only)

## Adding New Code

- **API endpoint:** add to a router in `backend/routers/` (or new router mounted
  in `backend/main.py` under `/api`).
- **Pydantic model:** `backend/schemas.py`.
- **Backend service/analysis step:** `backend/services/`; wire into
  `services/transcriber.py` pipeline if part of transcription.
- **Web component:** JS file in `frontend/js/components/`, add `<script>` in
  `index.html` (no bundler).
- **macOS logic:** put testable code in `MeetingTranscriberKit`; UI in
  `MeetingTranscriberApp/Views/`. Add matching test in `MeetingTranscriberKitTests`.
- **Python test:** `tests/unit|integration|e2e/`.
- **Bundle/path behavior:** `app_paths.py` (never import `config` there).
