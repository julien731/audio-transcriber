# Testing

## Frameworks

- **Python:** pytest (>=8.0) + pytest-asyncio (`asyncio_mode = auto`), pytest-cov (>=6.0)
- **HTTP:** httpx `AsyncClient` over `ASGITransport(app=app)` (in-process, no live server)
- **Swift (macOS):** custom executable harness (no XCTest/Testing — CLT-only toolchain). `TestRunner` singleton with `suite()`, `expect()`, `expectEqual()`, `expectNil()`; each suite is a `run*Tests()` func called from `main.swift`. Run via `swift run MeetingTranscriberKitTests`.

## Layout

```
tests/                          # Python (~376 test funcs)
  conftest.py                   # shared fixtures (see below)
  fixtures/*.json               # metadata (ready/processing/error), transcript
  unit/        test_*.py        # 20 files: schemas, job_queue, config,
                                #   transcriber, analyzers, provisioning, etc.
  integration/ test_*.py        # 8 files: meetings, jobs, analysis, service,
                                #   cancel, offline_gate (via httpx client)
  e2e/         test_flows.py    # full upload->artifacts flows
macos/Sources/
  MeetingTranscriberKitTests/   # 58 suites: APIClient, ServiceDiscovery,
                                #   Provisioning, Presentation, Upload, etc.
    main.swift                  # entry point, runs all suites, exits nonzero
    TestHarness.swift           # TestRunner + expect helpers
    MockURLProtocol.swift       # URLSession stubbing
  MeetingTranscriberIntegrationTests/  # real stub child + local HTTP
```

## Markers

`pytest.ini_options` defines `unit`, `integration`, `e2e` (directory-aligned).

## Key Fixtures (tests/conftest.py)

- `client` — async httpx client; monkeypatches `DATA_DIR`/`MEETINGS_DIR` on `config` + router modules to `tmp_path`, restores after.
- `_clean_job_queue` (autouse) — clears `job_queue` singleton after each test.
- `_isolate_app_support` (autouse) — redirects Application Support to tmp via `app_paths._APP_SUPPORT_OVERRIDE` monkeypatch (no real home writes).
- `data_dir` / `meetings_dir` — isolated tmp dirs.
- `sample_audio` — minimal valid WAV generated with `wave` (~1ms silence).
- `sample_metadata[_processing|_error]`, `sample_transcript` — load fixtures/*.json.
- `populated_meeting` — writes metadata + transcript + audio to disk, returns id.

## Patterns

- Test classes group by behavior: `class TestListMeetings`, `class TestCreateJob`.
- Mocking via `unittest.mock.patch` — e.g. `@patch("backend.routers.meetings.start_transcription")` to avoid real transcription; patch at import site.
- Concurrency tested with `ThreadPoolExecutor` (job_queue thread-safety).
- All test modules start with `from __future__ import annotations`.
- Swift: `runBlocking` semaphore bridges async APIClient to sync harness; `MockURLProtocol.handler` stubs HTTP responses + captures `lastRequest`.

## Coverage

- `[tool.coverage.run] source = ["backend", "config"]`
- `[tool.coverage.report] show_missing = true`, **`fail_under = 80`**
- CI runs `pytest --cov --cov-report=term-missing --cov-fail-under=80`.

## CI (.github/workflows/ci.yml)

Triggers: push + PR to `main`. Two jobs (both required gates):

```
Lint & Format  → ruff check .  +  ruff format --check .   (py 3.12)
Test & Coverage→ pytest --cov --cov-fail-under=80         (py 3.12)
```

Test job installs runtime deps (fastapi, uvicorn, multipart, dotenv, parselmouth) + test deps directly (not full requirements.txt — ML deps omitted).

## Not Covered by CI

- **Swift tests run manually only** — not invoked in any workflow. `release-macos.yml` does `swift build` (via build_app.sh) but does not run KitTests/IntegrationTests.
- **No frontend (JS) tests** — vanilla JS SPA, no test runner.
