# Plan: Debugging & observability — capture logs and diagnostics for app hangs

**Story**: #137
**Spec**: N/A (chore-scale feature; no formal spec)
**Branch**: feature/137-observability-diagnostics
**Date**: 2026-09-03
**Mode**: Standard — UI/OS-integration heavy (SwiftUI menu, NSSavePanel, Finder reveal, ditto zip); targeted tests where the harness allows (Python logging setup, Swift FileLog, ServiceSupervisor drain regression).

## Technical Decisions

### TD-1: Fix the pipe-drain deadlock in ServiceSupervisor
- **Context**: `ServiceSupervisor` drains the child's stdout only until the readiness handshake (scanner returned) and stderr only until the first error line. Afterward nothing reads the pipes; once the ~64KB OS pipe buffer fills with uvicorn logs / a traceback, the Python child blocks on `write()` → transcription stalls (spinner forever) and the blocked process resists SIGTERM → quit hangs.
- **Decision**: Rewrite both readers to drain their handle to EOF for the whole process lifetime. Keep parsing lines for the handshake (stdout) / error (stderr) but with a per-handle "already found" guard so, once found, later lines are only teed+drained and cannot overwrite `_handshake`/`_lastError`.
- **Alternatives considered**: Discard post-handshake output to `/dev/null` (simpler but loses native crash output that is exactly what we need for debugging). Rejected — we tee to a file instead.

### TD-2: Python owns a readable rotating log; Swift owns raw child capture
- **Context**: The bundled service uses default uvicorn logging to stderr with no file handler. Native crashes / PyInstaller bootstrap errors bypass Python logging entirely.
- **Decision**: Two complementary sinks in one `Logs/` dir. Python configures a `RotatingFileHandler` → `service.log` (structured, level-controlled). Swift tees the raw child stderr (and post-handshake stdout) → `service-stderr.log` (captures anything that bypasses Python logging).
- **Alternatives considered**: Single sink. Rejected — neither side alone captures both structured app logs and native/C-level output.

### TD-3: DiagnosticsExporter is pure Kit logic; presentation stays in the App layer
- **Context**: `MeetingTranscriberKit` is the testable, UI-independent layer and currently imports zero AppKit/SwiftUI (CONVENTIONS.md Layer 2).
- **Decision**: `DiagnosticsExporter.exportDiagnostics(to:)` takes a caller-supplied destination URL, stages `Logs/` + a `diagnostics.txt` system-info file, and zips via `/usr/bin/ditto`. NSSavePanel presentation and Finder reveal live in the App layer (AppDelegate/App.swift). App is not sandboxed, so ditto + save panel + Application Support access are permitted.

### TD-4: File logging scoped to the bundled entrypoint
- **Context**: Dev `run.py` uses auto-reload and console output; polluting it with file handlers is unnecessary.
- **Decision**: `configure_service_logging()` is called only from `service_main.py` (the bundled path). `run.py` is untouched.

## Files to Create or Modify

- `backend/services/app_paths.py` — add `logs_dir() -> Path` (`app_support_dir()/"Logs"`, created on access; honors `_APP_SUPPORT_OVERRIDE`).
- `backend/services/logging_setup.py` (new) — `configure_service_logging()`: root level INFO + `RotatingFileHandler` (5 MB × 3) → `logs_dir()/service.log`; idempotent; no stdout StreamHandler (protects handshake).
- `service_main.py` — call `configure_service_logging()` before uvicorn; pass `log_config=None` so uvicorn loggers propagate to root→file.
- `tests/unit/test_logging_setup.py` (new) — handler installed, INFO/uvicorn record reaches file, rotation, idempotent, no stdout handler; fixture removes handler + restores root level.
- `macos/Sources/MeetingTranscriberKit/Logging/FileLog.swift` (new) — thread-safe append + size rotation (.log → .log.1) via serial queue; `logsDirectory()` helper.
- `macos/Sources/MeetingTranscriberKit/Logging/AppLog.swift` (new) — wraps `os.Logger(subsystem:category:)` and mirrors to the app.log FileLog.
- `macos/Sources/MeetingTranscriberKit/Service/ServiceSupervisor.swift` — continuous-drain rewrite + tee to injected `FileLog? = nil`.
- `macos/Sources/MeetingTranscriberKit/Diagnostics/DiagnosticsExporter.swift` (new) — pure logic: stage logs + diagnostics.txt, zip via ditto to destination URL.
- `macos/Sources/MeetingTranscriberApp/AppState.swift` — AppLog lifecycle events; inject the stderr FileLog into the supervisor.
- `macos/Sources/MeetingTranscriberApp/AppDelegate.swift` — AppLog terminate path; NSSavePanel + reveal actions calling the exporter.
- `macos/Sources/MeetingTranscriberApp/App.swift` — Help `CommandGroup`: "Reveal Logs in Finder", "Export Diagnostics…".
- `macos/Sources/MeetingTranscriberKitTests/FileLogTests.swift` (new) + `main.swift` registration.
- `macos/Sources/MeetingTranscriberIntegrationTests/main.swift` — drain regression: stub emits >64KB on stderr AND on post-handshake stdout; supervisor stays responsive.

## Approach per AC

### AC1 — Deadlock fix
Rewrite the two `scanFor*` readers into continuous EOF drains; tee raw bytes to the injected FileLog; parse-once guard.

### AC2 — Service log file
`configure_service_logging()` + `logs_dir()`; wire into `service_main.py` with `log_config=None`; root level INFO.

### AC3 — Raw child capture
Supervisor tees child stderr (and post-handshake stdout) to `service-stderr.log` via FileLog.

### AC4 — App logging
`FileLog` + `AppLog`; emit events at start/ready/serviceFailed/shutdown/terminate.

### AC5 — Access to logs
Help menu → reveal `Logs/` in Finder; "Export Diagnostics…" → NSSavePanel (App) → `DiagnosticsExporter.exportDiagnostics(to:)` (Kit) → ditto zip with `diagnostics.txt` (app version, `operatingSystemVersionString`, timestamps).

## Commit Sequence

1. `[#137]` app_paths.logs_dir + logging_setup + service_main wiring + Python tests
2. `[#137]` Swift FileLog + AppLog + FileLog tests
3. `[#137]` ServiceSupervisor continuous-drain deadlock fix + integration regression
4. `[#137]` AppState/AppDelegate lifecycle logging
5. `[#137]` Help menu + DiagnosticsExporter (reveal + export zip)

## Risks and Trade-offs

- `uvicorn log_config=None`: uvicorn won't set levels, so root is raised to INFO explicitly; test asserts an INFO record lands in the file.
- Double-logging (`service.log` + `service-stderr.log`) is intentional belt-and-suspenders.
- Secrets: HF token is passed as a `token=` arg (`transcriber.py:93`), never logged. Audit scope includes the raw stderr tee and the exported zip, not just Python call sites.
- Swift drain regression lives in IntegrationTests, which CI does not run — validated manually (existing harness limitation, per TESTING.md).
- `logsDirectory()` path literal is duplicated across Python and Swift (unavoidable cross-language); kept in one place per language.

## Deviations from Plan

- `AppState` gained an explicit `init()` to inject the shared `serviceLog` into every supervisor generation — a stored-property default value can't reference another instance property, so the planned default-arg injection alone was insufficient.
- Added `DiagnosticsExporterTests` (Kit) beyond the drafted test list, exercising the now-pure exporter (zip created, non-empty, overwrite, summary header).
- Added `FileLog.flush()` (blocks until the serial queue drains) so tests and the export are deterministic.
- stdout is teed only *after* the handshake match (`teeBeforeMatch: false`) to skip the handshake JSON line; stderr is teed from the first byte (`teeBeforeMatch: true`) to capture startup failures. Both drain to EOF regardless.
- Help menu uses `CommandGroup(replacing: .help)` (the default "Help" item has no help book, so replacing it is cleaner than appending).
