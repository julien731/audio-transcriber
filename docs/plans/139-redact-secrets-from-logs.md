# Plan: Redact secrets from logs and the diagnostics export

**Story**: #139
**Spec**: N/A (follow-up to #137 / PR #138)
**Branch**: feature/139-redact-secrets-from-logs
**Date**: 2026-09-03
**Mode**: TDD — both redactors are pure functions with clear input/output.

## Technical Decisions

### TD-1: Attach the redacting filter to the handler, not the root logger
- **Context**: AC1 asks for a filter "installed on the root logger". A `logging.Filter` added via `Logger.addFilter` only runs for records logged *directly* on that logger — records propagated up from child loggers (notably `uvicorn`/`uvicorn.access`) skip it.
- **Decision**: Add the filter to the `RotatingFileHandler` (which lives on the root logger). Handler-level filters run for every record the handler emits, including propagated ones.
- **Alternatives considered**: `root.addFilter(...)` — rejected, misses propagated records; a `RedactingFormatter` subclass — works but AC explicitly asks for a filter, and a filter keeps message/traceback handling in one place.

### TD-2: Pre-render redacted traceback into `record.exc_text`
- **Context**: A filter runs before the Formatter renders `record.exc_info`, so redacting `record.msg` alone leaves the traceback unmasked.
- **Decision**: In the filter, format the exception via `logging.Formatter().formatException(record.exc_info)`, redact it, and assign to `record.exc_text`. `Formatter.format` reuses a non-empty `exc_text` instead of re-rendering, so the on-disk traceback is redacted.
- **Alternatives considered**: clearing `exc_info` — rejected, loses structured info other handlers may want.

### TD-3: Fail-open filter
- **Context**: `record.getMessage()` / `formatException()` can raise (e.g. mismatched `%` args). Filters run before the handler's emit try/except, so a raise escapes to the log call site.
- **Decision**: Wrap the redaction work in try/except and return `True` on failure — never drop or raise from the redaction filter.

### TD-4: Per-chunk Swift tee redaction
- **Context**: The raw child stderr tee (`service-stderr.log`) bypasses Python logging entirely. For stderr that never passes through Python (uncaught interpreter tracebacks, C-level/third-party direct prints), the Swift tee is the *sole* redaction sink, not a backstop.
- **Decision**: Redact each `availableData` chunk at the single tee write site. Accepted limitation: a token split across two reads could evade masking (low risk — tracebacks arrive as a burst).
- **Alternatives considered**: newline-buffered tee with EOF flush — more robust but adds a second buffer to a concurrency-sensitive drain loop; deferred for simplicity.

## Files to Create or Modify

- `backend/services/logging_setup.py` — add `_SECRET_PATTERNS`, `redact_secrets()`, `SecretRedactingFilter`; attach filter to the handler.
- `tests/unit/test_logging_setup.py` — message/traceback on-disk redaction, fail-open, `redact_secrets` unit.
- `macos/Sources/MeetingTranscriberKit/Logging/SecretRedaction.swift` (new) — `SecretRedaction.redact(_:)`.
- `macos/Sources/MeetingTranscriberKit/Service/ServiceSupervisor.swift` — redact chunk before teeing.
- `macos/Sources/MeetingTranscriberKit/Logging/AppLog.swift` — redact lines before writing (app.log is bundled too).
- `macos/Sources/MeetingTranscriberKitTests/SecretRedactionTests.swift` (new) + `main.swift` registration.
- `CHANGELOG.md` — Unreleased entry.

## Approach per AC

### AC1/AC2: Redacting filter on the root logger covering message + traceback
`SecretRedactingFilter` attached to the `RotatingFileHandler`; redacts `record.msg` (args collapsed via `getMessage()`), `record.exc_text` (pre-rendered), and `record.stack_info`.

### AC3: Raw stderr tee redacted Swift-side
`SecretRedaction.redact` applied at the tee write site in `drainAndTee`; `AppLog` writes also routed through it.

### AC4: Unit test on-disk
Log a record/exception containing `hf_TESTTOKEN`; assert the on-disk `service.log` contains `hf_***`, not the token.

### AC5: Diagnostics zip free of the token
Python on-disk test as automated coverage; documented manual check in the PR (seed token, export, `unzip -p | grep`). Shared `hf_TESTTOKEN` vector in Python and Swift tests to surface pattern drift.

## Commit Sequence

1. Python redacting filter + tests
2. Swift redactor + tee/AppLog redaction + tests
3. CHANGELOG + plan doc

## Risks and Trade-offs

- Per-chunk Swift redaction: token split across reads could evade masking (documented, low risk).
- Two pattern lists (Python + Swift) can drift; mitigated by cross-reference comments and a shared test token.
- Redaction is forward-only — existing/rotated on-disk log content is not rewritten (our code never logged tokens historically).
- app.log carries only Swift-authored lifecycle strings, but is routed through the redactor anyway as near-free defense-in-depth.

## Deviations from Plan

- The fail-open test (`test_filter_neutralizes_malformed_format_record`) exercises `SecretRedactingFilter.filter()` directly rather than through the logging pipeline: pytest's own capture handler re-raises a malformed-`%`-args record independent of our filter, so a full-pipeline test could not isolate our filter's behavior. The filter now also drops `record.args` when `getMessage()` raises, so no downstream formatter can raise either.
- The Swift redactor also covers `app.log` (via `AppLog.line`), not only the `service-stderr.log` tee — near-free defense-in-depth for the third bundled sink.
