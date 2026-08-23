# Plan: Set HuggingFace token in app settings

**Story**: #104
**Spec**: N/A (GitHub issue body empty; ACs proposed and approved in-thread)
**Branch**: execute-issue-104
**Date**: 2026-08-23
**Mode**: TDD for the Kit model (async logic, existing executable test harness); Standard for the SwiftUI view/scene wiring (no view-test infra in this repo)

## Technical Decisions

### TD-1: Reuse `ProvisioningController` instead of a new token client
- **Context**: The setup wizard already sets the token, kicks off the model download, polls progress, and honors the "backend never validates the token → never claim it was rejected" contract. Setting a token without downloading the pyannote models would let the UI falsely report "diarization enabled" when the weights are absent (they would otherwise download on-demand at the next transcription, or silently degrade if offline).
- **Decision**: A thin `SettingsTokenModel` (Kit, plain `final class`) composes a `ProvisioningController`. Saving a token delegates to `submit(token:)` (setToken + startModelDownload); clearing delegates to `submit("")` (Whisper-only re-provision). The model adds only an editing state machine and a `SettingsPhase` display mapping — the genuinely new, testable logic.
- **Alternatives considered**: A standalone `TokenSettingsController` calling `setToken` alone — rejected (re-implements the token/never-rejected contract and skips the model download, the exact gap this story must close).

### TD-2: Compose, don't subclass; keep the view trivial
- **Context**: SwiftUI views aren't unit-tested here (Command Line Tools toolchain, no XCTest).
- **Decision**: All real logic lives in `SettingsTokenModel` / `ProvisioningController` (both tested). `SettingsView` mirrors `SetupWizardView`'s `@State` + async-controller + `pollWhileDownloading` pattern and stays presentation-only.

## Files to Create or Modify

- `macos/Sources/MeetingTranscriberKit/Settings/SettingsTokenModel.swift` — composes `ProvisioningController`; `SettingsPhase` = `.idle(diarizationAvailable:)` | `.editing` | `.working(progress:)` | `.failed(message:)`; API `beginEditing()`, `cancelEditing(diarizationAvailable:)`, `save(token:) async`, `refresh() async`. Maps `SetupPhase` → `SettingsPhase`.
- `macos/Sources/MeetingTranscriberKitTests/SettingsTokenModelTests.swift` — unit suite (MockURLProtocol).
- `macos/Sources/MeetingTranscriberKitTests/main.swift` — register `runSettingsTokenModelTests()`.
- `macos/Sources/MeetingTranscriberApp/Views/SettingsView.swift` — SwiftUI Settings form.
- `macos/Sources/MeetingTranscriberApp/AppState.swift` — add `var client: APIClient?` (non-nil only for `.ready`).
- `macos/Sources/MeetingTranscriberApp/App.swift` — add `Settings { SettingsView(appState:) }` scene.

## Approach per AC

### AC1 / AC6: Open Settings; gated on `.ready`
`SettingsView(appState:)` observes `AppState`. `if case let .ready(client)` → fetch `provisioning()`, build `ProvisioningController` + `SettingsTokenModel`, render status from `.idle(diarizationAvailable:)`. Other phases → "Settings will be available once the transcription service is ready." Status load keyed via `.task(id:)` on client identity so it refreshes across service restarts.

### AC2: Enter token + Save → download + confirm
`beginEditing()` reveals a `SecureField`; Save → `model.save(token:)` (submit = setToken + startModelDownload) → `.working(progress)`; view polls `refresh()` while working (with `Task.isCancelled` guard); `.idle(diar:true)` confirms enabled.

### AC3: Clear token → Whisper-only
Clear → `save(token: "")`; confirmation derived from the returned `diarizationAvailable` (whitespace-only trims to empty → disabled).

### AC4: Write-only field
Field starts empty and is never populated from the backend; Save/Clear disabled while `.working`/busy.

### AC5: Failure handling
`.failed(message)` shows inline with Retry; message comes from `ProvisioningController` (`APIError.userMessage`) and never asserts token validity.

## Commit Sequence

1. `[#104] docs: add implementation plan`
2. `[#104] feat(macos): SettingsTokenModel composing ProvisioningController + unit suite`
3. `[#104] feat(macos): Settings window to set the HuggingFace token`

## Risks and Trade-offs

- Clearing the token re-runs a Whisper-only `startModelDownload` round-trip — fast (weights cached) but a provisioning round-trip; reuses the tested path.
- SwiftUI views untested here — logic concentrated in the tested Kit model.
- Unrelated placeholder "Blah" in `Info.plist` / `WindowGroup` title left out of scope.

## Deviations from Plan

- `SettingsTokenModel.settingsPhase(from:editing:)` is `public` (not internal) so the separate test module can unit-test the mapping directly. No behavior change.
- On a failed save, "Try again" re-sends the token via `save()` (setToken + startModelDownload) rather than a download-only retry. Functionally equivalent and slightly more robust (token is re-persisted); accepted by architect review as a Nit.
