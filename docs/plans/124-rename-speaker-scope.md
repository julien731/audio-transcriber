# Plan: When renaming a speaker, select all segments or just the current one

**Story**: #124
**Spec**: N/A (GitHub issue, no spec file)
**Branch**: feature/124-rename-speaker-scope
**Date**: 2026-08-24
**Mode**: Standard — thin SwiftUI view change plus one pure Kit helper; the harness has no view tests, so the helper carries the unit coverage.

## Technical Decisions

### TD-1: Reuse existing API paths, no backend change
- **Context**: Story asks for per-speaker "all vs single" rename scope, already present in the web client.
- **Decision**: Route "all segments" through the existing `APIClient.updateMeeting(id:update:)` with a merged `speakers` dict, and "this segment" through the existing `renameSegmentSpeaker(...)`. Both endpoints already exist and are used by the web client.
- **Alternatives considered**: A new dedicated endpoint — rejected; the web client achieves the same with the existing PATCH endpoints.

### TD-2: Extract the dict merge as a pure Kit helper
- **Context**: The "all segments" path builds an updated speakers dict; view logic is not unit-tested by the harness.
- **Decision**: Add `SpeakerPanel.speakers(renamingAll:to:in:)` returning a copy of the speakers dict with the target speaker overwritten (trimmed). Unit-test it.
- **Alternatives considered**: Inline the one-line merge in the view — rejected; codebase convention pushes logic into Kit for testability.

### TD-3: Recent-name menu buttons stay single-segment
- **Context**: The web recent-name chips only prefill the input; scope still applies. The macOS menu applies a recent name instantly to one segment.
- **Decision**: Keep the instant single-segment quick action; scope selection lives in the explicit Rename sheet.
- **Alternatives considered**: Prefill+open the sheet on recent-name tap — deferred; would remove the quick-correction affordance.

## Files to Create or Modify

- `macos/Sources/MeetingTranscriberKit/Presentation/SpeakerPanel.swift` — add pure helper `speakers(renamingAll:to:in:)`.
- `macos/Sources/MeetingTranscriberApp/Views/TranscriptTabView.swift` — add `RenameScope` state + radio-group `Picker` in the rename sheet; route Save by scope; add `renameAllSegments(_:to:)`.
- `macos/Sources/MeetingTranscriberKitTests/SpeakerPanelTests.swift` — add a suite for the helper.

## Approach per AC

### AC1: The user can choose the rename scope
Radio-group `Picker` in the rename sheet, bound to a new `RenameScope` state, reset each time the sheet opens.

### AC2: "This segment only" renames one segment
Existing `reassign(_:to:)` → `renameSegmentSpeaker(...)`.

### AC3: "All segments from this speaker" renames every segment for that speaker
`renameAllSegments(_:to:)` builds `SpeakerPanel.speakers(renamingAll:)` and PATCHes via `updateMeeting(speakers:)`.

### AC4: Default scope matches the web (all segments)
`@State private var renameScope: RenameScope = .allSegments`.

### AC5: Recent-name persistence unchanged
Both save paths call `prefs.addRecentSpeakerName(trimmed)`.

## Commit Sequence

1. `[#124] Add SpeakerPanel.speakers(renamingAll:) helper + tests`
2. `[#124] Add rename-scope picker to the macOS Transcript rename sheet`

## Risks and Trade-offs

- View wiring is not unit-tested by the harness (consistent with existing views); correctness rests on the pure-helper tests + QA.
- `SpeakerEditorView` already does bulk meeting-wide rename via the same `updateMeeting(speakers:)` mechanism; the new helper is a single-key merge (distinct shape), so no duplication/extraction.
- Recent-name menu buttons stay single-segment (see TD-3).

## Deviations from Plan

- Adopted the two architect plan-review advisories folded into the plan: `.radioGroup` picker style and acknowledging the pre-existing `SpeakerEditorView`.
- Reversed TD-3 after user feedback: recent-name menu buttons no longer rename a single segment instantly. They now prefill and open the rename sheet, so a recent name still goes through the scope choice (default all segments) — closer to the web chips' behavior.
- Made the dev stub service (`macos/scripts/stub_service.py`) persist speaker renames (both scopes) so the flow is verifiable without the real backend; previously all PATCHes were no-ops.
