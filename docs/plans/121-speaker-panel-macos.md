# Plan: Speakers panel with click-to-cycle (macOS)

**Story**: #121
**Spec**: docs/specs/native-macos-app.md
**Branch**: speaker-list-macos-app
**Date**: 2026-08-23
**Mode**: TDD for Kit helpers (pure, harness-testable) · Standard for the SwiftUI view (harness cannot exercise SwiftUI)

## Technical Decisions

### TD-1: Two pure Kit helpers, cursor state in the view
- **Context**: The story asks for two testable Kit helpers (ordered speaker list + summary; next-segment cursor resolution) while cycle state must persist per speaker across clicks.
- **Decision**: Put ordering/summary/next-segment logic in a stateless `enum SpeakerPanel` in Kit; keep the per-speaker cursor (`[speakerId: lastSegmentId]`) as `@State` in `TranscriptTabView`. `nextSegmentId` takes the current cursor as a parameter and returns the next target, so it stays pure and unit-testable.
- **Alternatives considered**: An observable stateful panel model — rejected as untestable in the CLT harness and heavier than a dict lookup.

### TD-2: Panel lives inside `TranscriptTabView`
- **Context**: AC9 requires the panel only on the Transcript tab; AC5 requires scrolling to a segment, which needs the `ScrollViewReader` proxy.
- **Decision**: Host the toggle + panel as overlays inside the existing `ScrollViewReader` in `TranscriptTabView` (only rendered on the Transcript tab). The row-select closure calls `proxy.scrollTo` directly, so repeated clicks on a single-segment speaker still re-center (a value-based `.onChange` would miss an unchanged target).
- **Alternatives considered**: Toggle in `TranscriptContainerView` header — rejected; the proxy and segment ids live in `TranscriptTabView`.

### TD-3: Reuse `SpeakerColor` for ordering, colors, naming
- **Context**: AC1/AC2 require first-appearance order and colors identical to transcript segments.
- **Decision**: `SpeakerPanel.rows` builds on `SpeakerColor.orderedSpeakerIds` + `assignments` + `isUnidentified` + `displayName` — the same source `TranscriptTabView.colors` uses, guaranteeing parity. No reimplementation.

## Files to Create or Modify

- `macos/Sources/MeetingTranscriberKit/Presentation/SpeakerPanel.swift` — new `enum SpeakerPanel` with `SpeakerRow`, `rows(in:speakers:)`, `summary(for:)`, `nextSegmentId(for:in:after:)`.
- `macos/Sources/MeetingTranscriberKitTests/SpeakerPanelTests.swift` — `runSpeakerPanelTests()` suite.
- `macos/Sources/MeetingTranscriberKitTests/main.swift` — register `runSpeakerPanelTests()`.
- `macos/Sources/MeetingTranscriberApp/Views/TranscriptTabView.swift` — toggle button + dismissible `SpeakersPanelView`, per-speaker `cursors`, `jump(to:proxy:)`.

## Approach per AC

### AC 1: Panel lists each distinct speaker once, ordered by first appearance
`SpeakerPanel.rows` maps `SpeakerColor.orderedSpeakerIds(in:)` (deduped, first-appearance).

### AC 2: Row shows the same color as the speaker's segments + display name
`colorHex` from `SpeakerColor.assignments(orderedSpeakerIds:)` (same computation as `TranscriptTabView.colors`); swatch via `Color(hex:)`.

### AC 3: Unnamed speakers render as "Unnamed speaker" with a flag, styled distinctly
`isUnnamed = SpeakerColor.isUnidentified(speakers[id] ?? id)`; `displayName` becomes "Unnamed speaker"; the row shows a "?" flag and a muted style.

### AC 4: Header summary "N speakers, all named" or "M of N unnamed"
`SpeakerPanel.summary(for:)` from `rows.count` and the unnamed subset; singular/plural handled.

### AC 5 / AC 6: Click cycles forward through the speaker's segments, wraps, first click → first segment, independent per speaker
`nextSegmentId(for:in:after:)`: filter segments by speaker in document order; nil cursor → first; else `(index(of cursor)+1) % count`; stale cursor → first. `TranscriptTabView.cursors[speakerId]` stores the last target per speaker (independent keys).

### AC 7: Clicking a speaker with no resolvable segment is a no-op
`nextSegmentId` returns nil for a speaker with no segments; `jump` guards on nil and does nothing.

### AC 8: Jump only — renaming stays on the per-segment menu
Panel rows only jump; the existing `speakerMenu` rename path is untouched.

### AC 9: Panel toggleable/dismissible, only on the Transcript tab
`showSpeakers` toggled by a floating button; close button + toggle dismiss it. Panel lives in `TranscriptTabView`, rendered only for the `.transcript` tab.

## Commit Sequence

1. `[#121] Add speaker panel plan doc`
2. `[#121] Add SpeakerPanel Kit helpers and tests`
3. `[#121] Add speakers panel with click-to-cycle to Transcript tab`

## Risks and Trade-offs

- Active-segment auto-scroll during playback can compete with a manual jump. The web client mitigates via `userScrolledAway`; on macOS auto-scroll only fires as audio advances, so the conflict is minor — accepted for this slice.
- Swift tests are not run in CI (manual only). Kit helpers are still fully covered via `swift run MeetingTranscriberKitTests`.
- AC6 per-speaker independence lives in view `@State`, not harness-testable; the underlying `nextSegmentId` is unit-tested, so correctness rests on trivial dict keying.

## Deviations from Spec

- None. The spec (native-macos-app.md) covers the transcript viewer generally; this slice restores the web speakers sidebar with the intentional click-to-cycle improvement described in the story.

## Deviations from Plan

_Populated after implementation._
