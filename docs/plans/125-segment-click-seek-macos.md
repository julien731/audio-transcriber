# Plan: Click a transcript segment to jump to its audio time

**Story**: #125
**Spec**: N/A
**Branch**: feature/125-segment-click-seek-macos
**Date**: 2026-08-24
**Mode**: Standard — UI-wiring change; the single unit-testable sync seam gets a test.

## Technical Decisions

### TD-1: Widen the seek target to the whole segment row
- **Context**: The web renders a large `▶ Play from here` affordance per segment (`transcript-viewer.js` `playFromSegment`). The macOS port only exposes a tiny borderless timecode `Button` (`TranscriptTabView.swift:91`); the rest of the segment is not click-to-seek. That discoverable "click the segment" target is the feature that got lost.
- **Decision**: Add `.contentShape(Rectangle())` + `.onTapGesture` to the segment row `VStack` so a click anywhere on the row seeks to `segment.start` and plays. Keep the existing timecode button, speaker `Menu`, and selectable `Text`.
- **Alternatives considered**: (a) Wrap the row in a `Button` — breaks `Menu`/text-selection hit-testing. (b) Attach the tap only to non-text chrome — would exclude the text (majority of the row) from the target, silently shrinking AC1; rejected.

### TD-2: Extract the active-segment predicate into a tested Kit seam
- **Context**: The transcript↔audio sync predicate (`start <= t && t < end`) lives inline in the view and is the only unit-testable logic in this story.
- **Decision**: Move it to `TranscriptSync.activeSegmentId(at:in:)` in `MeetingTranscriberKit`, mirroring the `SpeakerPanel` public-enum pattern, and unit-test the boundaries.
- **Alternatives considered**: Leave it inline (UI-only change, zero automated coverage) — rejected; architect concurred the extraction is consolidation, not premature abstraction (single occurrence).

## Files to Create or Modify

- `macos/Sources/MeetingTranscriberKit/Presentation/TranscriptSync.swift` — new `public enum TranscriptSync` with `public static func activeSegmentId(at:in:)`.
- `macos/Sources/MeetingTranscriberApp/Views/TranscriptTabView.swift` — whole-row tap-to-seek; route `activeSegmentId` through `TranscriptSync`.
- `macos/Sources/MeetingTranscriberKitTests/TranscriptSyncTests.swift` — boundary tests.
- `macos/Sources/MeetingTranscriberKitTests/main.swift` — register `runTranscriptSyncTests()`.
- `docs/macos-app.md` — checklist note that the whole segment is click-to-seek.

## Approach per AC

### AC 1: Clicking anywhere on a segment row seeks to its exact start time and plays
Add `.contentShape(Rectangle())` + `.onTapGesture { audio.seek(to: segment.start); audio.play() }` on the row `VStack`. `Menu` and the timecode `Button` own their hit-testing, so their taps don't trigger the row tap.

### AC 2: Active-segment highlight reflects the clicked segment
`AudioPlaybackController.seek(to:)` sets `currentTime` synchronously; the highlight recomputes via `TranscriptSync.activeSegmentId(at: audio.currentTime, in:)`, which resolves the clicked segment because its lower bound is inclusive.

### AC 3: Text stays selectable; speaker menu still opens without seeking
Keep `.textSelection(.enabled)` on the `Text` (click-drag selects; single click seeks) and the `Menu` untouched.

## Commit Sequence

1. `[#125]` Add `TranscriptSync.activeSegmentId` + tests (Kit).
2. `[#125]` Make the whole transcript segment click-to-seek (App).
3. `[#125]` Update macOS manual-check note.

## Risks and Trade-offs

- SwiftUI `.onTapGesture` on a row whose child `Text` is selectable: primary path is whole-row clickable. If tap-vs-selection genuinely conflicts on macOS, surface it back to the user as an AC change rather than silently narrowing the tap target.
- Tapping recenters via the existing `.onChange(of: activeSegmentId)` auto-scroll — intended (mirrors playback-follow).

## Deviations from Plan

_Populated after implementation._
