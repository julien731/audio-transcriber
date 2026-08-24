import Foundation
import MeetingTranscriberKit

private func seg(_ id: String, _ start: Double, _ end: Double) -> TranscriptSegment {
    TranscriptSegment(id: id, start: start, end: end, speaker: "SPEAKER_00", text: "x")
}

func runTranscriptSyncTests() {
    let segments = [seg("s1", 0, 2), seg("s2", 2, 4), seg("s3", 5, 6)]

    suite("TranscriptSync.activeSegmentId exact start boundary") {
        // Seeking to a segment's exact start must resolve to that segment (guards
        // the inclusive `<=` against a regression to `<`).
        expectEqual(TranscriptSync.activeSegmentId(at: 2, in: segments), "s2",
                    "exact start of s2 → s2")
        expectEqual(TranscriptSync.activeSegmentId(at: 0, in: segments), "s1",
                    "exact start of first segment → s1")
    }

    suite("TranscriptSync.activeSegmentId mid-segment") {
        expectEqual(TranscriptSync.activeSegmentId(at: 1, in: segments), "s1", "1s → s1")
        expectEqual(TranscriptSync.activeSegmentId(at: 3.9, in: segments), "s2", "3.9s → s2")
    }

    suite("TranscriptSync.activeSegmentId end boundary is exclusive") {
        // At t == end the next segment (if any) owns the instant, else nil.
        expectEqual(TranscriptSync.activeSegmentId(at: 2, in: segments), "s2",
                    "t == s1.end falls into s2")
        expectNil(TranscriptSync.activeSegmentId(at: 6, in: segments),
                  "t == last segment end → nil")
    }

    suite("TranscriptSync.activeSegmentId outside and gaps") {
        expectNil(TranscriptSync.activeSegmentId(at: -1, in: segments), "before first → nil")
        expectNil(TranscriptSync.activeSegmentId(at: 10, in: segments), "after last → nil")
        expectNil(TranscriptSync.activeSegmentId(at: 4.5, in: segments), "gap between s2 and s3 → nil")
        expectNil(TranscriptSync.activeSegmentId(at: 0, in: []), "no segments → nil")
    }
}
