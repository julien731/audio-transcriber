import Foundation
import MeetingTranscriberKit

private func seg(_ id: String, _ speaker: String) -> TranscriptSegment {
    TranscriptSegment(id: id, start: 0, end: 1, speaker: speaker, text: "x")
}

func runSpeakerPanelTests() {
    suite("SpeakerPanel.rows ordering + colors") {
        let segments = [seg("s1", "SPEAKER_01"), seg("s2", "SPEAKER_00"), seg("s3", "SPEAKER_01")]
        let rows = SpeakerPanel.rows(in: segments, speakers: ["SPEAKER_01": "Alice"])
        expectEqual(rows.count, 2, "distinct speakers only")
        expectEqual(rows.map(\.id), ["SPEAKER_01", "SPEAKER_00"], "first-appearance order")
        expectEqual(rows[0].colorHex, "#4A90D9", "first speaker → first palette color")
        expectEqual(rows[1].colorHex, "#D94A4A", "second speaker → second palette color")
        expectEqual(rows[0].displayName, "Alice", "named speaker shows mapped name")
        expect(!rows[0].isUnnamed, "named speaker not flagged unnamed")
    }

    suite("SpeakerPanel.rows unnamed speakers") {
        let segments = [seg("s1", "SPEAKER_00"), seg("s2", "UNKNOWN")]
        let rows = SpeakerPanel.rows(in: segments, speakers: [:])
        expect(rows[0].isUnnamed, "SPEAKER_00 flagged unnamed")
        expectEqual(rows[0].displayName, "Unnamed speaker", "SPEAKER_00 → Unnamed speaker label")
        expect(rows[1].isUnnamed, "UNKNOWN flagged unnamed")
        expectEqual(rows[1].displayName, "Unnamed speaker", "UNKNOWN → Unnamed speaker label")
    }

    suite("SpeakerPanel.summary") {
        let one = SpeakerPanel.rows(in: [seg("s1", "SPEAKER_00")], speakers: ["SPEAKER_00": "Alice"])
        expectEqual(SpeakerPanel.summary(for: one), "1 speaker, all named", "singular, all named")
        let twoNamed = SpeakerPanel.rows(in: [seg("s1", "SPEAKER_00"), seg("s2", "SPEAKER_01")],
                                         speakers: ["SPEAKER_00": "Alice", "SPEAKER_01": "Bob"])
        expectEqual(SpeakerPanel.summary(for: twoNamed), "2 speakers, all named", "plural, all named")
        let mixed = SpeakerPanel.rows(in: [seg("s1", "SPEAKER_00"), seg("s2", "SPEAKER_01"), seg("s3", "SPEAKER_02")],
                                      speakers: ["SPEAKER_00": "Alice"])
        expectEqual(SpeakerPanel.summary(for: mixed), "2 of 3 unnamed", "M of N unnamed")
    }

    suite("SpeakerPanel.nextSegmentId cycling") {
        let segments = [
            seg("a1", "SPEAKER_00"), seg("b1", "SPEAKER_01"),
            seg("a2", "SPEAKER_00"), seg("a3", "SPEAKER_00"),
        ]
        // First click → first segment.
        expectEqual(SpeakerPanel.nextSegmentId(for: "SPEAKER_00", in: segments, after: nil), "a1", "nil cursor → first")
        // Advance forward.
        expectEqual(SpeakerPanel.nextSegmentId(for: "SPEAKER_00", in: segments, after: "a1"), "a2", "advance a1 → a2")
        expectEqual(SpeakerPanel.nextSegmentId(for: "SPEAKER_00", in: segments, after: "a2"), "a3", "advance a2 → a3")
        // Wrap last → first.
        expectEqual(SpeakerPanel.nextSegmentId(for: "SPEAKER_00", in: segments, after: "a3"), "a1", "wrap last → first")
        // Stale cursor (belongs to another speaker) → first.
        expectEqual(SpeakerPanel.nextSegmentId(for: "SPEAKER_00", in: segments, after: "b1"), "a1", "stale cursor → first")
        // Independent cursors: SPEAKER_01 has a single segment, wraps to itself.
        expectEqual(SpeakerPanel.nextSegmentId(for: "SPEAKER_01", in: segments, after: nil), "b1", "other speaker first")
        expectEqual(SpeakerPanel.nextSegmentId(for: "SPEAKER_01", in: segments, after: "b1"), "b1", "single segment wraps to itself")
        // No resolvable segment → nil (no-op).
        expectNil(SpeakerPanel.nextSegmentId(for: "SPEAKER_99", in: segments, after: nil), "unknown speaker → nil")
    }
}
