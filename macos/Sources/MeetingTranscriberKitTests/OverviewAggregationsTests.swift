import Foundation
import MeetingTranscriberKit

private let insightDecoder = JSONCoding.makeDecoder()

private func emotion(_ id: String, _ speaker: String, start: Double, end: Double,
                     _ category: String, _ confidence: Double) -> EmotionAnnotation {
    try! insightDecoder.decode(EmotionAnnotation.self, from: Data("""
    {"segment_id":"\(id)","speaker":"\(speaker)","start":\(start),"end":\(end),
     "primary_emotion":"\(category)","confidence":\(confidence),"emotion_scores":{},"low_confidence":false}
    """.utf8))
}

private func interaction(_ type: String, a: String, b: String) -> InteractionEvent {
    try! insightDecoder.decode(InteractionEvent.self, from: Data("""
    {"event_type":"\(type)","timestamp":1.0,"speaker_a":"\(a)","speaker_b":"\(b)","duration":0.5}
    """.utf8))
}

private func segInteraction(_ id: String, hesitation: Double) -> SegmentInteraction {
    try! insightDecoder.decode(SegmentInteraction.self, from: Data("""
    {"segment_id":"\(id)","hesitation_before":\(hesitation)}
    """.utf8))
}

func runOverviewAggregationsTests() {
    suite("energyTrajectory (port of buildEnergyTrajectory)") {
        expect(OverviewAggregations.energyTrajectory([]).isEmpty, "empty → no windows")
        // One 300s window: engaged(+0.8) and frustrated(-0.4) at 10s and 20s → mean 0.2.
        let emotions = [
            emotion("s1", "A", start: 10, end: 12, "engaged", 0.8),
            emotion("s2", "B", start: 20, end: 22, "frustrated", 0.4),
            emotion("s3", "A", start: 305, end: 307, "neutral", 0.9), // second window, score 0
        ]
        let windows = OverviewAggregations.energyTrajectory(emotions)
        expectEqual(windows.count, 2, "two 300s windows (end≈307)")
        expectEqual(windows[0].count, 2, "first window has 2 emotions")
        expect(abs((windows[0].score ?? -9) - 0.2) < 1e-9, "mean of +0.8 and -0.4 = 0.2")
        expectEqual(windows[1].score, 0, "neutral → 0 energy")
        expectEqual(windows[0].start, 0, "window 0 start")
        expectEqual(windows[1].start, 300, "window 1 start")
    }

    suite("energyScore") {
        expectEqual(OverviewAggregations.energyScore(emotion("s", "A", start: 0, end: 1, "confident", 0.7)), 0.7, "positive")
        expectEqual(OverviewAggregations.energyScore(emotion("s", "A", start: 0, end: 1, "disengaged", 0.6)), -0.6, "negative")
        expectEqual(OverviewAggregations.energyScore(emotion("s", "A", start: 0, end: 1, "neutral", 0.9)), 0, "neutral → 0")
    }

    suite("interruptions (port of summarizeInterruptions)") {
        // b|a keying: speaker_b interrupts speaker_a. Two events A→(interrupts)B, one B→A.
        let events = [
            interaction("interruption", a: "B", b: "A"),
            interaction("interruption", a: "B", b: "A"),
            interaction("interruption", a: "A", b: "B"),
            interaction("overlap", a: "A", b: "B"), // ignored
        ]
        let summary = OverviewAggregations.interruptions(events, speakers: ["A": "Alice", "B": "Bob"])
        expectEqual(summary.total, 3, "3 interruptions, overlap ignored")
        // A made 2 (interrupted B twice), received 1; B made 1, received 2.
        expectEqual(summary.totals.first?.speakerId, "A", "A ranks first (made2+recv1=3)")
        expectEqual(summary.totals.first?.made, 2, "A made 2")
        expectEqual(summary.totals.first?.received, 1, "A received 1")
        expectEqual(summary.pairs.first?.count, 2, "top pair count 2")
        expectEqual(summary.pairs.first?.interrupterName, "Alice", "interrupter name resolved")
        expectEqual(summary.pairs.first?.interruptedName, "Bob", "interrupted name resolved")
    }

    suite("latencies (port of summarizeLatencies)") {
        let transcript = Transcript(segments: [
            TranscriptSegment(id: "s1", start: 0, end: 1, speaker: "A", text: "x"),
            TranscriptSegment(id: "s2", start: 1, end: 2, speaker: "B", text: "y"),
            TranscriptSegment(id: "s3", start: 2, end: 3, speaker: "A", text: "z"),
        ])
        let sis = [
            segInteraction("s1", hesitation: 1.0),
            segInteraction("s3", hesitation: 3.0),
            segInteraction("s2", hesitation: 0.5),
            segInteraction("sX", hesitation: 9.0), // no matching segment → skipped
            segInteraction("s2", hesitation: 0.0), // zero → skipped
        ]
        let result = OverviewAggregations.latencies(sis, transcript: transcript, speakers: ["A": "Alice"])
        expectEqual(result.count, 2, "two speakers with positive hesitation")
        expectEqual(result.first?.speakerId, "A", "A highest average")
        expect(abs((result.first?.average ?? 0) - 2.0) < 1e-9, "A avg (1.0+3.0)/2 = 2.0")
        expectEqual(result.first?.name, "Alice", "name resolved")
    }

    suite("SegmentInsights.isWordToneMismatch (badge port)") {
        let frustrated = emotion("s", "A", start: 0, end: 1, "frustrated", 0.9)
        expect(SegmentInsights.isWordToneMismatch(emotion: frustrated, text: "Yeah, sounds good."),
               "frustrated tone + agreement phrase → mismatch")
        expect(!SegmentInsights.isWordToneMismatch(emotion: frustrated, text: "I strongly disagree."),
               "no agreement phrase → no mismatch")
        let engaged = emotion("s", "A", start: 0, end: 1, "engaged", 0.9)
        expect(!SegmentInsights.isWordToneMismatch(emotion: engaged, text: "sounds good"),
               "non-mismatch emotion → no mismatch")
        expect(!SegmentInsights.isWordToneMismatch(emotion: nil, text: "sounds good"), "nil emotion → false")
    }

    suite("SegmentInsights.index + labels") {
        expectEqual(SegmentInsights.label(for: .frustrated), "Frustrated", "known label")
        expectEqual(SegmentInsights.label(for: .unknown), "Unknown", "fallback label")
    }
}
