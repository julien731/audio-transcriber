import Foundation

/// Per-segment insight helpers ported from audio-insights.js: emotion labels, the
/// word/tone-mismatch badge predicate (the only heuristic ported — OQ-5 badge),
/// and the per-segment index used to annotate transcript rows.
public enum SegmentInsights {
    public static let emotionLabels: [EmotionCategory: String] = [
        .neutral: "Neutral", .confident: "Confident", .frustrated: "Frustrated",
        .uncertain: "Uncertain", .engaged: "Engaged", .disengaged: "Disengaged",
    ]

    public static func label(for emotion: EmotionCategory) -> String {
        emotionLabels[emotion] ?? emotion.rawValue.capitalized
    }

    static let mismatchEmotions: Set<EmotionCategory> = [.frustrated, .uncertain]
    static let agreementPhrases: [String] = [
        "works for me", "sounds good", "that's fine", "thats fine",
        "no concerns", "no issues", "no problem", "i agree", "agreed",
        "sure thing", "fine by me", "looks good", "all good",
        "happy with that", "i'm good", "im good", "makes sense",
        "i'm okay with", "im okay with", "i'm fine with", "im fine with",
    ]

    /// utils/audio-insights `isWordToneMismatch`: a frustrated/uncertain tone over
    /// text that verbally agrees. The 4-line display predicate (plan Artifact B).
    public static func isWordToneMismatch(emotion: EmotionAnnotation?, text: String) -> Bool {
        guard let emotion, mismatchEmotions.contains(emotion.primaryEmotion) else { return false }
        let lower = text.lowercased()
        return agreementPhrases.contains { lower.contains($0) }
    }

    public struct SegmentInsight: Equatable {
        public var emotion: EmotionAnnotation?
        public var prosody: ProsodyAnnotation?
        public var interaction: SegmentInteraction?
    }

    public static func hasCompletedAnalysis(metadata: MeetingMetadata, analysis: AudioAnalysis?) -> Bool {
        metadata.audioAnalysisEnabled && analysis?.status == .completed
    }

    /// Index emotion/prosody/segment-interaction by segment id (audio-insights.js
    /// `indexBySegment`).
    public static func index(_ analysis: AudioAnalysis?) -> [String: SegmentInsight] {
        guard let analysis else { return [:] }
        var map: [String: SegmentInsight] = [:]
        for emotion in analysis.emotions { map[emotion.segmentId, default: .init()].emotion = emotion }
        for prosody in analysis.prosody { map[prosody.segmentId, default: .init()].prosody = prosody }
        for interaction in analysis.segmentInteractions { map[interaction.segmentId, default: .init()].interaction = interaction }
        return map
    }
}
