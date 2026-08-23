import Foundation

public struct EmotionAnnotation: Codable, Equatable {
    public let segmentId: String
    public let speaker: String
    public let start: Double
    public let end: Double
    public let primaryEmotion: EmotionCategory
    public let confidence: Double
    public let emotionScores: [String: Double]
    public let lowConfidence: Bool
}

public struct EmotionUnavailable: Codable, Equatable {
    public let segmentId: String
    public let reason: String
}

public struct ProsodyAnnotation: Codable, Equatable {
    public let segmentId: String
    public let speaker: String
    public let start: Double
    public let end: Double
    public let volumeMean: Double
    public let volumeVariance: Double
    public let pitchMean: Double
    public let pitchVariance: Double
    public let speakingRate: Double
    public let pauseRatio: Double
}

public struct ProsodyUnavailable: Codable, Equatable {
    public let segmentId: String
    public let reason: String
}

public struct InteractionEvent: Codable, Equatable {
    public let eventType: InteractionEventType
    public let timestamp: Double
    public let speakerA: String
    public let speakerB: String
    public let duration: Double
    public let context: String

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        eventType = try container.decode(InteractionEventType.self, forKey: .eventType)
        timestamp = try container.decode(Double.self, forKey: .timestamp)
        speakerA = try container.decode(String.self, forKey: .speakerA)
        speakerB = try container.decode(String.self, forKey: .speakerB)
        duration = try container.decode(Double.self, forKey: .duration)
        context = try container.decodeIfPresent(String.self, forKey: .context) ?? ""
    }
}

public struct SegmentInteraction: Codable, Equatable {
    public let segmentId: String
    public let precededByInterruption: Bool
    public let followedByInterruption: Bool
    public let hesitationBefore: Double

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        segmentId = try container.decode(String.self, forKey: .segmentId)
        precededByInterruption = try container.decodeIfPresent(Bool.self, forKey: .precededByInterruption) ?? false
        followedByInterruption = try container.decodeIfPresent(Bool.self, forKey: .followedByInterruption) ?? false
        hesitationBefore = try container.decodeIfPresent(Double.self, forKey: .hesitationBefore) ?? 0
    }
}

/// Mirrors schemas.py `AudioAnalysis`. All list fields default to empty and all
/// sub-status fields are optional, so a partial/degraded analysis decodes cleanly
/// (plan Artifact B: degraded states must render, not just the happy path).
public struct AudioAnalysis: Codable, Equatable {
    public let status: AudioAnalysisStatus
    public let reason: String?
    public let emotionStatus: AudioAnalysisStatus?
    public let emotionReason: String?
    public let emotions: [EmotionAnnotation]
    public let emotionUnavailable: [EmotionUnavailable]
    public let prosodyStatus: AudioAnalysisStatus?
    public let prosodyReason: String?
    public let prosody: [ProsodyAnnotation]
    public let prosodyUnavailable: [ProsodyUnavailable]
    public let interactionStatus: AudioAnalysisStatus?
    public let interactionReason: String?
    public let interactions: [InteractionEvent]
    public let segmentInteractions: [SegmentInteraction]
    public let dominantSpeakerLimitation: Bool

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        status = try c.decode(AudioAnalysisStatus.self, forKey: .status)
        reason = try c.decodeIfPresent(String.self, forKey: .reason)
        emotionStatus = try c.decodeIfPresent(AudioAnalysisStatus.self, forKey: .emotionStatus)
        emotionReason = try c.decodeIfPresent(String.self, forKey: .emotionReason)
        emotions = try c.decodeIfPresent([EmotionAnnotation].self, forKey: .emotions) ?? []
        emotionUnavailable = try c.decodeIfPresent([EmotionUnavailable].self, forKey: .emotionUnavailable) ?? []
        prosodyStatus = try c.decodeIfPresent(AudioAnalysisStatus.self, forKey: .prosodyStatus)
        prosodyReason = try c.decodeIfPresent(String.self, forKey: .prosodyReason)
        prosody = try c.decodeIfPresent([ProsodyAnnotation].self, forKey: .prosody) ?? []
        prosodyUnavailable = try c.decodeIfPresent([ProsodyUnavailable].self, forKey: .prosodyUnavailable) ?? []
        interactionStatus = try c.decodeIfPresent(AudioAnalysisStatus.self, forKey: .interactionStatus)
        interactionReason = try c.decodeIfPresent(String.self, forKey: .interactionReason)
        interactions = try c.decodeIfPresent([InteractionEvent].self, forKey: .interactions) ?? []
        segmentInteractions = try c.decodeIfPresent([SegmentInteraction].self, forKey: .segmentInteractions) ?? []
        dominantSpeakerLimitation = try c.decodeIfPresent(Bool.self, forKey: .dominantSpeakerLimitation) ?? false
    }
}
