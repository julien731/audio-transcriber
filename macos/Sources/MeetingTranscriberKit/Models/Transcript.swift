import Foundation

public struct TranscriptSegment: Codable, Equatable, Identifiable {
    public let id: String
    public let start: Double
    public let end: Double
    public var speaker: String
    public let text: String
    /// Per-segment detected language (multilingual path only); nil ⇒ fall back to
    /// `Transcript.language` (schemas.py: pre-feature/single-language transcripts).
    public let language: String?

    public init(id: String, start: Double, end: Double, speaker: String, text: String, language: String? = nil) {
        self.id = id
        self.start = start
        self.end = end
        self.speaker = speaker
        self.text = text
        self.language = language
    }
}

public struct Transcript: Codable, Equatable {
    public let segments: [TranscriptSegment]
    public let language: String

    public init(segments: [TranscriptSegment], language: String = "en") {
        self.segments = segments
        self.language = language
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        segments = try container.decode([TranscriptSegment].self, forKey: .segments)
        language = try container.decodeIfPresent(String.self, forKey: .language) ?? "en"
    }
}
