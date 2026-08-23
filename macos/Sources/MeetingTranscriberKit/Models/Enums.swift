import Foundation

// Enum raw values mirror backend/schemas.py exactly. Each is decoded leniently:
// an unknown value from a newer service falls back to a safe `.unknown`-style
// case so the client never hard-fails on forward-compatible additions.

public enum MeetingType: String, Codable, CaseIterable, Equatable {
    case interview, sales, client, other

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = MeetingType(rawValue: raw) ?? .other
    }
}

public enum MeetingStatus: String, Codable, Equatable {
    case processing, ready, error

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = MeetingStatus(rawValue: raw) ?? .error
    }
}

public enum JobStatus: String, Codable, Equatable {
    case pending, processing, completed, failed

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = JobStatus(rawValue: raw) ?? .failed
    }
}

public enum EmotionCategory: String, Codable, Equatable {
    case neutral, confident, frustrated, uncertain, engaged, disengaged
    case unknown

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = EmotionCategory(rawValue: raw) ?? .unknown
    }
}

public enum AudioAnalysisStatus: String, Codable, Equatable {
    case completed, failed, unavailable

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = AudioAnalysisStatus(rawValue: raw) ?? .unavailable
    }
}

public enum InteractionEventType: String, Codable, Equatable {
    case interruption, overlap
    case longPause = "long_pause"
    case hesitation
    case unknown

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = InteractionEventType(rawValue: raw) ?? .unknown
    }
}

public enum DownloadState: String, Codable, Equatable {
    case idle, downloading, completed, failed

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = DownloadState(rawValue: raw) ?? .idle
    }
}
