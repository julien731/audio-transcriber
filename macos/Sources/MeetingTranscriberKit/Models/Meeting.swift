import Foundation

/// List-view row (`GET /api/meetings`). Note the service omits `job_id` here; the
/// full `MeetingMetadata` (via `GET /api/meetings/{id}`) carries it, which is how
/// the app resumes polling an in-progress job after reopening (plan grounding).
public struct MeetingSummary: Codable, Equatable, Identifiable {
    public let id: String
    public let title: String
    public let type: MeetingType
    public let createdAt: Date
    public let durationSeconds: Double?
    public let status: MeetingStatus
}

/// Full per-meeting metadata (schemas.py `MeetingMetadata`). Resilient decoding:
/// every field with a Pydantic default is optional/defaulted here so older or
/// newer payloads decode without hard-failing.
public struct MeetingMetadata: Codable, Equatable, Identifiable {
    public let id: String
    public let title: String
    public let type: MeetingType
    public let createdAt: Date
    public let durationSeconds: Double?
    public let audioFilename: String
    public let status: MeetingStatus
    public let language: String
    public let expectedLanguages: [String]
    public let numSpeakers: Int?
    public let preprocessAudio: Bool
    public let audioAnalysisEnabled: Bool
    public let audioAnalysisStatus: AudioAnalysisStatus?
    public let jobId: String?
    public let speakers: [String: String]
    public let error: String?
    public let context: String

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        title = try c.decode(String.self, forKey: .title)
        type = try c.decodeIfPresent(MeetingType.self, forKey: .type) ?? .other
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        durationSeconds = try c.decodeIfPresent(Double.self, forKey: .durationSeconds)
        audioFilename = try c.decodeIfPresent(String.self, forKey: .audioFilename) ?? ""
        status = try c.decodeIfPresent(MeetingStatus.self, forKey: .status) ?? .processing
        language = try c.decodeIfPresent(String.self, forKey: .language) ?? "auto"
        expectedLanguages = try c.decodeIfPresent([String].self, forKey: .expectedLanguages) ?? []
        numSpeakers = try c.decodeIfPresent(Int.self, forKey: .numSpeakers)
        preprocessAudio = try c.decodeIfPresent(Bool.self, forKey: .preprocessAudio) ?? true
        audioAnalysisEnabled = try c.decodeIfPresent(Bool.self, forKey: .audioAnalysisEnabled) ?? false
        audioAnalysisStatus = try c.decodeIfPresent(AudioAnalysisStatus.self, forKey: .audioAnalysisStatus)
        jobId = try c.decodeIfPresent(String.self, forKey: .jobId)
        speakers = try c.decodeIfPresent([String: String].self, forKey: .speakers) ?? [:]
        error = try c.decodeIfPresent(String.self, forKey: .error)
        context = try c.decodeIfPresent(String.self, forKey: .context) ?? ""
    }
}

/// Meeting detail response (`GET /api/meetings/{id}`).
public struct MeetingDetail: Codable, Equatable {
    public let metadata: MeetingMetadata
    public let transcript: Transcript?
    public let audioAnalysis: AudioAnalysis?
}
