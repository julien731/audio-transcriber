import Foundation

/// Job progress (`GET /api/jobs/{id}`). `stage` is a free-form string on the wire
/// (schemas.py keeps it a plain str, though JobStage enumerates known values).
public struct JobInfo: Codable, Equatable, Identifiable {
    public let id: String
    public let meetingId: String
    public let status: JobStatus
    public let progress: Int
    public let stage: String
    public let error: String?
    public let createdAt: Date
    public let updatedAt: Date

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        meetingId = try c.decode(String.self, forKey: .meetingId)
        status = try c.decode(JobStatus.self, forKey: .status)
        progress = try c.decodeIfPresent(Int.self, forKey: .progress) ?? 0
        stage = try c.decodeIfPresent(String.self, forKey: .stage) ?? ""
        error = try c.decodeIfPresent(String.self, forKey: .error)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        updatedAt = try c.decode(Date.self, forKey: .updatedAt)
    }
}
