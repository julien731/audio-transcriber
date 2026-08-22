import Foundation

/// First-run provisioning status (`GET/POST /api/provisioning*`). Drives the
/// setup wizard: token entry, model-download progress, and the failure/retry +
/// continue-without-diarization contract (plan slice 4, BR-12/13/14).
public struct ProvisioningStatus: Codable, Equatable {
    public let provisioningCompleted: Bool
    public let modelsPresent: Bool
    public let whisperModel: String
    public let diarizationAvailable: Bool
    public let downloadState: DownloadState
    public let downloadProgress: Int
    public let downloadError: String?

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        provisioningCompleted = try c.decode(Bool.self, forKey: .provisioningCompleted)
        modelsPresent = try c.decode(Bool.self, forKey: .modelsPresent)
        whisperModel = try c.decode(String.self, forKey: .whisperModel)
        diarizationAvailable = try c.decode(Bool.self, forKey: .diarizationAvailable)
        downloadState = try c.decodeIfPresent(DownloadState.self, forKey: .downloadState) ?? .idle
        downloadProgress = try c.decodeIfPresent(Int.self, forKey: .downloadProgress) ?? 0
        downloadError = try c.decodeIfPresent(String.self, forKey: .downloadError)
    }
}
