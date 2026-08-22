import Foundation

/// The service readiness signal. Once `GET /api/health` *connects*, the service
/// is accepting requests — `status` is always `"ready"` at the HTTP layer, so
/// reachability (not a status flip) is the readiness condition (plan Artifact A).
public struct HealthResponse: Codable, Equatable {
    public let status: String
    public let provisioningCompleted: Bool
    public let diarizationAvailable: Bool

    enum CodingKeys: String, CodingKey {
        case status
        case provisioningCompleted = "provisioning_completed"
        case diarizationAvailable = "diarization_available"
    }
}

public enum HealthClient {
    /// Poll `GET /api/health` until it responds or the deadline passes. Readiness
    /// = a successful HTTP response, regardless of body (BR-2, EC-2).
    public static func waitUntilReady(
        baseURL: URL,
        session: URLSession = .shared,
        timeout: TimeInterval = 30,
        pollInterval: TimeInterval = 0.25
    ) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            if await isReachable(baseURL: baseURL, session: session) { return true }
            try? await Task.sleep(nanoseconds: UInt64(pollInterval * 1_000_000_000))
        } while Date() < deadline
        return false
    }

    public static func isReachable(baseURL: URL, session: URLSession = .shared) async -> Bool {
        var request = URLRequest(url: baseURL.appendingPathComponent("api/health"))
        request.timeoutInterval = 5
        do {
            let (_, response) = try await session.data(for: request)
            return (response as? HTTPURLResponse) != nil
        } catch {
            return false
        }
    }
}
