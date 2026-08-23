import Foundation

/// The `service.json` record the bundled service writes into Application Support
/// as a fallback discovery channel (the stdout handshake is primary). `nonce` is
/// optional for backward-compat with a service build that predates it.
public struct ServiceRecord: Equatable, Codable {
    public let port: Int
    public let pid: Int
    public let nonce: String?

    public init(port: Int, pid: Int, nonce: String?) {
        self.port = port
        self.pid = pid
        self.nonce = nonce
    }
}

/// Locates and validates the service discovery file. Ownership is pinned to the
/// child the app spawned: a record is trusted only when its `pid` (and, when the
/// app set one, its `nonce`) match — this rejects a stale file from a prior run
/// and a live-but-unrelated process after PID reuse (plan Artifact A, finding #3).
public enum ServiceDiscovery {
    public static let appName = "MeetingTranscriber"
    public static let fileName = "service.json"

    public static func appSupportDirectory(fileManager: FileManager = .default) -> URL {
        let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return base.appendingPathComponent(appName, isDirectory: true)
    }

    public static func serviceFileURL(directory: URL? = nil, fileManager: FileManager = .default) -> URL {
        (directory ?? appSupportDirectory(fileManager: fileManager)).appendingPathComponent(fileName)
    }

    public static func readRecord(at url: URL) -> ServiceRecord? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ServiceRecord.self, from: data)
    }

    /// Whether a discovered record belongs to the child we launched.
    public static func isOwned(_ record: ServiceRecord, expectedPID: Int32, expectedNonce: String?) -> Bool {
        guard record.pid == Int(expectedPID) else { return false }
        if let expectedNonce, !expectedNonce.isEmpty {
            return record.nonce == expectedNonce
        }
        return true
    }

    /// Guarded cleanup (finding #5): re-read the shared file and delete it only if
    /// it still identifies the terminating child — never remove a record that a
    /// restarted or unrelated process may have overwritten in the interim.
    @discardableResult
    public static func removeIfOwned(
        at url: URL,
        expectedPID: Int32,
        expectedNonce: String?,
        fileManager: FileManager = .default
    ) -> Bool {
        guard
            let record = readRecord(at: url),
            isOwned(record, expectedPID: expectedPID, expectedNonce: expectedNonce)
        else { return false }
        try? fileManager.removeItem(at: url)
        return true
    }
}
