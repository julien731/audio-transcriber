import Foundation

/// The startup handshake the bundled service prints on stdout (plan Artifact A,
/// service `service_main.py`). The `ready` line is authoritative for the port;
/// the optional `nonce` is echoed back from the `MT_SERVICE_NONCE` the app set
/// when it spawned the child, letting the app reject a stale/foreign process.
public struct ServiceHandshake: Equatable {
    public let event: String
    public let port: Int
    public let nonce: String?

    public init(event: String, port: Int, nonce: String?) {
        self.event = event
        self.port = port
        self.nonce = nonce
    }

    /// Parse one stdout line. Returns a handshake only for a JSON object with
    /// `event == "ready"` and an integer `port`; every other line (log noise,
    /// non-JSON) yields nil so the reader can keep scanning.
    public static func parse(line: String) -> ServiceHandshake? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{"), let data = trimmed.data(using: .utf8) else { return nil }
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            (object["event"] as? String) == "ready",
            let port = object["port"] as? Int
        else { return nil }
        return ServiceHandshake(event: "ready", port: port, nonce: object["nonce"] as? String)
    }

    /// The machine-readable error line the service emits on stderr when it cannot
    /// bind a port (service EC-1). Returned message is surfaced to the user.
    public static func parseError(line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{"), let data = trimmed.data(using: .utf8) else { return nil }
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            (object["event"] as? String) == "error"
        else { return nil }
        return (object["message"] as? String) ?? "The transcription service failed to start."
    }
}
