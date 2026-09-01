import Foundation

/// Errors surfaced by `APIClient`, mapped from the service's HTTP responses.
/// The service returns FastAPI's `{"detail": "..."}` body on error; `detail` is
/// carried through so the UI can show the service's own message (e.g. the
/// models-missing text for EC-7, or the transcript-not-ready text for EC-11).
public enum APIError: Error, Equatable {
    /// 404 — meeting/job/segment/template not found.
    case notFound(String)
    /// 409 — e.g. transcript not ready yet (EC-11), or meeting not processing.
    case conflict(String)
    /// 503 — new transcription rejected because models are absent (EC-7, BR-24/25).
    case modelsUnavailable(String)
    /// 400 — bad request (unsupported format, file too large).
    case badRequest(String)
    /// Any other non-2xx response.
    case server(status: Int, detail: String)
    /// Response body could not be decoded into the expected type.
    case decoding(String)
    /// Transport failure (offline, connection refused).
    case transport(String)
    /// Request exceeded its timeout. Emitted transiently while a CPU-bound local
    /// transcription saturates the machine and starves the service's event loop;
    /// `MeetingsStore` treats it as non-fatal so a background refresh doesn't raise
    /// a false alarm (issue #133).
    case timedOut

    /// A human-readable message suitable for a toast/inline error.
    public var userMessage: String {
        switch self {
        case let .notFound(m), let .conflict(m), let .modelsUnavailable(m), let .badRequest(m):
            return m
        case let .server(status, detail):
            return detail.isEmpty ? "The service returned an error (\(status))." : detail
        case .decoding:
            return "The service returned an unexpected response."
        case let .transport(m):
            return m
        case .timedOut:
            return "The request timed out."
        }
    }
}
