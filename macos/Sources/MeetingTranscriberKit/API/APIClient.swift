import Foundation

/// Thin async wrapper over the local service HTTP API. Every workflow action goes
/// through here — the app holds no transcription/workflow logic of its own
/// (BR-6). Base URL is the ephemeral localhost address discovered at launch.
public final class APIClient {
    private let baseURL: URL
    private let session: URLSession
    private let decoder = JSONCoding.makeDecoder()
    private let encoder = JSONCoding.makeEncoder()

    public init(baseURL: URL, session: URLSession = APIClient.defaultSession) {
        self.baseURL = baseURL
        self.session = session
    }

    /// The service is a local process doing CPU-bound transcription: a heavy
    /// compute burst can starve its event loop for tens of seconds, so the stock
    /// 60s per-request timeout produces false "request timed out" alarms on
    /// background refreshes (issue #133). 120s clears those bursts (the timer
    /// resets on the first received byte) while a genuinely hung service still
    /// errors within ~2 minutes. Launch reachability is unaffected — `HealthClient`
    /// keeps its own short probe.
    public static let defaultSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        return URLSession(configuration: config)
    }()

    // MARK: Meetings

    public func listMeetings() async throws -> [MeetingSummary] {
        try await get("api/meetings")
    }

    public func meeting(id: String) async throws -> MeetingDetail {
        try await get("api/meetings/\(encodePath(id))")
    }

    public func deleteMeeting(id: String) async throws {
        _ = try await send(method: "DELETE", path: "api/meetings/\(encodePath(id))")
    }

    public func updateMeeting(id: String, update: MeetingUpdate) async throws -> MeetingMetadata {
        try await sendJSON(method: "PATCH", path: "api/meetings/\(encodePath(id))", body: update)
    }

    public func renameSegmentSpeaker(id: String, segmentId: String, speakerName: String) async throws {
        let body = SegmentSpeakerUpdate(segmentId: segmentId, speakerName: speakerName)
        _ = try await send(method: "PATCH",
                           path: "api/meetings/\(encodePath(id))/segments/speaker",
                           body: try encoder.encode(body),
                           contentType: "application/json")
    }

    public func cancelTranscription(id: String) async throws {
        _ = try await send(method: "POST", path: "api/meetings/\(encodePath(id))/cancel")
    }

    public func retryTranscription(id: String) async throws -> StartResponse {
        try await decode(from: send(method: "POST", path: "api/meetings/\(encodePath(id))/retry"))
    }

    /// Multipart upload that starts a transcription (`POST /api/meetings`).
    public func createMeeting(_ upload: MeetingUpload) async throws -> StartResponse {
        let boundary = "MTBoundary-\(UUID().uuidString)"
        let body = upload.multipartBody(boundary: boundary)
        return try await decode(from: send(
            method: "POST",
            path: "api/meetings",
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)"
        ))
    }

    public func audioURL(id: String) -> URL {
        baseURL.appendingPathComponent("api/meetings/\(encodePath(id))/audio")
    }

    // MARK: Jobs

    public func job(id: String) async throws -> JobInfo {
        try await get("api/jobs/\(encodePath(id))")
    }

    // MARK: Analysis

    public func analysisPrompt(id: String, templateType: String, meetingContext: String?) async throws -> AnalysisPromptResponse {
        var items = [URLQueryItem(name: "template_type", value: templateType)]
        if let meetingContext { items.append(URLQueryItem(name: "meeting_context", value: meetingContext)) }
        return try await get("api/meetings/\(encodePath(id))/analysis-prompt", query: items)
    }

    // MARK: Service / provisioning

    public func health() async throws -> HealthResponse {
        try await get("api/health")
    }

    public func provisioning() async throws -> ProvisioningStatus {
        try await get("api/provisioning")
    }

    public func setToken(_ token: String) async throws -> ProvisioningStatus {
        try await sendJSON(method: "POST", path: "api/provisioning/token", body: TokenUpdate(hfToken: token))
    }

    public func startModelDownload() async throws -> ProvisioningStatus {
        try await decode(from: send(method: "POST", path: "api/provisioning/models"))
    }

    // MARK: - Request plumbing

    private func get<T: Decodable>(_ path: String, query: [URLQueryItem] = []) async throws -> T {
        try await decode(from: send(method: "GET", path: path, query: query))
    }

    private func sendJSON<Body: Encodable, T: Decodable>(method: String, path: String, body: Body) async throws -> T {
        let data = try encoder.encode(body)
        return try await decode(from: send(method: method, path: path, body: data, contentType: "application/json"))
    }

    private func send(
        method: String,
        path: String,
        query: [URLQueryItem] = [],
        body: Data? = nil,
        contentType: String? = nil
    ) async throws -> Data {
        var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        if !query.isEmpty { components.queryItems = query }
        var request = URLRequest(url: components.url!)
        request.httpMethod = method
        request.httpBody = body
        if let contentType { request.setValue(contentType, forHTTPHeaderField: "Content-Type") }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw Self.mapTransport(error)
        }
        guard let http = response as? HTTPURLResponse else {
            throw APIError.transport("The service returned no HTTP response.")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw Self.mapError(status: http.statusCode, data: data)
        }
        return data
    }

    private func decode<T: Decodable>(from data: Data) throws -> T {
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decoding(String(describing: error))
        }
    }

    private func encodePath(_ component: String) -> String {
        component.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? component
    }

    static func mapError(status: Int, data: Data) -> APIError {
        let detail = decodeDetail(data)
        switch status {
        case 400: return .badRequest(detail)
        case 404: return .notFound(detail)
        case 409: return .conflict(detail)
        case 503: return .modelsUnavailable(detail)
        default: return .server(status: status, detail: detail)
        }
    }

    static func decodeDetail(_ data: Data) -> String {
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let detail = object["detail"] as? String
        else { return "" }
        return detail
    }

    /// Classify a URL-loading failure. A timeout becomes the dedicated
    /// `.timedOut` case (matched on `NSURLErrorDomain`/`NSURLErrorTimedOut` rather
    /// than the locale-dependent message) so callers can treat it non-fatally;
    /// everything else keeps the human-readable transport message.
    static func mapTransport(_ error: Error) -> APIError {
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorTimedOut {
            return .timedOut
        }
        return .transport(transportMessage(error))
    }

    private static func transportMessage(_ error: Error) -> String {
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain,
           nsError.code == NSURLErrorNotConnectedToInternet || nsError.code == NSURLErrorCannotConnectToHost {
            return "Could not reach the transcription service."
        }
        return nsError.localizedDescription
    }
}
