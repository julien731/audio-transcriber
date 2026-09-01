import Foundation

/// In-process URL stub for APIClient unit tests. A handler inspects each request
/// (method, path, captured body) and returns a status + body. Runs on the URL
/// loading system's own queue, so tests can block the calling thread on a
/// semaphore while a request completes.
final class MockURLProtocol: URLProtocol {
    struct Stub {
        let status: Int
        let body: Data
        /// When set, the request fails via `didFailWithError` instead of returning
        /// a response — lets tests exercise transport failures (e.g. timeouts).
        var error: URLError? = nil
    }

    /// (request, capturedBody) -> Stub. Set before each test.
    nonisolated(unsafe) static var handler: ((URLRequest, Data) -> Stub)?
    /// Records the last request seen, for assertions on method/path/headers/body.
    nonisolated(unsafe) static var lastRequest: URLRequest?
    nonisolated(unsafe) static var lastBody: Data = Data()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let body = Self.readBody(from: request)
        Self.lastRequest = request
        Self.lastBody = body
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        let stub = handler(request, body)
        if let error = stub.error {
            client?.urlProtocol(self, didFailWithError: error)
            return
        }
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: stub.status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: stub.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    /// URLSession may hand the body over as `httpBody` or as `httpBodyStream`.
    private static func readBody(from request: URLRequest) -> Data {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let size = 4096
        var buffer = [UInt8](repeating: 0, count: size)
        while stream.hasBytesAvailable {
            let read = stream.read(&buffer, maxLength: size)
            if read <= 0 { break }
            data.append(buffer, count: read)
        }
        return data
    }

    static func makeSession() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        return URLSession(configuration: config)
    }
}
