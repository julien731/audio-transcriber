import Foundation

public enum ServiceError: Error, Equatable {
    case readinessTimeout
    case nonceMismatch
    case launchFailed(String)
    case startupError(String)
}

/// Spawns and supervises the bundled service child process (plan Artifact A,
/// BR-2/BR-4/BR-5, OQ-7). The app always launches its own child and holds the
/// handle, so ownership is unambiguous. `start` blocks until the stdout `ready`
/// handshake arrives (or times out); the caller runs it off the main thread.
///
/// Shutdown is signal-based because the service exposes no shutdown endpoint
/// (finding #10): SIGTERM → bounded wait → SIGKILL.
public final class ServiceSupervisor {
    public struct Launch {
        public let executableURL: URL
        public let arguments: [String]
        /// Per-launch coordination nonce, exported as `MT_SERVICE_NONCE`.
        public let nonce: String

        public init(executableURL: URL, arguments: [String], nonce: String) {
            self.executableURL = executableURL
            self.arguments = arguments
            self.nonce = nonce
        }
    }

    private let process = Process()
    private let stateLock = NSLock()
    private var _port: Int?
    private var _handshake: ServiceHandshake?
    private var _lastError: String?

    /// Invoked (off the main thread) if the child exits unexpectedly after startup.
    public var onUnexpectedTermination: ((Int32) -> Void)?

    public init() {}

    public var port: Int? {
        stateLock.lock(); defer { stateLock.unlock() }
        return _port
    }

    public var isRunning: Bool { process.isRunning }

    public var processIdentifier: Int32 { process.processIdentifier }

    /// Launch the child and block until it announces readiness. On timeout the
    /// child is terminated and `readinessTimeout` is thrown.
    @discardableResult
    public func start(_ launch: Launch, timeout: TimeInterval = 30) throws -> Int {
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = launch.executableURL
        process.arguments = launch.arguments
        var environment = ProcessInfo.processInfo.environment
        environment["MT_SERVICE_NONCE"] = launch.nonce
        process.environment = environment
        process.standardOutput = stdout
        process.standardError = stderr

        let ready = DispatchSemaphore(value: 0)
        scanForHandshake(on: stdout.fileHandleForReading, signaling: ready)
        scanForStartupError(on: stderr.fileHandleForReading)

        process.terminationHandler = { [weak self] proc in
            guard let self else { return }
            // Only meaningful as an "unexpected exit" once we were ready.
            if self.port != nil { self.onUnexpectedTermination?(proc.terminationStatus) }
        }

        do {
            try process.run()
        } catch {
            throw ServiceError.launchFailed(String(describing: error))
        }

        if ready.wait(timeout: .now() + timeout) == .timedOut {
            terminate()
            stateLock.lock(); let startupError = _lastError; stateLock.unlock()
            if let startupError { throw ServiceError.startupError(startupError) }
            throw ServiceError.readinessTimeout
        }

        stateLock.lock()
        let handshake = _handshake
        stateLock.unlock()

        guard let handshake else { throw ServiceError.readinessTimeout }
        if !launch.nonce.isEmpty, let nonce = handshake.nonce, nonce != launch.nonce {
            terminate()
            throw ServiceError.nonceMismatch
        }
        stateLock.lock(); _port = handshake.port; stateLock.unlock()
        return handshake.port
    }

    /// SIGTERM → wait up to `gracePeriod` → SIGKILL (finding #10). No-op if the
    /// child already exited (e.g. during a quit-confirmation dialog).
    public func terminate(gracePeriod: TimeInterval = 10) {
        guard process.isRunning else { return }
        process.terminate() // SIGTERM
        let deadline = Date().addingTimeInterval(gracePeriod)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
        }
    }

    // MARK: - stdout/stderr scanning

    private func scanForHandshake(on handle: FileHandle, signaling ready: DispatchSemaphore) {
        let queue = DispatchQueue(label: "com.nimblehq.MeetingTranscriber.service-stdout")
        queue.async { [weak self] in
            var buffer = Data()
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty { break } // EOF
                buffer.append(chunk)
                while let newline = buffer.firstIndex(of: 0x0A) {
                    let lineData = buffer.subdata(in: buffer.startIndex..<newline)
                    buffer.removeSubrange(buffer.startIndex...newline)
                    guard
                        let line = String(data: lineData, encoding: .utf8),
                        let handshake = ServiceHandshake.parse(line: line)
                    else { continue }
                    self?.stateLock.lock()
                    self?._handshake = handshake
                    self?.stateLock.unlock()
                    ready.signal()
                    return
                }
            }
        }
    }

    private func scanForStartupError(on handle: FileHandle) {
        let queue = DispatchQueue(label: "com.nimblehq.MeetingTranscriber.service-stderr")
        queue.async { [weak self] in
            var buffer = Data()
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty { break }
                buffer.append(chunk)
                while let newline = buffer.firstIndex(of: 0x0A) {
                    let lineData = buffer.subdata(in: buffer.startIndex..<newline)
                    buffer.removeSubrange(buffer.startIndex...newline)
                    guard
                        let line = String(data: lineData, encoding: .utf8),
                        let message = ServiceHandshake.parseError(line: line)
                    else { continue }
                    self?.stateLock.lock()
                    self?._lastError = message
                    self?.stateLock.unlock()
                    return
                }
            }
        }
    }
}
