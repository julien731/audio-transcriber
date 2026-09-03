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

    /// Optional sink for the child's raw stdout/stderr (story #137). When set, the
    /// readers tee the child's output here so native crashes and post-handshake
    /// logs that never reach Python's own log file are still captured.
    private let stderrLog: FileLog?

    /// Invoked (off the main thread) if the child exits unexpectedly after startup.
    public var onUnexpectedTermination: ((Int32) -> Void)?

    public init(stderrLog: FileLog? = nil) {
        self.stderrLog = stderrLog
    }

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

        // Read BOTH pipes to EOF for the full process lifetime. Stopping after the
        // handshake/first-error line (the previous behavior) left the pipes
        // undrained: once the ~64KB OS buffer filled with uvicorn logs or a
        // traceback, the child blocked on write() — hanging transcription and
        // resisting SIGTERM on quit (story #137). Parsing stops after the first
        // match (`parse-once`) so later lines can't overwrite the handshake/error.
        drainAndTee(
            stdout.fileHandleForReading,
            label: "com.nimblehq.MeetingTranscriber.service-stdout",
            // The handshake JSON is on stdout; only tee the post-handshake noise.
            teeBeforeMatch: false
        ) { [weak self] line in
            guard let self, let handshake = ServiceHandshake.parse(line: line) else { return false }
            self.stateLock.lock(); self._handshake = handshake; self.stateLock.unlock()
            ready.signal()
            return true
        }
        drainAndTee(
            stderr.fileHandleForReading,
            label: "com.nimblehq.MeetingTranscriber.service-stderr",
            // Capture stderr from the very first byte — startup failures live here.
            teeBeforeMatch: true
        ) { [weak self] line in
            guard let self, let message = ServiceHandshake.parseError(line: line) else { return false }
            self.stateLock.lock(); self._lastError = message; self.stateLock.unlock()
            return true
        }

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

    /// Continuously read `handle` to EOF, teeing raw output to `stderrLog` and
    /// parsing lines until `parse` returns true (the handshake / startup error).
    /// After that first match, parsing stops but draining continues so the pipe
    /// never fills and blocks the child. `teeBeforeMatch` controls whether output
    /// received before the match is teed (stderr: yes; stdout: only post-match, to
    /// skip the handshake JSON line).
    private func drainAndTee(
        _ handle: FileHandle,
        label: String,
        teeBeforeMatch: Bool,
        parse: @escaping (String) -> Bool
    ) {
        let queue = DispatchQueue(label: label)
        queue.async { [weak self] in
            var buffer = Data()
            var matched = false
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty { break } // EOF
                // Redact secrets before teeing: this raw child output bypasses
                // Python's logging redaction, and the file is bundled into the
                // shareable diagnostics zip (story #139). This single write site
                // serves both the stderr tee and the post-handshake stdout tee.
                // Per-chunk redaction accepts one boundary limitation: a token
                // split across two availableData reads could evade masking.
                if matched || teeBeforeMatch {
                    let redacted = SecretRedaction.redact(
                        String(decoding: chunk, as: UTF8.self)
                    )
                    self?.stderrLog?.write(Data(redacted.utf8))
                }
                if matched { continue } // keep draining + teeing, stop parsing
                buffer.append(chunk)
                while let newline = buffer.firstIndex(of: 0x0A) {
                    let lineData = buffer.subdata(in: buffer.startIndex..<newline)
                    buffer.removeSubrange(buffer.startIndex...newline)
                    guard let line = String(data: lineData, encoding: .utf8) else { continue }
                    if parse(line) { matched = true; break }
                }
            }
        }
    }
}
