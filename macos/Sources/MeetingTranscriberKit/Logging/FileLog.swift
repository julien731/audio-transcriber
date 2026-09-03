// Story #137 — debugging & observability.
// Plan: docs/plans/137-observability-diagnostics.md

import Foundation

/// Thread-safe, size-bounded file logger.
///
/// Appends UTF-8 to a file, rotating the current file to `<name>.1` once it would
/// exceed `maxBytes`. All I/O runs on a private serial queue, so the multiple
/// producers this serves — the app's own logging *and* the service stdout/stderr
/// tee, which run on separate background `DispatchQueue`s — never interleave a
/// line or race the `FileHandle`. Logging must never crash the app, so every I/O
/// failure is swallowed.
public final class FileLog {
    /// `~/Library/Application Support/MeetingTranscriber/Logs` — the single dir the
    /// Python service (`service.log`) and the app (`app.log`, `service-stderr.log`)
    /// both write into and the diagnostics export bundles.
    public static func logsDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let dir = base.appendingPathComponent("MeetingTranscriber/Logs", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private let fileURL: URL
    private let maxBytes: Int
    private let queue: DispatchQueue

    public init(fileName: String,
                directory: URL = FileLog.logsDirectory(),
                maxBytes: Int = 5 * 1024 * 1024) {
        self.fileURL = directory.appendingPathComponent(fileName)
        self.maxBytes = maxBytes
        self.queue = DispatchQueue(label: "com.nimblehq.MeetingTranscriber.filelog.\(fileName)")
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    /// The file this logger writes to (its `.1` backup sits alongside it).
    public var url: URL { fileURL }

    /// Append a line; a trailing newline is added.
    public func writeLine(_ message: String) {
        write(Data((message + "\n").utf8))
    }

    /// Append raw bytes exactly as received (used to tee child stdout/stderr).
    public func write(_ data: Data) {
        guard !data.isEmpty else { return }
        queue.async { [self] in
            rotateIfNeeded(adding: data.count)
            append(data)
        }
    }

    /// Block until all enqueued writes have been flushed. Used by tests and by the
    /// diagnostics export so the zip captures the latest lines.
    public func flush() {
        queue.sync {}
    }

    private func append(_ data: Data) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: fileURL.path) {
            fm.createFile(atPath: fileURL.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: fileURL) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
    }

    private func rotateIfNeeded(adding count: Int) {
        let fm = FileManager.default
        let attrs = try? fm.attributesOfItem(atPath: fileURL.path)
        let size = (attrs?[.size] as? NSNumber)?.intValue ?? 0
        guard size > 0, size + count > maxBytes else { return }
        let rotated = URL(fileURLWithPath: fileURL.path + ".1")
        try? fm.removeItem(at: rotated)
        try? fm.moveItem(at: fileURL, to: rotated)
    }
}
