// Story #137 — debugging & observability.
// Plan: docs/plans/137-observability-diagnostics.md

import Foundation

/// Bundles the logs the user can share for investigation into a single zip.
///
/// Pure, UI-independent logic (Kit stays free of AppKit/SwiftUI): the caller —
/// the App layer, via an NSSavePanel — supplies the destination URL. Staging a
/// copy first keeps the archive self-consistent even while the app keeps logging.
/// Relies on `/usr/bin/ditto`; the app is not sandboxed, so shelling out and
/// reading Application Support are permitted.
public enum DiagnosticsExporter {
    public enum DiagnosticsError: Error, Equatable {
        case zipFailed(Int)
    }

    /// Stage `logsDirectory` plus a `diagnostics.txt` summary and write a zip to
    /// `destination` (overwriting any existing file there).
    public static func exportDiagnostics(
        to destination: URL,
        logsDirectory: URL = FileLog.logsDirectory()
    ) throws {
        let fm = FileManager.default
        let staging = fm.temporaryDirectory
            .appendingPathComponent("MeetingTranscriber-Diagnostics-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }

        if let items = try? fm.contentsOfDirectory(at: logsDirectory, includingPropertiesForKeys: nil) {
            for item in items {
                try? fm.copyItem(at: item, to: staging.appendingPathComponent(item.lastPathComponent))
            }
        }
        try Data(summary().utf8).write(to: staging.appendingPathComponent("diagnostics.txt"))

        try zip(source: staging, destination: destination)
    }

    /// Human-readable environment summary embedded in the export.
    public static func summary() -> String {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "unknown"
        let build = info?["CFBundleVersion"] as? String ?? "unknown"
        return """
        MeetingTranscriber diagnostics
        Generated: \(ISO8601DateFormatter().string(from: Date()))
        App version: \(version) (build \(build))
        macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)
        Logs directory: \(FileLog.logsDirectory().path)
        """
    }

    private static func zip(source: URL, destination: URL) throws {
        try? FileManager.default.removeItem(at: destination)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-c", "-k", "--sequesterRsrc", source.path, destination.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw DiagnosticsError.zipFailed(Int(process.terminationStatus))
        }
    }
}
