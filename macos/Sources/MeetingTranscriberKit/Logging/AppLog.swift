// Story #137 — debugging & observability.
// Plan: docs/plans/137-observability-diagnostics.md

import Foundation
import os

/// App-side lifecycle logging. Each event is mirrored to the unified log
/// (`os.Logger`, visible in Console.app) *and* appended to a rotating `app.log`
/// file: os_log alone is awkward to export off an ad-hoc-signed build, and the
/// file survives app relaunch so a hang that needed a restart is still captured.
///
/// Messages are logged `.public` on purpose — these are lifecycle breadcrumbs
/// (launch, service start/ready/failure, quit), never secrets. See the story's
/// no-secrets truth before adding new call sites.
public enum AppLog {
    private static let logger = Logger(subsystem: "com.nimblehq.MeetingTranscriber", category: "app")
    private static let file = FileLog(fileName: "app.log")

    public static func info(_ message: String) {
        logger.info("\(message, privacy: .public)")
        file.writeLine(line("INFO", message))
    }

    public static func error(_ message: String) {
        logger.error("\(message, privacy: .public)")
        file.writeLine(line("ERROR", message))
    }

    /// Flush pending file writes (called before a diagnostics export).
    public static func flush() {
        file.flush()
    }

    private static func line(_ level: String, _ message: String) -> String {
        // app.log is bundled into the diagnostics export; redact as
        // defense-in-depth even though these are Swift-authored breadcrumbs that
        // never interpolate tokens today (story #139).
        let safe = SecretRedaction.redact(message)
        return "\(ISO8601DateFormatter().string(from: Date())) \(level) app: \(safe)"
    }
}
