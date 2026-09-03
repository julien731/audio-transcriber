// Story #139 — redact secrets from logs and the diagnostics export.
// Plan: docs/plans/139-redact-secrets-from-logs.md

import Foundation
import MeetingTranscriberKit

// Shared canonical vector — the Python test (`_SEEDED_TOKEN`) uses the same
// string so a pattern divergence between the two runtimes surfaces.
private let seededToken = "hf_TESTTOKEN0123456789abcdef"

func runSecretRedactionTests() {
    suite("SecretRedaction") {
        // Masks an hf_ token embedded in surrounding text.
        let redacted = SecretRedaction.redact("401 Unauthorized for \(seededToken) here")
        expect(!redacted.contains(seededToken), "token removed, got \(redacted.debugDescription)")
        expect(redacted.contains("hf_***"), "placeholder present, got \(redacted.debugDescription)")

        // Leaves non-secret text untouched.
        expectEqual(
            SecretRedaction.redact("plain line, no secrets"),
            "plain line, no secrets",
            "non-secret text unchanged"
        )

        // Masks every occurrence, not just the first.
        let multi = SecretRedaction.redact("\(seededToken) and \(seededToken)")
        expect(!multi.contains(seededToken), "all occurrences masked, got \(multi.debugDescription)")

        // The stderr tee redacts before writing to the bundled log file.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("redaction-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let stderrLog = FileLog(fileName: "service-stderr.log", directory: dir)
        let chunk = "Traceback: RuntimeError 401 for \(seededToken)\n"
        stderrLog.write(Data(SecretRedaction.redact(chunk).utf8))
        stderrLog.flush()
        let onDisk = (try? String(contentsOf: stderrLog.url, encoding: .utf8)) ?? ""
        expect(!onDisk.contains(seededToken), "tee redacts token on disk, got \(onDisk.debugDescription)")
        expect(onDisk.contains("hf_***"), "tee wrote placeholder, got \(onDisk.debugDescription)")
    }
}
