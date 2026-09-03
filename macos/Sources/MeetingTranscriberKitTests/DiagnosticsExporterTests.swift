// Story #137 — debugging & observability.
// Plan: docs/plans/137-observability-diagnostics.md

import Foundation
import MeetingTranscriberKit

func runDiagnosticsExporterTests() {
    suite("DiagnosticsExporter") {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("diag-tests-\(UUID().uuidString)", isDirectory: true)
        let logs = tmp.appendingPathComponent("Logs", isDirectory: true)
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        try? Data("service log body".utf8).write(to: logs.appendingPathComponent("service.log"))

        // Export produces a non-empty zip at the caller-supplied destination.
        let dest = tmp.appendingPathComponent("diag.zip")
        do {
            try DiagnosticsExporter.exportDiagnostics(to: dest, logsDirectory: logs)
        } catch {
            expect(false, "export threw \(error)")
        }
        expect(FileManager.default.fileExists(atPath: dest.path), "export wrote a zip")
        let size = ((try? FileManager.default.attributesOfItem(atPath: dest.path))?[.size] as? NSNumber)?.intValue ?? 0
        expect(size > 0, "zip is non-empty (size \(size))")

        // Overwrites an existing destination rather than failing.
        do {
            try DiagnosticsExporter.exportDiagnostics(to: dest, logsDirectory: logs)
            expect(true, "second export overwrote destination")
        } catch {
            expect(false, "second export threw \(error)")
        }

        // Summary carries the identifying header.
        expect(DiagnosticsExporter.summary().contains("MeetingTranscriber diagnostics"),
               "summary has header")

        // AC5 (story #139): a token that passed through the redacting write path
        // must be absent from the exported zip. Seed a log the way the stderr tee
        // would write it (raw child output run through SecretRedaction.redact),
        // export, extract, and grep the extracted tree for the raw token.
        let seededToken = "hf_TESTTOKEN0123456789abcdef"
        let teed = SecretRedaction.redact("Traceback: 401 Unauthorized for \(seededToken)\n")
        try? Data(teed.utf8).write(to: logs.appendingPathComponent("service-stderr.log"))

        let secretDest = tmp.appendingPathComponent("diag-secret.zip")
        do {
            try DiagnosticsExporter.exportDiagnostics(to: secretDest, logsDirectory: logs)
        } catch {
            expect(false, "secret export threw \(error)")
        }

        let extracted = tmp.appendingPathComponent("extracted-\(UUID().uuidString)", isDirectory: true)
        let ditto = Process()
        ditto.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        ditto.arguments = ["-x", "-k", secretDest.path, extracted.path]
        try? ditto.run()
        ditto.waitUntilExit()

        var zipBody = ""
        if let files = FileManager.default.enumerator(at: extracted, includingPropertiesForKeys: nil) {
            for case let url as URL in files {
                zipBody += (try? String(contentsOf: url, encoding: .utf8)) ?? ""
            }
        }
        expect(!zipBody.contains(seededToken), "exported zip is free of the seeded token")
        expect(zipBody.contains("hf_***"), "exported zip carries the redacted placeholder")
    }
}
