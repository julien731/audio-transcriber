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
    }
}
