// Story #137 — debugging & observability.
// Plan: docs/plans/137-observability-diagnostics.md

import Foundation
import MeetingTranscriberKit

func runFileLogTests() {
    suite("FileLog") {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("filelog-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // writeLine appends newline-terminated content.
        let log = FileLog(fileName: "app.log", directory: dir)
        log.writeLine("first")
        log.writeLine("second")
        log.flush()
        let contents = (try? String(contentsOf: log.url, encoding: .utf8)) ?? ""
        expect(contents == "first\nsecond\n", "writeLine appends lines, got \(contents.debugDescription)")

        // Raw write tees bytes verbatim.
        let raw = FileLog(fileName: "raw.log", directory: dir)
        raw.write(Data("abc".utf8))
        raw.flush()
        let rawContents = (try? String(contentsOf: raw.url, encoding: .utf8)) ?? ""
        expectEqual(rawContents, "abc", "raw write is verbatim")

        // Rotation: exceeding maxBytes moves the current file to <name>.1.
        let rot = FileLog(fileName: "rot.log", directory: dir, maxBytes: 256)
        for i in 0 ..< 200 { rot.writeLine("padding line \(i) \(String(repeating: "x", count: 40))") }
        rot.flush()
        let backup = URL(fileURLWithPath: rot.url.path + ".1")
        expect(FileManager.default.fileExists(atPath: backup.path), "rotation created rot.log.1")
        let currentSize = ((try? FileManager.default.attributesOfItem(atPath: rot.url.path))?[.size] as? NSNumber)?.intValue ?? -1
        expect(currentSize >= 0 && currentSize <= 256 + 200, "post-rotation file stays bounded, size=\(currentSize)")

        // Concurrent writers from multiple queues: no crash, every line lands.
        let concurrent = FileLog(fileName: "concurrent.log", directory: dir)
        let group = DispatchGroup()
        let writers = DispatchQueue(label: "writers", attributes: .concurrent)
        let count = 500
        for i in 0 ..< count {
            writers.async(group: group) { concurrent.writeLine("line-\(i)") }
        }
        group.wait()
        concurrent.flush()
        let lines = ((try? String(contentsOf: concurrent.url, encoding: .utf8)) ?? "")
            .split(separator: "\n", omittingEmptySubsequences: true)
        expectEqual(lines.count, count, "all concurrent lines written")
    }
}
