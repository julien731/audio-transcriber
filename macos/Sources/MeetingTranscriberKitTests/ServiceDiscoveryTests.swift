import Foundation
import MeetingTranscriberKit

private func makeTempDir() -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("mt-tests-\(UUID().uuidString)", isDirectory: true)
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func write(_ record: ServiceRecord, to url: URL) {
    let data = try! JSONEncoder().encode(record)
    try! data.write(to: url)
}

func runServiceDiscoveryTests() {
    suite("ServiceDiscovery.readRecord") {
        let dir = makeTempDir()
        let url = ServiceDiscovery.serviceFileURL(directory: dir)
        // Matches the service's on-disk shape {"port","pid","nonce"}.
        try! Data(#"{"port":5050,"pid":4242,"nonce":"n1"}"#.utf8).write(to: url)
        let record = ServiceDiscovery.readRecord(at: url)
        expectEqual(record?.port, 5050, "reads port")
        expectEqual(record?.pid, 4242, "reads pid")
        expectEqual(record?.nonce, "n1", "reads nonce")

        // Backward-compat: a record without a nonce still decodes.
        try! Data(#"{"port":1,"pid":2}"#.utf8).write(to: url)
        expectEqual(ServiceDiscovery.readRecord(at: url)?.pid, 2, "nonce-less record decodes")

        expectNil(ServiceDiscovery.readRecord(at: dir.appendingPathComponent("missing.json")),
                  "missing file → nil")
    }

    suite("ServiceDiscovery.isOwned") {
        let rec = ServiceRecord(port: 1, pid: 100, nonce: "abc")
        expect(ServiceDiscovery.isOwned(rec, expectedPID: 100, expectedNonce: "abc"), "pid+nonce match")
        expect(!ServiceDiscovery.isOwned(rec, expectedPID: 999, expectedNonce: "abc"), "pid mismatch rejected")
        expect(!ServiceDiscovery.isOwned(rec, expectedPID: 100, expectedNonce: "other"), "nonce mismatch rejected")
        // When the app set no nonce, pid alone decides (backward-compat).
        expect(ServiceDiscovery.isOwned(rec, expectedPID: 100, expectedNonce: nil), "nil expected nonce → pid only")
        expect(ServiceDiscovery.isOwned(rec, expectedPID: 100, expectedNonce: ""), "empty expected nonce → pid only")
    }

    suite("ServiceDiscovery.removeIfOwned (guarded cleanup)") {
        let dir = makeTempDir()
        let url = ServiceDiscovery.serviceFileURL(directory: dir)

        // Owned → removed.
        write(ServiceRecord(port: 1, pid: 7, nonce: "k"), to: url)
        expect(ServiceDiscovery.removeIfOwned(at: url, expectedPID: 7, expectedNonce: "k"), "removes owned record")
        expect(!FileManager.default.fileExists(atPath: url.path), "file gone")

        // A record overwritten by another process (different pid/nonce) is preserved.
        write(ServiceRecord(port: 2, pid: 8, nonce: "other"), to: url)
        expect(!ServiceDiscovery.removeIfOwned(at: url, expectedPID: 7, expectedNonce: "k"),
               "does not remove a foreign record")
        expect(FileManager.default.fileExists(atPath: url.path), "foreign file preserved")
    }
}
