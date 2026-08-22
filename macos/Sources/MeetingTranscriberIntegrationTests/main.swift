import Foundation
import MeetingTranscriberKit

// Integration suite (plan TD-2, test tier 2): drives the REAL stub child process
// and a live local HTTP server end to end. Runs in this environment via
// `swift run MeetingTranscriberIntegrationTests` from the package root.
//
// This is the D1 tracer-bullet assertion in code form: spawn the service, parse
// its stdout handshake, confirm /api/health is reachable, then SIGTERM it and
// confirm the process is gone and the guarded service.json cleanup fires.

var passed = 0
var failed = 0

func check(_ condition: Bool, _ message: String) {
    if condition {
        passed += 1
    } else {
        failed += 1
        FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8))
    }
}

let cwd = FileManager.default.currentDirectoryPath
let scriptPath = "\(cwd)/scripts/stub_service.py"
guard FileManager.default.fileExists(atPath: scriptPath) else {
    FileHandle.standardError.write(Data("stub service not found at \(scriptPath); run from macos/ root\n".utf8))
    exit(2)
}

let supervisor = ServiceSupervisor()
let nonce = UUID().uuidString
let launch = ServiceSupervisor.Launch(
    executableURL: URL(fileURLWithPath: "/usr/bin/env"),
    arguments: ["python3", scriptPath],
    nonce: nonce
)

do {
    let port = try supervisor.start(launch, timeout: 20)
    check(port > 0, "handshake yielded a port (\(port))")
    check(supervisor.isRunning, "child process is running")

    let baseURL = URL(string: "http://127.0.0.1:\(port)")!
    let ready = await HealthClient.waitUntilReady(baseURL: baseURL, timeout: 15)
    check(ready, "GET /api/health is reachable")

    // The stub echoes our nonce into service.json; discovery must recognize ownership.
    let record = ServiceDiscovery.readRecord(at: ServiceDiscovery.serviceFileURL())
    check(record?.nonce == nonce, "service.json carries our nonce")
    check(record.map { ServiceDiscovery.isOwned($0, expectedPID: supervisor.processIdentifier, expectedNonce: nonce) } ?? false,
          "discovery recognizes the spawned child as owned")

    let pid = supervisor.processIdentifier
    supervisor.terminate(gracePeriod: 5)
    check(!supervisor.isRunning, "child terminated on SIGTERM")

    let removed = ServiceDiscovery.removeIfOwned(
        at: ServiceDiscovery.serviceFileURL(),
        expectedPID: pid,
        expectedNonce: nonce
    )
    check(removed, "guarded cleanup removed our service.json")
} catch {
    check(false, "supervisor.start threw: \(error)")
}

let summary = "\n\(passed) passed, \(failed) failed\n"
FileHandle.standardOutput.write(Data(summary.utf8))
exit(failed == 0 ? 0 : 1)
