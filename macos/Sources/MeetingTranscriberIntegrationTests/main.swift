import Foundation
import MeetingTranscriberKit

// Integration suite (plan TD-2, test tier 2): drives REAL child processes and a
// live local HTTP server end to end. Runs via
// `swift run MeetingTranscriberIntegrationTests` from the package root.

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

func python(_ code: String, nonce: String) -> ServiceSupervisor.Launch {
    ServiceSupervisor.Launch(
        executableURL: URL(fileURLWithPath: "/usr/bin/env"),
        arguments: ["python3", "-c", code],
        nonce: nonce
    )
}

// MARK: - Scenario 1: full stub service (handshake + health + guarded cleanup)

func scenarioStubService() async {
    let cwd = FileManager.default.currentDirectoryPath
    let scriptPath = "\(cwd)/scripts/stub_service.py"
    guard FileManager.default.fileExists(atPath: scriptPath) else {
        FileHandle.standardError.write(Data("stub not found at \(scriptPath); run from macos/ root\n".utf8))
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
        check(port > 0, "stub: handshake yielded a port (\(port))")
        check(supervisor.isRunning, "stub: child is running")

        let baseURL = URL(string: "http://127.0.0.1:\(port)")!
        let ready = await HealthClient.waitUntilReady(baseURL: baseURL, timeout: 15)
        check(ready, "stub: /api/health reachable")

        let record = ServiceDiscovery.readRecord(at: ServiceDiscovery.serviceFileURL())
        check(record?.nonce == nonce, "stub: service.json carries our nonce")
        check(record.map { ServiceDiscovery.isOwned($0, expectedPID: supervisor.processIdentifier, expectedNonce: nonce) } ?? false,
              "stub: discovery recognizes the spawned child")

        let pid = supervisor.processIdentifier
        supervisor.terminate(gracePeriod: 5)
        check(!supervisor.isRunning, "stub: child terminated on SIGTERM")
        check(ServiceDiscovery.removeIfOwned(at: ServiceDiscovery.serviceFileURL(),
                                             expectedPID: pid, expectedNonce: nonce),
              "stub: guarded cleanup removed our service.json")
    } catch {
        check(false, "stub: supervisor.start threw: \(error)")
    }
}

// MARK: - Scenario 2: noisy/mixed stdout before the handshake

func scenarioNoisyStdout() {
    let nonce = UUID().uuidString
    let code = """
    import json, os, time
    print("INFO: booting uvicorn")
    print("  a partial log line")
    print(json.dumps({"event": "ready", "port": 45678, "nonce": os.environ.get("MT_SERVICE_NONCE", "")}), flush=True)
    time.sleep(30)
    """
    let supervisor = ServiceSupervisor()
    do {
        let port = try supervisor.start(python(code, nonce: nonce), timeout: 10)
        check(port == 45678, "noisy: handshake parsed past log noise (got \(port))")
    } catch {
        check(false, "noisy: start threw \(error)")
    }
    supervisor.terminate(gracePeriod: 2)
    check(!supervisor.isRunning, "noisy: terminated")
}

// MARK: - Scenario 3: SIGTERM-ignoring child forces SIGKILL escalation

func scenarioSigkillEscalation() {
    let nonce = UUID().uuidString
    let code = """
    import json, os, signal, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(json.dumps({"event": "ready", "port": 1, "nonce": os.environ.get("MT_SERVICE_NONCE", "")}), flush=True)
    time.sleep(60)
    """
    let supervisor = ServiceSupervisor()
    do {
        _ = try supervisor.start(python(code, nonce: nonce), timeout: 10)
        check(supervisor.isRunning, "sigkill: child running before terminate")
        let start = Date()
        supervisor.terminate(gracePeriod: 0.3) // SIGTERM ignored → escalates to SIGKILL
        check(!supervisor.isRunning, "sigkill: SIGTERM-ignoring child killed via SIGKILL")
        check(Date().timeIntervalSince(start) >= 0.3, "sigkill: waited the grace period before escalating")
    } catch {
        check(false, "sigkill: start threw \(error)")
    }
}

// MARK: - Scenario 4: nonce mismatch is rejected

func scenarioNonceMismatch() {
    // Child echoes a DIFFERENT nonce than the one the app set → start must reject.
    let code = """
    import json, time
    print(json.dumps({"event": "ready", "port": 2, "nonce": "not-ours"}), flush=True)
    time.sleep(30)
    """
    let supervisor = ServiceSupervisor()
    do {
        _ = try supervisor.start(python(code, nonce: "ours"), timeout: 10)
        check(false, "nonce: expected a mismatch error")
    } catch let error as ServiceError {
        check(error == .nonceMismatch, "nonce: mismatch rejected")
    } catch {
        check(false, "nonce: wrong error \(error)")
    }
    check(!supervisor.isRunning, "nonce: child terminated after rejection")
}

// MARK: - Scenario 5: post-handshake output flood must not deadlock (story #137)

func scenarioPostHandshakeFloodDoesNotBlock() {
    // Regression for the pipe-drain deadlock: after the handshake the child writes
    // far more than the ~64KB OS pipe buffer to BOTH stderr and stdout, then
    // touches a sentinel file. If the supervisor stopped draining (the old bug),
    // the child blocks on write() and never reaches the sentinel.
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent("supervisor-flood-\(UUID().uuidString)", isDirectory: true)
    try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }
    let sentinel = tmp.appendingPathComponent("done.txt")

    let nonce = UUID().uuidString
    let code = """
    import json, os, sys, time
    print(json.dumps({"event": "ready", "port": 45999, "nonce": os.environ.get("MT_SERVICE_NONCE", "")}), flush=True)
    big = "x" * (200 * 1024)  # 200 KB — well over the 64KB pipe buffer
    sys.stderr.write(big); sys.stderr.flush()
    sys.stdout.write(big); sys.stdout.flush()
    with open(r"\(sentinel.path)", "w") as fh:
        fh.write("done")
    time.sleep(30)
    """

    let stderrLog = FileLog(fileName: "service-stderr.log", directory: tmp)
    let supervisor = ServiceSupervisor(stderrLog: stderrLog)
    do {
        let port = try supervisor.start(python(code, nonce: nonce), timeout: 10)
        check(port == 45999, "flood: handshake parsed (got \(port))")

        // The child reaches the sentinel only if both pipes kept draining.
        let deadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: sentinel.path), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        check(FileManager.default.fileExists(atPath: sentinel.path),
              "flood: child finished writing >64KB to both pipes without blocking")

        stderrLog.flush()
        let size = ((try? FileManager.default.attributesOfItem(atPath: stderrLog.url.path))?[.size] as? NSNumber)?.intValue ?? 0
        check(size >= 200 * 1024, "flood: child output teed to service-stderr.log (size \(size))")
    } catch {
        check(false, "flood: start threw \(error)")
    }
    supervisor.terminate(gracePeriod: 2)
    check(!supervisor.isRunning, "flood: child terminated")
}

await scenarioStubService()
scenarioNoisyStdout()
scenarioSigkillEscalation()
scenarioNonceMismatch()
scenarioPostHandshakeFloodDoesNotBlock()

let summary = "\n\(passed) passed, \(failed) failed\n"
FileHandle.standardOutput.write(Data(summary.utf8))
exit(failed == 0 ? 0 : 1)
