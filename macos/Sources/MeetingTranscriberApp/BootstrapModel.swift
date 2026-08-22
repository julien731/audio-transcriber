import Foundation
import MeetingTranscriberKit

/// Drives the launch sequence for the D1 tracer bullet: spawn the (stub) service,
/// parse its handshake, poll `/api/health`, and publish a state the UI renders.
/// The real product bootstrap (setup wizard, provisioning) arrives in later
/// slices; this proves child-process plumbing + lifecycle only (plan Milestone D1).
@MainActor
final class BootstrapModel: ObservableObject {
    enum State: Equatable {
        case starting
        case ready(port: Int)
        case failed(String)
    }

    @Published private(set) var state: State = .starting

    private let supervisor = ServiceSupervisor()
    private let nonce = UUID().uuidString

    /// Resolve the service command. D1 launches the bundled stub Python service;
    /// `MT_STUB_SERVICE_PATH` overrides for `swift run`/dev outside a .app bundle.
    private func resolveLaunch() -> ServiceSupervisor.Launch? {
        let envOverride = ProcessInfo.processInfo.environment["MT_STUB_SERVICE_PATH"]
        let scriptPath: String?
        if let envOverride {
            scriptPath = envOverride
        } else if let resources = Bundle.main.resourceURL {
            scriptPath = resources.appendingPathComponent("service/stub_service.py").path
        } else {
            scriptPath = nil
        }
        guard let scriptPath else { return nil }
        return ServiceSupervisor.Launch(
            executableURL: URL(fileURLWithPath: "/usr/bin/env"),
            arguments: ["python3", scriptPath],
            nonce: nonce
        )
    }

    func start() {
        state = .starting
        supervisor.onUnexpectedTermination = { [weak self] status in
            Task { @MainActor in
                self?.state = .failed("The transcription service exited unexpectedly (status \(status)).")
            }
        }
        guard let launch = resolveLaunch() else {
            state = .failed("Could not locate the bundled transcription service.")
            return
        }
        Task.detached { [supervisor, nonce] in
            do {
                let port = try supervisor.start(launch)
                let baseURL = URL(string: "http://127.0.0.1:\(port)")!
                let ready = await HealthClient.waitUntilReady(baseURL: baseURL, timeout: 30)
                await MainActor.run {
                    self.state = ready ? .ready(port: port) : .failed("The service started but never became ready.")
                }
                _ = nonce
            } catch {
                await MainActor.run {
                    self.state = .failed(Self.describe(error))
                }
            }
        }
    }

    func restart() { start() }

    /// Guarded shutdown used on quit: SIGTERM→SIGKILL, then delete `service.json`
    /// only if it still identifies our child (finding #5).
    func shutdown() {
        let pid = supervisor.processIdentifier
        supervisor.terminate()
        ServiceDiscovery.removeIfOwned(
            at: ServiceDiscovery.serviceFileURL(),
            expectedPID: pid,
            expectedNonce: nonce
        )
    }

    private static func describe(_ error: Error) -> String {
        switch error {
        case ServiceError.readinessTimeout:
            return "The transcription service did not start in time."
        case ServiceError.nonceMismatch:
            return "A conflicting transcription service is already running."
        case let ServiceError.startupError(message):
            return message
        case let ServiceError.launchFailed(message):
            return "Could not launch the transcription service: \(message)"
        default:
            return String(describing: error)
        }
    }
}
