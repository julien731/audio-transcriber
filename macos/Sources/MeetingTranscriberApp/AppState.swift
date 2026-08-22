import Foundation
import MeetingTranscriberKit

/// Top-level app phase and lifecycle owner. Spawns and supervises the embedded
/// service (Artifact A), then routes to first-run setup or the main UI based on
/// the service's provisioning state (BR-2, BR-12/14).
@MainActor
final class AppState: ObservableObject {
    enum Phase {
        case starting
        case serviceFailed(String)
        case setup(ProvisioningController, APIClient)
        case ready(APIClient)
    }

    @Published private(set) var phase: Phase = .starting

    private var supervisor = ServiceSupervisor()
    private let nonce = UUID().uuidString

    // MARK: Launch

    func start() {
        phase = .starting
        // A `Process` cannot be re-run, so `restart()` needs a fresh supervisor;
        // reusing the prior one makes every relaunch fail at `process.run()`.
        supervisor = ServiceSupervisor()
        supervisor.onUnexpectedTermination = { [weak self] status in
            Task { @MainActor in
                self?.phase = .serviceFailed("The transcription service exited unexpectedly (status \(status)). Restart to continue.")
            }
        }
        guard let launch = resolveLaunch() else {
            phase = .serviceFailed("Could not locate the bundled transcription service.")
            return
        }
        Task { [supervisor] in
            do {
                let port = try await Task.detached { try supervisor.start(launch) }.value
                let baseURL = URL(string: "http://127.0.0.1:\(port)")!
                guard await HealthClient.waitUntilReady(baseURL: baseURL, timeout: 30) else {
                    self.phase = .serviceFailed("The service started but never became ready.")
                    return
                }
                await self.routeAfterReady(baseURL: baseURL)
            } catch {
                self.phase = .serviceFailed(Self.describe(error))
            }
        }
    }

    /// After the service is reachable, decide setup vs main from provisioning.
    private func routeAfterReady(baseURL: URL) async {
        let client = APIClient(baseURL: baseURL)
        do {
            let status = try await client.provisioning()
            if status.provisioningCompleted {
                phase = .ready(client)
            } else {
                phase = .setup(ProvisioningController(client: client, initial: status), client)
            }
        } catch {
            // Provisioning is unreachable but health passed — still let the user in
            // to browse existing meetings (offline-first, EC-7); new transcription
            // is gated server-side.
            phase = .ready(client)
        }
    }

    /// Called by the setup wizard when provisioning completes.
    func finishSetup(client: APIClient) {
        phase = .ready(client)
    }

    func restart() { start() }

    /// Guarded shutdown on quit: SIGTERM→SIGKILL then remove our service.json only
    /// if it still identifies our child (finding #5).
    func shutdown() {
        let pid = supervisor.processIdentifier
        supervisor.terminate()
        ServiceDiscovery.removeIfOwned(
            at: ServiceDiscovery.serviceFileURL(),
            expectedPID: pid,
            expectedNonce: nonce
        )
    }

    // MARK: Service command resolution

    /// D1: launch the bundled stub Python service; `MT_STUB_SERVICE_PATH` overrides
    /// for `swift run`/dev outside a .app. Milestone D0 swaps in the real native
    /// service executable under Contents/Resources/service/.
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
