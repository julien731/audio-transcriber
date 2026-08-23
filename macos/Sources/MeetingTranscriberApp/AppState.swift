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

    /// The live service client, available only once the app is `.ready` (BR-2).
    /// The Settings screen uses this to gate the token form; `.setup` deliberately
    /// does not qualify — the setup wizard owns the token during first run.
    var client: APIClient? {
        if case let .ready(client) = phase { return client }
        return nil
    }

    private var supervisor = ServiceSupervisor()
    private let nonce = UUID().uuidString
    /// Bumped on every start()/restart(). An in-flight launch only mutates state
    /// or keeps its child if it is still the current generation — so a superseded
    /// launch can't orphan a process or overwrite the fresh app state.
    private var launchGeneration = 0

    // MARK: Launch

    func start() {
        launchGeneration += 1
        let generation = launchGeneration

        // Tear down the previous supervisor before replacing it (a `Process` can't
        // be re-run). Detach its crash callback so a late exit can't overwrite
        // fresh state, and terminate its child off the main thread so it can't
        // orphan and interfere with the replacement.
        let previous = supervisor
        previous.onUnexpectedTermination = nil
        supervisor = ServiceSupervisor()
        if previous.isRunning { Task.detached { previous.terminate() } }

        phase = .starting
        supervisor.onUnexpectedTermination = { [weak self] status in
            Task { @MainActor in
                guard let self, self.launchGeneration == generation else { return }
                self.phase = .serviceFailed("The transcription service exited unexpectedly (status \(status)). Restart to continue.")
            }
        }
        guard let launch = resolveLaunch() else {
            phase = .serviceFailed("Could not locate the bundled transcription service.")
            return
        }
        Task { [supervisor] in
            do {
                let port = try await Task.detached { try supervisor.start(launch) }.value
                // A newer launch superseded this one: kill this child, touch nothing.
                guard self.launchGeneration == generation else {
                    await Task.detached { supervisor.terminate() }.value
                    return
                }
                let baseURL = URL(string: "http://127.0.0.1:\(port)")!
                // The handshake arrives in ~2s, but the HTTP server only comes up
                // after the service loads its full native ML dependency tree. On a
                // cold first launch, macOS validates hundreds of dylibs, which can
                // take a couple of minutes — so allow a generous readiness window
                // rather than declaring failure. Subsequent launches are fast.
                guard await HealthClient.waitUntilReady(baseURL: baseURL, timeout: 240) else {
                    // The child never served HTTP — terminate it so it can't linger.
                    await Task.detached { supervisor.terminate() }.value
                    if self.launchGeneration == generation {
                        self.phase = .serviceFailed("The service started but never became ready.")
                    }
                    return
                }
                guard self.launchGeneration == generation else {
                    await Task.detached { supervisor.terminate() }.value
                    return
                }
                await self.routeAfterReady(baseURL: baseURL, generation: generation)
            } catch {
                // Kill any child left by a failed start (no-op if already gone).
                await Task.detached { supervisor.terminate() }.value
                if self.launchGeneration == generation {
                    self.phase = .serviceFailed(Self.describe(error))
                }
            }
        }
    }

    /// After the service is reachable, decide setup vs main from provisioning.
    private func routeAfterReady(baseURL: URL, generation: Int) async {
        let client = APIClient(baseURL: baseURL)
        let resolved: Phase
        do {
            let status = try await client.provisioning()
            resolved = status.provisioningCompleted
                ? .ready(client)
                : .setup(ProvisioningController(client: client, initial: status), client)
        } catch {
            // Provisioning is unreachable but health passed — still let the user in
            // to browse existing meetings (offline-first, EC-7); new transcription
            // is gated server-side.
            resolved = .ready(client)
        }
        guard launchGeneration == generation else { return }
        phase = resolved
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
        // Deliberate shutdown: silence the crash handler so it can't flip the
        // phase to .serviceFailed for an intentional termination.
        supervisor.onUnexpectedTermination = nil
        supervisor.terminate()
        ServiceDiscovery.removeIfOwned(
            at: ServiceDiscovery.serviceFileURL(),
            expectedPID: pid,
            expectedNonce: nonce
        )
    }

    // MARK: Service command resolution

    /// Resolve the service to launch, preferring the real embedded PyInstaller
    /// service (D0) and falling back to the dev stub (D1). `MT_STUB_SERVICE_PATH`
    /// forces the stub for `swift run`/dev outside a .app bundle.
    private func resolveLaunch() -> ServiceSupervisor.Launch? {
        let fm = FileManager.default

        // 1) Explicit dev override → stub via python3.
        if let override = ProcessInfo.processInfo.environment["MT_STUB_SERVICE_PATH"] {
            return ServiceSupervisor.Launch(
                executableURL: URL(fileURLWithPath: "/usr/bin/env"),
                arguments: ["python3", override], nonce: nonce)
        }

        guard let resources = Bundle.main.resourceURL else { return nil }

        // 2) Real embedded self-contained service (D0): a native arm64 executable.
        let real = resources.appendingPathComponent("service/MeetingTranscriber/MeetingTranscriber")
        if fm.isExecutableFile(atPath: real.path) {
            return ServiceSupervisor.Launch(executableURL: real, arguments: [], nonce: nonce)
        }

        // 3) Dev stub (D1): python3 script.
        let stub = resources.appendingPathComponent("service/stub_service.py")
        if fm.fileExists(atPath: stub.path) {
            return ServiceSupervisor.Launch(
                executableURL: URL(fileURLWithPath: "/usr/bin/env"),
                arguments: ["python3", stub.path], nonce: nonce)
        }
        return nil
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
