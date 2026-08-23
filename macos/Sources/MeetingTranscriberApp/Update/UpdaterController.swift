import Foundation
import Sparkle

/// Tracks whether a transcription is in progress, so the updater can defer a
/// disruptive install until the machine is idle (plan Artifact C:
/// "don't update mid-transcription"). Updated by the meeting list on each load.
final class BusyState {
    static let shared = BusyState()
    private(set) var active = false
    /// Fired on a busy→idle transition (all transcriptions settled) so a deferred
    /// update can be applied. Registered by UpdaterController.
    var onBecameIdle: (() -> Void)?

    /// Update the busy flag and notify on the busy→idle edge.
    func set(active newValue: Bool) {
        let wasActive = active
        active = newValue
        if wasActive && !newValue { onBecameIdle?() }
    }
}

/// Wraps Sparkle's standard updater (plan Artifact C / slice 10). The feed URL and
/// EdDSA public key come from Info.plist (`SUFeedURL`, `SUPublicEDKey`); the app is
/// ad-hoc signed, so the EdDSA enclosure signature is the trust anchor.
///
/// This code compiles here but is only meaningfully exercised on a packaged,
/// signed build — the real v1→v2 install is proven at Milestone D0.
final class UpdaterController: NSObject, SPUUpdaterDelegate {
    private var controller: SPUStandardUpdaterController?
    private var pendingInstall: (() -> Void)?

    /// Whether auto-update is active. False when `SUPublicEDKey` isn't a real key
    /// yet (local/dev builds before D0 injects it) — in that state we do NOT start
    /// Sparkle, so an unconfigured updater can't brick the app at launch.
    private(set) var isEnabled = false

    override init() {
        super.init()
        // Resume a deferred install once transcriptions settle (busy→idle edge).
        BusyState.shared.onBecameIdle = { [weak self] in
            DispatchQueue.main.async { self?.applyPendingInstallIfIdle() }
        }
        guard Self.hasValidPublicKey else {
            NSLog("Sparkle: SUPublicEDKey not configured; auto-update disabled for this build.")
            return
        }
        // Construct without auto-starting so we can catch a start failure instead
        // of Sparkle showing its "updater failed to start" alert.
        let controller = SPUStandardUpdaterController(startingUpdater: false,
                                                      updaterDelegate: self,
                                                      userDriverDelegate: nil)
        do {
            try controller.updater.start()
            self.controller = controller
            isEnabled = true
        } catch {
            NSLog("Sparkle updater did not start: \(error.localizedDescription)")
        }
    }

    /// A real EdDSA public key is present (not absent, empty, or the placeholder).
    static var hasValidPublicKey: Bool {
        guard let key = Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") as? String else { return false }
        return !key.isEmpty && !key.hasPrefix("REPLACE_WITH")
    }

    func checkForUpdates() {
        controller?.checkForUpdates(nil)
    }

    // MARK: SPUUpdaterDelegate

    /// Postpone the update's relaunch/install while a transcription is running;
    /// the stored handler is invoked once the machine goes idle.
    func updater(_ updater: SPUUpdater,
                 shouldPostponeRelaunchForUpdate item: SUAppcastItem,
                 untilInvokingBlock installHandler: @escaping () -> Void) -> Bool {
        guard BusyState.shared.active else { return false }
        pendingInstall = installHandler
        return true
    }

    /// Call when transcriptions settle to apply a deferred install.
    func applyPendingInstallIfIdle() {
        guard !BusyState.shared.active, let handler = pendingInstall else { return }
        pendingInstall = nil
        handler()
    }
}
