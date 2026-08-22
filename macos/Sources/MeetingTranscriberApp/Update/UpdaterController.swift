import Foundation
import Sparkle

/// Tracks whether a transcription is in progress, so the updater can defer a
/// disruptive install until the machine is idle (plan Artifact C:
/// "don't update mid-transcription"). Updated by the meeting list on each load.
final class BusyState {
    static let shared = BusyState()
    var active = false
}

/// Wraps Sparkle's standard updater (plan Artifact C / slice 10). The feed URL and
/// EdDSA public key come from Info.plist (`SUFeedURL`, `SUPublicEDKey`); the app is
/// ad-hoc signed, so the EdDSA enclosure signature is the trust anchor.
///
/// This code compiles here but is only meaningfully exercised on a packaged,
/// signed build — the real v1→v2 install is proven at Milestone D0.
final class UpdaterController: NSObject, SPUUpdaterDelegate {
    private var controller: SPUStandardUpdaterController!
    private var pendingInstall: (() -> Void)?

    override init() {
        super.init()
        // The delegate is supplied at construction; `startingUpdater: true`
        // auto-starts the background updater.
        controller = SPUStandardUpdaterController(startingUpdater: true,
                                                  updaterDelegate: self,
                                                  userDriverDelegate: nil)
    }

    func checkForUpdates() {
        controller.checkForUpdates(nil)
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
