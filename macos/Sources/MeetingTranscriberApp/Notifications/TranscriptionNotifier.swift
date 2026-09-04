import AppKit
import UserNotifications
import MeetingTranscriberKit

/// Posts local notifications when a transcription settles (chore #144), mirroring
/// the web app's browser notifications. Thin shim over `UNUserNotificationCenter`;
/// the notification copy lives in Kit (`TranscriptionNotification`) where it is
/// unit-tested.
///
/// Every `UNUserNotificationCenter.current()` access is guarded on a real bundle
/// id: the center traps when the process has no main-bundle identifier, so
/// running the App target bare via `swift run MeetingTranscriberApp` would crash.
/// Un-bundled, this notifier is a silent no-op; the shipped .app has a bundle id.
struct UserNotificationNotifier {
    /// Ask the user for permission once. Idempotent — the system only prompts on
    /// the first call. No-op when un-bundled.
    func requestAuthorization() {
        guard Bundle.main.bundleIdentifier != nil else { return }
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, error in
            // Fires on an arbitrary background queue: keep it side-effect-free
            // (no MainActor state) to avoid a data race under the v5 language mode.
            if let error {
                AppLog.error("Notification authorization failed: \(error)")
            }
        }
    }

    /// Deliver `content`, unless the app is frontmost — the web app likewise skips
    /// the notification while its page is visible. No-op when un-bundled.
    func post(_ content: TranscriptionNotification) {
        guard Bundle.main.bundleIdentifier != nil else { return }
        guard !NSApplication.shared.isActive else { return }

        let notification = UNMutableNotificationContent()
        notification.title = content.title
        notification.body = content.body
        notification.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: notification,
            trigger: nil
        )
        UNUserNotificationCenter.current().add(request) { error in
            // Background queue — no MainActor state touched (see requestAuthorization).
            if let error {
                AppLog.error("Failed to post transcription notification: \(error)")
            }
        }
    }
}
