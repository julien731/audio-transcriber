import AppKit
import MeetingTranscriberKit

/// App lifecycle for the background-run model (plan TD-5/TD-6):
/// - closing the window does NOT quit the app (a transcription may be running);
/// - reopening focuses/recreates the window (single-instance behavior, EC-9);
/// - on quit, the embedded service is shut down cleanly (BR-5).
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let appState = AppState()
    let updater = UpdaterController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        AppLog.info("Application launched")
        appState.start()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            for window in sender.windows { window.makeKeyAndOrderFront(self) }
        }
        return true
    }

    /// Confirm before quitting if a transcription is active (BR-16, EC-5). The
    /// authoritative signal is the service's meeting list, so we check it async
    /// and reply via `terminateLater`.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard case let .ready(client) = appState.phase else { return .terminateNow }
        Task { @MainActor in
            let meetings = (try? await client.listMeetings()) ?? []
            guard QuitPolicy.needsConfirmation(meetings: meetings) else {
                sender.reply(toApplicationShouldTerminate: true)
                return
            }
            let alert = NSAlert()
            alert.messageText = QuitPolicy.confirmationTitle
            alert.informativeText = QuitPolicy.confirmationMessage
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Quit Anyway")
            alert.addButton(withTitle: "Cancel")
            let quit = alert.runModal() == .alertFirstButtonReturn
            AppLog.info("Quit requested with active transcription; user chose to \(quit ? "quit" : "cancel")")
            sender.reply(toApplicationShouldTerminate: quit)
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        appState.shutdown()
    }
}
