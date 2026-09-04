import AppKit
import MeetingTranscriberKit
import UniformTypeIdentifiers

/// App lifecycle for the background-run model (plan TD-5/TD-6):
/// - closing the window does NOT quit the app (a transcription may be running);
/// - reopening focuses/recreates the window (single-instance behavior, EC-9);
/// - on quit, the embedded service is shut down cleanly (BR-5).
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let appState = AppState()
    let updater = UpdaterController()
    private let notifier = UserNotificationNotifier()

    func applicationDidFinishLaunching(_ notification: Notification) {
        AppLog.info("Application launched")
        // Prompt once so completion notifications (chore #144) can be delivered.
        notifier.requestAuthorization()
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

    // MARK: - Diagnostics (Help menu, story #137)

    /// Open the Logs directory in Finder so the user can inspect or drag the files.
    func revealLogs() {
        NSWorkspace.shared.activateFileViewerSelecting([FileLog.logsDirectory()])
    }

    /// Save panel → zip the logs + a system summary → reveal the archive. Kit's
    /// DiagnosticsExporter does the packaging; presentation stays in the App layer.
    func exportDiagnostics() {
        AppLog.info("Exporting diagnostics")
        AppLog.flush()

        let panel = NSSavePanel()
        panel.nameFieldStringValue = "MeetingTranscriber-Diagnostics-\(Self.exportTimestamp()).zip"
        panel.allowedContentTypes = [.zip]
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let destination = panel.url else { return }

        do {
            try DiagnosticsExporter.exportDiagnostics(to: destination)
            NSWorkspace.shared.activateFileViewerSelecting([destination])
        } catch {
            AppLog.error("Diagnostics export failed: \(error)")
            let alert = NSAlert()
            alert.messageText = "Export Failed"
            alert.informativeText = "Could not create the diagnostics archive."
            alert.alertStyle = .warning
            alert.runModal()
        }
    }

    private static func exportTimestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter.string(from: Date())
    }
}
