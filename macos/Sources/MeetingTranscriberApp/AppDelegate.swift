import AppKit

/// App lifecycle for the background-run model (plan TD-5/TD-6):
/// - closing the window does NOT quit the app (a transcription may be running);
/// - reopening focuses/recreates the window (single-instance behavior, EC-9);
/// - on quit, the embedded service is shut down cleanly (BR-5).
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let bootstrap = BootstrapModel()

    func applicationDidFinishLaunching(_ notification: Notification) {
        bootstrap.start()
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

    func applicationWillTerminate(_ notification: Notification) {
        bootstrap.shutdown()
    }
}
