import SwiftUI

@main
struct MeetingTranscriberApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("Blah") {
            RootView(appState: appDelegate.appState)
                .frame(minWidth: 720, minHeight: 480)
        }
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") { appDelegate.updater.checkForUpdates() }
            }
            // Diagnostics access for debugging app hangs (story #137).
            CommandGroup(replacing: .help) {
                Button("Reveal Logs in Finder") { appDelegate.revealLogs() }
                Button("Export Diagnostics…") { appDelegate.exportDiagnostics() }
            }
        }

        // Standard macOS Settings window (⌘,) for the HuggingFace token (#104).
        Settings {
            SettingsView(appState: appDelegate.appState)
        }
    }
}
