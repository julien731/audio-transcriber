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
        }

        // Standard macOS Settings window (⌘,) for the HuggingFace token (#104).
        Settings {
            SettingsView(appState: appDelegate.appState)
        }
    }
}
