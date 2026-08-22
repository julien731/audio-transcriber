import SwiftUI

@main
struct MeetingTranscriberApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("Meeting Transcriber") {
            RootView(appState: appDelegate.appState)
                .frame(minWidth: 720, minHeight: 480)
        }
    }
}
