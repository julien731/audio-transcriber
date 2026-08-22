import SwiftUI

@main
struct MeetingTranscriberApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("Meeting Transcriber") {
            BootstrapView(model: appDelegate.bootstrap)
                .frame(minWidth: 480, minHeight: 320)
        }
        .windowResizability(.contentSize)
    }
}
