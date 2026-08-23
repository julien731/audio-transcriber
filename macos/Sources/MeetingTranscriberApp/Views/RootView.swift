import SwiftUI
import MeetingTranscriberKit

/// Routes the top-level app phase to the right screen (BR-2, EC-2).
struct RootView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        switch appState.phase {
        case .starting:
            ServiceStartingView()
        case let .serviceFailed(message):
            ServiceErrorView(message: message) { appState.restart() }
        case let .setup(controller, client):
            SetupWizardView(controller: controller) { appState.finishSetup(client: client) }
        case let .ready(client):
            MainView(client: client)
        }
    }
}

/// The service-startup spinner shown before the UI is ready.
struct ServiceStartingView: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView().controlSize(.large)
            Text("Starting the transcription service…").foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}

/// Clear error state with a Retry action (never a blank window — EC-2, BR-4).
struct ServiceErrorView: View {
    let message: String
    let onRetry: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 44)).foregroundStyle(.orange)
            Text("The transcription service isn’t running").font(.title2.weight(.semibold))
            Text(message).multilineTextAlignment(.center).foregroundStyle(.secondary)
            Button("Restart service", action: onRetry).keyboardShortcut(.defaultAction)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}
