import SwiftUI

/// The bootstrap screen shown until the service is ready. On failure it shows a
/// clear error with a Retry action rather than a blank window (EC-2, BR-4).
struct BootstrapView: View {
    @ObservedObject var model: BootstrapModel

    var body: some View {
        VStack(spacing: 20) {
            switch model.state {
            case .starting:
                ProgressView()
                    .controlSize(.large)
                Text("Starting the transcription service…")
                    .foregroundStyle(.secondary)
            case let .ready(port):
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(.green)
                Text("Service ready")
                    .font(.title2.weight(.semibold))
                Text("Listening on port \(port)")
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            case let .failed(message):
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(.orange)
                Text("The service could not start")
                    .font(.title2.weight(.semibold))
                Text(message)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                Button("Retry") { model.restart() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
