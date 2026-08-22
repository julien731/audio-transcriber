import SwiftUI
import MeetingTranscriberKit

/// Native first-run setup (plan slice 4, BR-12/13/14). Collects the optional
/// HuggingFace token and shows model-download progress. On a failed download it
/// shows the service's own message (never "token rejected" — BR-13) and offers
/// Retry or Continue without diarization.
struct SetupWizardView: View {
    let controller: ProvisioningController
    let onComplete: () -> Void

    @State private var phase: SetupPhase
    @State private var token: String = ""
    @State private var busy = false

    init(controller: ProvisioningController, onComplete: @escaping () -> Void) {
        self.controller = controller
        self.onComplete = onComplete
        _phase = State(initialValue: controller.phase)
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("Set up Meeting Transcriber")
                .font(.title.weight(.semibold))
            content
        }
        .frame(maxWidth: 460)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
        .task(id: isDownloading) { await pollWhileDownloading() }
        .onChange(of: phase) { newValue in
            if case .completed = newValue { onComplete() }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch phase {
        case .enteringToken:
            tokenEntry
        case let .downloading(progress):
            downloading(progress)
        case let .failed(message):
            failed(message)
        case .completed:
            ProgressView() // brief; onChange calls onComplete
        }
    }

    private var tokenEntry: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Enter your HuggingFace token to enable speaker diarization. This is optional — leave it blank to transcribe without identifying speakers. You can add it later.")
                .foregroundStyle(.secondary)
            SecureField("HuggingFace token (optional)", text: $token)
                .textFieldStyle(.roundedBorder)
            HStack {
                Button("Skip diarization") { submit(token: "") }
                    .disabled(busy)
                Spacer()
                Button("Continue") { submit(token: token) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(busy)
            }
        }
    }

    private func downloading(_ progress: Int) -> some View {
        VStack(spacing: 14) {
            Text("Downloading models…").font(.headline)
            ProgressView(value: Double(progress), total: 100)
            Text("\(progress)%").foregroundStyle(.secondary).monospacedDigit()
            Text("This one-time download can take a while.").font(.caption).foregroundStyle(.secondary)
        }
    }

    private func failed(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("The model download failed", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange).font(.headline)
            Text(message).foregroundStyle(.secondary)
            Text("A valid HuggingFace token is required for diarization. You can retry, or continue without diarization — transcription still works, but speakers won’t be separated.")
                .font(.callout).foregroundStyle(.secondary)
            HStack {
                Button("Continue without diarization") { continueWithoutDiarization() }
                    .disabled(busy)
                Spacer()
                Button("Retry") { retry() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(busy)
            }
        }
    }

    // MARK: Actions

    private var isDownloading: Bool {
        if case .downloading = phase { return true }
        return false
    }

    private func submit(token: String) {
        busy = true
        Task { phase = await controller.submit(token: token); busy = false }
    }

    private func retry() {
        busy = true
        Task { phase = await controller.retry(); busy = false }
    }

    private func continueWithoutDiarization() {
        busy = true
        Task { phase = await controller.continueWithoutDiarization(); busy = false }
    }

    /// While a download is in progress, poll provisioning until it finishes.
    private func pollWhileDownloading() async {
        while isDownloading && !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            phase = await controller.refresh()
        }
    }
}
