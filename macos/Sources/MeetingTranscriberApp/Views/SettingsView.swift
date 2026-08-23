import SwiftUI
import MeetingTranscriberKit

/// The app's Settings window (⌘,), reachable any time after first-run (#104).
/// Lets the user view whether speaker diarization is enabled and set, change, or
/// remove the HuggingFace token. All token/model-download logic is delegated to
/// `SettingsTokenModel` (which composes the setup wizard's `ProvisioningController`),
/// so this view stays presentation-only.
struct SettingsView: View {
    @ObservedObject var appState: AppState

    var body: some View {
        Group {
            if let client = appState.client {
                // Keyed on client identity so the form reloads if the service
                // restarts with a fresh client.
                TokenSettingsForm(client: client)
                    .id(ObjectIdentifier(client))
            } else {
                notReady
            }
        }
        .frame(width: 460)
        .padding(24)
    }

    private var notReady: some View {
        VStack(spacing: 10) {
            Image(systemName: "gearshape").font(.system(size: 32)).foregroundStyle(.secondary)
            Text("Settings will be available once the transcription service is ready.")
                .multilineTextAlignment(.center).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 20)
    }
}

/// The token form, active once the service is ready. Mirrors `SetupWizardView`'s
/// `@State` + async-model + poll-while-downloading pattern.
private struct TokenSettingsForm: View {
    let client: APIClient

    @State private var model: SettingsTokenModel?
    @State private var phase: SettingsPhase?
    @State private var token = ""
    @State private var busy = false
    @State private var lastDiarization = false
    @State private var loadError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Speaker Diarization").font(.title3.weight(.semibold))
            content
        }
        .task { await load() }
        .task(id: isWorking) { await pollWhileWorking() }
    }

    @ViewBuilder
    private var content: some View {
        if let loadError {
            failed(loadError, retryTitle: "Retry") { Task { await load() } }
        } else if let phase {
            switch phase {
            case let .idle(diarizationAvailable):
                idle(diarizationAvailable: diarizationAvailable)
            case .editing:
                editing
            case let .working(progress):
                working(progress)
            case let .failed(message):
                failed(message, retryTitle: "Try again") { save(token: token) }
            }
        } else {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Loading…").foregroundStyle(.secondary)
            }
        }
    }

    private func idle(diarizationAvailable: Bool) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(diarizationAvailable ? "Diarization is enabled" : "Diarization is disabled",
                  systemImage: diarizationAvailable ? "checkmark.circle.fill" : "person.crop.circle.badge.xmark")
                .foregroundStyle(diarizationAvailable ? Color.green : Color.secondary)
                .font(.headline)
            Text(diarizationAvailable
                 ? "Your HuggingFace token is set. New meetings are transcribed with separate speakers."
                 : "Add a HuggingFace token to separate speakers in your transcripts. Without one, meetings are transcribed as a single speaker.")
                .foregroundStyle(.secondary).font(.callout)
            HStack {
                Button(diarizationAvailable ? "Change token…" : "Add token…") { beginEditing() }
                    .disabled(busy)
                if diarizationAvailable {
                    Button("Remove token", role: .destructive) { save(token: "") }
                        .disabled(busy)
                }
            }
        }
    }

    private var editing: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Enter your HuggingFace token to enable speaker diarization. Leave it blank and save to disable diarization.")
                .foregroundStyle(.secondary).font(.callout)
            SecureField("HuggingFace token", text: $token)
                .textFieldStyle(.roundedBorder)
            HStack {
                Button("Cancel") { cancelEditing() }.disabled(busy)
                Spacer()
                Button("Save") { save(token: token) }
                    .keyboardShortcut(.defaultAction).disabled(busy)
            }
        }
    }

    private func working(_ progress: Int) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Setting up diarization…").font(.headline)
            ProgressView(value: Double(progress), total: 100)
            Text("\(progress)%").foregroundStyle(.secondary).monospacedDigit()
            Text("Downloading the speaker models. This one-time download can take a while.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func failed(_ message: String, retryTitle: String, retry: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Couldn’t set up diarization", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange).font(.headline)
            Text(message).foregroundStyle(.secondary).font(.callout)
            HStack {
                Button("Back") { cancelEditing() }.disabled(busy)
                Spacer()
                Button(retryTitle, action: retry)
                    .keyboardShortcut(.defaultAction).disabled(busy)
            }
        }
    }

    // MARK: Actions

    private var isWorking: Bool {
        if case .working = phase { return true }
        return false
    }

    private func load() async {
        loadError = nil
        do {
            let status = try await client.provisioning()
            let created = SettingsTokenModel(controller: ProvisioningController(client: client, initial: status))
            model = created
            phase = created.phase
            syncDiarization()
        } catch {
            loadError = (error as? APIError)?.userMessage ?? "Could not load settings."
        }
    }

    private func beginEditing() {
        token = ""
        model?.beginEditing()
        phase = model?.phase
    }

    private func cancelEditing() {
        model?.cancelEditing(diarizationAvailable: lastDiarization)
        phase = model?.phase
        token = ""
    }

    private func save(token value: String) {
        guard let model else { return }
        busy = true
        Task {
            phase = await model.save(token: value)
            syncDiarization()
            busy = false
        }
    }

    /// While a download is in progress, poll provisioning until it settles.
    private func pollWhileWorking() async {
        guard let model else { return }
        while isWorking && !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            phase = await model.refresh()
            syncDiarization()
        }
    }

    /// Remember the latest known diarization state (for Cancel) and clear the field
    /// once we return to a resting `.idle` state.
    private func syncDiarization() {
        if case let .idle(diarizationAvailable) = phase {
            lastDiarization = diarizationAvailable
            token = ""
        }
    }
}
