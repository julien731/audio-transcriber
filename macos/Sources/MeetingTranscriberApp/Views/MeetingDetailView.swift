import SwiftUI
import MeetingTranscriberKit

/// Meeting detail pane. Handles the processing (progress + cancel) and error
/// (retry) states here; the ready state shows the transcript viewer (slice 7).
struct MeetingDetailView: View {
    @StateObject private var store: MeetingDetailStore
    let onChanged: () -> Void

    init(client: APIClient, meetingId: String, onChanged: @escaping () -> Void) {
        _store = StateObject(wrappedValue: MeetingDetailStore(client: client, meetingId: meetingId))
        self.onChanged = onChanged
    }

    var body: some View {
        Group {
            if let detail = store.detail {
                content(for: detail)
            } else if store.isLoading {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Color.clear
            }
        }
        .task {
            store.onStatusSettled = onChanged
            await store.load()
        }
        .onDisappear { store.stop() }
        .overlay(alignment: .bottom) {
            if let error = store.errorMessage {
                ErrorBanner(message: error) { store.errorMessage = nil }
            }
        }
    }

    @ViewBuilder
    private func content(for detail: MeetingDetail) -> some View {
        switch detail.metadata.status {
        case .processing:
            processing(detail)
        case .error:
            errorState(detail)
        case .ready:
            readyState(detail)
        }
    }

    private func processing(_ detail: MeetingDetail) -> some View {
        VStack(spacing: 16) {
            ProgressView(value: Double(store.job?.progress ?? 0), total: 100) {
                Text(JobStagePresentation.label(for: store.job?.stage ?? ""))
            }
            .frame(maxWidth: 360)
            Text("\(store.job?.progress ?? 0)%").foregroundStyle(.secondary).monospacedDigit()
            Button("Cancel transcription", role: .destructive) { Task { await store.cancel() } }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle(detail.metadata.title)
    }

    private func errorState(_ detail: MeetingDetail) -> some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 40)).foregroundStyle(.orange)
            Text("Transcription failed").font(.title3.weight(.semibold))
            Text(detail.metadata.error ?? "The transcription did not complete.")
                .foregroundStyle(.secondary).multilineTextAlignment(.center)
            Button("Retry") { Task { await store.retry() } }.keyboardShortcut(.defaultAction)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding(40)
        .navigationTitle(detail.metadata.title)
    }

    private func readyState(_ detail: MeetingDetail) -> some View {
        // Transcript viewer + audio + tabs arrive in slices 7–9. For now confirm
        // the transcript is present.
        VStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill").font(.system(size: 36)).foregroundStyle(.green)
            Text(detail.metadata.title).font(.title3.weight(.semibold))
            Text("\(detail.transcript?.segments.count ?? 0) segments · transcript viewer arrives next")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle(detail.metadata.title)
    }
}
