import SwiftUI
import MeetingTranscriberKit

/// Main workspace: sidebar meeting list + detail pane. The detail is a
/// placeholder until the transcript viewer (slice 7); upload arrives in slice 6.
struct MainView: View {
    @StateObject private var store: MeetingsStore
    @State private var selection: String?
    @State private var showingUpload = false

    init(client: APIClient) {
        _store = StateObject(wrappedValue: MeetingsStore(client: client))
    }

    var body: some View {
        NavigationSplitView {
            MeetingListView(store: store, selection: $selection) { showingUpload = true }
                .navigationTitle("Meetings")
                .frame(minWidth: 280)
        } detail: {
            if let selection {
                MeetingDetailPlaceholder(meetingId: selection)
            } else {
                EmptyDetailView()
            }
        }
        .overlay(alignment: .bottom) {
            if let error = store.errorMessage {
                ErrorBanner(message: error) { store.errorMessage = nil }
            }
        }
        .sheet(isPresented: $showingUpload) {
            Text("Upload arrives in slice 6.").padding(40)
        }
    }
}

/// Empty detail state (macOS 13-compatible; ContentUnavailableView is 14+).
private struct EmptyDetailView: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform").font(.system(size: 40)).foregroundStyle(.secondary)
            Text("Select a meeting").font(.title3.weight(.semibold))
            Text("Choose a meeting to view its transcript.").foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Temporary detail pane; replaced by the transcript viewer in slice 7.
private struct MeetingDetailPlaceholder: View {
    let meetingId: String
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "doc.text").font(.system(size: 36)).foregroundStyle(.secondary)
            Text("Meeting \(meetingId)").font(.headline)
            Text("Transcript viewer arrives in the next slices.").foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct ErrorBanner: View {
    let message: String
    let onDismiss: () -> Void

    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.red)
            Text(message).font(.callout)
            Spacer()
            Button("Dismiss", action: onDismiss).buttonStyle(.borderless)
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .padding()
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }
}
