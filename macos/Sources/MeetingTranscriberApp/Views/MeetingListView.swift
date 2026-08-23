import SwiftUI
import MeetingTranscriberKit

/// Sidebar list of meetings with status badges, delete-with-confirm, and refresh
/// (BR-7 parity: list + delete). Sorted server-side (date desc).
struct MeetingListView: View {
    @ObservedObject var store: MeetingsStore
    @Binding var selection: String?
    let onNewMeeting: () -> Void

    @State private var pendingDelete: MeetingSummary?

    var body: some View {
        List(selection: $selection) {
            if store.meetings.isEmpty && !store.isLoading {
                emptyState
            }
            ForEach(store.meetings) { meeting in
                MeetingRow(meeting: meeting)
                    .tag(meeting.id)
                    .contextMenu {
                        Button(role: .destructive) { pendingDelete = meeting } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
            }
        }
        .overlay { if store.isLoading && store.meetings.isEmpty { ProgressView() } }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: onNewMeeting) { Label("New Meeting", systemImage: "plus") }
            }
            ToolbarItem {
                Button { Task { await store.load() } } label: { Label("Refresh", systemImage: "arrow.clockwise") }
            }
        }
        .confirmationDialog(
            "Delete this meeting?",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            presenting: pendingDelete
        ) { meeting in
            Button("Delete “\(meeting.title)”", role: .destructive) {
                Task {
                    if selection == meeting.id { selection = nil }
                    await store.delete(id: meeting.id)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("This permanently removes the meeting, its transcript, and audio. This can’t be undone.")
        }
        .task { await store.load() }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Text("No meetings yet").font(.headline)
            Text("Upload a recording to get started.").font(.callout).foregroundStyle(.secondary)
            Button("New Meeting", action: onNewMeeting)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 24)
    }
}

private struct MeetingRow: View {
    let meeting: MeetingSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(meeting.title).font(.body.weight(.medium)).lineLimit(1)
                Spacer()
                StatusBadgeView(badge: meeting.status.badge)
            }
            HStack(spacing: 8) {
                Text(meeting.type.displayName)
                Text("·")
                Text(Formatters.meetingDate(meeting.createdAt))
                if meeting.durationSeconds != nil {
                    Text("·")
                    Text(Formatters.duration(meeting.durationSeconds))
                }
            }
            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(.vertical, 2)
    }
}
