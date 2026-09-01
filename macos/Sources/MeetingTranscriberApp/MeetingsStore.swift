import Foundation
import MeetingTranscriberKit

/// Loads and mutates the meeting list through the service API (BR-6, BR-7). The
/// authoritative data lives in the service; this is a thin, refreshable cache.
@MainActor
final class MeetingsStore: ObservableObject {
    @Published private(set) var meetings: [MeetingSummary] = []
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    let client: APIClient

    init(client: APIClient) {
        self.client = client
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            meetings = try await client.listMeetings()
            // Feeds the updater's defer-while-busy gate; set(active:) fires the
            // busy→idle edge so a deferred update resumes when work settles (Artifact C).
            BusyState.shared.set(active: QuitPolicy.needsConfirmation(meetings: meetings))
            errorMessage = nil
        } catch {
            // A background refresh can time out while a local transcription
            // saturates the machine; that's transient — keep the cached list and
            // stay silent rather than raising a false alarm (issue #133). The next
            // refresh recovers.
            if (error as? APIError) == .timedOut { return }
            errorMessage = Self.message(for: error)
        }
    }

    func delete(id: String) async {
        do {
            try await client.deleteMeeting(id: id)
            meetings.removeAll { $0.id == id }
        } catch {
            errorMessage = Self.message(for: error)
        }
    }

    private static func message(for error: Error) -> String {
        (error as? APIError)?.userMessage ?? "Could not reach the transcription service."
    }
}
