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
            errorMessage = nil
        } catch {
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
