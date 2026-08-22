import Foundation
import MeetingTranscriberKit

/// Loads a meeting's detail and, while it is processing, polls its job to show
/// progress and auto-transition to the transcript on completion (BR-9). Also
/// drives cancel (during processing) and meeting-level retry (on error) — BR-7.
@MainActor
final class MeetingDetailStore: ObservableObject {
    @Published private(set) var detail: MeetingDetail?
    @Published private(set) var job: JobInfo?
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    let client: APIClient
    let meetingId: String
    private var pollTask: Task<Void, Never>?
    /// Fired when the meeting reaches a terminal state, so the list can refresh.
    var onStatusSettled: (() -> Void)?

    init(client: APIClient, meetingId: String) {
        self.client = client
        self.meetingId = meetingId
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let detail = try await client.meeting(id: meetingId)
            self.detail = detail
            if detail.metadata.status == .processing, let jobId = detail.metadata.jobId {
                startPolling(jobId: jobId)
            }
        } catch {
            errorMessage = Self.message(for: error)
        }
    }

    /// Resume polling an in-progress job after the window is reopened (BR-17).
    private func startPolling(jobId: String) {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                do {
                    let job = try await self.client.job(id: jobId)
                    self.job = job
                    if job.status.isTerminal {
                        await self.reloadAfterSettle()
                        return
                    }
                } catch {
                    // A missing job after a service restart: reconcile via detail
                    // (recover_stuck_meetings flips it to ERROR server-side).
                    await self.reloadAfterSettle()
                    return
                }
                try? await Task.sleep(nanoseconds: 2_500_000_000)
            }
        }
    }

    private func reloadAfterSettle() async {
        if let refreshed = try? await client.meeting(id: meetingId) {
            detail = refreshed
        }
        onStatusSettled?()
    }

    func cancel() async {
        do {
            try await client.cancelTranscription(id: meetingId)
            pollTask?.cancel()
            await load()
        } catch {
            errorMessage = Self.message(for: error)
        }
    }

    func retry() async {
        do {
            let start = try await client.retryTranscription(id: meetingId)
            await load()
            startPolling(jobId: start.jobId)
            onStatusSettled?()
        } catch {
            errorMessage = Self.message(for: error)
        }
    }

    /// Re-fetch detail after an edit (speaker rename, metadata change) without
    /// touching the poll loop.
    func reloadDetail() async {
        if let refreshed = try? await client.meeting(id: meetingId) {
            detail = refreshed
        }
    }

    func setError(_ message: String) { errorMessage = message }

    func stop() { pollTask?.cancel() }

    private static func message(for error: Error) -> String {
        (error as? APIError)?.userMessage ?? "Could not reach the transcription service."
    }
}
