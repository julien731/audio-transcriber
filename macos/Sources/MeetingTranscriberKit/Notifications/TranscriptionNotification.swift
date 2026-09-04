import Foundation

/// UI-agnostic content for a "transcription settled" user notification, mirroring
/// the web app's `sendNotification('Transcription complete'/'Transcription failed')`
/// (frontend/js/components/transcript-viewer.js). The App layer maps this onto a
/// `UNNotificationRequest`; keeping the copy here makes it unit-testable in Kit.
public struct TranscriptionNotification: Equatable {
    public let title: String
    public let body: String

    public init(title: String, body: String) {
        self.title = title
        self.body = body
    }
}

public extension JobStatus {
    /// Notification content for a settled job, or `nil` while still in progress.
    /// - Parameters:
    ///   - meetingTitle: the meeting's title, woven into the body for context.
    ///   - error: the job's failure message, used verbatim for `.failed` when present.
    func settledNotification(meetingTitle: String, error: String?) -> TranscriptionNotification? {
        switch self {
        case .completed:
            return TranscriptionNotification(
                title: "Transcription complete",
                body: "\(meetingTitle) has been transcribed and is ready to view."
            )
        case .failed:
            return TranscriptionNotification(
                title: "Transcription failed",
                body: error ?? "An error occurred while transcribing \(meetingTitle)."
            )
        case .pending, .processing:
            return nil
        }
    }
}
