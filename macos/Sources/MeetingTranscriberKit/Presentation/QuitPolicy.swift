import Foundation

/// Whether quitting should warn the user (BR-16, EC-5). Job state is in-memory in
/// the service and lost on shutdown, so an active transcription means the user
/// should confirm before quitting.
public enum QuitPolicy {
    public static func needsConfirmation(meetings: [MeetingSummary]) -> Bool {
        meetings.contains { $0.status == .processing }
    }

    public static let confirmationTitle = "A transcription is still running"
    public static let confirmationMessage =
        "Quitting now stops the transcription and the in-progress job will be lost. Quit anyway?"
}
