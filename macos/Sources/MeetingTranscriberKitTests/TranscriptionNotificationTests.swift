import Foundation
import MeetingTranscriberKit

func runTranscriptionNotificationTests() {
    suite("JobStatus.settledNotification") {
        let completed = JobStatus.completed.settledNotification(meetingTitle: "Q3 Sync", error: nil)
        expectEqual(completed?.title, "Transcription complete", "completed title")
        expect(completed?.body.contains("Q3 Sync") == true, "completed body names the meeting")

        let failed = JobStatus.failed.settledNotification(meetingTitle: "Q3 Sync", error: "GPU out of memory")
        expectEqual(failed?.title, "Transcription failed", "failed title")
        expectEqual(failed?.body, "GPU out of memory", "failed body uses the error verbatim")

        let failedNoError = JobStatus.failed.settledNotification(meetingTitle: "Q3 Sync", error: nil)
        expectEqual(failedNoError?.title, "Transcription failed", "failed (no error) title")
        expect(failedNoError?.body.contains("Q3 Sync") == true, "failed fallback body names the meeting")

        expectNil(JobStatus.processing.settledNotification(meetingTitle: "Q3 Sync", error: nil), "processing → nil")
        expectNil(JobStatus.pending.settledNotification(meetingTitle: "Q3 Sync", error: nil), "pending → nil")
    }
}
