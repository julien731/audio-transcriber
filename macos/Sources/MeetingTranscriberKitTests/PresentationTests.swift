import Foundation
import MeetingTranscriberKit

func runPresentationTests() {
    suite("Formatters.duration") {
        expectEqual(Formatters.duration(nil), "—", "nil → dash")
        expectEqual(Formatters.duration(-5), "—", "negative → dash")
        expectEqual(Formatters.duration(45), "45s", "seconds only")
        expectEqual(Formatters.duration(125), "2m 05s", "minutes + zero-padded seconds")
        expectEqual(Formatters.duration(3661), "1h 1m", "hours + minutes")
    }

    suite("Formatters.timecode") {
        expectEqual(Formatters.timecode(0), "0:00", "zero")
        expectEqual(Formatters.timecode(65), "1:05", "m:ss")
        expectEqual(Formatters.timecode(3725), "1:02:05", "h:mm:ss")
    }

    suite("MeetingType.displayName") {
        expectEqual(MeetingType.interview.displayName, "Interview", "interview")
        expectEqual(MeetingType.sales.displayName, "Sales", "sales")
    }

    suite("MeetingStatus.badge") {
        expectEqual(MeetingStatus.ready.badge, StatusBadge(label: "Ready", severity: .positive), "ready")
        expectEqual(MeetingStatus.processing.badge, StatusBadge(label: "Processing", severity: .warning), "processing")
        expectEqual(MeetingStatus.error.badge, StatusBadge(label: "Error", severity: .danger), "error")
    }

    suite("JobStatus.isTerminal") {
        expect(JobStatus.completed.isTerminal, "completed terminal")
        expect(JobStatus.failed.isTerminal, "failed terminal")
        expect(!JobStatus.processing.isTerminal, "processing not terminal")
        expect(!JobStatus.pending.isTerminal, "pending not terminal")
    }
}
