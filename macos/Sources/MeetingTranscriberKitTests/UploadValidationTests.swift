import Foundation
import MeetingTranscriberKit

func runUploadValidationTests() {
    suite("UploadValidation.validate") {
        expectNil(UploadValidation.validate(filename: "meeting.mp3", byteCount: 1_000), "mp3 ok")
        expectNil(UploadValidation.validate(filename: "REC.WAV", byteCount: 1_000), "case-insensitive ext")
        expectEqual(UploadValidation.validate(filename: "notes.txt", byteCount: 10),
                    .unsupportedFormat("txt"), "txt rejected")
        expectEqual(UploadValidation.validate(filename: "noext", byteCount: 10),
                    .unsupportedFormat(""), "missing ext rejected")
        expectEqual(UploadValidation.validate(filename: "big.mp4", byteCount: UploadValidation.maxUploadBytes + 1),
                    .tooLarge, "over-size rejected")
        expectNil(UploadValidation.validate(filename: "exact.mp4", byteCount: UploadValidation.maxUploadBytes),
                  "exactly max is allowed")
    }

    suite("UploadValidation.Failure.message") {
        expect(UploadValidation.Failure.tooLarge.message.contains("500 MB"), "size message")
        expect(UploadValidation.Failure.unsupportedFormat("x").message.contains("mp3"), "format message lists types")
    }

    suite("Languages") {
        expectEqual(Languages.all.count, 18, "18 supported languages")
        expectEqual(Languages.name(for: "fr"), "French", "code → name")
        expectEqual(Languages.name(for: "xx"), "XX", "unknown code → uppercased")
    }

    suite("JobStagePresentation.label") {
        expectEqual(JobStagePresentation.label(for: "transcribing"), "Transcribing", "known stage")
        expectEqual(JobStagePresentation.label(for: "diarizing"), "Identifying speakers", "friendly label")
        expectEqual(
            JobStagePresentation.label(for: "downloading_align_model"),
            "Downloading alignment model", "align-model download stage (#145)")
        expectEqual(JobStagePresentation.label(for: ""), "Working", "empty → generic")
        expectEqual(JobStagePresentation.label(for: "custom_stage"), "Custom Stage", "unknown → humanized")
    }
}
