import Foundation
import MeetingTranscriberKit

func runPlainTextRendererTests() {
    suite("PlainTextRenderer.render") {
        let transcript = Transcript(segments: [
            TranscriptSegment(id: "s1", start: 0, end: 3, speaker: "SPEAKER_00", text: "Hello there"),
            TranscriptSegment(id: "s2", start: 65, end: 70, speaker: "SPEAKER_01", text: "Hi back"),
        ], language: "en")
        let text = PlainTextRenderer.render(transcript: transcript, speakers: ["SPEAKER_00": "Alice"])
        let lines = text.split(separator: "\n").map(String.init)
        expectEqual(lines[0], "[0:00] Alice: Hello there", "named speaker + timecode")
        expectEqual(lines[1], "[1:05] SPEAKER_01: Hi back", "unnamed falls back to id")
    }
}
