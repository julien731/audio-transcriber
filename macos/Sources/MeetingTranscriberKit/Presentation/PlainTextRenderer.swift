import Foundation

/// Renders the Plain Text tab, matching the web client's `renderPlainTextTab`
/// format: `[time] Speaker: text` per segment (plan Artifact B, `present`).
public enum PlainTextRenderer {
    public static func render(transcript: Transcript, speakers: [String: String]) -> String {
        transcript.segments.map { segment in
            let name = SpeakerColor.displayName(for: segment.speaker, speakers: speakers)
            return "[\(Formatters.timecode(segment.start))] \(name): \(segment.text)"
        }.joined(separator: "\n")
    }
}
