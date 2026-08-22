import Foundation

/// Speaker display helpers ported verbatim from the web client (utils.js) so the
/// native app matches it exactly (plan Artifact B, `present`). Colors are keyed
/// by first-appearance index, not by hashing the id.
public enum SpeakerColor {
    /// Matches `SPEAKER_COLORS` in utils.js (order-significant).
    public static let palette: [String] = [
        "#4A90D9", "#D94A4A", "#4AD97A", "#D9A84A",
        "#9B4AD9", "#4AD9D9", "#D94A9B", "#7AD94A",
    ]

    public static func hex(forIndex index: Int) -> String {
        palette[((index % palette.count) + palette.count) % palette.count]
    }

    /// Assign each speaker id a stable palette color by first appearance order in
    /// the transcript (mirrors the web client's `speakerIds` ordering).
    public static func assignments(orderedSpeakerIds ids: [String]) -> [String: String] {
        var result: [String: String] = [:]
        for (index, id) in ids.enumerated() {
            result[id] = hex(forIndex: index)
        }
        return result
    }

    /// First-appearance order of speaker ids across transcript segments.
    public static func orderedSpeakerIds(in segments: [TranscriptSegment]) -> [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for segment in segments where !seen.contains(segment.speaker) {
            seen.insert(segment.speaker)
            ordered.append(segment.speaker)
        }
        return ordered
    }

    /// utils.js `isUnidentifiedSpeaker`: empty, "UNKNOWN", or `SPEAKER_<digits>`.
    public static func isUnidentified(_ name: String?) -> Bool {
        guard let name, !name.isEmpty else { return true }
        if name == "UNKNOWN" { return true }
        return matchesSpeakerLabel(name)
    }

    /// Resolve a speaker id to its display name (metadata mapping, else the id).
    public static func displayName(for speakerId: String, speakers: [String: String]) -> String {
        let mapped = speakers[speakerId]
        if let mapped, !mapped.isEmpty { return mapped }
        return speakerId
    }

    /// Count of distinct speakers still unnamed (mirrors utils.js
    /// `getUnnamedSpeakersInfo`) — used to warn before generating an analysis.
    public static func unnamedSpeakers(in segments: [TranscriptSegment], speakers: [String: String]) -> (unnamed: Int, total: Int) {
        let ids = orderedSpeakerIds(in: segments)
        let unnamed = ids.filter { isUnidentified(speakers[$0] ?? $0) }.count
        return (unnamed, ids.count)
    }

    private static func matchesSpeakerLabel(_ name: String) -> Bool {
        guard name.hasPrefix("SPEAKER_") else { return false }
        let suffix = name.dropFirst("SPEAKER_".count)
        return !suffix.isEmpty && suffix.allSatisfy { $0.isNumber }
    }
}
