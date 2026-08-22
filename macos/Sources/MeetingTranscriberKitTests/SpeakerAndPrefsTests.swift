import Foundation
import MeetingTranscriberKit

private func segment(_ id: String, _ speaker: String) -> TranscriptSegment {
    TranscriptSegment(id: id, start: 0, end: 1, speaker: speaker, text: "x")
}

func runSpeakerColorTests() {
    suite("SpeakerColor.palette + index") {
        expectEqual(SpeakerColor.palette.count, 8, "8 colors")
        expectEqual(SpeakerColor.hex(forIndex: 0), "#4A90D9", "first color")
        expectEqual(SpeakerColor.hex(forIndex: 8), "#4A90D9", "wraps at 8")
        expectEqual(SpeakerColor.hex(forIndex: 9), "#D94A4A", "wraps to second")
    }

    suite("SpeakerColor.ordered + assignments") {
        let segments = [segment("s1", "SPEAKER_01"), segment("s2", "SPEAKER_00"), segment("s3", "SPEAKER_01")]
        let ordered = SpeakerColor.orderedSpeakerIds(in: segments)
        expectEqual(ordered, ["SPEAKER_01", "SPEAKER_00"], "first-appearance order, deduped")
        let colors = SpeakerColor.assignments(orderedSpeakerIds: ordered)
        expectEqual(colors["SPEAKER_01"], "#4A90D9", "first speaker → first color")
        expectEqual(colors["SPEAKER_00"], "#D94A4A", "second speaker → second color")
    }

    suite("SpeakerColor.isUnidentified") {
        expect(SpeakerColor.isUnidentified(nil), "nil unidentified")
        expect(SpeakerColor.isUnidentified(""), "empty unidentified")
        expect(SpeakerColor.isUnidentified("UNKNOWN"), "UNKNOWN unidentified")
        expect(SpeakerColor.isUnidentified("SPEAKER_00"), "SPEAKER_00 unidentified")
        expect(SpeakerColor.isUnidentified("SPEAKER_12"), "SPEAKER_12 unidentified")
        expect(!SpeakerColor.isUnidentified("Alice"), "named is identified")
        expect(!SpeakerColor.isUnidentified("SPEAKER_"), "SPEAKER_ without digits is not the pattern")
        expect(!SpeakerColor.isUnidentified("SPEAKER_1A"), "non-numeric suffix not the pattern")
    }

    suite("SpeakerColor.unnamedSpeakers") {
        let segments = [segment("s1", "SPEAKER_00"), segment("s2", "SPEAKER_01"), segment("s3", "SPEAKER_00")]
        let info = SpeakerColor.unnamedSpeakers(in: segments, speakers: ["SPEAKER_00": "Alice"])
        expectEqual(info.total, 2, "two distinct speakers")
        expectEqual(info.unnamed, 1, "SPEAKER_01 still unnamed")
        let allNamed = SpeakerColor.unnamedSpeakers(in: segments, speakers: ["SPEAKER_00": "Alice", "SPEAKER_01": "Bob"])
        expectEqual(allNamed.unnamed, 0, "all named → zero")
    }

    suite("SpeakerColor.displayName") {
        expectEqual(SpeakerColor.displayName(for: "SPEAKER_00", speakers: ["SPEAKER_00": "Alice"]), "Alice", "mapped name")
        expectEqual(SpeakerColor.displayName(for: "SPEAKER_00", speakers: [:]), "SPEAKER_00", "fallback to id")
        expectEqual(SpeakerColor.displayName(for: "SPEAKER_00", speakers: ["SPEAKER_00": ""]), "SPEAKER_00", "empty mapping → id")
    }
}

func runPreferencesTests() {
    func isolatedPrefs() -> Preferences {
        let suite = "mt-prefs-\(UUID().uuidString)"
        return Preferences(defaults: UserDefaults(suiteName: suite)!)
    }

    suite("Preferences.theme") {
        let prefs = isolatedPrefs()
        expectEqual(prefs.theme, .system, "defaults to system")
        prefs.theme = .dark
        expectEqual(prefs.theme, .dark, "persists theme")
    }

    suite("Preferences.recentSpeakerNames") {
        let prefs = isolatedPrefs()
        prefs.addRecentSpeakerName("  ")
        expect(prefs.recentSpeakerNames.isEmpty, "blank ignored")
        prefs.addRecentSpeakerName("Alice")
        prefs.addRecentSpeakerName("Bob")
        prefs.addRecentSpeakerName("Alice") // move to front, dedup
        expectEqual(prefs.recentSpeakerNames, ["Alice", "Bob"], "move-to-front + dedup")
        for i in 0..<12 { prefs.addRecentSpeakerName("N\(i)") }
        expectEqual(prefs.recentSpeakerNames.count, Preferences.recentNamesLimit, "capped at 10")
        expectEqual(prefs.recentSpeakerNames.first, "N11", "most recent first")
    }
}
