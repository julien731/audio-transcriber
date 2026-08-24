import Foundation

/// Speakers-panel model for the Transcript tab (story #121). Restores the web
/// client's speakers sidebar (`transcript-viewer.js:321-412`) with an intentional
/// improvement: clicking a speaker cycles to their *next* passage rather than
/// always jumping to the first. Ordering, colors, and naming reuse `SpeakerColor`
/// so the panel matches the transcript segments exactly.
public enum SpeakerPanel {
    /// One row of the panel: a distinct speaker with their palette color and
    /// display name. `isUnnamed` drives the "Unnamed speaker" label + flag styling.
    public struct SpeakerRow: Equatable, Identifiable {
        public let id: String
        public let displayName: String
        public let colorHex: String
        public let isUnnamed: Bool

        public init(id: String, displayName: String, colorHex: String, isUnnamed: Bool) {
            self.id = id
            self.displayName = displayName
            self.colorHex = colorHex
            self.isUnnamed = isUnnamed
        }
    }

    /// Distinct speakers in first-appearance order (matching segment colors), each
    /// resolved to a display name and palette color. Unnamed speakers surface as
    /// "Unnamed speaker" (mirrors the web sidebar).
    public static func rows(in segments: [TranscriptSegment], speakers: [String: String]) -> [SpeakerRow] {
        let ids = SpeakerColor.orderedSpeakerIds(in: segments)
        let colors = SpeakerColor.assignments(orderedSpeakerIds: ids)
        return ids.map { id in
            let unnamed = SpeakerColor.isUnidentified(speakers[id] ?? id)
            return SpeakerRow(
                id: id,
                displayName: unnamed ? "Unnamed speaker" : SpeakerColor.displayName(for: id, speakers: speakers),
                colorHex: colors[id] ?? "#888888",
                isUnnamed: unnamed
            )
        }
    }

    /// Header summary: "N speakers, all named" or "M of N unnamed"
    /// (mirrors `transcript-viewer.js:331-333`).
    public static func summary(for rows: [SpeakerRow]) -> String {
        let total = rows.count
        let unnamed = rows.filter(\.isUnnamed).count
        if unnamed == 0 {
            return "\(total) speaker\(total == 1 ? "" : "s"), all named"
        }
        return "\(unnamed) of \(total) unnamed"
    }

    /// Resolve the next segment to jump to for a speaker, cycling forward through
    /// their segments in document order. `currentSegmentId` is the caller's per-
    /// speaker cursor (the last segment jumped to, or nil on the first click):
    /// - no segments for the speaker → nil (caller treats as a no-op).
    /// - nil / stale cursor → the speaker's first segment.
    /// - otherwise → the segment after the cursor, wrapping from the last to the first.
    public static func nextSegmentId(
        for speakerId: String,
        in segments: [TranscriptSegment],
        after currentSegmentId: String?
    ) -> String? {
        let ids = segments.filter { $0.speaker == speakerId }.map(\.id)
        guard !ids.isEmpty else { return nil }
        guard let current = currentSegmentId, let index = ids.firstIndex(of: current) else {
            return ids.first
        }
        return ids[(index + 1) % ids.count]
    }

    /// Speakers map for the "all segments from this speaker" rename scope (story
    /// #124): a copy of `existing` with `speakerId` mapped to the trimmed
    /// `newName`, mirroring the web client's `speakers[speakerId] = newName`
    /// (`speaker-editor.js:62`). Other entries are preserved; the target is added
    /// when absent. The caller still guards against an empty name.
    public static func speakers(
        renamingAll speakerId: String,
        to newName: String,
        in existing: [String: String]
    ) -> [String: String] {
        var updated = existing
        updated[speakerId] = newName.trimmingCharacters(in: .whitespaces)
        return updated
    }
}
