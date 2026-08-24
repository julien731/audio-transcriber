import Foundation

/// Transcript ↔ audio sync helpers (story #125, BR-8). The two directions of sync
/// share one seam: click a segment to seek to its exact start time (wired in the
/// view), and — here — resolve which segment is active for a given playback time so
/// the transcript highlights it. Mirrors the web client's
/// `highlightCurrentSegment` predicate (`transcript-viewer.js`).
public enum TranscriptSync {
    /// The id of the segment active at `time` (audio playback position), or nil
    /// when `time` falls before the first segment, after the last, or in a gap
    /// between segments. The lower bound is inclusive and the upper bound exclusive
    /// (`start <= time < end`), so seeking to a segment's exact `start` resolves to
    /// that segment.
    public static func activeSegmentId(at time: Double, in segments: [TranscriptSegment]) -> String? {
        segments.first { $0.start <= time && time < $0.end }?.id
    }
}
