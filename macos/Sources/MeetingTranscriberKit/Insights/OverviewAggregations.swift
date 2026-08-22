import Foundation

/// Overview aggregations `port`ed verbatim from the web client (audio-insights.js
/// + overview-viewer.js). These are analytical derivations, not formatting, so
/// they carry fixture contract tests to guard against divergence (plan Artifact B,
/// Decision 3). Field names and math mirror the JS exactly.
public enum OverviewAggregations {
    static let energyPositive: Set<EmotionCategory> = [.engaged, .confident]
    static let energyNegative: Set<EmotionCategory> = [.disengaged, .frustrated]
    public static let energyWindowSeconds: Double = 300

    // MARK: Energy trajectory

    public struct EnergyWindow: Equatable {
        public let index: Int
        public let start: Double
        public let end: Double
        /// Mean energy in the window, or nil when the window has no emotions.
        public let score: Double?
        public let count: Int
    }

    public static func energyScore(_ emotion: EmotionAnnotation) -> Double {
        if energyPositive.contains(emotion.primaryEmotion) { return emotion.confidence }
        if energyNegative.contains(emotion.primaryEmotion) { return -emotion.confidence }
        return 0
    }

    public static func energyTrajectory(_ emotions: [EmotionAnnotation]) -> [EnergyWindow] {
        guard !emotions.isEmpty else { return [] }
        let end = emotions.map(\.end).max() ?? 0
        guard end > 0 else { return [] }
        let windowCount = max(1, Int(end / energyWindowSeconds) + 1)
        var buckets = Array(repeating: [Double](), count: windowCount)
        for emotion in emotions {
            let idx = min(Int(emotion.start / energyWindowSeconds), windowCount - 1)
            buckets[idx].append(energyScore(emotion))
        }
        return buckets.enumerated().map { index, scores in
            EnergyWindow(
                index: index,
                start: Double(index) * energyWindowSeconds,
                end: Double(index + 1) * energyWindowSeconds,
                score: scores.isEmpty ? nil : scores.reduce(0, +) / Double(scores.count),
                count: scores.count
            )
        }
    }

    // MARK: Interruptions

    public struct SpeakerInterruptionCount: Equatable {
        public let speakerId: String
        public let name: String
        public let made: Int
        public let received: Int
    }

    public struct InterruptionPair: Equatable {
        public let interrupter: String
        public let interrupted: String
        public let interrupterName: String
        public let interruptedName: String
        public let count: Int
    }

    public struct InterruptionSummary: Equatable {
        public let total: Int
        public let totals: [SpeakerInterruptionCount]
        public let pairs: [InterruptionPair]
    }

    public static func interruptions(_ interactions: [InteractionEvent], speakers: [String: String]) -> InterruptionSummary {
        var pairCounts: [String: Int] = [:]
        var pairOrder: [String] = []
        var made: [String: Int] = [:]
        var received: [String: Int] = [:]
        // Preserve JS insertion order (V8 stable sort + Object.keys/Set order): the
        // web `speakerIds` is made-keys (first-made order) then received-keys.
        var madeOrder: [String] = []
        var receivedOrder: [String] = []
        var total = 0

        for event in interactions where event.eventType == .interruption {
            // Web keys as `${speaker_b}|${speaker_a}`: b interrupts a.
            let key = "\(event.speakerB)|\(event.speakerA)"
            if pairCounts[key] == nil { pairOrder.append(key) }
            pairCounts[key, default: 0] += 1
            if made[event.speakerB] == nil { madeOrder.append(event.speakerB) }
            made[event.speakerB, default: 0] += 1
            if received[event.speakerA] == nil { receivedOrder.append(event.speakerA) }
            received[event.speakerA, default: 0] += 1
            total += 1
        }

        var order = madeOrder
        for id in receivedOrder where !order.contains(id) { order.append(id) }
        let totals = order.map { id in
            SpeakerInterruptionCount(speakerId: id, name: speakers[id] ?? id,
                                     made: made[id] ?? 0, received: received[id] ?? 0)
        }.stableSorted { ($0.made + $0.received) > ($1.made + $1.received) }

        let pairs = pairOrder.map { key -> InterruptionPair in
            let parts = key.split(separator: "|", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
            let interrupter = parts.first ?? ""
            let interrupted = parts.count > 1 ? parts[1] : ""
            return InterruptionPair(interrupter: interrupter, interrupted: interrupted,
                                    interrupterName: speakers[interrupter] ?? interrupter,
                                    interruptedName: speakers[interrupted] ?? interrupted,
                                    count: pairCounts[key] ?? 0)
        }.stableSorted { $0.count > $1.count }

        return InterruptionSummary(total: total, totals: totals, pairs: pairs)
    }

    // MARK: Response latency

    public struct LatencyEntry: Equatable {
        public let speakerId: String
        public let name: String
        public let average: Double
        public let count: Int
    }

    public static func latencies(_ segmentInteractions: [SegmentInteraction],
                                 transcript: Transcript?,
                                 speakers: [String: String]) -> [LatencyEntry] {
        guard let transcript else { return [] }
        var segmentsById: [String: TranscriptSegment] = [:]
        for segment in transcript.segments { segmentsById[segment.id] = segment }

        var bySpeaker: [String: [Double]] = [:]
        var speakerOrder: [String] = []
        for si in segmentInteractions where si.hesitationBefore > 0 {
            guard let seg = segmentsById[si.segmentId] else { continue }
            if bySpeaker[seg.speaker] == nil { speakerOrder.append(seg.speaker) }
            bySpeaker[seg.speaker, default: []].append(si.hesitationBefore)
        }

        return speakerOrder.map { speakerId in
            let values = bySpeaker[speakerId] ?? []
            return LatencyEntry(speakerId: speakerId, name: speakers[speakerId] ?? speakerId,
                                average: values.reduce(0, +) / Double(values.count), count: values.count)
        }.stableSorted { $0.average > $1.average }
    }
}

extension Array {
    /// Stable sort (Swift's `sorted` is not guaranteed stable). Preserves original
    /// order for equal elements, matching the web client's V8 stable sort.
    func stableSorted(by areInIncreasingOrder: (Element, Element) -> Bool) -> [Element] {
        enumerated().sorted { lhs, rhs in
            if areInIncreasingOrder(lhs.element, rhs.element) { return true }
            if areInIncreasingOrder(rhs.element, lhs.element) { return false }
            return lhs.offset < rhs.offset
        }.map(\.element)
    }
}
