import SwiftUI
import MeetingTranscriberKit

/// Overview tab: energy/emotion trajectory, interruption summary, and response
/// latency — rendered from the ported aggregations (plan Artifact B, slice 8).
/// Handles the opted-out and unavailable states, not just the happy path.
struct OverviewTabView: View {
    let detail: MeetingDetail

    private var analysis: AudioAnalysis? { detail.audioAnalysis }
    private var speakers: [String: String] { detail.metadata.speakers }

    var body: some View {
        Group {
            if !detail.metadata.audioAnalysisEnabled {
                message("Audio analysis was not run for this meeting.",
                        detail: "Enable audio analysis when uploading to get emotion and interaction insights.")
            } else if let analysis, analysis.status == .completed {
                completed(analysis)
            } else {
                message("Audio analysis is unavailable.",
                        detail: analysis?.reason ?? "Analysis did not produce results for this meeting.")
            }
        }
    }

    private func completed(_ analysis: AudioAnalysis) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                if analysis.dominantSpeakerLimitation {
                    Label("A single speaker dominates this meeting, so interruption and turn-taking signals are sparse.",
                          systemImage: "info.circle")
                        .font(.callout).foregroundStyle(.secondary)
                        .padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                }
                section("Energy & emotion trajectory") {
                    TrajectoryView(windows: OverviewAggregations.energyTrajectory(analysis.emotions))
                }
                section("Interruptions") {
                    InterruptionSummaryView(
                        summary: OverviewAggregations.interruptions(analysis.interactions, speakers: speakers))
                }
                section("Average response latency") {
                    LatencyView(entries: OverviewAggregations.latencies(analysis.segmentInteractions,
                                                                        transcript: detail.transcript,
                                                                        speakers: speakers))
                }
            }
            .padding(20)
        }
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content()
        }
    }

    private func message(_ title: String, detail: String) -> some View {
        VStack(spacing: 8) {
            Text(title).font(.headline)
            Text(detail).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding(40)
    }
}

/// Diverging bars per time window (positive = engaged/confident, negative =
/// disengaged/frustrated).
private struct TrajectoryView: View {
    let windows: [OverviewAggregations.EnergyWindow]

    var body: some View {
        if windows.allSatisfy({ $0.score == nil }) {
            Text("No emotion data to chart.").font(.callout).foregroundStyle(.secondary)
        } else {
            VStack(spacing: 6) {
                ForEach(windows, id: \.index) { window in
                    HStack(spacing: 8) {
                        Text(Formatters.timecode(window.start)).font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary).frame(width: 56, alignment: .leading)
                        DivergingBar(score: window.score ?? 0)
                        Text(window.score.map { String(format: "%+.2f", $0) } ?? "—")
                            .font(.caption.monospacedDigit()).frame(width: 48, alignment: .trailing)
                    }
                }
                HStack(spacing: 12) {
                    legend(.green, "Positive")
                    legend(.orange, "Negative")
                }.font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private func legend(_ color: Color, _ text: String) -> some View {
        HStack(spacing: 4) { Circle().fill(color).frame(width: 7, height: 7); Text(text) }
    }
}

private struct DivergingBar: View {
    let score: Double // -1...1

    var body: some View {
        GeometryReader { geo in
            let mid = geo.size.width / 2
            let magnitude = min(abs(score), 1) * mid
            ZStack(alignment: .leading) {
                Rectangle().fill(.quaternary).frame(height: 10)
                Rectangle().fill(score >= 0 ? Color.green : Color.orange)
                    .frame(width: magnitude, height: 10)
                    .offset(x: score >= 0 ? mid : mid - magnitude)
            }
        }
        .frame(height: 10)
    }
}

private struct InterruptionSummaryView: View {
    let summary: OverviewAggregations.InterruptionSummary

    var body: some View {
        if summary.total == 0 {
            Text("No interruptions detected.").font(.callout).foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 6) {
                Text("\(summary.total) interruption\(summary.total == 1 ? "" : "s") total").font(.callout)
                ForEach(summary.pairs, id: \.interrupter) { pair in
                    Text("\(pair.interrupterName) interrupted \(pair.interruptedName) ")
                        + Text("×\(pair.count)").foregroundColor(.secondary)
                }
                .font(.callout)
            }
        }
    }
}

private struct LatencyView: View {
    let entries: [OverviewAggregations.LatencyEntry]

    var body: some View {
        if entries.isEmpty {
            Text("No response-latency data.").font(.callout).foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(entries, id: \.speakerId) { entry in
                    HStack {
                        Text(entry.name)
                        Spacer()
                        Text(String(format: "%.1fs avg", entry.average)).monospacedDigit()
                        Text("(\(entry.count))").foregroundStyle(.secondary)
                    }.font(.callout)
                }
            }
        }
    }
}
