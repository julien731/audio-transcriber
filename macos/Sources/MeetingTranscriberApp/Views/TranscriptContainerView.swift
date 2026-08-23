import SwiftUI
import MeetingTranscriberKit

/// Ready-meeting viewer: audio player + tabbed content (Transcript / Overview /
/// Plain Text / Analysis). Overview and Analysis are filled in slices 8–9.
struct TranscriptContainerView: View {
    @ObservedObject var store: MeetingDetailStore
    let detail: MeetingDetail

    @StateObject private var audio: AudioPlaybackController
    @State private var tab: Tab = .transcript
    @State private var editingMeta = false
    @State private var editingSpeakers = false

    enum Tab: String, CaseIterable { case transcript = "Transcript", overview = "Overview", plainText = "Plain Text", analysis = "Analysis" }

    init(store: MeetingDetailStore, detail: MeetingDetail) {
        self.store = store
        self.detail = detail
        _audio = StateObject(wrappedValue: AudioPlaybackController(url: store.client.audioURL(id: detail.metadata.id)))
    }

    private var speakers: [String: String] { detail.metadata.speakers }

    var body: some View {
        VStack(spacing: 0) {
            header
            AudioBar(audio: audio)
            Divider()
            Picker("", selection: $tab) {
                ForEach(Tab.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(8)
            Divider()
            content
        }
        .navigationTitle(detail.metadata.title)
        .sheet(isPresented: $editingMeta) {
            MeetingEditView(client: store.client, metadata: detail.metadata) {
                Task { await store.reloadDetail() }
            }
        }
        .sheet(isPresented: $editingSpeakers) {
            SpeakerEditorView(client: store.client, meetingId: detail.metadata.id, speakers: speakers) {
                Task { await store.reloadDetail() }
            }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(detail.metadata.title).font(.headline)
                Text(detail.metadata.type.displayName).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button { editingSpeakers = true } label: { Label("Speakers", systemImage: "person.2") }
            Button { editingMeta = true } label: { Label("Edit", systemImage: "pencil") }
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }

    @ViewBuilder
    private var content: some View {
        switch tab {
        case .transcript:
            if let transcript = detail.transcript {
                let showInsights = SegmentInsights.hasCompletedAnalysis(metadata: detail.metadata, analysis: detail.audioAnalysis)
                TranscriptTabView(store: store, transcript: transcript, speakers: speakers,
                                  language: detail.metadata.language, audio: audio,
                                  insights: showInsights ? SegmentInsights.index(detail.audioAnalysis) : [:])
            } else { emptyTranscript }
        case .overview:
            OverviewTabView(detail: detail)
        case .plainText:
            if let transcript = detail.transcript {
                PlainTextTabView(text: PlainTextRenderer.render(transcript: transcript, speakers: speakers))
            } else { emptyTranscript }
        case .analysis:
            AnalysisTabView(client: store.client, detail: detail)
        }
    }

    private var emptyTranscript: some View {
        Text("No transcript available.").foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Playback controls: a light pill with a circular play/pause button, a
/// waveform scrubber, elapsed time, and a speed selector below.
private struct AudioBar: View {
    @ObservedObject var audio: AudioPlaybackController

    private var progress: Double {
        audio.duration > 0 ? min(max(audio.currentTime / audio.duration, 0), 1) : 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 14) {
                Button { audio.toggle() } label: {
                    ZStack {
                        Circle().fill(Color.accentColor)
                        Image(systemName: audio.isPlaying ? "pause.fill" : "play.fill")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(.white)
                            // Nudge the play triangle to sit optically centered.
                            .offset(x: audio.isPlaying ? 0 : 1)
                    }
                    .frame(width: 34, height: 34)
                }
                .buttonStyle(.plain)

                WaveformScrubber(progress: progress) { fraction in
                    audio.seek(to: fraction * max(audio.duration, 0.1))
                }
                .frame(height: 36)

                Text(Formatters.timecode(audio.currentTime))
                    .monospacedDigit().font(.caption).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.primary.opacity(0.05)))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Color.primary.opacity(0.08)))

            speedControl
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }

    private var speedControl: some View {
        Menu {
            ForEach([0.75, 1.0, 1.25, 1.5, 2.0], id: \.self) { speed in
                Button("\(String(format: "%.2g", speed))×") { audio.rate = Float(speed) }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "speedometer")
                Text("Speed: \(String(format: "%.2g", Double(audio.rate)))×")
                Image(systemName: "chevron.down").font(.caption2)
            }
            .font(.caption)
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.primary.opacity(0.05)))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Color.primary.opacity(0.08)))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
    }
}

/// A static pseudo-waveform that doubles as the scrub bar: bars up to `progress`
/// are tinted with the accent color, the rest are muted. Tap or drag to seek.
private struct WaveformScrubber: View {
    let progress: Double
    let onScrub: (Double) -> Void

    private let bars: [CGFloat] = WaveformScrubber.makeBars(count: 100)

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 0) {
                ForEach(bars.indices, id: \.self) { index in
                    Capsule()
                        .fill(isPlayed(index) ? Color.accentColor : Color.primary.opacity(0.22))
                        .frame(width: 2, height: max(3, geo.size.height * bars[index]))
                        .frame(maxWidth: .infinity)
                }
            }
            .frame(maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0).onChanged { value in
                    onScrub(min(max(value.location.x / geo.size.width, 0), 1))
                }
            )
        }
    }

    private func isPlayed(_ index: Int) -> Bool {
        Double(index) / Double(bars.count) <= progress
    }

    /// Deterministic bar heights (0.16...1) that read like a real voice-memo
    /// waveform. Uses a splitmix64 PRNG per index for genuine, non-repeating
    /// randomness — stable across renders and independent of the actual (here
    /// silent) audio samples.
    static func makeBars(count: Int) -> [CGFloat] {
        func random(_ index: Int) -> Double {
            var z = UInt64(index) &+ 0x9E37_79B9_7F4A_7C15
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z ^= z >> 31
            return Double(z >> 11) * (1.0 / 9_007_199_254_740_992.0)
        }
        let floor = 0.16
        return (0..<count).map { CGFloat(floor + (1 - floor) * random($0)) }
    }
}
