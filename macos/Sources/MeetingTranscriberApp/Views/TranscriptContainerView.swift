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
                TranscriptTabView(store: store, transcript: transcript, speakers: speakers,
                                  language: detail.metadata.language, audio: audio)
            } else { emptyTranscript }
        case .overview:
            OverviewTabView(detail: detail)
        case .plainText:
            if let transcript = detail.transcript {
                PlainTextTabView(text: PlainTextRenderer.render(transcript: transcript, speakers: speakers))
            } else { emptyTranscript }
        case .analysis:
            AnalysisTabView(client: store.client, meetingId: detail.metadata.id, meetingType: detail.metadata.type)
        }
    }

    private var emptyTranscript: some View {
        Text("No transcript available.").foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Playback controls: play/pause, scrubber, time, and speed.
private struct AudioBar: View {
    @ObservedObject var audio: AudioPlaybackController

    var body: some View {
        HStack(spacing: 12) {
            Button { audio.toggle() } label: {
                Image(systemName: audio.isPlaying ? "pause.fill" : "play.fill").frame(width: 20)
            }
            .buttonStyle(.borderless)

            Text(Formatters.timecode(audio.currentTime)).monospacedDigit().font(.caption)
            Slider(value: Binding(
                get: { audio.currentTime },
                set: { audio.seek(to: $0) }
            ), in: 0...max(audio.duration, 0.1))
            Text(Formatters.timecode(audio.duration)).monospacedDigit().font(.caption).foregroundStyle(.secondary)

            Menu("\(String(format: "%.2gx", Double(audio.rate)))") {
                ForEach([0.75, 1.0, 1.25, 1.5, 2.0], id: \.self) { speed in
                    Button("\(String(format: "%.2g", speed))×") { audio.rate = Float(speed) }
                }
            }
            .frame(width: 56)
        }
        .padding(.horizontal, 16).padding(.vertical, 8)
    }
}
