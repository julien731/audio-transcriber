import SwiftUI
import MeetingTranscriberKit

/// Transcript segments synced to audio (BR-8): click a timecode to seek, the
/// active segment highlights as playback advances, and each segment's speaker can
/// be reassigned (single-segment scope — BR-7). Per-segment language badge shown
/// when present.
struct TranscriptTabView: View {
    @ObservedObject var store: MeetingDetailStore
    let transcript: Transcript
    let speakers: [String: String]
    let language: String
    @ObservedObject var audio: AudioPlaybackController
    var insights: [String: SegmentInsights.SegmentInsight] = [:]

    @State private var renaming: TranscriptSegment?
    @State private var customName = ""
    private let prefs = Preferences()

    private var colors: [String: String] {
        SpeakerColor.assignments(orderedSpeakerIds: SpeakerColor.orderedSpeakerIds(in: transcript.segments))
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(transcript.segments) { segment in
                        segmentRow(segment)
                            .id(segment.id)
                    }
                }
                .padding(16)
            }
            .onChange(of: activeSegmentId) { id in
                guard let id else { return }
                withAnimation { proxy.scrollTo(id, anchor: .center) }
            }
        }
        .sheet(item: $renaming) { segment in
            renameSheet(segment)
        }
    }

    private func segmentRow(_ segment: TranscriptSegment) -> some View {
        let isActive = segment.id == activeSegmentId
        let color = Color(hex: colors[segment.speaker] ?? "#888888")
        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                speakerMenu(segment, color: color)
                Button(Formatters.timecode(segment.start)) { audio.seek(to: segment.start); audio.play() }
                    .buttonStyle(.borderless).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                if let badge = languageBadge(segment) {
                    Text(badge).font(.caption2).padding(.horizontal, 5).padding(.vertical, 1)
                        .background(.quaternary, in: Capsule())
                }
                Spacer()
            }
            Text(segment.text)
                .textSelection(.enabled)
                .foregroundStyle(isActive ? .primary : .secondary)
            if let insight = insights[segment.id] {
                SegmentInsightsRow(insight: insight, text: segment.text)
            }
        }
        .padding(10)
        .background(isActive ? color.opacity(0.12) : .clear, in: RoundedRectangle(cornerRadius: 8))
    }

    private func speakerMenu(_ segment: TranscriptSegment, color: Color) -> some View {
        let name = SpeakerColor.displayName(for: segment.speaker, speakers: speakers)
        return Menu {
            if !prefs.recentSpeakerNames.isEmpty {
                Section("Recent") {
                    ForEach(prefs.recentSpeakerNames, id: \.self) { recent in
                        Button(recent) { reassign(segment, to: recent) }
                    }
                }
            }
            Button("Rename…") { customName = ""; renaming = segment }
        } label: {
            HStack(spacing: 5) {
                Circle().fill(color).frame(width: 8, height: 8)
                Text(name).font(.caption.weight(.medium))
                    .foregroundStyle(SpeakerColor.isUnidentified(speakers[segment.speaker]) ? .orange : color)
            }
        }
        .menuStyle(.borderlessButton).fixedSize()
    }

    private func renameSheet(_ segment: TranscriptSegment) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Rename speaker for this segment").font(.headline)
            TextField("Name", text: $customName).textFieldStyle(.roundedBorder).frame(width: 260)
            HStack {
                Button("Cancel") { renaming = nil }
                Spacer()
                Button("Save") {
                    reassign(segment, to: customName)
                    renaming = nil
                }
                .keyboardShortcut(.defaultAction)
                .disabled(customName.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(20).frame(width: 320)
    }

    private func languageBadge(_ segment: TranscriptSegment) -> String? {
        guard let code = segment.language ?? (language.isEmpty ? nil : language), code != "auto" else { return nil }
        return Languages.name(for: code)
    }

    private var activeSegmentId: String? {
        let t = audio.currentTime
        return transcript.segments.first { $0.start <= t && t < $0.end }?.id
    }

    private func reassign(_ segment: TranscriptSegment, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        Task {
            do {
                try await store.client.renameSegmentSpeaker(id: store.meetingId, segmentId: segment.id, speakerName: trimmed)
                prefs.addRecentSpeakerName(trimmed)
                await store.reloadDetail()
            } catch {
                store.setError((error as? APIError)?.userMessage ?? "Could not rename the speaker.")
            }
        }
    }
}

/// Inline per-segment insight badges (emotion, prosody, interaction, mismatch).
private struct SegmentInsightsRow: View {
    let insight: SegmentInsights.SegmentInsight
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            if let emotion = insight.emotion {
                Text(SegmentInsights.label(for: emotion.primaryEmotion) + (emotion.lowConfidence ? "?" : ""))
                    .font(.caption2.weight(.medium))
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(.quaternary, in: Capsule())
                    .foregroundStyle(emotion.lowConfidence ? .secondary : .primary)
                    .help("Tone: \(SegmentInsights.label(for: emotion.primaryEmotion)) (confidence \(Int(emotion.confidence * 100))%)")
            }
            if let prosody = insight.prosody {
                ProsodyIndicator(prosody: prosody)
            }
            if SegmentInsights.isWordToneMismatch(emotion: insight.emotion, text: text) {
                Image(systemName: "exclamationmark.bubble")
                    .font(.caption2).foregroundStyle(.orange)
                    .help("Word/tone mismatch: agreement wording over a frustrated or uncertain tone.")
            }
            if let interaction = insight.interaction,
               interaction.precededByInterruption || interaction.followedByInterruption || interaction.hesitationBefore > 0 {
                Image(systemName: "arrow.triangle.2.circlepath")
                    .font(.caption2).foregroundStyle(.secondary)
                    .help(interactionTooltip(interaction))
            }
        }
    }

    private func interactionTooltip(_ interaction: SegmentInteraction) -> String {
        var parts: [String] = []
        if interaction.precededByInterruption { parts.append("interrupted") }
        if interaction.followedByInterruption { parts.append("interrupts next") }
        if interaction.hesitationBefore > 0 { parts.append(String(format: "%.1fs pause before", interaction.hesitationBefore)) }
        return parts.joined(separator: ", ")
    }
}

/// Three-bar prosody indicator (volume / pitch / rate), tooltip with details.
private struct ProsodyIndicator: View {
    let prosody: ProsodyAnnotation

    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<3, id: \.self) { _ in
                RoundedRectangle(cornerRadius: 1).fill(.tertiary).frame(width: 3, height: 8)
            }
        }
        .help(tooltip)
    }

    private var tooltip: String {
        String(format: "Volume %.2f · Pitch %.0f Hz · Rate %.0f wpm · Pause %.2f",
               prosody.volumeMean, prosody.pitchMean, prosody.speakingRate, prosody.pauseRatio)
    }
}
