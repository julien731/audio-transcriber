import SwiftUI
import MeetingTranscriberKit

/// Analysis tab: request the fully-assembled prompt from the service (BR-10 —
/// assembly is server-side) and let the user copy it. Warns first when speakers
/// are still unnamed (web parity, feature #74).
struct AnalysisTabView: View {
    let client: APIClient
    let detail: MeetingDetail

    @State private var templateType: TemplateType
    @State private var meetingContext: String
    @State private var prompt: String?
    @State private var generating = false
    @State private var error: String?

    enum TemplateType: String, CaseIterable, Identifiable {
        case interview, sales, client, other, prototype
        var id: String { rawValue }
        var label: String { rawValue.capitalized }
    }

    init(client: APIClient, detail: MeetingDetail) {
        self.client = client
        self.detail = detail
        _templateType = State(initialValue: TemplateType(rawValue: detail.metadata.type.rawValue) ?? .other)
        _meetingContext = State(initialValue: detail.metadata.context)
    }

    private var unnamed: (unnamed: Int, total: Int) {
        SpeakerColor.unnamedSpeakers(in: detail.transcript?.segments ?? [], speakers: detail.metadata.speakers)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if unnamed.unnamed > 0 {
                    Label("\(unnamed.unnamed) of \(unnamed.total) speakers are still unnamed. The prompt will use raw labels like SPEAKER_00 until you rename them on the Transcript tab.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.callout).foregroundStyle(.orange)
                        .padding(10).background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
                }
                HStack {
                    Picker("Template", selection: $templateType) {
                        ForEach(TemplateType.allCases) { Text($0.label).tag($0) }
                    }
                    .frame(width: 220)
                    Spacer()
                    Button { Task { await generate() } } label: {
                        Label("Generate prompt", systemImage: "sparkles")
                    }
                    .disabled(generating)
                    if generating { ProgressView().controlSize(.small) }
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Meeting context (optional, overrides saved context)").font(.caption).foregroundStyle(.secondary)
                    TextEditor(text: $meetingContext).frame(height: 60).border(.quaternary)
                }
                if let error {
                    Text(error).font(.callout).foregroundStyle(.red)
                }
                if let prompt {
                    HStack {
                        Text("Assembled prompt").font(.headline)
                        Spacer()
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(prompt, forType: .string)
                        } label: { Label("Copy", systemImage: "doc.on.doc") }
                    }
                    Text(prompt)
                        .textSelection(.enabled)
                        .font(.body.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
                }
            }
            .padding(20)
        }
    }

    private func generate() async {
        generating = true
        defer { generating = false }
        error = nil
        do {
            let response = try await client.analysisPrompt(
                id: detail.metadata.id,
                templateType: templateType.rawValue,
                meetingContext: meetingContext
            )
            prompt = response.prompt
        } catch let apiError as APIError {
            // 409 (transcript not ready) surfaces the service's own message (EC-11).
            self.error = apiError.userMessage
        } catch {
            self.error = "Could not generate the prompt."
        }
    }
}
