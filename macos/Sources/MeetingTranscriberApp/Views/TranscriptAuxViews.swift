import SwiftUI
import MeetingTranscriberKit

/// Plain Text tab: the assembled `[time] Speaker: text` transcript, copyable.
struct PlainTextTabView: View {
    let text: String

    var body: some View {
        VStack(alignment: .trailing, spacing: 8) {
            HStack {
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                } label: { Label("Copy", systemImage: "doc.on.doc") }
            }
            ScrollView {
                Text(text.isEmpty ? "No transcript." : text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .font(.body.monospaced())
            }
        }
        .padding(16)
    }
}

/// Meeting-wide speaker rename (BR-7). Edits the `speakers` map and PATCHes it.
struct SpeakerEditorView: View {
    let client: APIClient
    let meetingId: String
    let speakers: [String: String]
    let onSaved: () -> Void
    @Environment(\.dismiss) private var dismiss

    @State private var names: [String: String]
    @State private var saving = false
    @State private var error: String?
    private let prefs = Preferences()

    init(client: APIClient, meetingId: String, speakers: [String: String], onSaved: @escaping () -> Void) {
        self.client = client
        self.meetingId = meetingId
        self.speakers = speakers
        self.onSaved = onSaved
        _names = State(initialValue: speakers)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Rename speakers").font(.headline)
            if names.isEmpty {
                Text("No named speakers yet. Rename individual segments on the Transcript tab.")
                    .foregroundStyle(.secondary)
            }
            ForEach(names.keys.sorted(), id: \.self) { id in
                HStack {
                    Text(id).font(.caption.monospaced()).foregroundStyle(.secondary).frame(width: 110, alignment: .leading)
                    TextField("Name", text: Binding(
                        get: { names[id] ?? "" },
                        set: { names[id] = $0 }
                    )).textFieldStyle(.roundedBorder)
                }
            }
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                Button("Save") { Task { await save() } }.keyboardShortcut(.defaultAction).disabled(saving)
            }
        }
        .padding(20).frame(width: 420)
    }

    private func save() async {
        saving = true; defer { saving = false }
        do {
            _ = try await client.updateMeeting(id: meetingId, update: MeetingUpdate(speakers: names))
            names.values.forEach { prefs.addRecentSpeakerName($0) }
            onSaved(); dismiss()
        } catch {
            self.error = (error as? APIError)?.userMessage ?? "Could not save speakers."
        }
    }
}

/// Edit title / type / context (BR-7). PATCHes the meeting.
struct MeetingEditView: View {
    let client: APIClient
    let metadata: MeetingMetadata
    let onSaved: () -> Void
    @Environment(\.dismiss) private var dismiss

    @State private var title: String
    @State private var type: MeetingType
    @State private var context: String
    @State private var saving = false
    @State private var error: String?

    init(client: APIClient, metadata: MeetingMetadata, onSaved: @escaping () -> Void) {
        self.client = client
        self.metadata = metadata
        self.onSaved = onSaved
        _title = State(initialValue: metadata.title)
        _type = State(initialValue: metadata.type)
        _context = State(initialValue: metadata.context)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Edit meeting").font(.headline)
            TextField("Title", text: $title).textFieldStyle(.roundedBorder)
            Picker("Type", selection: $type) {
                ForEach(MeetingType.allCases, id: \.self) { Text($0.displayName).tag($0) }
            }
            Text("Context").font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $context).frame(height: 100).border(.quaternary)
            if let error { Text(error).font(.caption).foregroundStyle(.red) }
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                Button("Save") { Task { await save() } }.keyboardShortcut(.defaultAction).disabled(saving)
            }
        }
        .padding(20).frame(width: 420)
    }

    private func save() async {
        saving = true; defer { saving = false }
        do {
            _ = try await client.updateMeeting(id: metadata.id,
                                               update: MeetingUpdate(title: title, type: type, context: context))
            onSaved(); dismiss()
        } catch {
            self.error = (error as? APIError)?.userMessage ?? "Could not save changes."
        }
    }
}

/// Overview tab — filled in slice 8 (ported aggregations).
struct OverviewTabView: View {
    let detail: MeetingDetail
    var body: some View {
        Text("Overview arrives in slice 8.").foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// Analysis tab — filled in slice 9 (assembled prompt).
struct AnalysisTabView: View {
    let client: APIClient
    let meetingId: String
    let meetingType: MeetingType
    var body: some View {
        Text("Analysis arrives in slice 9.").foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
