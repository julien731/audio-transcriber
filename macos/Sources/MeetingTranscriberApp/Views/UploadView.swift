import SwiftUI
import UniformTypeIdentifiers
import MeetingTranscriberKit

/// Upload form (BR-7). Picks an audio file, collects options, validates on the
/// client (mirroring the server), and starts a transcription. On success the new
/// meeting is selected and polled in the detail pane (BR-9 auto-navigate).
struct UploadView: View {
    let client: APIClient
    let onCreated: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    @State private var fileURL: URL?
    @State private var fileByteCount = 0
    @State private var title = ""
    @State private var meetingType: MeetingType = .other
    @State private var selectedLanguages: Set<String> = []
    @State private var numSpeakers = "auto"
    @State private var preprocess = true
    @State private var audioAnalysis = false
    @State private var context = ""

    @State private var validationError: String?
    @State private var submitting = false
    @State private var importing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("New Meeting").font(.title2.weight(.semibold)).padding([.top, .horizontal], 20)
            Form {
                Section {
                    HStack {
                        Button("Choose Audio File…") { importing = true }
                        if let fileURL {
                            Text(fileURL.lastPathComponent).lineLimit(1).foregroundStyle(.secondary)
                        }
                    }
                    if let validationError {
                        Text(validationError).font(.caption).foregroundStyle(.red)
                    }
                }
                Section {
                    TextField("Title (optional)", text: $title)
                    Picker("Type", selection: $meetingType) {
                        ForEach(MeetingType.allCases, id: \.self) { Text($0.displayName).tag($0) }
                    }
                    TextField("Speakers (number or “auto”)", text: $numSpeakers)
                    Toggle("Preprocess audio", isOn: $preprocess)
                    Toggle("Run audio analysis (emotion, prosody, interactions)", isOn: $audioAnalysis)
                }
                Section("Expected languages (optional)") {
                    LanguageMultiSelect(selected: $selectedLanguages)
                }
                Section("Context (optional)") {
                    TextEditor(text: $context).frame(minHeight: 60)
                }
            }
            .formStyle(.grouped)
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                Button("Start Transcription") { Task { await submit() } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(fileURL == nil || submitting)
                if submitting { ProgressView().controlSize(.small) }
            }
            .padding(20)
        }
        .frame(width: 520, height: 560)
        .fileImporter(isPresented: $importing,
                      allowedContentTypes: [.audio, .mpeg4Movie, .movie, .mp3, .wav],
                      allowsMultipleSelection: false) { result in
            handleImport(result)
        }
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        guard case let .success(urls) = result, let url = urls.first else { return }
        let byteCount = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        if let failure = UploadValidation.validate(filename: url.lastPathComponent, byteCount: byteCount) {
            validationError = failure.message
            fileURL = nil
        } else {
            validationError = nil
            fileURL = url
            fileByteCount = byteCount
            if title.isEmpty { title = url.deletingPathExtension().lastPathComponent }
        }
    }

    private func submit() async {
        guard let fileURL else { return }
        submitting = true
        defer { submitting = false }
        let needsAccess = fileURL.startAccessingSecurityScopedResource()
        defer { if needsAccess { fileURL.stopAccessingSecurityScopedResource() } }
        guard let data = try? Data(contentsOf: fileURL) else {
            validationError = "Could not read the selected file."
            return
        }
        let upload = MeetingUpload(
            fileData: data,
            filename: fileURL.lastPathComponent,
            title: title,
            meetingType: meetingType,
            expectedLanguages: Languages.all.map(\.code).filter { selectedLanguages.contains($0) },
            numSpeakers: Int(numSpeakers.trimmingCharacters(in: .whitespaces)),
            preprocessAudio: preprocess,
            audioAnalysisEnabled: audioAnalysis,
            context: context
        )
        do {
            let start = try await client.createMeeting(upload)
            onCreated(start.meetingId)
            dismiss()
        } catch {
            validationError = (error as? APIError)?.userMessage ?? "Upload failed."
        }
    }
}

private struct LanguageMultiSelect: View {
    @Binding var selected: Set<String>

    private let columns = [GridItem(.adaptive(minimum: 110), spacing: 6)]

    var body: some View {
        LazyVGrid(columns: columns, alignment: .leading, spacing: 6) {
            ForEach(Languages.all) { language in
                Toggle(language.name, isOn: Binding(
                    get: { selected.contains(language.code) },
                    set: { on in if on { selected.insert(language.code) } else { selected.remove(language.code) } }
                ))
                .toggleStyle(.checkbox)
            }
        }
    }
}
