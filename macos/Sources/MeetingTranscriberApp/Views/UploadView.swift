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
    @State private var numSpeakers: Int?
    @State private var preprocess = true
    @State private var audioAnalysis = false
    @State private var context = ""

    @State private var validationError: String?
    @State private var submitting = false
    @State private var importing = false
    @State private var isDropTargeted = false
    /// nil while the availability check is in flight or has failed — we render no
    /// warning in that case to avoid a false alarm.
    @State private var diarizationAvailable: Bool?
    @State private var showingSettings = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("New Meeting").font(.title2.weight(.semibold)).padding([.top, .horizontal], 20)
            Form {
                if diarizationAvailable == false {
                    Section {
                        diarizationWarning
                            .listRowInsets(EdgeInsets())
                            .listRowBackground(Color.clear)
                    }
                }
                Section {
                    dropZone
                        .listRowInsets(EdgeInsets())
                        .listRowBackground(Color.clear)
                    if let validationError {
                        Text(validationError).font(.caption).foregroundStyle(.red)
                    }
                }
                Section {
                    TextField("Title (optional)", text: $title)
                    Picker("Type", selection: $meetingType) {
                        ForEach(MeetingType.allCases, id: \.self) { Text($0.displayName).tag($0) }
                    }
                    Picker("Speakers", selection: $numSpeakers) {
                        Text("Auto").tag(Int?.none)
                        ForEach(1...10, id: \.self) { count in
                            Text("\(count)").tag(Int?.some(count))
                        }
                    }
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
        .task { await refreshDiarization() }
        // Nested over this upload sheet (macOS 13-compatible; no SettingsLink).
        // Re-check on dismissal so the warning clears once a token is added.
        .sheet(isPresented: $showingSettings, onDismiss: { Task { await refreshDiarization() } }) {
            SettingsSheet(client: client) { showingSettings = false }
        }
    }

    /// Callout shown when speaker diarization is disabled (no HuggingFace token),
    /// so the user isn't surprised by a single-speaker transcript. Links to the
    /// Settings sheet where the token is set.
    private var diarizationWarning: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 4) {
                Text("Speaker diarization is disabled").font(.callout.weight(.semibold))
                Text("This meeting will be transcribed as a single speaker. Add a HuggingFace token in Settings to separate speakers.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Button("Open Settings…") { showingSettings = true }
                    .buttonStyle(.link)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
    }

    /// Read the current diarization availability from provisioning (the same
    /// signal Settings uses). A failed read leaves the state nil so no warning is
    /// shown rather than a spurious one.
    private func refreshDiarization() async {
        diarizationAvailable = try? await client.provisioning().diarizationAvailable
    }

    /// Drag-and-drop target that mirrors the "Choose Audio File…" picker. Dropping a
    /// file runs the same validation as the importer, so both paths share `accept(url:)`.
    private var dropZone: some View {
        VStack(spacing: 8) {
            Image(systemName: fileURL == nil ? "waveform" : "checkmark.circle.fill")
                .font(.system(size: 28))
                .foregroundStyle(fileURL == nil ? (isDropTargeted ? Color.accentColor : .secondary) : .green)
            if let fileURL {
                Text(fileURL.lastPathComponent)
                    .lineLimit(1).truncationMode(.middle)
                Button("Choose a different file…") { importing = true }
                    .buttonStyle(.link)
            } else {
                Text("Drag & drop an audio file here").font(.callout)
                Text("or").font(.caption).foregroundStyle(.secondary)
                Button("Choose Audio File…") { importing = true }
                Text("mp3, mp4, m4a, wav, webm").font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isDropTargeted ? Color.accentColor.opacity(0.08) : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(
                    isDropTargeted ? Color.accentColor : Color.secondary.opacity(0.35),
                    style: StrokeStyle(lineWidth: 1.5, dash: [6, 4])
                )
        )
        .animation(.easeInOut(duration: 0.15), value: isDropTargeted)
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first, url.isFileURL else { return false }
            accept(url: url)
            return true
        } isTargeted: { isDropTargeted = $0 }
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        guard case let .success(urls) = result, let url = urls.first else { return }
        accept(url: url)
    }

    /// Validates a picked or dropped file and updates form state. Reads the file size
    /// without security-scoped access, which is safe because the app is not sandboxed.
    private func accept(url: URL) {
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
            numSpeakers: numSpeakers,
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
