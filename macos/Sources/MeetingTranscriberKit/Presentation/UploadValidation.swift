import Foundation

/// Client-side upload validation mirroring the server's rules (meetings.py) so the
/// user gets immediate feedback before a request is sent (BR-7: upload + client
/// validation). The server re-validates; this is a UX shortcut, not the gate.
public enum UploadValidation {
    public static let allowedExtensions: Set<String> = ["mp3", "mp4", "m4a", "wav", "webm"]
    public static let maxUploadBytes = 500 * 1024 * 1024 // 500 MB

    public enum Failure: Equatable {
        case unsupportedFormat(String)
        case tooLarge

        public var message: String {
            switch self {
            case .unsupportedFormat:
                return "Unsupported file format. Allowed: mp3, mp4, m4a, wav, webm."
            case .tooLarge:
                return "File too large (max 500 MB)."
            }
        }
    }

    public static func fileExtension(of filename: String) -> String {
        (filename as NSString).pathExtension.lowercased()
    }

    /// Returns the first validation failure, or nil if the file is acceptable.
    public static func validate(filename: String, byteCount: Int) -> Failure? {
        let ext = fileExtension(of: filename)
        if !allowedExtensions.contains(ext) { return .unsupportedFormat(ext) }
        if byteCount > maxUploadBytes { return .tooLarge }
        return nil
    }
}

/// Languages the uploader can select (mirrors meetings.py SUPPORTED_LANGUAGES),
/// with display names for the picker. 0/1 selected ⇒ single-language pipeline;
/// 2+ ⇒ multilingual (server-side routing).
public struct SupportedLanguage: Equatable, Identifiable {
    public let code: String
    public let name: String
    public var id: String { code }
}

public enum Languages {
    public static let all: [SupportedLanguage] = [
        ("en", "English"), ("fr", "French"), ("de", "German"), ("es", "Spanish"),
        ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("ja", "Japanese"),
        ("zh", "Chinese"), ("ko", "Korean"), ("ru", "Russian"), ("th", "Thai"),
        ("ar", "Arabic"), ("hi", "Hindi"), ("tr", "Turkish"), ("pl", "Polish"),
        ("vi", "Vietnamese"), ("id", "Indonesian"),
    ].map { SupportedLanguage(code: $0.0, name: $0.1) }

    public static func name(for code: String) -> String {
        all.first { $0.code == code }?.name ?? code.uppercased()
    }
}
