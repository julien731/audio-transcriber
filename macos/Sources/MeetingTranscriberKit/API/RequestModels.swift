import Foundation

/// PATCH body for meeting title/type/speakers/context. Nil fields are omitted
/// (synthesized `encodeIfPresent`), matching the partial-update endpoint.
public struct MeetingUpdate: Encodable, Equatable {
    public var title: String?
    public var type: MeetingType?
    public var speakers: [String: String]?
    public var context: String?

    public init(title: String? = nil, type: MeetingType? = nil,
                speakers: [String: String]? = nil, context: String? = nil) {
        self.title = title
        self.type = type
        self.speakers = speakers
        self.context = context
    }
}

public struct SegmentSpeakerUpdate: Encodable, Equatable {
    public let segmentId: String
    public let speakerName: String

    public init(segmentId: String, speakerName: String) {
        self.segmentId = segmentId
        self.speakerName = speakerName
    }
}

public struct TokenUpdate: Encodable, Equatable {
    public let hfToken: String
    public init(hfToken: String) { self.hfToken = hfToken }
}

/// Response of `POST /api/meetings` and `.../retry`.
public struct StartResponse: Decodable, Equatable {
    public let meetingId: String
    public let jobId: String
}

/// A pending upload assembled into a multipart/form-data body for `POST /meetings`.
public struct MeetingUpload {
    public var fileData: Data
    public var filename: String
    public var title: String
    public var meetingType: MeetingType
    public var expectedLanguages: [String]
    /// nil ⇒ "auto"; otherwise a fixed speaker count.
    public var numSpeakers: Int?
    public var preprocessAudio: Bool
    public var audioAnalysisEnabled: Bool
    public var context: String

    public init(fileData: Data, filename: String, title: String = "",
                meetingType: MeetingType = .other, expectedLanguages: [String] = [],
                numSpeakers: Int? = nil, preprocessAudio: Bool = true,
                audioAnalysisEnabled: Bool = false, context: String = "") {
        self.fileData = fileData
        self.filename = filename
        self.title = title
        self.meetingType = meetingType
        self.expectedLanguages = expectedLanguages
        self.numSpeakers = numSpeakers
        self.preprocessAudio = preprocessAudio
        self.audioAnalysisEnabled = audioAnalysisEnabled
        self.context = context
    }

    func multipartBody(boundary: String) -> Data {
        var body = Data()
        func boundaryLine() { body.append("--\(boundary)\r\n") }

        func field(_ name: String, _ value: String) {
            boundaryLine()
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            body.append("\(value)\r\n")
        }

        // File part first.
        boundaryLine()
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n")
        body.append("Content-Type: application/octet-stream\r\n\r\n")
        body.append(fileData)
        body.append("\r\n")

        field("title", title)
        field("meeting_type", meetingType.rawValue)
        // Repeat the field for each language (FastAPI list-from-form).
        for language in expectedLanguages { field("expected_languages", language) }
        field("num_speakers", numSpeakers.map(String.init) ?? "auto")
        field("preprocess_audio", preprocessAudio ? "true" : "false")
        field("audio_analysis_enabled", audioAnalysisEnabled ? "true" : "false")
        field("context", context)

        body.append("--\(boundary)--\r\n")
        return body
    }
}

private extension Data {
    mutating func append(_ string: String) {
        append(Data(string.utf8))
    }
}
