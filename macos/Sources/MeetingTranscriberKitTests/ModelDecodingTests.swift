import Foundation
import MeetingTranscriberKit

private let decoder = JSONCoding.makeDecoder()

private func decode<T: Decodable>(_ type: T.Type, _ json: String) -> T? {
    try? decoder.decode(T.self, from: Data(json.utf8))
}

func runModelDecodingTests() {
    suite("MeetingSummary list") {
        let json = """
        [{"id":"m1","title":"Standup","type":"other","created_at":"2026-08-22T10:31:59.123456",
          "duration_seconds":123.5,"status":"ready"},
         {"id":"m2","title":"No duration","type":"sales","created_at":"2026-08-22T09:00:00",
          "duration_seconds":null,"status":"processing"}]
        """
        let list = decode([MeetingSummary].self, json)
        expectEqual(list?.count, 2, "decodes two rows")
        expectEqual(list?[0].id, "m1", "id")
        expectEqual(list?[0].type, .other, "type")
        expectEqual(list?[0].status, .ready, "status")
        expectEqual(list?[0].durationSeconds, 123.5, "duration")
        expectNil(list?[1].durationSeconds, "null duration → nil")
        expect(list?[0].createdAt != nil, "parses fractional-second date")
    }

    suite("MeetingDetail full") {
        let json = """
        {"metadata":{"id":"m1","title":"Interview","type":"interview",
          "created_at":"2026-08-22T10:31:59","duration_seconds":600.0,"audio_filename":"audio.mp3",
          "status":"ready","language":"en","expected_languages":["en","fr"],"num_speakers":2,
          "preprocess_audio":true,"audio_analysis_enabled":true,"audio_analysis_status":"completed",
          "job_id":"j1","speakers":{"SPEAKER_00":"Alice"},"error":null,"context":"hiring"},
         "transcript":{"segments":[
            {"id":"s1","start":0.0,"end":2.0,"speaker":"SPEAKER_00","text":"Hi","language":"en"},
            {"id":"s2","start":2.0,"end":4.0,"speaker":"SPEAKER_01","text":"Bonjour","language":"fr"}],
          "language":"en"},
         "audio_analysis":null}
        """
        let detail = decode(MeetingDetail.self, json)
        expectEqual(detail?.metadata.title, "Interview", "metadata title")
        expectEqual(detail?.metadata.expectedLanguages, ["en", "fr"], "expected languages")
        expectEqual(detail?.metadata.numSpeakers, 2, "num speakers")
        expectEqual(detail?.metadata.jobId, "j1", "job id present in metadata")
        expectEqual(detail?.metadata.speakers["SPEAKER_00"], "Alice", "speaker map")
        expectEqual(detail?.transcript?.segments.count, 2, "two segments")
        expectEqual(detail?.transcript?.segments[1].language, "fr", "per-segment language")
        expectNil(detail?.audioAnalysis, "null audio_analysis → nil")
    }

    suite("MeetingMetadata minimal (defaults applied)") {
        // A sparse payload: only required-ish fields present.
        let json = """
        {"id":"m3","title":"Sparse","created_at":"2026-08-22T10:31:59"}
        """
        let meta = decode(MeetingMetadata.self, json)
        expectEqual(meta?.type, .other, "type defaults to other")
        expectEqual(meta?.status, .processing, "status defaults to processing")
        expectEqual(meta?.language, "auto", "language defaults to auto")
        expectEqual(meta?.preprocessAudio, true, "preprocess defaults true")
        expectEqual(meta?.audioAnalysisEnabled, false, "audio analysis defaults false")
        expectEqual(meta?.context, "", "context defaults empty")
        expect(meta?.speakers.isEmpty == true, "speakers default empty")
    }

    suite("JobInfo") {
        let json = """
        {"id":"j1","meeting_id":"m1","status":"processing","progress":42,"stage":"transcribing",
         "error":null,"created_at":"2026-08-22T10:31:59","updated_at":"2026-08-22T10:32:10"}
        """
        let job = decode(JobInfo.self, json)
        expectEqual(job?.status, .processing, "status")
        expectEqual(job?.progress, 42, "progress")
        expectEqual(job?.stage, "transcribing", "stage")
        expectEqual(job?.meetingId, "m1", "meeting id")
    }

    suite("ProvisioningStatus") {
        let json = """
        {"provisioning_completed":false,"models_present":false,"whisper_model":"large-v3",
         "diarization_available":false,"download_state":"failed","download_progress":30,
         "download_error":"connection reset"}
        """
        let status = decode(ProvisioningStatus.self, json)
        expectEqual(status?.downloadState, .failed, "download state")
        expectEqual(status?.downloadProgress, 30, "progress")
        expectEqual(status?.downloadError, "connection reset", "error surfaced")
        expectEqual(status?.whisperModel, "large-v3", "model")
    }

    suite("AudioAnalysis degraded / partial") {
        // Emotion completed, prosody unavailable, dominant-speaker limitation set.
        let json = """
        {"status":"completed","reason":null,
         "emotion_status":"completed",
         "emotions":[{"segment_id":"s1","speaker":"SPEAKER_00","start":0.0,"end":2.0,
            "primary_emotion":"frustrated","confidence":0.8,"emotion_scores":{"frustrated":0.8},
            "low_confidence":false}],
         "emotion_unavailable":[{"segment_id":"s2","reason":"unsupported language"}],
         "prosody_status":"unavailable","prosody_reason":"no pitch",
         "interactions":[{"event_type":"long_pause","timestamp":5.0,"speaker_a":"SPEAKER_00",
            "speaker_b":"SPEAKER_01","duration":1.2}],
         "dominant_speaker_limitation":true}
        """
        let analysis = decode(AudioAnalysis.self, json)
        expectEqual(analysis?.status, .completed, "overall status")
        expectEqual(analysis?.emotions.first?.primaryEmotion, .frustrated, "emotion category")
        expectEqual(analysis?.emotionUnavailable.first?.reason, "unsupported language", "emotion-unavailable")
        expectEqual(analysis?.prosodyStatus, .unavailable, "prosody unavailable")
        expect(analysis?.prosody.isEmpty == true, "missing prosody list defaults empty")
        expectEqual(analysis?.interactions.first?.eventType, .longPause, "long_pause maps")
        expectEqual(analysis?.interactions.first?.context, "", "missing context defaults empty")
        expectEqual(analysis?.dominantSpeakerLimitation, true, "dominant speaker limitation")
    }

    suite("Enum forward-compat") {
        // A value the client doesn't know must not hard-fail.
        let json = """
        {"status":"completed","emotions":[{"segment_id":"s","speaker":"x","start":0,"end":1,
          "primary_emotion":"ecstatic","confidence":0.5,"emotion_scores":{},"low_confidence":true}]}
        """
        let analysis = decode(AudioAnalysis.self, json)
        expectEqual(analysis?.emotions.first?.primaryEmotion, .unknown, "unknown emotion → .unknown")
    }

    suite("AnalysisPromptResponse") {
        let prompt = decode(AnalysisPromptResponse.self, #"{"prompt":"Analyze this."}"#)
        expectEqual(prompt?.prompt, "Analyze this.", "prompt body")
    }
}
