import Foundation
import MeetingTranscriberKit

/// Run an async operation to completion from the synchronous harness. Safe here
/// because MockURLProtocol services requests off the calling thread.
private func runBlocking(_ operation: @escaping () async -> Void) {
    let semaphore = DispatchSemaphore(value: 0)
    Task { await operation(); semaphore.signal() }
    semaphore.wait()
}

private func makeClient() -> APIClient {
    APIClient(baseURL: URL(string: "http://127.0.0.1:9999")!, session: MockURLProtocol.makeSession())
}

private func stub(_ status: Int, _ json: String) {
    MockURLProtocol.handler = { _, _ in .init(status: status, body: Data(json.utf8)) }
}

func runAPIClientTests() {
    suite("APIClient.listMeetings") {
        stub(200, #"[{"id":"m1","title":"A","type":"other","created_at":"2026-08-22T10:00:00","duration_seconds":null,"status":"ready"}]"#)
        runBlocking {
            do {
                let meetings = try await makeClient().listMeetings()
                expectEqual(meetings.count, 1, "one meeting")
                expectEqual(meetings.first?.id, "m1", "id")
                expectEqual(MockURLProtocol.lastRequest?.httpMethod, "GET", "GET method")
                expect(MockURLProtocol.lastRequest?.url?.path == "/api/meetings", "path")
            } catch { expect(false, "unexpected error: \(error)") }
        }
    }

    suite("APIClient error mapping") {
        let cases: [(Int, APIError)] = [
            (404, .notFound("Meeting not found")),
            (409, .conflict("Transcript is not ready yet")),
            (503, .modelsUnavailable("models missing")),
            (400, .badRequest("File too large (max 500MB)")),
            (500, .server(status: 500, detail: "boom")),
        ]
        for (status, expected) in cases {
            let detail = expected.userMessage
            stub(status, #"{"detail":"\#(detail)"}"#)
            runBlocking {
                do {
                    _ = try await makeClient().meeting(id: "m1")
                    expect(false, "expected error for status \(status)")
                } catch let error as APIError {
                    expectEqual(error, expected, "maps \(status)")
                } catch { expect(false, "wrong error type for \(status): \(error)") }
            }
        }
    }

    suite("APIClient.createMeeting multipart") {
        stub(200, #"{"meeting_id":"m9","job_id":"j9"}"#)
        let upload = MeetingUpload(
            fileData: Data("AUDIO".utf8), filename: "rec.mp3", title: "Weekly",
            meetingType: .sales, expectedLanguages: ["en", "fr"], numSpeakers: 3,
            preprocessAudio: false, audioAnalysisEnabled: true, context: "notes"
        )
        runBlocking {
            do {
                let result = try await makeClient().createMeeting(upload)
                expectEqual(result.meetingId, "m9", "meeting id")
                expectEqual(result.jobId, "j9", "job id")
                let contentType = MockURLProtocol.lastRequest?.value(forHTTPHeaderField: "Content-Type") ?? ""
                expect(contentType.hasPrefix("multipart/form-data; boundary="), "multipart content type")
                let body = String(data: MockURLProtocol.lastBody, encoding: .utf8) ?? ""
                expect(body.contains("name=\"file\"; filename=\"rec.mp3\""), "file part present")
                expect(body.contains("name=\"meeting_type\"\r\n\r\nsales"), "meeting_type field")
                expect(body.contains("name=\"expected_languages\"\r\n\r\nen"), "language en repeated")
                expect(body.contains("name=\"expected_languages\"\r\n\r\nfr"), "language fr repeated")
                expect(body.contains("name=\"num_speakers\"\r\n\r\n3"), "num_speakers")
                expect(body.contains("name=\"preprocess_audio\"\r\n\r\nfalse"), "preprocess false")
                expect(body.contains("name=\"audio_analysis_enabled\"\r\n\r\ntrue"), "audio analysis true")
                expect(body.contains("AUDIO"), "raw audio bytes present")
            } catch { expect(false, "unexpected error: \(error)") }
        }
    }

    suite("APIClient.updateMeeting PATCH json") {
        stub(200, #"{"id":"m1","title":"Renamed","created_at":"2026-08-22T10:00:00"}"#)
        runBlocking {
            do {
                let meta = try await makeClient().updateMeeting(id: "m1", update: MeetingUpdate(title: "Renamed"))
                expectEqual(meta.title, "Renamed", "returns updated meta")
                expectEqual(MockURLProtocol.lastRequest?.httpMethod, "PATCH", "PATCH")
                let body = String(data: MockURLProtocol.lastBody, encoding: .utf8) ?? ""
                expect(body.contains("\"title\":\"Renamed\""), "title in body")
                expect(!body.contains("\"type\""), "nil fields omitted")
            } catch { expect(false, "unexpected error: \(error)") }
        }
    }

    suite("APIClient.setToken json") {
        stub(200, #"{"provisioning_completed":false,"models_present":false,"whisper_model":"large-v3","diarization_available":true,"download_state":"idle","download_progress":0,"download_error":null}"#)
        runBlocking {
            do {
                let status = try await makeClient().setToken("hf_abc")
                expectEqual(status.diarizationAvailable, true, "diarization available")
                let body = String(data: MockURLProtocol.lastBody, encoding: .utf8) ?? ""
                expect(body.contains("\"hf_token\":\"hf_abc\""), "token snake_cased in body")
            } catch { expect(false, "unexpected error: \(error)") }
        }
    }

    suite("APIClient.analysisPrompt query") {
        stub(200, #"{"prompt":"ready"}"#)
        runBlocking {
            do {
                let prompt = try await makeClient().analysisPrompt(id: "m1", templateType: "sales", meetingContext: "ctx")
                expectEqual(prompt.prompt, "ready", "prompt body")
                let query = MockURLProtocol.lastRequest?.url?.query ?? ""
                expect(query.contains("template_type=sales"), "template_type query")
                expect(query.contains("meeting_context=ctx"), "meeting_context query")
            } catch { expect(false, "unexpected error: \(error)") }
        }
    }

    suite("APIClient.audioURL") {
        let url = makeClient().audioURL(id: "m1")
        expectEqual(url.path, "/api/meetings/m1/audio", "audio url path")
    }
}
