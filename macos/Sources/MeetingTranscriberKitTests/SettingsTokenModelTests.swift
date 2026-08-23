import Foundation
import MeetingTranscriberKit

private func runBlockingS(_ operation: @escaping () async -> Void) {
    let semaphore = DispatchSemaphore(value: 0)
    Task { await operation(); semaphore.signal() }
    semaphore.wait()
}

private func stClient() -> APIClient {
    APIClient(baseURL: URL(string: "http://127.0.0.1:9998")!, session: MockURLProtocol.makeSession())
}

private func stStatusJSON(completed: Bool = false, present: Bool = false, diar: Bool = false,
                          state: String = "idle", progress: Int = 0, error: String? = nil) -> String {
    let errorField = error.map { "\"\($0)\"" } ?? "null"
    return """
    {"provisioning_completed":\(completed),"models_present":\(present),"whisper_model":"large-v3",
     "diarization_available":\(diar),"download_state":"\(state)","download_progress":\(progress),
     "download_error":\(errorField)}
    """
}

private func stDecode(_ json: String) -> ProvisioningStatus {
    try! JSONCoding.makeDecoder().decode(ProvisioningStatus.self, from: Data(json.utf8))
}

private func model(initial: ProvisioningStatus) -> SettingsTokenModel {
    SettingsTokenModel(controller: ProvisioningController(client: stClient(), initial: initial))
}

func runSettingsTokenModelTests() {
    suite("SettingsPhase mapping from SetupPhase") {
        expectEqual(SettingsTokenModel.settingsPhase(from: .completed(diarizationAvailable: true), editing: false),
                    .idle(diarizationAvailable: true), "completed → idle keeps diarization flag")
        expectEqual(SettingsTokenModel.settingsPhase(from: .downloading(progress: 42), editing: true),
                    .working(progress: 42), "downloading → working carries progress")
        expectEqual(SettingsTokenModel.settingsPhase(from: .failed(message: "boom"), editing: true),
                    .failed(message: "boom"), "failed → failed surfaces the message")
        expectEqual(SettingsTokenModel.settingsPhase(from: .enteringToken, editing: true),
                    .editing, "enteringToken while editing → editing")
        expectEqual(SettingsTokenModel.settingsPhase(from: .enteringToken, editing: false),
                    .idle(diarizationAvailable: false), "enteringToken not editing → idle disabled")
    }

    suite("initial phase reflects completed provisioning") {
        let m = model(initial: stDecode(stStatusJSON(completed: true, present: true, diar: true)))
        expectEqual(m.phase, .idle(diarizationAvailable: true), "starts idle with current diarization state")
        m.beginEditing()
        expectEqual(m.phase, .editing, "beginEditing reveals the token field")
    }

    suite("save(non-empty) sets token, downloads, then completes on refresh") {
        MockURLProtocol.handler = { request, _ in
            if request.url!.path.hasSuffix("/provisioning/token") {
                return .init(status: 200, body: Data(stStatusJSON(diar: true).utf8))
            }
            if request.url!.path.hasSuffix("/provisioning/models") {
                return .init(status: 200, body: Data(stStatusJSON(state: "downloading", progress: 5).utf8))
            }
            // GET /provisioning (refresh) → completed with diarization enabled.
            return .init(status: 200, body: Data(stStatusJSON(completed: true, present: true, diar: true).utf8))
        }
        runBlockingS {
            let m = model(initial: stDecode(stStatusJSON(completed: true, present: true, diar: false)))
            m.beginEditing()
            let saved = await m.save(token: "hf_abc")
            expectEqual(saved, .working(progress: 5), "save → working (download started)")
            let refreshed = await m.refresh()
            expectEqual(refreshed, .idle(diarizationAvailable: true), "refresh → idle, diarization enabled")
        }
    }

    suite("save(empty) clears the token and disables diarization") {
        MockURLProtocol.handler = { request, body in
            if request.url!.path.hasSuffix("/provisioning/token") {
                let sent = String(data: body, encoding: .utf8) ?? ""
                expect(sent.contains("\"hf_token\":\"\""), "posts an EMPTY token to clear it")
                return .init(status: 200, body: Data(stStatusJSON(diar: false).utf8))
            }
            // startModelDownload → completed Whisper-only.
            return .init(status: 200, body: Data(stStatusJSON(completed: true, present: true, diar: false).utf8))
        }
        runBlockingS {
            let m = model(initial: stDecode(stStatusJSON(completed: true, present: true, diar: true)))
            let saved = await m.save(token: "")
            expectEqual(saved, .idle(diarizationAvailable: false), "cleared → idle, diarization disabled")
        }
    }

    suite("save(whitespace-only) is treated as a clear") {
        MockURLProtocol.handler = { request, body in
            if request.url!.path.hasSuffix("/provisioning/token") {
                let sent = String(data: body, encoding: .utf8) ?? ""
                expect(sent.contains("\"hf_token\":\"\""), "whitespace trims to an EMPTY token")
                return .init(status: 200, body: Data(stStatusJSON(diar: false).utf8))
            }
            return .init(status: 200, body: Data(stStatusJSON(completed: true, present: true, diar: false).utf8))
        }
        runBlockingS {
            let m = model(initial: stDecode(stStatusJSON(completed: true, present: true, diar: true)))
            let saved = await m.save(token: "   ")
            expectEqual(saved, .idle(diarizationAvailable: false), "whitespace-only save disables diarization")
        }
    }

    suite("save failure surfaces a generic message, never a token claim") {
        MockURLProtocol.handler = nil // forces a transport error
        runBlockingS {
            let m = model(initial: stDecode(stStatusJSON(completed: true, present: true, diar: false)))
            let saved = await m.save(token: "whatever")
            if case let .failed(message) = saved {
                expect(!message.lowercased().contains("token"), "message does not claim the token was rejected")
                expect(!message.lowercased().contains("reject"), "message does not say rejected")
            } else {
                expect(false, "expected a failed phase, got \(saved)")
            }
        }
    }

    suite("cancelEditing restores the known status") {
        let m = model(initial: stDecode(stStatusJSON(completed: true, present: true, diar: true)))
        m.beginEditing()
        expectEqual(m.phase, .editing, "editing")
        m.cancelEditing(diarizationAvailable: true)
        expectEqual(m.phase, .idle(diarizationAvailable: true), "cancel returns to idle without saving")
    }
}
