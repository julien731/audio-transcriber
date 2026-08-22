import Foundation
import MeetingTranscriberKit

private func runBlockingP(_ operation: @escaping () async -> Void) {
    let semaphore = DispatchSemaphore(value: 0)
    Task { await operation(); semaphore.signal() }
    semaphore.wait()
}

private func client() -> APIClient {
    APIClient(baseURL: URL(string: "http://127.0.0.1:9999")!, session: MockURLProtocol.makeSession())
}

private func provisioningJSON(completed: Bool = false, present: Bool = false, diar: Bool = false,
                              state: String = "idle", progress: Int = 0, error: String? = nil) -> String {
    let errorField = error.map { "\"\($0)\"" } ?? "null"
    return """
    {"provisioning_completed":\(completed),"models_present":\(present),"whisper_model":"large-v3",
     "diarization_available":\(diar),"download_state":"\(state)","download_progress":\(progress),
     "download_error":\(errorField)}
    """
}

func runProvisioningControllerTests() {
    suite("SetupPhase.from status mapping") {
        expectEqual(SetupPhase.from(decodeStatus(provisioningJSON())), .enteringToken, "idle → enteringToken")
        expectEqual(SetupPhase.from(decodeStatus(provisioningJSON(state: "downloading", progress: 40))),
                    .downloading(progress: 40), "downloading maps progress")
        expectEqual(SetupPhase.from(decodeStatus(provisioningJSON(state: "failed", error: "connection reset"))),
                    .failed(message: "connection reset"), "failed surfaces error")
        expectEqual(SetupPhase.from(decodeStatus(provisioningJSON(completed: true, present: true, diar: true))),
                    .completed(diarizationAvailable: true), "completed keeps diarization flag")
    }

    suite("submit(token) starts download") {
        // setToken → then startModelDownload returns downloading.
        var call = 0
        MockURLProtocol.handler = { request, _ in
            call += 1
            if request.url!.path.hasSuffix("/provisioning/token") {
                return .init(status: 200, body: Data(provisioningJSON(diar: true).utf8))
            }
            return .init(status: 200, body: Data(provisioningJSON(state: "downloading", progress: 5).utf8))
        }
        runBlockingP {
            let controller = ProvisioningController(client: client(), initial: decodeStatus(provisioningJSON()))
            let phase = await controller.submit(token: "hf_abc")
            expectEqual(phase, .downloading(progress: 5), "submit → downloading")
            expect(call == 2, "called setToken then startModelDownload")
        }
    }

    suite("failed download → continue without diarization clears token") {
        // First the download is failed; continueWithoutDiarization posts an empty
        // token then restarts, and provisioning completes Whisper-only.
        MockURLProtocol.handler = { request, body in
            if request.url!.path.hasSuffix("/provisioning/token") {
                let sent = String(data: body, encoding: .utf8) ?? ""
                expect(sent.contains("\"hf_token\":\"\""), "posts an EMPTY token")
                return .init(status: 200, body: Data(provisioningJSON(diar: false).utf8))
            }
            // startModelDownload → completed Whisper-only.
            return .init(status: 200, body: Data(provisioningJSON(completed: true, present: true, diar: false).utf8))
        }
        runBlockingP {
            let failed = decodeStatus(provisioningJSON(state: "failed", error: "401 unauthorized"))
            let controller = ProvisioningController(client: client(), initial: failed)
            expectEqual(controller.phase, .failed(message: "401 unauthorized"),
                        "starts in failed with the GENERIC service error (never 'token rejected')")
            let phase = await controller.continueWithoutDiarization()
            expectEqual(phase, .completed(diarizationAvailable: false), "Whisper-only completion, diarization disabled")
        }
    }

    suite("retry re-posts model download") {
        MockURLProtocol.handler = { _, _ in
            .init(status: 200, body: Data(provisioningJSON(state: "downloading", progress: 10).utf8))
        }
        runBlockingP {
            let controller = ProvisioningController(client: client(),
                                                    initial: decodeStatus(provisioningJSON(state: "failed", error: "network")))
            let phase = await controller.retry()
            expectEqual(phase, .downloading(progress: 10), "retry → downloading")
        }
    }

    suite("transport failure surfaces a generic message, not a token claim") {
        MockURLProtocol.handler = nil // forces a transport error
        runBlockingP {
            let controller = ProvisioningController(client: client(), initial: decodeStatus(provisioningJSON()))
            let phase = await controller.submit(token: "whatever")
            if case let .failed(message) = phase {
                expect(!message.lowercased().contains("token"), "message does not claim the token was rejected")
            } else {
                expect(false, "expected a failed phase")
            }
        }
    }
}

private func decodeStatus(_ json: String) -> ProvisioningStatus {
    try! JSONCoding.makeDecoder().decode(ProvisioningStatus.self, from: Data(json.utf8))
}
