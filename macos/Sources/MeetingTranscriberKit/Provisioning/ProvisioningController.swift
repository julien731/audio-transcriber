import Foundation

/// First-run setup state machine (plan slice 4, BR-12/13/14, EC-4/6/7).
///
/// The backend does NOT validate the HuggingFace token — `set_token` stores any
/// non-empty string and reports `diarization_available` purely from its presence
/// (provisioning.py). A bad token only surfaces later as a generic model-download
/// failure. So this controller never claims "token rejected": on a failed
/// download it exposes the service's own `download_error` plus two actions —
/// **retry** and **continue without diarization** (which clears the token so
/// provisioning needs only the Whisper model).
public enum SetupPhase: Equatable {
    /// Show the token field (first entry or re-enter). Empty is allowed.
    case enteringToken
    case downloading(progress: Int)
    /// Generic download failure; offers retry / continue-without-diarization.
    case failed(message: String)
    case completed(diarizationAvailable: Bool)

    public static func from(_ status: ProvisioningStatus) -> SetupPhase {
        if status.provisioningCompleted {
            return .completed(diarizationAvailable: status.diarizationAvailable)
        }
        switch status.downloadState {
        case .downloading:
            return .downloading(progress: status.downloadProgress)
        case .completed:
            // Download finished but the completed flag isn't set yet — treat as
            // still working; the next poll flips to `.completed`.
            return .downloading(progress: 100)
        case .failed:
            return .failed(message: status.downloadError ?? "The model download failed. Check your connection and try again.")
        case .idle:
            return .enteringToken
        }
    }
}

public final class ProvisioningController {
    public private(set) var phase: SetupPhase
    private let client: APIClient

    public init(client: APIClient, initial: ProvisioningStatus) {
        self.client = client
        self.phase = SetupPhase.from(initial)
    }

    /// Store the token (possibly empty) and kick off the model download.
    @discardableResult
    public func submit(token: String) async -> SetupPhase {
        do {
            _ = try await client.setToken(token.trimmingCharacters(in: .whitespacesAndNewlines))
            let status = try await client.startModelDownload()
            phase = SetupPhase.from(status)
        } catch {
            phase = .failed(message: Self.message(for: error))
        }
        return phase
    }

    /// Poll provisioning once; advances downloading → completed/failed.
    @discardableResult
    public func refresh() async -> SetupPhase {
        do {
            phase = SetupPhase.from(try await client.provisioning())
        } catch {
            phase = .failed(message: Self.message(for: error))
        }
        return phase
    }

    /// Retry the model download without changing the token.
    @discardableResult
    public func retry() async -> SetupPhase {
        do {
            phase = SetupPhase.from(try await client.startModelDownload())
        } catch {
            phase = .failed(message: Self.message(for: error))
        }
        return phase
    }

    /// Clear the token and re-provision Whisper-only (diarization disabled, BR-13).
    @discardableResult
    public func continueWithoutDiarization() async -> SetupPhase {
        await submit(token: "")
    }

    private static func message(for error: Error) -> String {
        (error as? APIError)?.userMessage ?? "The model download failed. Check your connection and try again."
    }
}
