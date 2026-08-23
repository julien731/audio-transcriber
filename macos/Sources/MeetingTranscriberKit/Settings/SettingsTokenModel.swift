import Foundation

/// State for the Settings screen where a user views and changes the HuggingFace
/// token after first-run setup (#104).
///
/// It composes `ProvisioningController` so the token/model-download contract lives
/// in exactly one place: saving delegates to `submit(token:)` (setToken +
/// startModelDownload) and clearing to `submit("")` (Whisper-only re-provision).
/// The backend never validates the token, so — like the setup wizard — this model
/// never claims a token was "rejected"; a bad token only surfaces later as a
/// generic model-download failure.
public enum SettingsPhase: Equatable {
    /// Not editing: show current diarization status and a "Change token" action.
    case idle(diarizationAvailable: Bool)
    /// The token field is visible for entry.
    case editing
    /// A model download is in progress after a save; poll `refresh()` until done.
    case working(progress: Int)
    /// Generic save/download failure; the field can be retried.
    case failed(message: String)
}

public final class SettingsTokenModel {
    public private(set) var phase: SettingsPhase
    private let controller: ProvisioningController

    public init(controller: ProvisioningController) {
        self.controller = controller
        self.phase = Self.settingsPhase(from: controller.phase, editing: false)
    }

    /// Map the wizard's `SetupPhase` onto the settings phase. A completed download
    /// returns to `.idle`; an in-progress one is `.working`; a failure is `.failed`.
    /// The backend's `enteringToken` (idle) state only appears before any save — it
    /// stays `.editing` while the user is typing, otherwise `.idle` (no token yet).
    public static func settingsPhase(from setup: SetupPhase, editing: Bool) -> SettingsPhase {
        switch setup {
        case let .completed(diarizationAvailable):
            return .idle(diarizationAvailable: diarizationAvailable)
        case let .downloading(progress):
            return .working(progress: progress)
        case let .failed(message):
            return .failed(message: message)
        case .enteringToken:
            return editing ? .editing : .idle(diarizationAvailable: false)
        }
    }

    /// Reveal the token field so the user can enter or replace the token.
    public func beginEditing() {
        phase = .editing
    }

    /// Leave the token field without saving, restoring the known status.
    public func cancelEditing(diarizationAvailable: Bool) {
        phase = .idle(diarizationAvailable: diarizationAvailable)
    }

    /// Store the token (trimmed by the controller; empty clears it) and start the
    /// model download. On success the phase is `.working` — poll `refresh()` until
    /// it settles to `.idle`; on error it is `.failed` with a generic message.
    @discardableResult
    public func save(token: String) async -> SettingsPhase {
        let setup = await controller.submit(token: token)
        phase = Self.settingsPhase(from: setup, editing: true)
        return phase
    }

    /// Poll provisioning once while a download is in progress (`.working`).
    @discardableResult
    public func refresh() async -> SettingsPhase {
        let setup = await controller.refresh()
        phase = Self.settingsPhase(from: setup, editing: true)
        return phase
    }
}
