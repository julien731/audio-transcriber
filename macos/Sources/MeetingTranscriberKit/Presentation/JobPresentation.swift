import Foundation

/// Human-readable labels for the job stages the service reports (schemas.py
/// JobStage). Unknown/empty stages fall back to a generic label.
public enum JobStagePresentation {
    private static let labels: [String: String] = [
        "uploading": "Uploading",
        "preprocessing": "Preparing audio",
        "transcribing": "Transcribing",
        "aligning": "Aligning timestamps",
        "diarizing": "Identifying speakers",
        "emotion_analysis": "Analyzing emotion",
        "prosody_extraction": "Extracting prosody",
        "interaction_analysis": "Analyzing interactions",
    ]

    public static func label(for stage: String) -> String {
        labels[stage] ?? (stage.isEmpty ? "Working" : stage.replacingOccurrences(of: "_", with: " ").capitalized)
    }
}
