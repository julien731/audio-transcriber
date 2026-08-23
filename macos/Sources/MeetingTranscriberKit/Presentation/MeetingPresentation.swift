import Foundation

public extension MeetingType {
    var displayName: String {
        switch self {
        case .interview: return "Interview"
        case .sales: return "Sales"
        case .client: return "Client"
        case .other: return "Other"
        }
    }
}

/// UI-agnostic severity for a status badge; the SwiftUI layer maps it to a color.
public enum BadgeSeverity: Equatable {
    case neutral, positive, warning, danger
}

public struct StatusBadge: Equatable {
    public let label: String
    public let severity: BadgeSeverity

    public init(label: String, severity: BadgeSeverity) {
        self.label = label
        self.severity = severity
    }
}

public extension MeetingStatus {
    var badge: StatusBadge {
        switch self {
        case .processing: return StatusBadge(label: "Processing", severity: .warning)
        case .ready: return StatusBadge(label: "Ready", severity: .positive)
        case .error: return StatusBadge(label: "Error", severity: .danger)
        }
    }
}

public extension JobStatus {
    var isTerminal: Bool { self == .completed || self == .failed }
}
