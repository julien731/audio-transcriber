import SwiftUI
import MeetingTranscriberKit

/// Renders a `StatusBadge`, mapping the UI-agnostic severity to a color.
struct StatusBadgeView: View {
    let badge: StatusBadge

    var body: some View {
        Text(badge.label)
            .font(.caption.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            // A material base keeps the label legible over any row background —
            // the default list fill, the blue key-window selection, and the gray
            // inactive-selection highlight alike — with a faint severity tint on
            // top for colour identity. Avoids guessing the selection state, which
            // would need macOS 14's backgroundProminence (the app targets 13).
            .background {
                Capsule().fill(.regularMaterial)
                Capsule().fill(color.opacity(0.12))
            }
            .foregroundStyle(color)
    }

    private var color: Color {
        switch badge.severity {
        case .neutral: return .secondary
        case .positive: return .green
        case .warning: return .orange
        case .danger: return .red
        }
    }
}
