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
            .background(color.opacity(0.15), in: Capsule())
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
