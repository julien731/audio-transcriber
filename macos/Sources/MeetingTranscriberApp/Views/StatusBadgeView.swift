import SwiftUI
import MeetingTranscriberKit

/// Renders a `StatusBadge`, mapping the UI-agnostic severity to a color.
struct StatusBadgeView: View {
    let badge: StatusBadge
    /// True when the badge sits on a selected List row, whose accent-colored
    /// highlight flips text to white. The severity-colored badge is unreadable
    /// there, so invert it to white-on-translucent like the surrounding row text.
    var isSelected: Bool = false

    var body: some View {
        Text(badge.label)
            .font(.caption.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(fillStyle, in: Capsule())
            .foregroundStyle(textColor)
    }

    private var textColor: Color { isSelected ? .white : color }

    private var fillStyle: Color {
        isSelected ? Color.white.opacity(0.25) : color.opacity(0.15)
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
