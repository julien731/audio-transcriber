import Foundation

/// App-local presentation preferences over `UserDefaults` (BR-11) — theme and
/// recently-used speaker names. Replaces the web client's `localStorage`. The
/// store is injectable so it is unit-testable with an isolated suite.
public final class Preferences {
    public enum Theme: String, CaseIterable {
        case system, light, dark
    }

    private enum Key {
        static let theme = "theme"
        static let recentSpeakerNames = "recentSpeakerNames"
    }

    public static let recentNamesLimit = 10

    private let defaults: UserDefaults

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    public var theme: Theme {
        get { Theme(rawValue: defaults.string(forKey: Key.theme) ?? "") ?? .system }
        set { defaults.set(newValue.rawValue, forKey: Key.theme) }
    }

    public var recentSpeakerNames: [String] {
        defaults.stringArray(forKey: Key.recentSpeakerNames) ?? []
    }

    /// Mirror utils.js `addRecentSpeakerName`: ignore blank, move-to-front,
    /// de-duplicate, cap at 10 (most-recent first).
    public func addRecentSpeakerName(_ name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var names = recentSpeakerNames.filter { $0 != trimmed }
        names.insert(trimmed, at: 0)
        if names.count > Self.recentNamesLimit { names = Array(names.prefix(Self.recentNamesLimit)) }
        defaults.set(names, forKey: Key.recentSpeakerNames)
    }
}
