// swift-tools-version:6.0
import PackageDescription

// Native macOS app for Meeting Transcriber (spec: docs/specs/native-macos-app.md,
// plan: docs/plans/94-native-macos-app.md). A thin SwiftUI front end over the
// bundled local transcription service; no transcription/workflow logic lives
// here — everything is driven through the service HTTP API.
//
// Targets:
//   MeetingTranscriberKit               — testable, UI-independent logic.
//   MeetingTranscriberApp               — SwiftUI @main shell (assembled into a .app by scripts/build_app.sh).
//   MeetingTranscriberKitTests          — unit suite; runs via `swift run MeetingTranscriberKitTests`.
//   MeetingTranscriberIntegrationTests  — integration suite (real stub child + local HTTP); `swift run …`.
//
// The dev toolchain here is Command Line Tools only (no XCTest/Testing module),
// so tests are executables run with `swift run`, not `swift test` (plan TD-2).
// Language mode v5: a localhost client + child-process coordinator; Swift 6
// strict-concurrency isolation would add noise to a UI shell without value.
//
// Sparkle is intentionally NOT a dependency yet — it is introduced in slice 10
// so early milestones build without a network fetch.
let package = Package(
    name: "MeetingTranscriber",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "MeetingTranscriberKit", targets: ["MeetingTranscriberKit"]),
        .executable(name: "MeetingTranscriberApp", targets: ["MeetingTranscriberApp"]),
        .executable(name: "MeetingTranscriberKitTests", targets: ["MeetingTranscriberKitTests"]),
        .executable(name: "MeetingTranscriberIntegrationTests", targets: ["MeetingTranscriberIntegrationTests"]),
    ],
    targets: [
        .target(
            name: "MeetingTranscriberKit",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "MeetingTranscriberApp",
            dependencies: ["MeetingTranscriberKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "MeetingTranscriberKitTests",
            dependencies: ["MeetingTranscriberKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "MeetingTranscriberIntegrationTests",
            dependencies: ["MeetingTranscriberKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
