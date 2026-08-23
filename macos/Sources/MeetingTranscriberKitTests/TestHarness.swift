import Foundation

// Minimal test harness (plan TD-2). The dev toolchain here is Command Line Tools
// only — XCTest and swift-testing are unavailable — so tests run as an
// executable via `swift run MeetingTranscriberKitTests`. Suites read like normal
// test cases and are portable to XCTest on a full-Xcode machine later.

final class TestRunner {
    static let shared = TestRunner()

    private(set) var passed = 0
    private(set) var failed = 0
    private var currentSuite = ""

    func suite(_ name: String, _ body: () -> Void) {
        currentSuite = name
        body()
    }

    func expect(_ condition: Bool, _ message: @autoclosure () -> String,
                file: String = #fileID, line: Int = #line) {
        if condition {
            passed += 1
        } else {
            failed += 1
            FileHandle.standardError.write(
                Data("FAIL [\(currentSuite)] \(message()) (\(file):\(line))\n".utf8)
            )
        }
    }

    func expectEqual<T: Equatable>(_ actual: T, _ expected: T, _ message: String = "",
                                   file: String = #fileID, line: Int = #line) {
        expect(actual == expected,
               "\(message.isEmpty ? "" : message + ": ")expected \(expected), got \(actual)",
               file: file, line: line)
    }

    func expectNil<T>(_ value: T?, _ message: String = "",
                      file: String = #fileID, line: Int = #line) {
        expect(value == nil, "\(message) expected nil, got \(String(describing: value))",
               file: file, line: line)
    }

    func finish() -> Never {
        let summary = "\n\(passed) passed, \(failed) failed\n"
        FileHandle.standardOutput.write(Data(summary.utf8))
        exit(failed == 0 ? 0 : 1)
    }
}

/// Convenience free functions so suites stay terse.
func suite(_ name: String, _ body: () -> Void) { TestRunner.shared.suite(name, body) }
func expect(_ condition: Bool, _ message: @autoclosure () -> String,
            file: String = #fileID, line: Int = #line) {
    TestRunner.shared.expect(condition, message(), file: file, line: line)
}
func expectEqual<T: Equatable>(_ actual: T, _ expected: T, _ message: String = "",
                               file: String = #fileID, line: Int = #line) {
    TestRunner.shared.expectEqual(actual, expected, message, file: file, line: line)
}
func expectNil<T>(_ value: T?, _ message: String = "",
                  file: String = #fileID, line: Int = #line) {
    TestRunner.shared.expectNil(value, message, file: file, line: line)
}
