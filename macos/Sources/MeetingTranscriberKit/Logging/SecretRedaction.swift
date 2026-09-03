// Story #139 — redact secrets from logs and the diagnostics export.
// Plan: docs/plans/139-redact-secrets-from-logs.md

import Foundation

/// Masks secret-shaped tokens before text is written to a shareable log file.
///
/// The raw child stderr tee (`service-stderr.log`) and the app's own `app.log`
/// are both bundled into the user-shareable diagnostics zip, and the raw stderr
/// bypasses Python's logging formatter entirely — so for third-party output that
/// never passes through Python (e.g. an uncaught traceback printed by
/// `huggingface_hub`/`pyannote`), this is the *sole* redaction sink, not a
/// backstop. Keep the pattern list in sync with the Python side in
/// `backend/services/logging_setup.py` (`_SECRET_PATTERNS`); both are exercised
/// with the shared `hf_TESTTOKEN...` test vector so a divergence surfaces.
public enum SecretRedaction {
    /// (pattern, replacement) pairs. Add more credential shapes here.
    private static let patterns: [(NSRegularExpression, String)] = {
        let specs = [
            (#"hf_[A-Za-z0-9]+"#, "hf_***"),
        ]
        return specs.compactMap { pattern, replacement in
            (try? NSRegularExpression(pattern: pattern)).map { ($0, replacement) }
        }
    }()

    /// Return `text` with every known secret shape masked. Safe on any string.
    public static func redact(_ text: String) -> String {
        var result = text
        for (regex, replacement) in patterns {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(
                in: result, range: range, withTemplate: replacement
            )
        }
        return result
    }
}
