import Foundation

/// Shared JSON coders for the service API. Field names are snake_case on the wire
/// (Pydantic), so we convert to/from camelCase and avoid per-type CodingKeys.
/// Dates are Pydantic `datetime` ISO-8601 strings, often with fractional seconds
/// and no timezone (e.g. `2026-08-22T10:31:59.123456`); the lenient strategy
/// below accepts those plus timezone-qualified and whole-second forms.
public enum JSONCoding {
    public static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let raw = try container.decode(String.self)
            if let date = lenientDate(from: raw) { return date }
            throw DecodingError.dataCorruptedError(in: container,
                                                   debugDescription: "Unrecognized date: \(raw)")
        }
        return decoder
    }

    public static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }

    static func lenientDate(from raw: String) -> Date? {
        for formatter in cachedFormatters {
            if let date = formatter.date(from: raw) { return date }
        }
        return nil
    }

    private static let cachedFormatters: [DateFormatter] = {
        let patterns = [
            "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXXXX",
            "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
            "yyyy-MM-dd'T'HH:mm:ss.SSSXXXXX",
            "yyyy-MM-dd'T'HH:mm:ss.SSS",
            "yyyy-MM-dd'T'HH:mm:ssXXXXX",
            "yyyy-MM-dd'T'HH:mm:ss",
        ]
        return patterns.map { pattern in
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(identifier: "UTC")
            formatter.dateFormat = pattern
            return formatter
        }
    }()
}
