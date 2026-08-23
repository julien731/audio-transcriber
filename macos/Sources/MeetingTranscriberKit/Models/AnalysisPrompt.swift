import Foundation

/// The fully-assembled analysis prompt (`GET /api/meetings/{id}/analysis-prompt`).
/// Assembly is server-side (service BR-16/BR-17); the app requests it ready-to-use
/// and never reimplements template substitution (plan Artifact B, thin client).
public struct AnalysisPromptResponse: Codable, Equatable {
    public let prompt: String
}

/// The rendered Audio Analysis Context markdown (`.../analysis-context`).
public struct AnalysisContextResponse: Codable, Equatable {
    public let context: String
}

/// Template body (`GET /api/templates/{type}`).
public struct TemplateResponse: Codable, Equatable {
    public let template: String
}
