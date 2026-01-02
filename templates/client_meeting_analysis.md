# Client Meeting Analysis Template

Use this prompt with a meeting transcription to extract decisions, action items, and project status.

---

## Prompt

You are analyzing a client meeting transcript for Nimble, a digital product consulting company. This is an ongoing engagement, and the meeting covers project status, planning, and/or problem-solving.

**Client:** {{CLIENT_NAME}}
**Project:** {{PROJECT_NAME}}
**Attendees:** {{NAMES_AND_ROLES}}
**Date:** {{DATE}}

### Instructions

Extract key information for project management and follow-up. Be precise about what was decided vs. what was discussed vs. what remains open. Attribute action items to specific people with deadlines where mentioned.

### Output Format

```markdown
# Client Meeting: {{CLIENT_NAME}} — {{PROJECT_NAME}}

**Date:** {{DATE}}
**Duration:** [extracted from transcript]

**Attendees:**
- [Name, Role] — Nimble
- [Name, Role] — Client

---

## Summary

[3-5 sentences: What was this meeting about? What's the headline? Any major shifts or concerns?]

---

## Decisions Made

Explicit decisions that were agreed upon in this meeting:

| # | Decision | Context/Rationale |
|---|----------|-------------------|
| 1 | [What was decided] | [Why, if discussed] |
| 2 | [What was decided] | [Why, if discussed] |

---

## Action Items

### Nimble

| # | Action | Owner | Deadline | Notes |
|---|--------|-------|----------|-------|
| 1 | [Task] | [Name] | [Date or "ASAP" or "TBD"] | [Context] |
| 2 | [Task] | [Name] | [Date] | [Context] |

### Client

| # | Action | Owner | Deadline | Notes |
|---|--------|-------|----------|-------|
| 1 | [Task] | [Name] | [Date] | [Context] |
| 2 | [Task] | [Name] | [Date] | [Context] |

---

## Open Questions

Issues raised but not resolved — need follow-up or further discussion:

| # | Question | Owner | Context |
|---|----------|-------|---------|
| 1 | [Question or unresolved issue] | [Who needs to answer] | [Background] |
| 2 | [Question] | [Owner] | [Background] |

---

## Project Status

### Progress Since Last Meeting
[What was completed or advanced?]

### Current Blockers
| Blocker | Impact | Owner | Path to Resolution |
|---------|--------|-------|-------------------|
| [Issue] | High/Med/Low | [Name] | [What needs to happen] |

### Upcoming Milestones
| Milestone | Target Date | Status | Risk Level |
|-----------|-------------|--------|------------|
| [Milestone] | [Date] | On Track / At Risk / Blocked | 🟢 / 🟡 / 🔴 |

### Scope Changes
[Any changes to scope discussed? Additions, cuts, deferrals?]

| Change | Type | Impact | Status |
|--------|------|--------|--------|
| [Feature/requirement] | Addition / Removal / Deferral | [Effort, timeline] | Agreed / Under Discussion |

---

## Technical Discussions

[Summarize any technical topics covered: architecture decisions, implementation approaches, tradeoffs discussed]

### Decisions
- [Technical decision 1]
- [Technical decision 2]

### Still Under Discussion
- [Topic needing more exploration]

---

## Risks & Concerns

| Risk/Concern | Raised By | Severity | Mitigation |
|--------------|-----------|----------|------------|
| [Issue] | [Name] | High/Med/Low | [What we'll do about it] |

---

## Client Sentiment

**Overall mood:** Positive / Neutral / Concerned / Frustrated
**Confidence in project:** High / Moderate / Low
**Key concerns:** [What's on their mind?]

**Notable quotes:**
> "[Anything important they said that's worth remembering]"

---

## Context for Next Meeting

[What should be prepared? What topics will we need to address?]

---

## Private Notes

[Anything not for client consumption: internal observations, concerns, politics to navigate, relationship dynamics]
```

---

## Transcript

[PASTE TRANSCRIPT HERE]
