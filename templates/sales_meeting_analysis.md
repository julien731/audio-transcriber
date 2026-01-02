# Sales Meeting Analysis Template

Use this prompt with a meeting transcription to extract sales intelligence and next steps.

---

## Prompt

You are analyzing a sales meeting transcript for Nimble, a digital product consulting company offering web/mobile development, UX/UI design, and product management services.

**Meeting type:** {{DISCOVERY / PROPOSAL / NEGOTIATION / FOLLOW-UP}}
**Prospect company:** {{COMPANY_NAME}}
**Attendees:** {{NAMES_AND_ROLES}}
**Date:** {{DATE}}

### Instructions

Extract actionable sales intelligence from this conversation. Be specific with quotes and timestamps where relevant. Distinguish between what was explicitly stated vs. what you're inferring.

### Output Format

```markdown
# Sales Meeting: {{COMPANY_NAME}}

**Date:** {{DATE}}
**Meeting type:** {{TYPE}}
**Duration:** [extracted from transcript]

**Attendees:**
- [Name, Role] — Nimble
- [Name, Role] — Prospect

---

## TL;DR

[3-4 sentences: What's the opportunity? Where are we in the process? What's the critical next step?]

---

## Deal Snapshot

| Field | Value |
|-------|-------|
| **Opportunity** | [Brief description of what they need] |
| **Estimated size** | [If discussed: budget, team size, duration] |
| **Timeline** | [When do they want to start? Deadline pressures?] |
| **Stage** | Discovery / Qualified / Proposal Sent / Negotiation / Verbal Yes |
| **Temperature** | 🔥 Hot / ☀️ Warm / ❄️ Cool |

---

## Pain Points & Needs

[What problems are they trying to solve? Why now?]

| Pain Point | Severity | Quote/Evidence |
|------------|----------|----------------|
| [Issue 1] | High/Med/Low | "[Quote]" |
| [Issue 2] | High/Med/Low | "[Quote]" |

**Underlying motivation:**
[What's really driving this? Business pressure, competitive threat, internal politics, new leadership?]

---

## Decision Making

**Decision maker(s):**
- [Name, Role] — [What's their stake? What do they care about?]

**Influencers:**
- [Name, Role] — [Technical evaluator? Budget holder? End user?]

**Buying process:**
[What's their process? Who else needs to approve? Procurement involved?]

**Competition:**
[Are they talking to others? Who? How do we compare?]

---

## Budget Signals

| Signal | Interpretation |
|--------|----------------|
| [What was said/implied] | [What it means] |

**Budget range:** [Explicit or inferred]
**Budget owner:** [Who controls the money?]
**Fiscal considerations:** [End of quarter? Budget cycle? Approval thresholds?]

---

## Objections & Concerns

| Objection | Severity | How We Addressed | Resolved? |
|-----------|----------|------------------|-----------|
| [Concern 1] | High/Med/Low | [Our response] | Yes/Partially/No |
| [Concern 2] | High/Med/Low | [Our response] | Yes/Partially/No |

**Unspoken concerns:**
[What might they be worried about but didn't say?]

---

## Sentiment Analysis

### Overall Enthusiasm
**Level:** High / Moderate / Low / Mixed
**Evidence:** [What signals are you reading?]

### Reaction to Pricing/Scope
**Level:** Comfortable / Cautious / Concerned / Not Discussed
**Evidence:** [Specific reactions when costs came up]

### Confidence in Nimble
**Level:** High / Building / Skeptical / Unclear
**Evidence:** [Trust signals, concerns about our capabilities]

### Urgency
**Level:** Urgent / Normal / Low Priority / Unclear
**Evidence:** [Timeline pressure, competing priorities]

### Momentum
**Trend:** Gaining / Steady / Losing / Stalled
**Evidence:** [Are we moving forward or spinning?]

---

## What We Committed To

- [ ] [Action item 1] — Owner: [Name] — Due: [Date]
- [ ] [Action item 2] — Owner: [Name] — Due: [Date]

---

## What They Committed To

- [ ] [Action item 1] — Owner: [Name] — Due: [Date]
- [ ] [Action item 2] — Owner: [Name] — Due: [Date]

---

## Key Quotes

> "[Verbatim quote that's useful for proposal or follow-up]"
> — [Name], [context]

> "[Another key quote]"
> — [Name], [context]

---

## Strategic Notes

**What's working:**
[What resonated? What should we double down on?]

**What to adjust:**
[What fell flat? What should we do differently?]

**Risks to the deal:**
[What could derail this?]

**Recommended next steps:**
1. [Immediate action]
2. [Follow-up action]
3. [Preparation for next meeting]
```

---

## Transcript

[PASTE TRANSCRIPT HERE]
