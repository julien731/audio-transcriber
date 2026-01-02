# Interview Analysis Template

Use this prompt with a meeting transcription to generate structured interview notes.

---

## Prompt

You are analyzing a job interview transcript for Nimble, a digital product consulting company. The candidate is interviewing for a {{ROLE}} position.

**Interviewer(s):** {{INTERVIEWER_NAMES}}
**Candidate:** {{CANDIDATE_NAME}}
**Date:** {{DATE}}

### Instructions

Analyze the transcript and produce a structured evaluation. Be specific — cite examples from the conversation to support your assessments. Flag areas where you lack sufficient signal.

### Output Format

```markdown
# Interview Summary: {{CANDIDATE_NAME}}

**Role:** {{ROLE}}
**Date:** {{DATE}}
**Interviewer(s):** {{INTERVIEWER_NAMES}}
**Duration:** [extracted from transcript]

---

## Quick Take

[2-3 sentences: Who is this person? What's their standout strength? What's the main concern?]

---

## Evaluation Criteria

### 1. Problem Understanding & Communication
**Signal strength:** Strong / Moderate / Weak / Insufficient
- How well did they understand questions before answering?
- Did they ask clarifying questions when appropriate?
- Were their answers structured and easy to follow?

**Evidence:**
> [Quote or paraphrase from transcript]

**Notes:** [Your assessment]

---

### 2. Research & Discovery Approach
**Signal strength:** Strong / Moderate / Weak / Insufficient
- Did they demonstrate a clear methodology for gathering information?
- How do they approach ambiguity?
- Do they validate assumptions?

**Evidence:**
> [Quote or paraphrase from transcript]

**Notes:** [Your assessment]

---

### 3. Technical/Domain Expertise
**Signal strength:** Strong / Moderate / Weak / Insufficient
- Did they demonstrate relevant technical knowledge for the role?
- Were their technical recommendations sensible and well-reasoned?
- Do they understand tradeoffs?

**Evidence:**
> [Quote or paraphrase from transcript]

**Notes:** [Your assessment]

---

### 4. Planning & Execution
**Signal strength:** Strong / Moderate / Weak / Insufficient
- Can they break down complex work into actionable steps?
- Are their estimates realistic?
- Do they consider dependencies and risks?

**Evidence:**
> [Quote or paraphrase from transcript]

**Notes:** [Your assessment]

---

### 5. Leadership & Collaboration
**Signal strength:** Strong / Moderate / Weak / Insufficient
- How do they work with others?
- Can they influence without authority?
- How do they handle disagreements?

**Evidence:**
> [Quote or paraphrase from transcript]

**Notes:** [Your assessment]

---

### 6. Attention to Detail & Quality
**Signal strength:** Strong / Moderate / Weak / Insufficient
- Do they catch their own mistakes?
- Do they hold themselves to high standards?
- Are they precise in their communication?

**Evidence:**
> [Quote or paraphrase from transcript]

**Notes:** [Your assessment]

---

### 7. Risk Awareness
**Signal strength:** Strong / Moderate / Weak / Insufficient
- Do they proactively identify risks?
- Are their risk assessments realistic (not over/under-stated)?
- Do they propose mitigations?

**Evidence:**
> [Quote or paraphrase from transcript]

**Notes:** [Your assessment]

---

## Red Flags

[List any concerns: evasive answers, inconsistencies, attitude issues, missing core competencies, etc. If none, state "None identified."]

---

## Questions Left Unanswered

[Topics that weren't covered or need follow-up in next round]

---

## Culture & Team Fit

[Would they thrive at Nimble? In a consulting environment? With clients?]

---

## Recommendation

**Verdict:** Strong Hire / Hire / Lean Hire / Lean No Hire / No Hire / Need More Signal

**Rationale:**
[3-5 sentences summarizing the decision. What tips the balance?]

**If proceeding, focus next round on:**
[Specific areas to probe deeper]
```

---

## Transcript

[PASTE TRANSCRIPT HERE]
