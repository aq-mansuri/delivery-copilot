# Delivery Copilot — Product Requirements

**Status:** v1 in build
**Client:** Priya Sharma, VP Engineering, ~400-person insurance company
**Owner:** Forward-deployed engineering

---

## 1. Problem

Every Monday, engineering leads spend roughly half a day assembling a delivery
status deck for executive review. The numbers are consistently slightly wrong.

The client's opening ask was "put AI on top of Jira." Discovery established that
this is not the problem.

**What the executives actually act on** — established by asking what happens in
the room rather than what the deck contains:

1. Will we hit the dates committed to the board?
2. What is blocked that needs a decision from this room?

Burndown charts are skipped. The RAG status per epic is whatever the lead felt
that morning, so two equally-late teams show amber and red. The "top risks" slide
is copy-pasted from the previous week.

**The highest-value signal is one nobody currently produces.** A ticket can sit
in "In Progress" for three weeks with no commits, no comments and no transitions.
It looks healthy on the board. It is dead. Catching that is what the leads would
actually value, and it is an inference from *absence of activity* — a changelog
query, not a judgement call.

## 2. Constraints

These come from the client and are not negotiable design preferences.

| Constraint | Source | Consequence |
|---|---|---|
| No unapproved writes to Jira | Regulated insurer; audit trail must show a human made each change | Agent proposes, a named lead approves, every decision logged |
| Every claim traceable to a source | "'The system says' is not an answer I can give the board" | Citations mandatory; unattributable answers are rejected, not warned |
| Recall over precision on risk | A false positive is noise; a missed slip costs the VP her credibility | Thresholds bias toward flagging |
| `Target Release` is ~40% blank and semantically inconsistent | Existing Jira data | No delivery-date prediction in v1; coverage stated explicitly |

## 3. Scope

### v1 delivers

- Ingest 6 Jira projects and ~200 labelled Confluence pages
- Deterministic risk engine: stale in-progress, blocked dependency, sprint
  carryover, overdue, missing delivery date
- Weekly narrative summary, every claim carrying a source link
- Ad-hoc question answering over the same corpus, with citations
- Proposed at-risk flags that a named lead approves or rejects, with an audit log
- Explicit refusal when the corpus cannot answer a question

### v1 explicitly does not

- Write to Jira without approval
- Predict delivery dates or confidence percentages
- Clean or backfill the `Target Release` field
- Send email or replace the executive deck
- Perform historical trend analysis

Non-goals exist to stop the week-three asks, not to describe work nobody
proposed.

## 4. The architectural decision

**Rules find the problems. The model explains them.**

| Deterministic | LLM |
|---|---|
| Stale in-progress detection | Exec-readable narrative generation |
| Sprint carryover counting | Ad-hoc Q&A with citations |
| Overdue and slip arithmetic | Classifying free-text comments into blocker categories |
| Data-quality gaps | Deciding when to decline |

Risk detection is a pure function over issue changelogs. It is faster, cheaper,
auditable, and identical on every run — properties a model cannot offer and that
an auditor will ask for. The model is scoped to the four jobs it is genuinely
better at.

The model never decides what is at risk, and never writes to Jira.

## 5. Success criteria

| Measure | Target | Current |
|---|---|---|
| Retrieval recall@5 | ≥ 80% | 62.5% |
| Cross-source questions (all required pages) | ≥ 80% | ~50% |
| Unanswerable questions declined | 100% | 2/2 live |
| Answers with unattributable claims reaching a user | 0 | 0 |
| Lead editing time before sending | < 10 min | not yet measured |

Recall is the metric that matters: if the right passage is not retrieved, no
prompting recovers it. Precision is recoverable — a model can ignore an
irrelevant passage.

## 6. Known gaps

- **Claim-level grounding is unmeasured.** Paragraph-level citation means
  traceable, not supported. Live output has produced recommendations ("formally
  log the SLA breach") citing passages that state facts but make no
  recommendation. See ADR-008.
- **Temporal questions are unsupported.** "What changed most recently" needs
  version diffing; retrieval cannot order by time.
- **Cross-source recall is measured any-of, not all-of.** A partial answer to a
  two-source question currently scores as a hit.
- **Determinism is not guaranteed.** The installed SDK no longer accepts
  `temperature`; runs rely on model defaults.

## 7. Open questions for the client

1. Who are the named approvers, and does approval authority vary by project?
2. Is a weekly batch sufficient, or do mid-week ad-hoc questions need live data?
3. Will the team commit to a `Target Release` definition? That unlocks date
   prediction in v2 and is a few hours of cleanup per team, not an engineering
   problem.
