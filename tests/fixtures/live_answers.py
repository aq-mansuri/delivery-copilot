"""Answers captured verbatim from Claude against the seed corpus.

Every other fixture in this project was written by hand. These were recorded
from a live run, which makes them the only ones that can disprove an assumption
about how a model actually formats an answer.

The first live run rejected 3 of 4 answerable questions. Each rejection is
preserved here as a regression fixture, because the causes were invisible from
reading the code:

  - citations placed AFTER the full stop, which the sentence splitter credited
    to the following sentence
  - markdown headings, table rows and rules treated as claims
  - paragraph-level citation, which a per-sentence rule rejects wholesale
"""

RELEASE_PLAN = """## Q4 2026 Release Plan — Summary

### Committed Items (in priority order)
Three items carry board commitments this quarter [1]:
1. **Claims vendor integration**
2. **Compliance export** (for the regulator)
3. **Partner authentication** (claims portal)

### Sequencing Risk
Partner authentication must land before claims portal work can proceed, as the \
portal depends on the platform team's auth endpoint. The compliance export has \
no upstream dependencies and runs in parallel. [3]

### Schedule
| Item | Target | Confidence |
|---|---|---|
| Claims vendor integration | Late Q4 | Low |
| Compliance export | Mid Q4 | High |
[4]

**Bottom line:** Only one of three board-committed items is currently on track. [2]"""

HOLDING_UP = """Based on the context, two of the three board-committed items are \
currently blocked:

**Claims Vendor Integration** is stalled because the external vendor has not \
responded to confirm the callback schema, leaving INS-101 unable to advance. [5] \
INS-102, also under this epic, depends on work held by the platform team. [5]

**Partner Authentication** is held outside this group and cannot progress. [3]

In summary, the release is being held up by an unresponsive external vendor and \
by a platform-team dependency. [3][5]"""

ACME_CROSS_SOURCE = """## Acme Integration Status & Contractual Risk

### Current Status

INS-101 is fully stalled — untouched for three weeks — pending schema \
confirmation from Acme. [1] Two follow-up requests have gone unanswered. [1]

### Contractual Risk

There is a clear, documented contractual breach. Acme's contract commits to a \
**5 business day turnaround** on integration support requests, and that \
commitment has not been met. [5] The current agreement runs to **31 March \
2027**. [5]

---

**Bottom line:** Acme is in breach of contracted SLA. [5]"""

CONTROLS_OUTSTANDING = """Based on the security requirements documentation, three \
controls remain unsatisfied before the claims vendor integration may process \
live policyholder data [3]:

1. **Mutual TLS** — specified but certificates have not been issued [3]
2. **Penetration testing** — of the callback endpoint has not been scheduled [3]
3. **Audit trail for approval actions** — designed but not implemented [3]

Note: No exceptions have been raised against any of these controls [2]."""

# The presupposition case. The model named the false premise before declining —
# better behaviour than the prompt strictly required.
PRESUPPOSITION_REFUSAL = """The context explicitly states that **no exceptions \
have been raised for this integration** [2]. The question presupposes that a \
security exception exists and was approved — the context does not support this \
premise.

INSUFFICIENT_EVIDENCE"""

OFF_TOPIC_REFUSAL = "INSUFFICIENT_EVIDENCE"


# Second live run. The lead-in paragraph below was rejected; everything after it
# is cited.
HOLDING_UP_WITH_LEAD_IN = """Based on the context, two of the three \
board-committed items are blocked.

**Claims Vendor Integration** is stalled waiting for an external vendor to \
confirm the callback schema — no response has been received [5]. Specifically, \
INS-101 cannot advance until that confirmation arrives [5].

**Partner Authentication** is held outside this group [3] and has no committed \
target date [4].

In summary, the two active blockers are an unresponsive vendor and an externally \
held dependency on the platform team [3][5]."""

# Better than the protocol asked for — names the false premise — but omits the
# marker, so it fell through to the grounding check and was logged as a failure.
PROSE_REFUSAL_NO_MARKER = """The context explicitly states that **no exceptions \
have been raised for this integration** [2]. There is therefore no security \
exception to approve, and no approver to identify.

The question presupposes an exception exists — the context does not support that \
premise."""
