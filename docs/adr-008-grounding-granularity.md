# ADR-008: Grounding is checked per paragraph, not per sentence

**Status:** Accepted
**Supersedes:** the per-sentence rule shipped in the first version of
`check_grounding`.

## Context

The first live run against Claude refused every scenario. A diagnostic across six
questions gave: 1 answered, 2 correctly declined (both negatives), 3 rejected by
the grounding checker.

All three rejections had the same two causes, neither visible from reading the
code.

**1. A splitter bug, not a strictness problem.** Models write the citation after
the full stop:

    ...runs in parallel. [3]

Splitting on `(?<=[.!?])\s+` starts the next sentence with "[3]". Sentence N
loses the citation it earned; sentence N+1 inherits one it did not. Every
rejection was this single off-by-one.

**2. Models cite per paragraph.** Three sentences of connected reasoning, then
[3]. That is normal, readable prose. A per-sentence rule rejects it wholesale.

Markdown made both worse: headings, table rows and `---` were being treated as
uncited claims.

## Decision

- Trailing citations are pulled back onto the sentence they follow.
- Markdown structure (headings, table rows, rules, bare list markers) is not a
  claim.
- The default unit of enforcement is the **paragraph**: every paragraph making
  claims must carry at least one resolvable citation.
- `per_sentence=True` is retained for evals.

## The tradeoff, stated plainly

A paragraph citing [3] is traceable — a lead can click through and check it. But
a paragraph with one real citation could contain three sentences the source does
not support, and this check passes it.

Per-sentence is a stronger guarantee that produced a 100% refusal rate. A system
nobody can use is not a safe system; it is a false negative wearing a safety
jacket, and its failure mode is that the lead stops opening the tool.

Paragraph-level enforcement plus the Day 5 groundedness eval is the chosen
balance. The eval measures what this check lets through, rather than assuming it
lets nothing through.

## Consequences

- Four answers recorded from the live run are now regression fixtures in
  `tests/fixtures/live_answers.py` — the only fixtures in this project recorded
  rather than imagined.
- Two fabrication guards run alongside them, so a future "fix" cannot pass by
  disabling the check.
- Day 5 must measure claim-level groundedness to quantify the gap this ADR
  knowingly opens.
