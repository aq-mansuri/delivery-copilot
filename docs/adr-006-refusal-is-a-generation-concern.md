# ADR-006: Refusal belongs at generation time, not retrieval time

**Status:** Accepted
**Supersedes:** the working assumption that a relevance floor would handle
unanswerable questions.

## Context

Five eval cases ask questions the corpus genuinely cannot answer, including
"Who approved the Acme security exception?" — which presupposes an exception
that does not exist.

The intuitive fix was a score floor: return nothing when the best match scores
below a threshold. This was measured before implementing.

## Measurement

Best raw score per query across the 16-case eval set:

| set | BM25 range | max cosine |
|---|---|---|
| answerable (11 cases) | 1.87 – 7.25 | — |
| unanswerable (5 cases) | 2.54 – 5.21 | 0.71 |

The distributions overlap substantially. Any floor rejecting "Who approved the
Acme security exception?" (5.21) also rejects "What's blocking INS-101?" (1.87)
and "What has changed since the previous release update?" (2.42).

## Why

These negatives ask for absent **facts** about present **topics**. "Acme",
"security", "O7" and "release" are all real corpus terms, so retrieval correctly
finds highly relevant pages. There is no signal at the retrieval layer that the
specific fact is missing, because relevance and answerability are different
properties.

## Decision

Keep configurable `min_bm25` / `min_cosine` floors — they remove genuinely
off-topic queries cheaply, which is worth having. Do **not** rely on them for
refusal.

Refusal is a generation-time obligation: the model is given retrieved context
and must decline when that context does not contain the answer. This is
enforced by prompt and measured by a groundedness eval (every claim traceable to
a retrieved chunk), not by a retrieval threshold.

## Consequences

- Negative-case performance cannot improve until the agent exists. Retrieval
  eval scores 0% on that category by design, and that is not a bug to chase.
- Groundedness evaluation becomes mandatory rather than optional, since it is
  now the only defence against confabulation.
- Client-facing framing: the system may retrieve relevant pages and still say it
  cannot answer. That is correct behaviour and should be explained before it is
  observed, or it reads as a failure.


## Postscript: the inverse failure, found on a live tenant

This ADR is about making the system decline when it does not know. The opposite
failure took longer to notice, because it looks identical from the outside.

Asked **"INS-12 is blocked?"** the service replied *"The available Jira and
Confluence content does not contain enough information to answer this."*

Every step before the last was correct. The model called
`get_risk_findings(issue_key="INS-12")`, the deterministic rules had genuinely
found nothing, and the tool reported that. The answering model then read

    No risk findings for INS-12. This means the rules found nothing, not that
    the check failed.

as *missing context* and declined under rule 1 of the answering prompt.

Absence of evidence read as evidence of absence, inverted. The risk rules are
**exhaustive over what they check**, so "no findings" is not a gap in the
context — it is the answer, and the answer is no. "We don't know" and "no, it is
fine" are opposite answers to a delivery lead, and the system had computed the
second and reported the first.

Note the tool had already tried to prevent exactly this: the sentence "not that
the check failed" was written for it. It was not enough, because it reads as a
caveat about a negative rather than as a positive assertion.

### Where the fix went, and why not the prompt

The obvious fix is a new rule in `SYSTEM_PROMPT`: "a reported absence is
evidence". It was rejected. That prompt governs refusal on genuinely
unanswerable questions, which is a baselined metric at 100% (and the reason this
ADR exists) — loosening it risks the behaviour the whole design is for, across
every question, to fix one.

The tool's wording was changed instead. `get_risk_findings` only ever speaks
when the rules have actually run over a complete sync, so it **cannot** loosen
refusal anywhere else; the blast radius is exactly the case being fixed. It now
states the negative in the question's own words, names what was checked so the
scope of the "no" is explicit, and says the result is definitive.

Measured after the change: "INS-12 is blocked?" answered *"No."* with a
citation, "Is INS-22 blocked?" answered *"Yes, blocked by INS-5"*, and the
adversarial fabrication probes held 6/6 with both unanswerable questions still
declining.

### Then the confident negative was wrong

The first wording ended "It is not blocked and not at risk." A blocker was then
added to INS-12 on the live tenant — as a **status change only**, with no issue
link — and the sentence was produced verbatim for a ticket whose status was
literally `Blocked`. The rules had found nothing because `rule_blocked_dependency`
reads links, and the wording had turned a narrow true negative into a broad
false one.

That is the sharpest lesson in this file. Making a negative more assertive is
exactly what creates the risk of a confident wrong answer, and a confident wrong
answer is worse than the refusal it replaced. The negative now enumerates what
the rules actually checked, says "not blocked (no blocking issue is linked to it
and its status is not a blocked one)", and explicitly declines to generalise:
"they are not a judgement that the work is healthy in every other respect."

The underlying gap was real and is fixed separately — see `rule_blocked_status`
in ADR-001. A status a human deliberately set was invisible to an engine that
only read links.

### The test that agreed with the code

`test_risk_tool_distinguishes_clean_from_broken` asserted the exact string
"found nothing, not that the check failed". Its docstring stated the right
requirement — "'No findings' must not read like 'the check failed'" — and its
assertion pinned an implementation detail, so it kept passing while the sentence
it was pinning was being misread downstream. It now asserts the property.

Worth re-reading CLAUDE.md's line about asking *why* a test passes. This one
passed because the string had not changed, which is not the same as the
requirement being met.
