# Delivery Copilot — case study

## The ask, and why it was the wrong one

Priya, VP Engineering at a ~400-person insurer: *"Our delivery status reporting
is a mess. Every Monday my leads spend half a day pulling Jira data into slides,
and the numbers are always slightly wrong. Someone mentioned we could just put AI
on top of it."*

Five discovery questions changed the shape of the project.

Asking what the executives actually *do* in the room — rather than what the deck
contains — established that they act on two things: whether the board-committed
dates will hold, and what is blocked that needs a decision from that room. The
burndown charts are skipped. The RAG status is whatever the lead felt that
morning, so two equally-late teams show amber and red.

Asking what "blocked" means produced the finding the project was built on:

> *"A ticket sits in In Progress for three weeks with no commits, no comments, no
> status change. It looks healthy on the board. It's dead. If your thing could
> catch that, my leads would actually care."*

That is not an AI problem. It is an inference from *absence of activity* — a
changelog query. The most valuable half of the product turned out to be
deterministic, and recognising that set the architecture.

Asking about error cost set the rest of it. False positives are noise; a missed
slip costs her credibility with the board. And: *"This cannot silently write to
Jira. We're a regulated insurer. If your system starts editing issues on its own,
my CISO kills the project in week one."*

## What was built

A service that ingests Jira and Confluence, computes delivery risk with
deterministic rules, answers questions with a citation on every claim, and
proposes changes a named lead approves — with the approval recorded before
anything is written.

**Rules find the problems. The model explains them.** Stale detection, blocked
dependencies, sprint carryover and slip arithmetic are pure functions over issue
changelogs. The model generates the narrative, answers questions, classifies
free-text blockers, and decides when to decline. It never decides what is at
risk and never writes to Jira.

Each of Priya's constraints became a type rather than a convention. A partial
sync is a different class from a complete one, so it cannot reach the report
generator. An approved action is only constructible through an approval that
demands a named actor. The tool registry holds no write client at all, so a
write tool physically cannot mutate anything.

## Outcomes

| | |
|---|---|
| Claims traceable to their cited source | 85.3% |
| Claims asserting more than the source (overreach) | 8.8% |
| Claims in no retrieved passage | 2.0% |
| Unanswerable questions declined | 100% |
| Adversarial fabrication probes held | 6/6 |
| Cost per question | $0.005 |
| Time to answer | ~6s |

The number that moved most: a prompt change targeting three observed failure
patterns — strengthening vocabulary ("cannot progress" → "blocked"), conclusions
the source does not draw ("commitment not met" → "confirmed breach"), and an
unsolicited recommendations section — took supported claims from 62% to 89%
against a measured run-to-run swing of 8 points.

That improvement is quotable only because the noise was measured first.

## Three things that went wrong, and what they taught

**The eval instrument was wrong twice before the system was.** A retrieval score
floor looked like the obvious way to make the system decline unanswerable
questions. Measuring first showed the score distributions overlap, because those
questions ask for absent facts about present topics. Later, a per-sentence
grounding rule produced a 100% refusal rate — traced to an off-by-one in a
sentence splitter, since models write the citation after the full stop. A system
that refuses everything looks maximally safe and is unusable.

**Two bugs cancelled each other out invisibly.** Contract tests written against
fixtures from documentation all passed. Diffing one real Jira payload showed the
seeder created issue links backwards while the reader parsed them correctly. Had
the system only ever been tested against its own seed data, both would have
survived indefinitely under a green suite.

**The judge needed judging.** Groundedness is measured by a model, so the
measurement is only as good as the grader. Calibrating it against labelled cases
caught no model errors and two of mine — an underspecified taxonomy, then a
fixture set that lagged its own rubric. A held-out set, written without checking
the judge first, then showed the fitted set's 100% was partly co-adaptation. The
honest number is 88%, with no dangerous misses.

## What was not delivered, and why

Delivery-date prediction was scoped out on day one. `Target Release` is blank on
~40% of issues and means different things to different teams, so any confidence
percentage would be a figure with nothing underneath it. Told to Priya as: the
data does not support it yet, here is what would change that, and here is what
the board can have instead — named at-risk epics with explicit coverage
("this covers 61% of open issues") and a link behind every claim.

Also open and stated rather than hidden: approvals do not survive a restart, the
`X-Actor` header is self-asserted so the audit trail records a claim rather than
an identity, and temporal questions are unsupported because retrieval cannot
order by time.

## Stack

Python, FastAPI, LangGraph, Claude (Sonnet answering, Haiku judging), hybrid
BM25 + dense retrieval fused by reciprocal rank, React and Vite, Docker.
362 tests, all runnable offline.
