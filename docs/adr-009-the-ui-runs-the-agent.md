# ADR-009: The UI runs the agent, and write intent is a keyword gate

**Status:** Accepted

## Context

For most of the build, `/ask` was a retrieval path: retrieve → answer →
validate → stream. No tool loop. The agent graph (ADR-007), the tool registry
(ADR-002) and the approval gate all existed, were tested, and were reachable
only from `scripts/demo_agent.py`.

The failure this produced is worth recording precisely, because nothing was
broken in the ordinary sense and the whole test suite was green.

Typing **"Flag INS-101 as at risk"** into the product returned a grounded
paragraph *about* INS-101, with five citations, in six seconds. It did exactly
what the code said to do. "Waiting on you" stayed empty, because nothing on that
path can propose anything. Three specifics underneath it:

- `make_flag_tool()` appeared nowhere in `app/` — only in `tests/` and the demo
  script.
- `build_services()` constructed no `ToolRegistry` at all.
- It passed `findings=[]`, so even the read-only risk tool would have returned
  nothing. The risk engine — the flagship half of the system — was not connected
  to the API.

A guarantee that can only be demonstrated from a script is a guarantee the
client has no reason to believe. The approval gate was the strongest claim in
the system and the product could not show it.

## Decision

Two changes.

**1. `/ask` runs the graph.** One endpoint, one code path, the same one the
tests pin. Not a second agent-flavoured endpoint beside the retrieval one.

**2. Whether the write tools are *offered* is decided by a keyword gate**
(`app/agent/intent.py`), not by a model call and not by a UI mode switch.

## Rationale

### Why the whole UI goes through the graph

The alternative was routing: retrieval for questions, the graph for requests.
It was rejected because the router is the part that breaks. "What is at risk
right now?" is a question, and answering it correctly requires the risk engine —
so a router keyed on request-versus-question sends the single most important
question in the product down the path that cannot answer it, and it answers from
Confluence instead. Documentation describes intent; the rules describe reality,
and preferring the wrong one silently is the failure mode the tool descriptions
already warn the model about.

One path also means one thing to test, one thing to trace, and one thing to
explain in a security review.

The cost is one extra model call on a question needing no tools: the reasoning
turn, then the answer node. That was judged worth it, and part of it was
recovered — see below.

### Why query decomposition moved rather than being dropped

Decomposition is a measured retrieval gain (`run_eval.py --decompose`), and it
lived in the retrieval path being replaced. It moved into the graph's `retrieve`
node. Dropping a measured gain as a side effect of a refactor is how a number in
a README stops being true.

### Why write intent is not a model call

The project rule is that a thing that can be computed is computed. The usual
objection — "intent classification needs a model" — does not apply here, because
this gate cannot cause harm in either direction:

- **Missed request.** Write tools are withheld, the question is answered as
  asked, the user rephrases. One wasted round-trip.
- **Spurious detection.** A write tool is *offered*. The model must still choose
  it; whatever it produces is a `ProposedAction`; that still requires a named
  human. The floor is a proposal somebody rejects — noise, not risk.

The security property is unchanged and lives where it always did: `ToolRegistry`
holds no client capable of writing to Jira (ADR-002). This gate adjusts how much
is offered, never what can be done.

It also matches the actions that exist rather than the idea of change. "Close
INS-101" returns False, because there is no `close_issue` proposal to make;
detecting it would offer tools that cannot serve it and produce a confusing
refusal instead of a plain answer.

## Two bugs this surfaced

Both were invisible while the graph ran only under scripted tests.

**The `retrieve` node never sent its passages to the model.** It stored hits in
state; the `reason` node sent the bare question. Its own docstring claimed the
opposite — that pre-retrieving "guarantees the model has context on its first
turn". So the model opened every run by calling `search_documentation` to fetch
what had already been fetched. Offline that extra call is free and scripted, so
no test noticed and no reader doubted the comment. Live it was a third of the
latency and the bill. Fixing it paid back most of the extra reasoning turn.

**Citation numbering shown to the user could be wrong.** The answer node puts
tool output first, so after a tool round `[1]` is the risk engine — but the
evidence rail had been sent the retrieval list. Clicking `[1]` lit an unrelated
Confluence page. A citation that resolves to the wrong source is worse than no
citation: it is checkable, and it checks out wrong. The graph now publishes the
evidence list the citations index into, and the stream sends it again before the
answer.

## Consequences

- `Services` holds a `SyncResult` rather than a findings list, so the sync gate
  (ADR-004) reaches the agent: a `PartialSync` yields a risk tool that reports
  nothing and says why, instead of serving a fraction that reads as the whole
  picture.
- The service boots with findings. Offline that is a sandbox tenant
  (`app/core/sandbox.py`), and `/health` says which source is in use. A failed
  live sync never falls back to sandbox data — invented issues shown to someone
  who believes they are looking at their own tenant is the failure they would be
  least able to detect.
- Proposals are submitted to the `ApprovalStore` by the API, not by a graph
  node. A node that can enqueue an approvable action is one refactor away from
  being a node that applies one.

## Two more, found the first time it was pointed at a real tenant

Recorded here rather than in their own ADRs because both are the same mistake as
the one above: a claim nothing was checking.

**The audit trail asserted a Jira write that never happened.** `applied` meant
"the configured writer returned without error", and the Decisions panel rendered
that as ", written to Jira" — while the shipped writer was `LoggingWriter`, which
holds no Jira client. The fix is a separate `AuditEntry.written_to_jira`, read
from the writer's own `writes_to_jira` declaration and defaulting to False, so an
undeclared writer under-reports rather than inventing a change. It is stored per
row, not derived at render time, because an audit row is a durable claim about a
past event: rows written during a dry-run pilot must still say so after a real
writer is deployed.

**`rule_overdue` had no tests and did not work.** Jira returns `duedate` as a
bare calendar day — the only date-only field in the API — so the shared parser
produced a naive datetime, `issue.due_date >= now` raised TypeError inside the
rule, and `run_sync`'s per-project error handling turned that into a
`PartialSync` naming a permission problem that did not exist. One issue with a
due date took its entire project down, and every issue after it went unread.

The gate behaved correctly throughout, which is the point worth keeping: the
service degraded to "no findings, here is the operator message" rather than
serving a partial picture. But the diagnosis it offered was wrong, and a rule
that cannot evaluate one issue must skip that issue rather than abort the
project. Fixed at the parser (`_parse_due_date`, which also maps the day to its
end — otherwise every overdue issue is flagged a day early) and again in the
rule.


## Postscript: what "provide writing to Jira" actually required

`JiraIssueWriter` is one PUT and one POST. The design around it is the part
worth defending.

**Two switches, not one.** `can_write_to_jira` requires credentials, project
keys *and* `JIRA_ALLOW_WRITES`. Reading a client's tenant is how a pilot starts;
the day someone adds project keys must not be the day the service gains the
ability to edit their tickets. Only `1/true/yes/on` count — `bool(os.getenv(...))`
reads `"false"` as True, which is how a flag guarding writes gets enabled by
somebody trying to disable it.

**`update.labels[].add`, never `fields.labels`.** Sending `fields` replaces the
whole array, so flagging one issue at-risk would delete every other label a team
had put on it — silently, returning 204. The human approved *adding* a label; a
destructive edit is not a smaller-scoped version of that. It is the mutation
test worth keeping: swap `update` for `fields` and a test must fail.

**The registry still holds nothing that can write.** Nothing reachable from the
model imports this module. `ToolRegistry` gained no field; the one path in is
`apply_action`. A test asserts that, because the guarantee is an absence and
absences are what refactors quietly fill in.

## Postscript: seeding, and the rule that cannot be seeded

Four of the five risk rules can be given real data —
`scripts/seed_jira.py --sprints` creates sprints on the project's scrum board
and walks issues through them, so the carryover entries the rule counts are
genuine Jira history rather than fixtures.

`stale_in_progress` cannot. Jira refuses to backdate `created`, comments or
changelog entries, so nothing on a sandbox is older than the sandbox. The
tempting fix is dropping `stale_days_high` to 2 until something fires, and it is
the wrong one: it ships thresholds tuned to fake data, and the first thing a
client does is argue about those numbers.

Every rule takes `now` as a parameter — which was justified on Day 1 as making
time-dependent logic testable, and turns out to be the same property that makes
a fresh sandbox demonstrable. `RISK_CLOCK_OFFSET_DAYS` moves the clock forward,
`/health` reports the offset, and the UI states it in its own banner rather than
among the other caveats: the others say how much of the system is live, this one
changes what the numbers mean.
