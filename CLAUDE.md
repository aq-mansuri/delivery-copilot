# Delivery Copilot — working instructions

Agentic delivery-risk service over Jira and Confluence, built for a regulated
insurer. Read `docs/PRD.md` for the problem and `docs/adr-*.md` for decisions
already made and their reasoning.

## The rule that governs everything

**Rules find the problems. The model explains them.**

Before adding an LLM call, ask whether the thing can be computed. Stale-ticket
detection is a changelog query, not an inference. If a deterministic version
exists, it wins: faster, cheaper, auditable, identical every run.

The model is scoped to four jobs — narrative generation, Q&A with citations,
classifying free-text into categories, and deciding when to decline.

## Constraints encoded in types, not in discipline

These are client requirements, not preferences. Each is enforced structurally so
that code written later by someone who has not read this file still cannot break
them.

| Rule | How it is enforced |
|---|---|
| A partial sync cannot produce a report | `build_report` accepts `CompleteSync`; `PartialSync` stores data under `partial_findings` so it cannot duck-type |
| A partial sync cannot answer a risk question either | `risk_tool_for` returns a tool with no findings and an operator message; the agent is the second consumer of the same data (ADR-009) |
| Unapproved changes cannot reach Jira | `apply_action` accepts `ApprovedAction`, constructible only via `ApprovalStore.approve`; `ToolRegistry` holds no write client at all |
| No claim without a source | `check_grounding` validates every citation against the passages supplied; ungrounded answers are rejected, not warned |
| The audit trail never claims a write that did not happen | `AuditEntry.written_to_jira` is recorded per row from the writer's own `writes_to_jira`, defaulting to False; `applied` only ever meant "the writer returned" |
| A status a human set is not invisible to the engine | `rule_blocked_status` fires on `RiskConfig.blocked_status_names` — the one rule allowed to read a status name, because `statusCategory` flattens Blocked into In Progress (ADR-001 postscript) |
| A citation resolves to what the answer meant | the graph publishes `evidence` — the list the citations index into — and the stream sends it; numbering `hits` is wrong after any tool round |

When adding a constraint, prefer making the bad state unrepresentable over
adding a check somebody must remember to call.

## Testing

- Every test runs offline, and `tests/conftest.py` enforces it by refusing
  outbound sockets rather than trusting the rule. Added after a `/sync` test
  branched on `settings().has_jira`, read the developer's live tenant and passed
  — two of its assertions were meaningless and nobody could tell. If the code
  under test branches on configuration, pin the environment with a fixture.
- **Mutation-test anything load-bearing.** Break the behaviour deliberately,
  confirm the right tests fail, restore. A green suite proves nothing; a suite
  that fails for the correct reason proves something.
- Fixtures recorded from reality live in `tests/fixtures/jira_payloads.py`
  (`TestAgainstRealTenantPayloads`) and `tests/fixtures/live_answers.py`. These
  are the only fixtures that can disprove an assumption rather than restate one.
  Do not "tidy" them to look like the hand-written ones.
- When a test passes, occasionally ask *why*. "The right chunk happened to rank
  first" is not "the code selected the right chunk."

## An empty result is an answer, not a gap

The rules are exhaustive over what they check, so "no findings for INS-12" is
not missing context — it is the answer, and the answer is no. Phrase every
empty tool result as a positive assertion that names what was checked and says
it is definitive. A caveat ("this means the rules found nothing, not that the
check failed") is not enough: the answering model classified that as missing
context and declined, so a lead asking "is INS-12 blocked?" was told the system
did not know when it did (ADR-006 postscript).

Fix this kind of thing in the **tool**, not in `answering.SYSTEM_PROMPT`. That
prompt governs refusal on genuinely unanswerable questions, which is baselined
at 100%; a tool only speaks when its rules actually ran, so its blast radius is
the case you are fixing.

**Anchor the negative to when the data was read.** Findings are a snapshot;
`POST /sync` refreshes them and `/health` reports the age. A confident "not
blocked" from a startup sync was returned twice for an issue blocked minutes
earlier. The risk tool's negative now carries the sync timestamp.

**Scope the negative to what was actually checked.** The first version of this
ended "it is not blocked and not at risk" and was then produced for a ticket
whose status was `Blocked` — the rules had missed it because they only read
links. Making a negative more assertive is what creates the risk of a confident
wrong answer, and that is worse than the refusal it replaced. Enumerate the
rules that did not match and say the set is not a judgement of overall health.

## Measure before changing

Every tuning change is made against a number, never an impression.

    python scripts/run_eval.py                  # retrieval baseline
    python scripts/run_eval.py --decompose      # with query decomposition
    python scripts/diagnose_grounding.py --live # per-paragraph grounding verdicts

`docs/baseline.json` is the before-picture. Its whole value is that it predates
the fixes.

Two corrections already came from measuring first: a retrieval score floor
cannot separate answerable from unanswerable questions (ADR-006), and the
grounding checker's per-sentence rule caused a 100% refusal rate (ADR-008).

## The UI is the whole system, not half of it

`/ask` runs the agent graph. It was a retrieval-only path for a while, which
meant the tool loop and the approval gate were real, tested, and reachable only
from `scripts/demo_agent.py` — so "Flag INS-101 as at risk" came back as a
grounded answer *about* INS-101 and nothing was ever proposed. ADR-009 has the
detail. When adding a capability, ask where a person clicks it; a guarantee only
a script can demonstrate is one a client has no reason to believe.

`write_intent()` decides whether write tools are offered. Keep it in step with
`apply_action`'s branches: add an action there, add its phrasing there, add a
case to `tests/test_intent.py`.

## Running it

    pytest                              # offline, no API key
    cd frontend && npm run check        # answer renderer, offline
    cd frontend && npm run build        # Vite builds into app/api/static
    uvicorn app.api.main:app            # the service, UI included
    python scripts/demo.py              # rules + retrieval, no model
    python scripts/demo_agent.py        # full agent, scripted
    python scripts/demo_agent.py --live # full agent, real Claude

The UI is a built bundle. Editing `frontend/src` changes nothing a browser sees
until `npm run build` runs.

Offline, findings come from the sandbox tenant in `app/core/sandbox.py`. Set
`JIRA_PROJECT_KEYS` to sync a real one; the two are never mixed, and a failed
live sync serves no findings rather than falling back to invented ones.

Documentation is always the bundled corpus — `ConfluenceClient` is written and
tested but not wired in. `/health` reports it, along with which writer is loaded
and whether the evaluation clock has been shifted. Anything the UI shows must
distinguish what was recorded from what was done: `applied` means "the writer
returned", never "Jira changed".

`JiraIssueWriter` is the only code that mutates a tenant. It needs credentials,
project keys AND `JIRA_ALLOW_WRITES` — credentials alone never imply permission.
Labels go through `update.labels[].add`; `fields.labels` replaces the array and
silently deletes every label the team put there.

## Seeding, and the one rule you cannot seed

`scripts/seed_jira.py --project INS --count 14 --sprints 3`. Sprint carryover
needs `--sprints`: it creates real sprints on the project's scrum board and
walks issues through them, so the changelog entries the rule counts are genuine.

`scripts/seed_blocked_status.py --project INS --pairs 4` adds a `Blocked` status
to the project's workflow and blocks issues behind in-progress ones. It refuses
to edit a workflow scheme shared by more than one project. The status is
presentation only — `Blocked` is category `IN_PROGRESS`, so the rules cannot
tell it from `In Progress`, and `rule_blocked_dependency` finds these through
their **links** as it always did. Never key a rule off a status name; do print
`issue.status_name` in a finding's detail, because "In Progress" beside a ticket
marked Blocked is a claim the reader can falsify.

Stale detection cannot be seeded at all — Jira refuses to backdate `created`,
comments and changelog entries. Do not lower `stale_days_high` to make it fire;
that ships thresholds tuned to fake data and loses the argument in week three.
Shift the clock instead (`RISK_CLOCK_OFFSET_DAYS`, or pass `now=` to
`evaluate`), which is why every rule takes `now` as a parameter.

Against a real tenant, always run `scripts/check_jira_connection.py` first. It
reports whether changelogs are arriving and whether the custom field id is right
— both fail silently otherwise.

## Atlassian specifics that cost time to learn

- `/rest/api/3/search` is deprecated. Use `/rest/api/3/search/jql` with
  `nextPageToken`; there is no `total`.
- `/search/jql` returns only `id` unless `fields` is requested explicitly.
- `fields.updated` is bumped by automation. Use `Issue.last_activity_at()`,
  computed from changelog and comments (ADR-003).
- Custom field ids are per-tenant. Never hardcode; see `scripts/find_field.py`.
- Link direction: in a *response*, `inwardIssue` is the blocker. In a *write
  request*, the semantics invert. Both are verified against a live tenant.
- Confluence paginates by cursor and the original filters must not be re-sent
  with it.

## Style

- Comments explain *why*, especially where the obvious implementation is wrong.
  A comment restating the code is noise.
- Error messages say what to do next. `PartialSync.operator_message()` and the
  403 handler naming Browse Projects are the pattern.
- Thresholds live in config (`RiskConfig`), never scattered through rules —
  every one is a number a client will argue about in week three.
