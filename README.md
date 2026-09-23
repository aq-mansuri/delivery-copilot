# Delivery Copilot

An agentic service over Jira and Confluence that surfaces delivery risk, answers
questions with citations, and proposes changes a named human approves.

Built as a one-week exercise against a realistic brief: an insurance company
whose engineering leads spend half of every Monday assembling a status deck
whose numbers are always slightly wrong.

```bash
docker compose up --build        # needs ANTHROPIC_API_KEY in .env
```

---

## The decision the whole thing rests on

**Rules find the problems. The model explains them.**

The client asked for "AI on top of Jira". The highest-value signal they
described — a ticket sitting In Progress for three weeks with no commits, no
comments, no transitions, looking healthy on the board — is a changelog query,
not an inference.

| Deterministic | LLM |
|---|---|
| Stale detection, blocked dependencies, blocked-with-no-blocker-recorded, sprint carryover, slip arithmetic | Narrative generation, Q&A with citations, classifying free text, deciding when to decline |

Risk detection is a pure function over issue changelogs: faster, cheaper,
auditable, identical on every run. The model never decides what is at risk, and
never writes to Jira.

## Three constraints, encoded in types rather than discipline

The client is a regulated insurer. Each rule below is structural, so code
written later by someone who has not read this cannot break it.

**A partial sync cannot produce a report.** `build_report` accepts
`CompleteSync`; `PartialSync` stores its data under `partial_findings` so it
cannot duck-type into reporting code. A report that looks complete while
silently omitting 200 unread issues is the failure the client named.

**Unapproved changes cannot reach Jira.** `apply_action` accepts only
`ApprovedAction`, constructible solely through `ApprovalStore.approve`, which
requires a named actor. `ToolRegistry` holds no write client at all — the
security property is an absence, not a check.

**No claim without a source.** Every citation is validated against the passages
actually retrieved. An answer with an unattributable claim is rejected, not
warned. A warned answer still gets read, and the reader is about to brief a
board.

All three are reachable from the product, not only from a script. That sentence
is there because for most of the build they were not: `/ask` was a retrieval
path with no tool loop, so "Flag INS-101 as at risk" returned a grounded answer
*about* INS-101 and proposed nothing. Everything was tested and green. A
guarantee the UI cannot demonstrate is a guarantee a client has no reason to
believe — [ADR-009](docs/adr-009-the-ui-runs-the-agent.md), which also records
the two bugs that only became visible once the graph ran outside its own tests.

## Measured, not asserted

Nothing here was tuned by impression. Numbers from
`scripts/check_regressions.py --live --runs 3`, 102 claims:

| | |
|---|---|
| Retrieval recall@5 | 62.5% (cross-source 100%, identifier 100%) |
| Claims supported by their cited source | 85.3% |
| Overreach — stronger than the source | 8.8% |
| Unsupported — in no retrieved passage | 2.0% |
| Unanswerable questions declined | 100% (3/3) |
| Adversarial probes held | 6/6 |
| Cost per answer / per evaluated question | $0.005 / $0.015 |

Two of these exist because measuring first contradicted the obvious design:

- A **retrieval score floor cannot produce refusal** (ADR-006). Measured,
  answerable questions scored BM25 1.87–7.25 and unanswerable ones 2.54–5.21 —
  overlapping, because negatives ask for absent *facts* about present *topics*.
  Refusal moved to generation time.
- **Per-sentence grounding produced a 100% refusal rate** (ADR-008). The cause
  was an off-by-one: models write the citation after the full stop, so a naive
  sentence split credited it to the next sentence. A 100% refusal rate is not
  safety; it is a false negative wearing a safety jacket.

The judge that produces the groundedness numbers is itself calibrated against
18 labelled cases plus 8 held out. Held-out detection is 88% with no dangerous
misses — every unsupported claim was flagged; the errors are over-flagging,
which is the safe direction. The fitted set scores 100%, and that gap is
co-adaptation, so **88% is the number to quote**.

## What is not production-ready

Stated plainly because a README that implies more than the code delivers is how
a client finds out during a demo. Full detail in [DEPLOY.md](DEPLOY.md).

- **Writes are off unless two switches are thrown.** `JiraIssueWriter` is real
  and verified against a live tenant, but `build_services` injects it only when
  credentials, project keys *and* `JIRA_ALLOW_WRITES` are all present.
  Credentials alone never imply permission. The default writer is a dry run that
  declares itself, `/health` reports which is loaded, and every audit row carries
  `written_to_jira` — so the Decisions panel says "recorded — dry run" rather
  than claiming a change. It used to claim the change.
- **Stale detection needs a shifted clock on a fresh sandbox.** Jira will not
  backdate `created`, comments or changelog entries, so nothing on a newly
  seeded tenant has sat still long enough to be stale. `RISK_CLOCK_OFFSET_DAYS`
  moves the evaluation clock forward rather than lowering the thresholds, which
  would ship numbers tuned to fake data. Off by default; `/health` and the UI
  both say when it is on.
- **Documentation is always the bundled corpus.** `ConfluenceClient` is written
  and tested but not wired into the service, so with `JIRA_PROJECT_KEYS` set the
  risk panel shows your issues while documentation answers describe the sample
  space. `/health` reports both sources and the UI says so.
- **Approvals do not survive a restart.** `ApprovalStore` is in memory.
- **Findings refresh only when asked.** `POST /sync` re-reads Jira and the UI
  shows how old the current read is, because there is no push from Jira and
  re-syncing per question would make every answer wait on a full project read.
  This was worse than a limitation before it was built: the service answered
  "INS-12 is not blocked" twice about an issue that had just been blocked, from
  a snapshot taken at startup. `JIRA_PROJECT_KEYS` unset means the sandbox
  tenant in `app/core/sandbox.py`; `/health` reports which and when.
- **Write intent is keyword-matched** (`app/agent/intent.py`). It decides only
  whether write tools are *offered*; a miss costs a round-trip and a false
  positive costs a proposal someone rejects. Neither can reach Jira.
- **The vector index rebuilds per process.**
- **`X-Actor` is self-asserted.** The audit trail records a claim, not an
  identity. Behind an SSO proxy it becomes real; today it is a log.
- Temporal questions ("what changed most recently") are unsupported — retrieval
  cannot order by time.
- Cross-source recall is measured any-of, not all-of, so the reported 100% is
  "found one of two required pages".

## Layout

```
app/
  risk/          deterministic rules — the flagship logic
  rag/           chunking, hybrid retrieval, decomposition, evaluation
  agent/         answering, tools, approval gate, graph, judge
  integrations/  Jira and Confluence clients (+ the one writer)
  core/          sync gate, tracing, regression thresholds, config
  api/           FastAPI service + built React UI
frontend/        React source (Vite builds into app/api/static)
scripts/         things you run: seeding, evals, diagnostics, demos
docs/            PRD, 9 ADRs, eval set, regression baseline
tests/           495 tests, all offline
```

## Running things

```bash
pytest                                       # 495 tests, no API key needed
python scripts/demo.py                       # rules + retrieval, no model
python scripts/demo_agent.py                 # full agent, scripted
python scripts/demo_agent.py --live          # full agent, real Claude
python scripts/check_regressions.py          # retrieval, instant
python scripts/check_regressions.py --live --runs 3
python scripts/run_groundedness.py --live    # per-claim verdicts
python scripts/check_adversarial.py --live   # fabrication probes
python scripts/demo_tracing.py --live        # cost and latency
python scripts/seed_jira.py --project INS --count 14 --sprints 3
python scripts/seed_blocked_status.py --project INS --pairs 4
cd frontend && npm run check                 # answer renderer, offline
```

Every test runs with no API key, no network and no bill — `FakeLLM` and
`FakeEmbedder` are protocol implementations, not mocks. `tests/conftest.py`
enforces it by refusing outbound sockets, which was added after a `/sync` test
branched on `settings().has_jira`, read the developer's live tenant, and passed
locally while it would have failed in CI.

## Reading order

For anyone assessing the engineering rather than the output:

1. [docs/adr-002](docs/adr-002-no-unapproved-writes.md) and
   [adr-004](docs/adr-004-partial-syncs-cannot-produce-reports.md) — constraints
   as types
2. [docs/adr-006](docs/adr-006-refusal-is-a-generation-concern.md) and
   [adr-008](docs/adr-008-grounding-granularity.md) — two designs reversed by
   measurement
3. `app/agent/judge.py` — the judge, and why it is calibrated before it is
   trusted
4. `tests/fixtures/jira_payloads.py` (`TestAgainstRealTenantPayloads`) and
   `tests/fixtures/live_answers.py` — the only fixtures recorded rather than
   imagined

That last point is the one I would defend hardest. Every other fixture was
written from documentation and memory; a live payload diff found that my seeder
created issue links backwards while the reader parsed them correctly — two
errors that cancelled out invisibly, and would have stayed hidden under a green
suite forever.
