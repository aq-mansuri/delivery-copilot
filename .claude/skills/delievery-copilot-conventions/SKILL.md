---
name: delivery-copilot-conventions
description: Use when writing, reviewing, or extending any code in the Delivery Copilot project — adding risk rules, agent tools, graph nodes, approval logic, or tests. Covers the rules-versus-model split that governs where LLM calls are allowed, encoding client constraints in types rather than checks, the agent graph's termination discipline, mutation testing, and how the approval gate and audit trail must behave. Trigger on risk rule, agent tool, LangGraph node, approval, audit, ProposedAction, sync gate, or adding an LLM call.
---

# Delivery Copilot conventions

## Where the model is allowed

**Rules find the problems. The model explains them.**

Before adding an LLM call, establish that the thing cannot be computed. Stale
detection is a changelog query. Sprint carryover is a count. Slip arithmetic is
arithmetic. Handing any of these to a model gives something slower, costlier and
occasionally wrong on exactly the signal the client said they cannot afford to
miss.

The model's four jobs: narrative generation, Q&A with citations, classifying
free-text into fixed categories, and deciding when to decline.

Blocker classification is the one legitimate exception, and it is scoped: the
rules decide *whether* an issue is blocked; the model decides *what kind*.

## Encode constraints in types

Client requirements are made structurally unbreakable, not enforced by
discipline. A convention survives until the second caller is written by someone
who has not read the file.

Established patterns:

- `CompleteSync` vs `PartialSync` — `build_report` accepts only the former.
  `PartialSync` stores data under `partial_findings`, deliberately not
  `findings`, so it cannot duck-type into reporting code. Both are frozen.
- `ApprovedAction` — constructible only via `ApprovalStore.approve`, which
  requires a named actor. `apply_action` accepts nothing else.
- `ToolRegistry` holds **no Jira write client**. A write tool returns a
  `ProposedAction` because there is nothing there it could write with. The
  security property is an absence, not a check.

Add both a type signature and a runtime `isinstance` guard. The signature catches
the editor and CI; the guard catches dynamic callers a type checker never sees.

## The approval gate

- Audit record is written **before** the API call. If the write succeeds and
  logging dies, a regulator sees a change with no approval record. Recording
  first can leave an approved-but-unapplied entry, which is untidy and *visible*.
  Choose the failure you can see.
- Rejections are logged too. "A lead said no" is evidence, and it protects the
  lead when someone asks six months later why nothing was done.
- A proposal with no citations is refused at intake — there would be nothing for
  the approver to evaluate.

## The agent graph

A bounded state machine, not a ReAct loop (ADR-007). The model picks *which*
tool; whether to continue is a conditional edge. `MAX_TOOL_ROUNDS` gives cost per
question a quotable ceiling.

**Reducers and control flow do not mix.** `tool_calls` accumulates for tracing;
`pending_calls` holds only the current turn. The termination edge reads
`pending_calls`. Reading the accumulated field made every turn after the first
look like it requested tools — an infinite loop created by a reducer.

The `answer` node re-asks rather than reusing the reasoning turn's text. One
extra call, and it guarantees everything user-facing passes the grounding check.
Post-tool turns are where a model summarizes without citing.

Tool output is citable evidence (`Passage.from_tool`). Numbering only retrieval
chunks produced answers whose `[1]` pointed at an unrelated page.

Write tools are withheld unless `allow_writes` is set. Not offering a tool is
more reliable than instructing a model not to use one.

## Tool descriptions are prompt, not documentation

Say when **not** to use a tool. `search_documentation` explicitly tells the model
not to use search for risk questions — documentation describes intent, the risk
engine describes reality.

Phrase empty results so the model can distinguish "nothing is wrong" from "the
tool failed". Ambiguity there produces hedged answers.

Tool errors return as results, never exceptions, so the model can self-correct.

## Testing

- Everything runs offline. `FakeLLM` records calls so tests assert what was
  actually sent; it raises a pointed error when scripted responses run out,
  which usually means a loop is not terminating.
- **Mutation-test load-bearing behaviour**: break it, confirm the right tests
  fail, restore.
- Recorded fixtures (`TestAgainstRealTenantPayloads`, `live_answers.py`) are the
  only ones that can disprove an assumption. Do not normalize them.
- Ask why a test passes. Three bugs in this project survived green suites:
  a reducer loop, an inverted seeder that cancelled out an inverted reader, and
  citations resolving against the wrong list.

## Style

- `RiskConfig` holds every threshold. Each one is a number a client will argue
  about in week three; that argument should be a config change.
- Comments explain why, especially where the obvious implementation is wrong.
- Error messages say what to do next — see `PartialSync.operator_message()`.
