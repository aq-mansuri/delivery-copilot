# ADR-007: A bounded state machine, not a free-running agent loop

**Status:** Accepted

## Context

The agent needs to call tools before answering. The default pattern is a ReAct
loop: think, act, observe, repeat until the model decides it is finished.

## Decision

A LangGraph state machine with a hard ceiling (`MAX_TOOL_ROUNDS = 3`). The model
chooses *which* tool. Whether to continue is a conditional edge — our code.

## Rationale

- **Bounded cost.** Cost per question has a maximum we can quote to a client.
  A model deciding its own stopping condition cannot see the bill.
- **Reproducibility.** The same question takes the same path. A loop that runs
  two steps on Monday and nine on Tuesday makes regressions unmeasurable.
- **Reviewability.** The flow is a diagram. A client security review asking
  "what can this agent do" gets an answer with edges, not a description of a
  model's judgement.
- **Guaranteed termination**, including on the paths the model gets wrong.

## Implementation notes

The final answer is produced by a separate `answer` node that re-asks rather
than reusing the reasoning turn's text. Costs one extra call; buys the guarantee
that everything user-facing passes the grounding check. Post-tool turns are
exactly where a model summarizes without citing.

Write tools are withheld unless `allow_writes` is set. Not offering a tool is
more reliable than instructing a model not to use one.

## Consequence discovered during implementation

State reducers and control flow interact badly. `tool_calls` accumulates for
tracing; the termination check originally read it, so after the first round
every turn looked like it had requested tools — an infinite loop created by a
reducer, not by the model. Fixed by splitting `pending_calls` (current turn
only, not accumulated) from `tool_calls` (full history). Both behaviours are now
pinned by tests.
