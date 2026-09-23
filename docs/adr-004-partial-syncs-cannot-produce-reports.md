# ADR-004: Partial syncs are a distinct type, not a flagged result

**Status:** Accepted

## Context

Risk findings are computed while streaming issues from Jira. A sync can end early:
429 exhaustion, a 403 on one of six projects, a timeout.

If findings from an aborted run reach the report, the report *looks complete* while
silently omitting issues. That is the specific failure the client named — telling the
board you're on track when you aren't — and it is worse than a loud crash, which at
least announces itself.

## Decision

`run_sync` returns `CompleteSync | PartialSync`. These are separate types.
`build_report` accepts `CompleteSync` only, enforced by signature and by a runtime
`isinstance` check for dynamic callers.

`PartialSync` stores its data under `partial_findings`, not `findings`, so it cannot
duck-type into reporting code.

Both types are frozen, so a caller cannot mutate `status` and proceed.

## Rejected alternative

A single `SyncResult` with a `status` field and a convention that consumers check it.
Rejected because the convention is unenforced: it survives until the second consumer
is written, and the failure mode when it breaks is silent.

## Consequences

- A permission error on one project blocks the whole week's report. Accepted — that
  is the intended behaviour, and `operator_message()` names the fix so the gap is
  short-lived.
- Per-project boundaries are preserved in `projects_covered` / `projects_incomplete`,
  so partial results remain useful for diagnosis without being publishable.
- The gate sits upstream of the LLM, so an incomplete sync never costs tokens.
