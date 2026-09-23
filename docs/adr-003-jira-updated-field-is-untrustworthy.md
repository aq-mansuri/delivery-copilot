# ADR-003: `updated` is not an activity signal

**Status:** Accepted

## Context

Jira bumps `fields.updated` on automation rules, bulk edits, and field syncs. A dead
ticket can show a recent `updated` timestamp — which is precisely the failure the
client described ("it looks healthy on the board").

## Decision

`Issue.last_activity_at()` computes the max of created, comment timestamps, and
changelog entry timestamps. Risk rules read that, never `updated`.

## Consequences

- Requires `expand=changelog` on every search, increasing payload size materially.
  Accepted: the changelog *is* the product.
- Covered by an explicit regression test so nobody "optimizes" it back.
