# ADR-002: Read-only by default; writes require human approval

**Status:** Accepted

## Context

Client is a regulated insurer. Tickets carry compliance sign-offs and the audit trail
must show a human made each change. Client stated directly that autonomous writes
would end the project.

## Decision

The agent emits `ProposedAction` objects. Nothing reaches Jira's write API until a
named human approves, and every approval/rejection is persisted with actor and
timestamp.

## Consequences

- Agent design is constrained to propose-and-confirm; no autonomous loops.
- Approval log becomes a product feature (audit evidence), not just plumbing.
- Slower than full automation. This is the point.
