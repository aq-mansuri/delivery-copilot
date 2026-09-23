# ADR-001: Risk detection is deterministic, not model-based

**Status:** Accepted

## Context

The client asked for "AI on top of Jira." The highest-value signal she described —
a ticket sitting In Progress for weeks with no real activity — is an inference from
*absence of activity* in the changelog.

## Decision

All risk detection is pure functions over issue changelogs. No LLM participates in
deciding whether something is at risk.

## Rationale

- **Auditable.** A regulated insurer must be able to answer "why was this flagged."
  "The model decided" is not an answer; a rule ID and a threshold is.
- **Deterministic.** Same input, same output. Required for eval baselines and for
  client trust after the first disagreement.
- **Cost and latency.** 3,000 issues x 4 rules is milliseconds and $0.
- **Recall.** The client cannot tolerate false negatives on this signal. A rule with
  a tuned threshold gives a guarantee a prompt cannot.

## Consequences

- Thresholds become a negotiation surface — mitigated by injectable `RiskConfig`.
- Rules cannot catch novel risk phrasing in free text — accepted; comment
  *classification* is delegated to the LLM, detection is not.


## Postscript: the signal the engine could not see

`rule_blocked_dependency` reads issue **links**. On a live tenant a ticket was
moved to a `Blocked` status with no link recorded, and the engine reported
nothing — correctly, by its own definition, and uselessly to the lead who had
just marked it.

Worse, the agent turned that silence into "INS-12 is not blocked", stated as a
definitive result. Confidently wrong about a ticket whose own status said
otherwise is the worst output this system can produce.

`rule_blocked_status` closes it. Two things about it are worth defending:

**It reads a status NAME, which every other rule is forbidden from doing.** The
prohibition exists because names differ per project and per tenant, and
`statusCategory` is the stable field. But the category is exactly what destroys
this signal: `Blocked`, `On Hold`, `Impediment` and `In Progress` are all
`indeterminate`. The name carries information the category throws away. So the
name is read, and the vocabulary lives in `RiskConfig.blocked_status_names`
where a client can change it — the same treatment as `reported_link_types`, and
the same lesson as never hardcoding a custom field id.

**It is MEDIUM, and stands down when a link already explains the block.** A
status with no linked blocker says *that* something is wrong without saying
what, so there is no dependency to chase and no date to assess — which is what
the detail tells the reader to fix. When a link names the blocker, that is
strictly more useful, and firing both would put two findings on one problem. A
report that double-counts is one a delivery lead learns to discount.

### The stand-down has to be narrower than it first looks

The first version suppressed whenever `rule_blocked_dependency` returned
anything. That rule reports two different things under one id: HIGH means
"blocked by X" and names what the issue is waiting on; MEDIUM means "linked to
X", which is usually the **reverse** direction — this issue blocks something
else — and explains nothing about why it cannot move.

So a ticket a human had marked `Blocked`, whose only link pointed downstream,
was reported as "Linked to 1 unfinished issue". True, pointing the wrong way,
and never mentioning that somebody had flagged it. An explicit human signal
swallowed by a weaker automatic one.

The stand-down now keys on the blocking links themselves — unresolved, of a
reported type, and pointing *at* this issue — rather than on whether the other
rule happened to return something. The lesson generalises: when one rule defers
to another, the condition must be the fact that justifies deferring, not the
other rule's return value. Return values bundle cases together; facts do not.
