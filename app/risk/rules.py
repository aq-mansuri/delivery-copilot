"""Deterministic risk rules.

No LLM in this module. On purpose.

These rules find the problems; the model's job is only to explain them. Every
rule here is a pure function over an Issue, which makes them fast, free,
unit-testable, and identical on every run — properties you cannot get from a
model and that an auditor will ask for.

Tuning bias: false positives are cheap noise, false negatives cost Priya her
credibility with the board. Thresholds lean toward flagging.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

from app.models.sync import SyncAbort, SyncGapReason, SyncRecorder, SyncResult
from app.models.domain import (
    Citation,
    Issue,
    IssueLink,
    IssueStatusCategory,
    RiskFinding,
    RiskLevel,
)


@dataclass(frozen=True)
class RiskConfig:
    """Thresholds live in config, not scattered through the rules.

    Every one of these is a number a client will want to argue about in week
    three. Making them injectable means that argument is a config change.
    """

    stale_days_medium: int = 7
    stale_days_high: int = 14
    sprint_carryover_high: int = 2
    base_url: str = "https://example.atlassian.net"

    # Which link types produce findings at all. "relates to" is excluded by
    # default: it is the most-used and least-meaningful link in most tenants,
    # and a rule that fires on half the backlog trains leads to ignore the
    # report. Recall matters, but only for signals that carry information.
    reported_link_types: frozenset[str] = frozenset({"blocks", "depends_on"})

    # Status names that mean "this cannot move". Injected, and compared
    # case-insensitively.
    #
    # This is the one place a rule is allowed to look at a status NAME rather
    # than its category, and it needs the exception: `statusCategory` flattens
    # Blocked, On Hold, Impediment and In Progress all to `indeterminate`, so
    # the category cannot see a block a human recorded deliberately. The name
    # carries information the category throws away.
    #
    # It is configuration precisely because names differ per tenant. The list
    # below is a default, not a truth — `find_field.py`'s lesson applied to
    # statuses.
    blocked_status_names: frozenset[str] = frozenset(
        {
            "blocked",
            "on hold",
            "impediment",
            "impeded",
            "waiting",
            "waiting for customer",
            "waiting for support",
            "waiting for approval",
            "pending",
        }
    )


def _normalize_status(name: str) -> str:
    """Compare status names the way a human reads them, not byte for byte.

    Tenants produce " Blocked ", "blocked" and "BLOCKED" for the same column,
    and a rule that misses one because of a trailing space is a rule that
    silently under-reports on somebody's project.
    """
    return " ".join((name or "").split()).strip().lower()


def _issue_url(issue: Issue, cfg: RiskConfig) -> str:
    return f"{cfg.base_url}/browse/{issue.key}"


def _cite(issue: Issue, cfg: RiskConfig, excerpt: str | None = None) -> Citation:
    return Citation(
        source_type="issue",
        source_id=issue.key,
        url=_issue_url(issue, cfg),
        excerpt=excerpt,
    )


def _cite_link(link: IssueLink, cfg: RiskConfig) -> Citation:
    return Citation(
        source_type="issue",
        source_id=link.target_key,
        url=f"{cfg.base_url}/browse/{link.target_key}",
        excerpt=link.target_summary,
    )


def rule_stale_in_progress(
    issue: Issue, cfg: RiskConfig, now: datetime | None = None
) -> RiskFinding | None:
    """The rule Priya actually wants.

    "A ticket sits in 'In Progress' for three weeks with no commits, no
    comments, no status change. It looks healthy on the board. It's dead."

    Note this reads last_activity_at(), not issue.updated_at — see the docstring
    on the model for why that distinction is load-bearing.
    """
    if issue.status_category is not IssueStatusCategory.IN_PROGRESS:
        return None

    now = now or datetime.now(timezone.utc)
    idle_days = (now - issue.last_activity_at()).days

    if idle_days >= cfg.stale_days_high:
        level = RiskLevel.HIGH
    elif idle_days >= cfg.stale_days_medium:
        level = RiskLevel.MEDIUM
    else:
        return None

    return RiskFinding(
        issue_key=issue.key,
        rule_id="stale_in_progress",
        level=level,
        detail=(
            # The tenant's own status name, not the category the rule matched
            # on. `indeterminate` covers "In Progress", "Blocked", "Escalated"
            # and whatever else a client has invented, and firing on the
            # category is correct — but printing "In Progress" beside a ticket
            # marked Blocked is a statement the reader can check and find false,
            # which costs the other findings their credibility too.
            f"{issue.status_name} with no comments, transitions, or updates for "
            f"{idle_days} days (assignee: {issue.assignee_id or 'unassigned'})."
        ),
        citations=[_cite(issue, cfg, excerpt=issue.summary)],
    )


def rule_missing_target_release(
    issue: Issue, cfg: RiskConfig, now: datetime | None = None
) -> RiskFinding | None:
    """Data-quality rule, and an honest one.

    We are explicitly NOT cleaning this field (see non-goals). But an issue with
    no Target Release cannot be included in any date-commitment answer, so the
    system has to say so out loud rather than quietly omitting it. Silent
    omission is how you end up telling a board you're on track when you aren't.
    """
    if issue.status_category is IssueStatusCategory.DONE:
        return None
    if issue.target_release:
        return None

    return RiskFinding(
        issue_key=issue.key,
        rule_id="missing_target_release",
        level=RiskLevel.MEDIUM,
        detail=(
            "No Target Release set — this issue is excluded from delivery-date "
            "reporting and its slip risk cannot be assessed."
        ),
        citations=[_cite(issue, cfg)],
    )


def rule_sprint_carryover(
    issue: Issue, cfg: RiskConfig, now: datetime | None = None
) -> RiskFinding | None:
    """Counts how many times an issue has been moved between sprints.

    Derived from the changelog, so it's a fact rather than a lead's impression.
    Repeated carryover is the strongest early signal of a slip.
    """
    moves = [e for e in issue.changelog if e.field.lower() in {"sprint", "sprints"}]
    if len(moves) < cfg.sprint_carryover_high:
        return None

    return RiskFinding(
        issue_key=issue.key,
        rule_id="sprint_carryover",
        level=RiskLevel.HIGH,
        detail=(
            f"Carried over {len(moves)} times between sprints "
            f"(currently: {issue.sprint_name or 'no sprint'})."
        ),
        citations=[_cite(issue, cfg)],
    )


def _aware(moment: datetime) -> datetime:
    """Treat a naive datetime as UTC rather than raising on comparison.

    The parser (`_parse_due_date`) is where this is supposed to be settled, and
    it is. This is the second line, and it earns its place: `Issue` is a plain
    model that anything can construct — a fixture, a future integration, a
    script — and comparing one naive datetime raised TypeError *inside a rule*,
    which `run_sync` catches per project. One issue with a due date therefore
    aborted its entire project: every issue after it went unread, and the
    result was a PartialSync naming a permission problem that did not exist.

    A rule that cannot evaluate an issue should skip that issue, never take the
    project down with it.
    """
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def rule_overdue(
    issue: Issue, cfg: RiskConfig, now: datetime | None = None
) -> RiskFinding | None:
    now = _aware(now or datetime.now(timezone.utc))
    if issue.due_date is None:
        return None
    if issue.status_category is IssueStatusCategory.DONE:
        # A shipped ticket that missed its date is history, not risk. Reporting
        # it fills the page with items nobody can act on.
        return None

    due = _aware(issue.due_date)
    if due >= now:
        return None

    return RiskFinding(
        issue_key=issue.key,
        rule_id="overdue",
        level=RiskLevel.HIGH,
        detail=f"Past due date by {(now - due).days} days.",
        citations=[_cite(issue, cfg)],
    )


MAX_LISTED_LINKS = 5


def rule_blocked_dependency(
    issue: Issue, cfg: RiskConfig, now: datetime | None = None
) -> RiskFinding | None:
    """An issue held up by work that isn't finished.

    The failure this catches: the blocked ticket itself looks fine. Assignee,
    sprint, recent activity — and it cannot move, because another team hasn't
    shipped the thing it depends on. Nothing on the board says so.

    Severity turns on direction. Blocked *by* unfinished work is a direct threat
    to this issue's date (HIGH). Merely blocking something else, or relating to
    it, is context rather than a slip signal (MEDIUM).
    """
    if issue.status_category is IssueStatusCategory.DONE:
        return None

    unresolved = [
        link
        for link in issue.links
        if link.link_type in cfg.reported_link_types and not link.is_resolved()
    ]
    if not unresolved:
        return None

    blocking = [link for link in unresolved if link.target_blocks_this]

    # Report on the same set we counted. Listing every unresolved link under a
    # "blocked by 1" headline makes the sentence contradict itself.
    reported = blocking or unresolved
    # Sorted so the same input always produces the same string — required for
    # eval baselines and for week-over-week report diffing (see ADR-001).
    reported = sorted(reported, key=lambda link: link.target_key)

    shown = reported[:MAX_LISTED_LINKS]
    listed = ", ".join(
        f"{link.target_key} ("
        f"{link.target_status_category.value if link.target_status_category else 'status unknown'}"
        f")"
        for link in shown
    )
    if len(reported) > len(shown):
        listed += f", +{len(reported) - len(shown)} more"

    if blocking:
        level = RiskLevel.HIGH
        detail = f"Blocked by {len(blocking)} unfinished issue(s): {listed}."
    else:
        level = RiskLevel.MEDIUM
        detail = f"Linked to {len(unresolved)} unfinished issue(s): {listed}."

    return RiskFinding(
        issue_key=issue.key,
        rule_id="blocked_dependency",
        level=level,
        detail=detail,
        citations=[_cite(issue, cfg, excerpt=issue.summary)]
        + [_cite_link(link, cfg) for link in reported],
    )


def rule_blocked_status(
    issue: Issue, cfg: RiskConfig, now: datetime | None = None
) -> RiskFinding | None:
    """A ticket somebody moved to Blocked, with no blocker linked.

    Found on a live tenant, and it is the failure that worries me most of the
    set. INS-12's status was `Blocked` and it had no issue links, so the
    link-based rule correctly found nothing — and the agent reported "INS-12 is
    not blocked" as a definitive result, about a ticket whose own status said
    otherwise. Confidently wrong is worse than silent.

    A person deliberately moving a ticket to Blocked is the strongest signal in
    this file: every other rule infers a problem, this one is somebody stating
    it. Ignoring it because the API shape is less convenient than a link would
    be exactly backwards.

    MEDIUM rather than HIGH, and the reason is in the detail: nobody recorded
    *what* is blocking it, so there is no dependency to chase and no date to
    assess. The actionable output is "link the blocker", which is what the
    message says.
    """
    if issue.status_category is IssueStatusCategory.DONE:
        return None
    if _normalize_status(issue.status_name) not in cfg.blocked_status_names:
        return None

    # Stand down only when a link actually identifies what this issue is
    # waiting on. Then `rule_blocked_dependency` names the blocker and its
    # state, which is strictly more useful, and firing both would put two
    # findings on one problem — a report that double-counts is one a delivery
    # lead learns to discount.
    #
    # Keyed on the blocking links themselves, NOT on whether that rule returned
    # something. It also reports the reverse direction ("Linked to X", i.e. this
    # issue blocks something else), which explains nothing about why this one
    # cannot move. Suppressing on that swallowed a human's explicit Blocked
    # marking and replaced it with a weaker automatic signal pointing the wrong
    # way.
    explains_the_block = any(
        link.target_blocks_this
        and link.link_type in cfg.reported_link_types
        and not link.is_resolved()
        for link in issue.links
    )
    if explains_the_block:
        return None

    return RiskFinding(
        issue_key=issue.key,
        rule_id="blocked_status",
        level=RiskLevel.MEDIUM,
        detail=(
            f"Status is {issue.status_name!r}, but no blocking issue is linked "
            f"— so what it is waiting on is not recorded anywhere, and the "
            f"dependency cannot be chased or dated."
        ),
        citations=[_cite(issue, cfg, excerpt=issue.summary)],
    )


logger = logging.getLogger(__name__)


ALL_RULES = [
    rule_stale_in_progress,
    rule_missing_target_release,
    rule_sprint_carryover,
    rule_overdue,
    rule_blocked_dependency,
    rule_blocked_status,
]


def evaluate(
    issues: list[Issue], cfg: RiskConfig, now: datetime | None = None
) -> list[RiskFinding]:
    """Run every rule against every issue.

    3,000 open issues x 4 pure functions. No batching, no queue, no cache.
    Priya's scale does not justify anything more, and over-engineering this
    would be the wrong instinct to show a client.
    """
    now = now or datetime.now(timezone.utc)
    findings: list[RiskFinding] = []
    for issue in issues:
        for rule in ALL_RULES:
            if (finding := rule(issue, cfg, now)) is not None:
                findings.append(finding)
    return findings


async def evaluate_stream(
    issues: AsyncIterator[Issue],
    cfg: RiskConfig,
    recorder: SyncRecorder,
    now: datetime | None = None,
) -> SyncResult:
    """Stream issues through the rules, binding the output to sync completeness.

    Returns a SyncResult rather than a list of findings, and that is the whole
    design: there is no signature here that hands back bare findings, so no
    caller can accidentally route around the gate.

    An upstream failure does not propagate. It is recorded as a gap and the
    stream ends, which leaves a PARTIAL result — findings a lead can triage,
    that the report generator will refuse. Re-raising would be the safe-looking
    choice and the worse one: it throws away work, and it tempts the caller
    into a `try/except: report_anyway` that silently reintroduces the bug.

    Cancellation is the exception: it is re-raised per asyncio convention. That
    is still safe, because no SyncResult is produced at all, and no result means
    no report.
    """
    now = now or datetime.now(timezone.utc)
    findings: list[RiskFinding] = []

    try:
        async for issue in issues:
            recorder.record_issue_seen()
            for rule in ALL_RULES:
                if (finding := rule(issue, cfg, now)) is not None:
                    findings.append(finding)
    except asyncio.CancelledError:
        recorder.record_gap(
            SyncGapReason.ABORTED,
            scope="sync",
            detail=(
                f"Cancelled after {recorder.issues_seen} issues; the remainder "
                f"were never read."
            ),
        )
        raise
    except SyncAbort as exc:
        recorder.record_gap(exc.gap_reason, scope=exc.scope, detail=str(exc))
    except Exception as exc:  # any failure at all means data is missing
        logger.exception("sync aborted after %d issues", recorder.issues_seen)
        recorder.record_gap(
            SyncGapReason.UPSTREAM_ERROR,
            scope="sync",
            detail=f"{type(exc).__name__}: {exc}",
        )

    return recorder.finish(findings, now)
