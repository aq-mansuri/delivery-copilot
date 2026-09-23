"""The offline tenant.

Same argument as `app/rag/corpus.py`, applied to Jira. The running service
needs risk findings to have anything to say about delivery state, and until a
client's credentials are in the environment there is no Jira to get them from.
Before this existed the API shipped `findings=[]`, so the risk tool answered
"nothing is at risk" with total confidence — the worst available failure, since
it is indistinguishable from good news.

The issues here are the ones the Confluence corpus talks about, so the two
halves of the sandbox agree with each other: the release plan says the claims
vendor integration is stalled, and INS-101 is the stalled one.

`scripts/` holds things you run; `app/` holds things that run. Both demos import
from here rather than keeping their own copy, which is also how the copies
stopped drifting.

## This is never mixed with real data

`build_services()` uses the sandbox only when no Atlassian credentials are
configured, and `/health` says which it is using. Falling back to invented
issues when a real sync fails would put fabricated delivery state in front of a
lead who believes they are looking at their own tenant.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.sync_result import CompleteSync
from app.models.domain import (
    ChangelogEntry,
    Comment,
    Issue,
    IssueLink,
    IssueStatusCategory,
)
from app.risk.rules import RiskConfig, evaluate

SANDBOX_BASE_URL = "https://acme.atlassian.net"
SANDBOX_PROJECTS = ["INS"]


def sandbox_config() -> RiskConfig:
    return RiskConfig(base_url=SANDBOX_BASE_URL)


def sandbox_issues(now: datetime | None = None) -> list[Issue]:
    """Four issues, each carrying one thing worth demonstrating."""
    now = now or datetime.now(timezone.utc)
    return [
        # Looks healthy on the board. Assignee, sprint, recent `updated`.
        # Hasn't actually moved in 23 days.
        Issue(
            key="INS-101",
            project_key="INS",
            summary="Integrate claims vendor callback",
            status_name="In Progress",
            status_category=IssueStatusCategory.IN_PROGRESS,
            assignee_id="acc-dev-1",
            sprint_name="Sprint 14",
            target_release="Q4-2026",
            created_at=now - timedelta(days=23),
            updated_at=now,  # automation bumped this today
        ),
        # Genuinely blocked by another team, and nothing on the board says so.
        Issue(
            key="INS-102",
            project_key="INS",
            summary="Enable partner auth on claims portal",
            status_name="In Progress",
            status_category=IssueStatusCategory.IN_PROGRESS,
            assignee_id="acc-dev-2",
            target_release="Q4-2026",
            created_at=now - timedelta(days=10),
            updated_at=now,
            comments=[
                Comment(
                    id="c1",
                    body="Waiting on the platform team's auth endpoint.",
                    created_at=now - timedelta(days=2),
                )
            ],
            links=[
                IssueLink(
                    target_key="PLAT-7",
                    target_summary="Ship partner auth endpoint",
                    target_status_category=IssueStatusCategory.IN_PROGRESS,
                    link_type="blocks",
                    target_blocks_this=True,
                )
            ],
        ),
        # Carried over three sprints. Fact from the changelog, not an impression.
        Issue(
            key="INS-103",
            project_key="INS",
            summary="Migrate legacy policy records",
            status_name="In Progress",
            status_category=IssueStatusCategory.IN_PROGRESS,
            assignee_id="acc-dev-3",
            target_release=None,  # the 40% problem
            sprint_name="Sprint 14",
            created_at=now - timedelta(days=45),
            updated_at=now,
            comments=[
                Comment(
                    id="c2",
                    body="Still scoping.",
                    created_at=now - timedelta(days=1),
                )
            ],
            changelog=[
                ChangelogEntry(field="Sprint", at=now - timedelta(days=28)),
                ChangelogEntry(field="Sprint", at=now - timedelta(days=14)),
                ChangelogEntry(field="Sprint", at=now - timedelta(days=1)),
            ],
        ),
        # Healthy. Should produce no findings at all — a rule set that flags
        # everything is a rule set nobody reads.
        Issue(
            key="INS-104",
            project_key="INS",
            summary="Add audit logging to approvals",
            status_name="In Progress",
            status_category=IssueStatusCategory.IN_PROGRESS,
            assignee_id="acc-dev-1",
            target_release="Q4-2026",
            created_at=now - timedelta(days=6),
            updated_at=now,
            comments=[
                Comment(
                    id="c3",
                    body="PR open.",
                    created_at=now - timedelta(days=1),
                )
            ],
        ),
    ]


def sandbox_sync(now: datetime | None = None) -> CompleteSync:
    """The sandbox's findings, in the type the rest of the system expects.

    A `CompleteSync` rather than a bare list, so the offline path and the live
    path hand the tool registry the same shape and the sync gate is exercised
    in both. Complete is honest here: there is no project this run failed to
    read.
    """
    now = now or datetime.now(timezone.utc)
    issues = sandbox_issues(now)
    return CompleteSync(
        started_at=now,
        issues_seen=len(issues),
        projects_requested=list(SANDBOX_PROJECTS),
        projects_covered=list(SANDBOX_PROJECTS),
        findings=evaluate(issues, sandbox_config(), now),
    )
