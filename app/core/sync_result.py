"""Sync results.

The design constraint, from ADR-002's reasoning applied to a different problem:
encode the rule in the type, not in the discipline of whoever calls it.

A partial sync — rate-limited, permission-denied on one project, aborted — must
not produce a report. The obvious implementation is a `status` field plus a
convention that consumers check it. That convention survives exactly until
someone adds a second consumer and forgets.

So `CompleteSync` and `PartialSync` are separate types. The report generator's
signature accepts only the former. A partial sync isn't a report that's flagged
as unreliable; it is not a thing a report can be made from.

Note `PartialSync.partial_findings` is deliberately NOT named `findings`. If it
were, a partial result would duck-type cleanly into any code reaching for
`.findings` and the whole guarantee would leak.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.models.domain import RiskFinding


class SyncStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class PartialReason(str, Enum):
    """Why a sync came up short. Each maps to different client-facing wording.

    A 403 is a provisioning problem the client can fix in five minutes; a rate
    limit is ours to handle. Collapsing them into "sync failed" wastes a
    support round-trip.
    """

    RATE_LIMITED = "rate_limited"
    PERMISSION_DENIED = "permission_denied"
    UPSTREAM_ERROR = "upstream_error"
    TIMEOUT = "timeout"
    ABORTED = "aborted"


class _SyncBase(BaseModel):
    started_at: datetime
    finished_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    issues_seen: int
    projects_requested: list[str]

    model_config = {"frozen": True}


class CompleteSync(_SyncBase):
    """Every requested project was read to the end. Safe to report from."""

    status: Literal[SyncStatus.COMPLETE] = SyncStatus.COMPLETE
    findings: list[RiskFinding]
    projects_covered: list[str]


class PartialSync(_SyncBase):
    """Coverage is incomplete. Cannot be reported from.

    The findings collected so far are kept — they're useful for debugging and
    for showing an operator what was reached — but under a name no reporting
    code will accidentally pick up.
    """

    status: Literal[SyncStatus.PARTIAL] = SyncStatus.PARTIAL
    partial_findings: list[RiskFinding]
    projects_covered: list[str]
    projects_incomplete: list[str]
    reason: PartialReason
    detail: str

    def operator_message(self) -> str:
        """What to show a human. Names the fix where there is one."""
        base = (
            f"Sync incomplete: {len(self.projects_incomplete)} of "
            f"{len(self.projects_requested)} projects were not fully read "
            f"({', '.join(self.projects_incomplete)})."
        )
        hint = {
            PartialReason.PERMISSION_DENIED: (
                " The service account lacks Browse Projects permission. This is "
                "fixable in Jira project settings."
            ),
            PartialReason.RATE_LIMITED: (
                " Jira rate limits were exhausted. Retry with a smaller project "
                "set or a longer window."
            ),
        }.get(self.reason, "")
        return base + hint


SyncResult = CompleteSync | PartialSync
