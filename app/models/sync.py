"""Sync completeness — the gate between "we collected data" and "we can report".

The failure this module exists to prevent is the one Priya named: telling the
board you are on track when you aren't. A sync that hit a 429 ceiling, a 403 on
one of the six projects, or an abort has *not* seen every issue. Its findings
may be perfectly correct and still be a lie by omission, because the issues it
never read are exactly the ones nobody is looking at.

The design rule here is that completeness is not a flag a caller remembers to
check. Findings are not reachable in report-ready form except through a
`SyncResult` with zero gaps, and the report generator accepts nothing else. A
caller who forgets to check gets a `TypeError`, not a wrong report.

Python cannot make this literally unconstructible — someone who imports `_SEAL`
or calls `object.__new__` can forge the token. What it can do is ensure no
*accidental* path exists: every honest route from findings to report passes
through `SyncResult.reportable()`, and forging one has to be deliberate,
obvious in review, and unjustifiable in a regulated audit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.models.domain import RiskFinding


class SyncOutcome(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class SyncGapReason(str, Enum):
    """Every way a sync can end up having not seen everything.

    These are not error categories for logging. Each one is a concrete class of
    issue that is missing from the result set.
    """

    # Backoff gave up. Some page of some project was never fetched.
    RATE_LIMIT_EXHAUSTED = "rate_limit_exhausted"

    # 403 on Browse Projects. An entire project is invisible, and the report
    # would silently scope itself to five of six projects.
    PERMISSION_DENIED = "permission_denied"

    # 401. Nothing was read at all, or reading stopped partway.
    AUTH_FAILED = "auth_failed"

    # 5xx ceiling, transport error, malformed payload.
    UPSTREAM_ERROR = "upstream_error"

    # One issue's JSON did not map to the domain model. The sync continues —
    # one bad issue must not kill 3,000 — but the result is not complete.
    # This is the quiet one: it raises no exception, so without a recorded gap
    # a dropped issue is indistinguishable from an issue with no risk.
    UNMAPPABLE_ISSUE = "unmappable_issue"

    # Cancelled, timed out, or shut down mid-stream.
    ABORTED = "aborted"


class SyncGap(BaseModel):
    """One concrete hole in the data, with enough detail to act on it.

    `scope` is what is missing — a project key, a JQL fragment, an issue key —
    because "the sync was partial" is not something a lead can do anything
    about, whereas "project CLAIMS returned 403" is.
    """

    model_config = {"frozen": True}

    reason: SyncGapReason
    scope: str
    detail: str
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SyncAbort(RuntimeError):
    """An upstream failure that leaves a *known kind* of hole in the data.

    Integration layers raise this instead of a bare error so the gap reason
    travels with the exception. By the time a failure reaches the risk engine,
    "403" has become an exception with no memory of which project vanished —
    and the scope is the only part a lead can act on.

    Lives in the model, not in the Jira client, so the risk engine can catch it
    without importing an integration. Rules stay pure domain code.
    """

    def __init__(
        self, message: str, *, gap_reason: SyncGapReason, scope: str = "sync"
    ) -> None:
        self.gap_reason = gap_reason
        self.scope = scope
        super().__init__(message)


class PartialSyncError(RuntimeError):
    """Raised when something tries to report on an incomplete sync.

    Carries the gaps so the caller can tell the user *what* is missing rather
    than just refusing.
    """

    def __init__(self, gaps: tuple[SyncGap, ...]) -> None:
        self.gaps = gaps
        summary = "; ".join(f"{g.reason.value} on {g.scope}" for g in gaps[:5])
        if len(gaps) > 5:
            summary += f"; +{len(gaps) - 5} more"
        super().__init__(
            f"Refusing to generate a report from a partial sync "
            f"({len(gaps)} gap(s)): {summary}"
        )


# Module-private construction token. Not exported in __all__, and the only
# legitimate holders are ReportableFindings' factory below.
_SEAL = object()


class ReportableFindings:
    """Proof that these findings came from a sync that saw everything.

    Deliberately not a pydantic model: `BaseModel.model_validate` would let any
    caller conjure one from a dict, which is precisely the bypass this type
    exists to close. A plain class with a sealed constructor has no such door.

    The only way to obtain one is `SyncResult.reportable()`, which refuses when
    gaps exist. Holding an instance therefore *is* the completeness guarantee —
    the report generator does not need to check anything, and cannot forget to.
    """

    __slots__ = ("completed_at", "findings", "issues_seen")

    def __init__(
        self,
        findings: tuple[RiskFinding, ...],
        issues_seen: int,
        completed_at: datetime,
        *,
        _seal: object = None,
    ) -> None:
        if _seal is not _SEAL:
            raise TypeError(
                "ReportableFindings cannot be constructed directly. Obtain one "
                "from SyncResult.reportable(), which refuses partial syncs. "
                "If you are trying to report on a partial sync, the answer is "
                "no — fix the gap or scope the report to what was actually read."
            )
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "issues_seen", issues_seen)
        object.__setattr__(self, "completed_at", completed_at)

    def __setattr__(self, *_: object) -> None:
        raise AttributeError("ReportableFindings is immutable")

    def __repr__(self) -> str:
        return (
            f"ReportableFindings(findings={len(self.findings)}, "
            f"issues_seen={self.issues_seen})"
        )


class SyncResult:
    """Findings bound to the completeness of the sync that produced them.

    There is no public `.findings`. That is the whole point: a caller cannot
    write `generate_report(result.findings)` because that attribute does not
    exist. The findings leave this object either through `reportable()` — which
    enforces completeness — or through `findings_for_triage()`, whose return
    value the report generator rejects by type.
    """

    __slots__ = ("_findings", "completed_at", "gaps", "issues_seen")

    def __init__(
        self,
        findings: tuple[RiskFinding, ...],
        gaps: tuple[SyncGap, ...],
        issues_seen: int,
        completed_at: datetime,
    ) -> None:
        self._findings = findings
        self.gaps = gaps
        self.issues_seen = issues_seen
        self.completed_at = completed_at

    @property
    def outcome(self) -> SyncOutcome:
        return SyncOutcome.PARTIAL if self.gaps else SyncOutcome.COMPLETE

    def reportable(self) -> ReportableFindings:
        """The only door to the report generator. Locked when gaps exist."""
        if self.gaps:
            raise PartialSyncError(self.gaps)
        return ReportableFindings(
            self._findings, self.issues_seen, self.completed_at, _seal=_SEAL
        )

    def findings_for_triage(self) -> tuple[RiskFinding, ...]:
        """Findings from a run that may be incomplete. Not reportable.

        A partial sync's findings are still useful — a lead triaging today wants
        to see the eleven blocked tickets we *did* find. What they must not do
        is put them in front of the board as a complete picture, so this returns
        a bare tuple that `generate_report` refuses by type.
        """
        return self._findings

    def __repr__(self) -> str:
        return (
            f"SyncResult(outcome={self.outcome.value}, "
            f"findings={len(self._findings)}, gaps={len(self.gaps)}, "
            f"issues_seen={self.issues_seen})"
        )


class SyncRecorder:
    """Mutable gap log for one sync, and the only way to build a SyncResult.

    Passed down into the client so the layer that *discovers* a hole is the
    layer that records it. A 403 is known in `_get`; by the time it reaches the
    agent it is just an exception with no idea which project vanished.
    """

    def __init__(self) -> None:
        self._gaps: list[SyncGap] = []
        self._issues_seen = 0

    def record_gap(
        self, reason: SyncGapReason, scope: str, detail: str
    ) -> None:
        self._gaps.append(SyncGap(reason=reason, scope=scope, detail=detail))

    def record_issue_seen(self) -> None:
        self._issues_seen += 1

    @property
    def gaps(self) -> tuple[SyncGap, ...]:
        return tuple(self._gaps)

    @property
    def issues_seen(self) -> int:
        return self._issues_seen

    def finish(
        self, findings: list[RiskFinding], now: datetime | None = None
    ) -> SyncResult:
        return SyncResult(
            findings=tuple(findings),
            gaps=tuple(self._gaps),
            issues_seen=self._issues_seen,
            completed_at=now or datetime.now(timezone.utc),
        )
