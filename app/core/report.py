"""Report generation entry point.

The signature is the gate. `build_report` takes `CompleteSync` — not
`SyncResult`, not a findings list. A partial sync cannot be passed here without
a type error, and the runtime guard catches the dynamic callers a type checker
never sees (JSON off a queue, a scheduler, the API layer).

This module does no LLM work. It assembles the structured payload the narrative
generator will later turn into prose. Keeping the gate upstream of the model
means a partial sync never costs a token.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from app.core.sync_result import CompleteSync
from app.models.domain import RiskFinding, RiskLevel


class UnreportableSyncError(RuntimeError):
    """Raised when something tries to report from an incomplete sync.

    Loud on purpose. The whole point of this machinery is that an incomplete
    sync fails visibly rather than producing a report that looks finished and
    silently omits 200 issues.
    """


class ReportSection(BaseModel):
    level: RiskLevel
    findings: list[RiskFinding]


class DeliveryReport(BaseModel):
    issues_seen: int
    projects_covered: list[str]
    sections: list[ReportSection]
    rule_counts: dict[str, int]

    def total_findings(self) -> int:
        return sum(len(s.findings) for s in self.sections)


def build_report(sync: CompleteSync) -> DeliveryReport:
    if not isinstance(sync, CompleteSync):
        raise UnreportableSyncError(
            f"build_report requires a CompleteSync, got {type(sync).__name__}. "
            "An incomplete sync cannot produce a report — publishing one would "
            "understate risk by omitting unread issues."
        )

    by_level = {
        level: [f for f in sync.findings if f.level is level]
        for level in (RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW)
    }

    return DeliveryReport(
        issues_seen=sync.issues_seen,
        projects_covered=sync.projects_covered,
        sections=[
            ReportSection(level=level, findings=found)
            for level, found in by_level.items()
            if found
        ],
        rule_counts=dict(Counter(f.rule_id for f in sync.findings)),
    )
