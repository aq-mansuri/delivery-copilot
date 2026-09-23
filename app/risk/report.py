"""Report generation — the one place findings become something a board sees.

Every other layer can be sloppy about completeness. This one cannot, so it is
the layer that refuses. It accepts a `ReportableFindings` and nothing else:
there is no overload that takes a bare list, no `force=True`, and no
`allow_partial` parameter. Adding one would defeat the gate, which is why the
type signature and the runtime check say the same thing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.domain import RiskFinding, RiskLevel
from app.models.sync import ReportableFindings


class DeliveryReport(BaseModel):
    """Only ever built from a complete sync — see `generate_report`."""

    generated_at: datetime
    issues_seen: int
    findings: list[RiskFinding]
    counts_by_level: dict[str, int] = Field(default_factory=dict)

    @property
    def high_risk(self) -> list[RiskFinding]:
        return [f for f in self.findings if f.level is RiskLevel.HIGH]


def generate_report(bundle: Any) -> DeliveryReport:
    """Build the report. Accepts only proof of a complete sync.

    The isinstance check is not defensive noise. The type hint is erased at
    runtime, so without it `generate_report(result.findings_for_triage())` would
    happily produce a report that is missing an entire project. The check turns
    that mistake into a TypeError at the call site instead of a wrong number in
    front of the board.
    """
    if not isinstance(bundle, ReportableFindings):
        raise TypeError(
            f"generate_report requires ReportableFindings, got "
            f"{type(bundle).__name__}. Findings alone are not reportable — pass "
            f"SyncResult.reportable(), which refuses partial syncs."
        )

    counts = {level.value: 0 for level in RiskLevel}
    for finding in bundle.findings:
        counts[finding.level.value] += 1

    return DeliveryReport(
        generated_at=bundle.completed_at,
        issues_seen=bundle.issues_seen,
        findings=list(bundle.findings),
        counts_by_level=counts,
    )
