"""Runs a sync across projects and evaluates risk rules as issues stream in.

Per-project boundaries matter here. If project four 403s, projects one through
three are still fully read — that's useful information for an operator, and it
narrows the fix to one project's permissions. A single try/except around the
whole run would lose that.

The result is still PARTIAL. Partial coverage is partial coverage; knowing
*which* projects were complete doesn't make the report publishable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.core.sync_result import (
    CompleteSync,
    PartialReason,
    PartialSync,
    SyncResult,
)
from app.integrations.atlassian.client import AtlassianError, JiraClient
from app.models.domain import Issue, RiskFinding
from app.risk.rules import ALL_RULES, RiskConfig

logger = logging.getLogger(__name__)


def _evaluate_one(issue: Issue, cfg: RiskConfig, now: datetime) -> list[RiskFinding]:
    return [f for rule in ALL_RULES if (f := rule(issue, cfg, now)) is not None]


def _classify(exc: Exception) -> tuple[PartialReason, str]:
    message = str(exc)
    lowered = message.lower()
    if "403" in message or "permission" in lowered:
        return PartialReason.PERMISSION_DENIED, message
    if "rate" in lowered or "429" in message:
        return PartialReason.RATE_LIMITED, message
    if "timeout" in lowered or "timed out" in lowered:
        return PartialReason.TIMEOUT, message
    return PartialReason.UPSTREAM_ERROR, message


async def run_sync(
    client: JiraClient,
    project_keys: list[str],
    cfg: RiskConfig,
    *,
    now: datetime | None = None,
) -> SyncResult:
    started_at = datetime.now(timezone.utc)
    now = now or started_at

    findings: list[RiskFinding] = []
    covered: list[str] = []
    incomplete: list[str] = []
    issues_seen = 0
    failure: tuple[PartialReason, str] | None = None

    for project_key in project_keys:
        try:
            # Findings accumulate; issues do not. A finding is a few hundred
            # bytes and there are far fewer of them than issues, so ranking and
            # dedup downstream stay cheap without holding the corpus.
            async for issue in client.search_issues(
                f'project = "{project_key}" AND statusCategory != Done'
            ):
                issues_seen += 1
                findings.extend(_evaluate_one(issue, cfg, now))
            covered.append(project_key)

        except AtlassianError as exc:
            logger.warning("project %s incomplete: %s", project_key, exc)
            incomplete.append(project_key)
            # First failure wins for reason attribution; later ones are
            # usually the same root cause and overwriting hides it.
            failure = failure or _classify(exc)

    if incomplete:
        reason, detail = failure or (PartialReason.UPSTREAM_ERROR, "unknown")
        return PartialSync(
            started_at=started_at,
            issues_seen=issues_seen,
            projects_requested=project_keys,
            partial_findings=findings,
            projects_covered=covered,
            projects_incomplete=incomplete,
            reason=reason,
            detail=detail,
        )

    return CompleteSync(
        started_at=started_at,
        issues_seen=issues_seen,
        projects_requested=project_keys,
        findings=findings,
        projects_covered=covered,
    )
