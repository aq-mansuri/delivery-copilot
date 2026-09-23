"""Tests for the partial-sync gate.

The guarantee under test: an incomplete sync cannot produce a report. Not
"produces a flagged report" — cannot produce one.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.core.report import DeliveryReport, UnreportableSyncError, build_report
from app.core.sync_result import (
    CompleteSync,
    PartialReason,
    PartialSync,
    SyncStatus,
)
from app.models.domain import Citation, RiskFinding, RiskLevel

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def finding(key="INS-1", rule_id="stale_in_progress", level=RiskLevel.HIGH):
    return RiskFinding(
        issue_key=key,
        rule_id=rule_id,
        level=level,
        detail="In Progress with no activity for 21 days.",
        citations=[
            Citation(
                source_type="issue",
                source_id=key,
                url=f"https://x.atlassian.net/browse/{key}",
            )
        ],
    )


def complete(**overrides) -> CompleteSync:
    defaults = dict(
        started_at=NOW,
        issues_seen=120,
        projects_requested=["INS", "CLM"],
        findings=[finding("INS-1"), finding("CLM-2", level=RiskLevel.MEDIUM)],
        projects_covered=["INS", "CLM"],
    )
    defaults.update(overrides)
    return CompleteSync(**defaults)


def partial(**overrides) -> PartialSync:
    defaults = dict(
        started_at=NOW,
        issues_seen=80,
        projects_requested=["INS", "CLM"],
        partial_findings=[finding("INS-1")],
        projects_covered=["INS"],
        projects_incomplete=["CLM"],
        reason=PartialReason.PERMISSION_DENIED,
        detail="403 on project CLM",
    )
    defaults.update(overrides)
    return PartialSync(**defaults)


class TestGate:
    def test_partial_sync_cannot_produce_a_report(self):
        with pytest.raises(UnreportableSyncError):
            build_report(partial())

    def test_complete_sync_reports_normally(self):
        report = build_report(complete())
        assert isinstance(report, DeliveryReport)
        assert report.total_findings() == 2
        assert report.projects_covered == ["INS", "CLM"]

    def test_bare_findings_list_is_rejected(self):
        """Guards the shortcut someone will try in six months."""
        with pytest.raises(UnreportableSyncError):
            build_report([finding("INS-1")])

    def test_error_message_names_the_consequence(self):
        """An operator reading this must understand the risk, not just the type."""
        with pytest.raises(UnreportableSyncError) as exc:
            build_report(partial())
        assert "understate risk" in str(exc.value)


class TestTypesAreDistinct:
    def test_partial_does_not_expose_a_findings_attribute(self):
        """If PartialSync had `.findings`, it would duck-type into reporting
        code and the guarantee would leak."""
        assert not hasattr(partial(), "findings")
        assert hasattr(partial(), "partial_findings")

    def test_status_is_fixed_per_type(self):
        assert complete().status is SyncStatus.COMPLETE
        assert partial().status is SyncStatus.PARTIAL

    def test_results_are_immutable(self):
        """A caller must not be able to flip status to COMPLETE and proceed."""
        with pytest.raises(Exception):
            partial().status = SyncStatus.COMPLETE


class TestOperatorMessage:
    def test_permission_error_names_the_fix(self):
        message = partial().operator_message()
        assert "CLM" in message
        assert "Browse Projects permission" in message

    def test_rate_limit_suggests_a_different_remedy(self):
        message = partial(reason=PartialReason.RATE_LIMITED).operator_message()
        assert "rate limits" in message
        assert "Browse Projects" not in message


class TestReportShape:
    def test_findings_grouped_by_level_highest_first(self):
        report = build_report(complete())
        assert [s.level for s in report.sections] == [
            RiskLevel.HIGH,
            RiskLevel.MEDIUM,
        ]

    def test_empty_levels_are_omitted(self):
        report = build_report(complete(findings=[finding("INS-1")]))
        assert [s.level for s in report.sections] == [RiskLevel.HIGH]

    def test_rule_counts_support_week_over_week_comparison(self):
        report = build_report(
            complete(
                findings=[
                    finding("INS-1", rule_id="stale_in_progress"),
                    finding("INS-2", rule_id="stale_in_progress"),
                    finding("INS-3", rule_id="blocked_dependency"),
                ]
            )
        )
        assert report.rule_counts == {
            "stale_in_progress": 2,
            "blocked_dependency": 1,
        }
