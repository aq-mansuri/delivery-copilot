"""Tests for rule_blocked_dependency.

The first test is the one that matters — it's a regression guard for a bug where
the headline count and the listed issues disagreed.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.domain import Issue, IssueLink, IssueStatusCategory, RiskLevel
from app.risk.rules import RiskConfig, rule_blocked_dependency

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
CFG = RiskConfig()


def make_issue(links, **overrides) -> Issue:
    defaults = dict(
        key="INS-101",
        project_key="INS",
        summary="Integrate claims vendor callback",
        status_name="In Progress",
        status_category=IssueStatusCategory.IN_PROGRESS,
        created_at=NOW - timedelta(days=5),
        updated_at=NOW,
        links=links,
    )
    defaults.update(overrides)
    return Issue(**defaults)


def link(key, *, blocks_this=False, category=IssueStatusCategory.IN_PROGRESS,
         link_type="blocks") -> IssueLink:
    return IssueLink(
        target_key=key,
        target_summary=f"Work item {key}",
        target_status_category=category,
        link_type=link_type,
        target_blocks_this=blocks_this,
    )


class TestCountMatchesList:
    def test_headline_count_matches_the_issues_listed(self):
        """Regression: 'Blocked by 1' must not be followed by three issue keys."""
        issue = make_issue([
            link("INS-5", blocks_this=True),
            link("INS-9", blocks_this=False),
            link("INS-12", blocks_this=False),
        ])
        finding = rule_blocked_dependency(issue, CFG, NOW)
        assert finding.level is RiskLevel.HIGH
        assert "Blocked by 1 unfinished issue(s)" in finding.detail
        assert "INS-5" in finding.detail
        assert "INS-9" not in finding.detail
        assert "INS-12" not in finding.detail

    def test_citations_cover_exactly_the_reported_links(self):
        issue = make_issue([
            link("INS-5", blocks_this=True),
            link("INS-9", blocks_this=False),
        ])
        finding = rule_blocked_dependency(issue, CFG, NOW)
        cited = {c.source_id for c in finding.citations}
        assert cited == {"INS-101", "INS-5"}


class TestDirection:
    def test_blocked_by_open_work_is_high(self):
        issue = make_issue([link("INS-5", blocks_this=True)])
        assert rule_blocked_dependency(issue, CFG, NOW).level is RiskLevel.HIGH

    def test_blocking_others_is_medium_context(self):
        issue = make_issue([link("INS-5", blocks_this=False)])
        assert rule_blocked_dependency(issue, CFG, NOW).level is RiskLevel.MEDIUM

    def test_resolved_blocker_clears_the_flag(self):
        issue = make_issue([
            link("INS-5", blocks_this=True, category=IssueStatusCategory.DONE)
        ])
        assert rule_blocked_dependency(issue, CFG, NOW) is None

    def test_unknown_status_counts_as_unresolved(self):
        """Recall over precision: can't see it, assume it's a dependency."""
        issue = make_issue([link("INS-5", blocks_this=True, category=None)])
        finding = rule_blocked_dependency(issue, CFG, NOW)
        assert finding is not None
        assert "status unknown" in finding.detail


class TestNoise:
    def test_relates_to_links_do_not_fire_by_default(self):
        """'Relates to' is the most-used, least-meaningful link in most tenants."""
        issue = make_issue([link("INS-5", link_type="relates_to")])
        assert rule_blocked_dependency(issue, CFG, NOW) is None

    def test_relates_to_can_be_opted_in_per_client(self):
        cfg = RiskConfig(
            reported_link_types=frozenset({"blocks", "depends_on", "relates_to"})
        )
        issue = make_issue([link("INS-5", link_type="relates_to")])
        assert rule_blocked_dependency(issue, cfg, NOW) is not None

    def test_long_link_lists_are_truncated(self):
        issue = make_issue([
            link(f"INS-{i}", blocks_this=True) for i in range(20, 32)
        ])
        detail = rule_blocked_dependency(issue, CFG, NOW).detail
        assert "+7 more" in detail
        assert "Blocked by 12" in detail

    def test_done_issues_are_exempt(self):
        issue = make_issue(
            [link("INS-5", blocks_this=True)],
            status_category=IssueStatusCategory.DONE,
        )
        assert rule_blocked_dependency(issue, CFG, NOW) is None


class TestDeterminism:
    def test_output_is_stable_regardless_of_payload_order(self):
        """Jira does not guarantee link order; the report must not change week
        to week because of it (ADR-001 determinism claim)."""
        a = make_issue([
            link("INS-9", blocks_this=True), link("INS-5", blocks_this=True)
        ])
        b = make_issue([
            link("INS-5", blocks_this=True), link("INS-9", blocks_this=True)
        ])
        assert (
            rule_blocked_dependency(a, CFG, NOW).detail
            == rule_blocked_dependency(b, CFG, NOW).detail
        )
