"""Tests for the deterministic risk engine.

Note what's being asserted: not just "the rule fires", but the specific failure
modes Priya described. A test named after a client sentence is a test that
survives refactoring.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.domain import (
    ChangelogEntry,
    Comment,
    Issue,
    IssueLink,
    IssueStatusCategory,
    RiskLevel,
)
from app.risk.rules import (
    RiskConfig,
    evaluate,
    rule_blocked_dependency,
    rule_blocked_status,
    rule_missing_target_release,
    rule_overdue,
    rule_sprint_carryover,
    rule_stale_in_progress,
)

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
CFG = RiskConfig()


def make_issue(**overrides) -> Issue:
    defaults = dict(
        key="INS-101",
        project_key="INS",
        summary="Integrate claims vendor callback",
        status_name="In Progress",
        status_category=IssueStatusCategory.IN_PROGRESS,
        assignee_id="u-dev-1",
        target_release="Q4-2026",
        created_at=NOW - timedelta(days=60),
        updated_at=NOW,
    )
    defaults.update(overrides)
    return Issue(**defaults)


class TestStaleInProgress:
    def test_dead_ticket_that_looks_healthy_is_flagged_high(self):
        """Priya: 'three weeks, no commits, no comments. It looks healthy.'"""
        issue = make_issue(
            updated_at=NOW,  # Jira says it was touched today...
            created_at=NOW - timedelta(days=21),
            comments=[],
            changelog=[],  # ...but no human has done anything in 21 days.
        )
        finding = rule_stale_in_progress(issue, CFG, NOW)
        assert finding is not None
        assert finding.level is RiskLevel.HIGH
        assert "21 days" in finding.detail

    def test_jira_updated_field_does_not_mask_staleness(self):
        """Regression guard: automation bumping `updated` must not hide a dead ticket."""
        issue = make_issue(
            updated_at=NOW,
            created_at=NOW - timedelta(days=30),
            changelog=[],
        )
        assert rule_stale_in_progress(issue, CFG, NOW) is not None

    def test_recent_comment_clears_the_flag(self):
        issue = make_issue(
            created_at=NOW - timedelta(days=30),
            comments=[
                Comment(
                    id="c1",
                    body="Vendor confirmed the schema, resuming.",
                    created_at=NOW - timedelta(days=1),
                )
            ],
        )
        assert rule_stale_in_progress(issue, CFG, NOW) is None

    def test_todo_items_are_not_stale_candidates(self):
        issue = make_issue(
            status_category=IssueStatusCategory.TODO,
            created_at=NOW - timedelta(days=90),
        )
        assert rule_stale_in_progress(issue, CFG, NOW) is None

    @pytest.mark.parametrize(
        "idle_days,expected",
        [(6, None), (7, RiskLevel.MEDIUM), (13, RiskLevel.MEDIUM), (14, RiskLevel.HIGH)],
    )
    def test_threshold_boundaries(self, idle_days, expected):
        issue = make_issue(created_at=NOW - timedelta(days=idle_days))
        finding = rule_stale_in_progress(issue, CFG, NOW)
        assert (finding.level if finding else None) is expected


class TestMissingTargetRelease:
    def test_blank_field_is_surfaced_not_silently_dropped(self):
        issue = make_issue(target_release=None)
        finding = rule_missing_target_release(issue, CFG, NOW)
        assert finding is not None
        assert "excluded from delivery-date reporting" in finding.detail

    def test_done_issues_are_exempt(self):
        issue = make_issue(
            target_release=None, status_category=IssueStatusCategory.DONE
        )
        assert rule_missing_target_release(issue, CFG, NOW) is None


class TestSprintCarryover:
    def test_repeated_carryover_is_high_risk(self):
        issue = make_issue(
            changelog=[
                ChangelogEntry(field="Sprint", at=NOW - timedelta(days=28)),
                ChangelogEntry(field="Sprint", at=NOW - timedelta(days=14)),
                ChangelogEntry(field="Sprint", at=NOW - timedelta(days=1)),
            ]
        )
        finding = rule_sprint_carryover(issue, CFG, NOW)
        assert finding is not None
        assert finding.level is RiskLevel.HIGH


class TestEvaluate:
    def test_every_finding_carries_a_citation(self):
        """Non-negotiable: no unattributed claims reach the report."""
        issues = [
            make_issue(key="INS-1", created_at=NOW - timedelta(days=30)),
            make_issue(key="INS-2", target_release=None),
        ]
        findings = evaluate(issues, CFG, NOW)
        assert findings
        assert all(f.citations for f in findings)
        assert all(f.citations[0].url.endswith(f.issue_key) for f in findings)

    def test_one_issue_can_produce_multiple_findings(self):
        issue = make_issue(
            key="INS-9",
            target_release=None,
            created_at=NOW - timedelta(days=30),
        )
        rule_ids = {f.rule_id for f in evaluate([issue], CFG, NOW)}
        assert {"stale_in_progress", "missing_target_release"} <= rule_ids


class TestOverdue:
    """The rule that had no tests, and broke on the first real tenant.

    `Issue.due_date` came back offset-naive from Jira — `duedate` is the one
    date-only field the API returns — so `issue.due_date >= now` raised
    TypeError and took the whole sync down. Nothing here caught it because
    nothing here exercised it, and every hand-written fixture built a
    timezone-aware datetime directly.
    """

    def test_not_overdue_before_the_due_date(self):
        issue = make_issue(due_date=NOW + timedelta(days=3))
        assert rule_overdue(issue, CFG, NOW) is None

    def test_not_overdue_during_the_due_date_itself(self):
        """Jira's `duedate` is a calendar day, and a deadline of the 20th is
        met by work finished on the 20th. Comparing against midnight flags an
        issue as overdue from the first second of the day it is due — a day
        early, every time, on a number a client will check."""
        due = datetime(2026, 9, 15, 23, 59, 59, tzinfo=timezone.utc)
        midday = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
        assert rule_overdue(make_issue(due_date=due), CFG, midday) is None

    def test_overdue_once_the_day_has_passed(self):
        due = datetime(2026, 9, 15, 23, 59, 59, tzinfo=timezone.utc)
        next_day = datetime(2026, 9, 16, 9, 0, tzinfo=timezone.utc)
        finding = rule_overdue(make_issue(due_date=due), CFG, next_day)
        assert finding is not None
        assert finding.rule_id == "overdue"
        assert finding.level is RiskLevel.HIGH

    def test_done_issues_are_never_overdue(self):
        """A shipped ticket that missed its date is history, not risk. Flagging
        it fills the report with items nobody can act on."""
        issue = make_issue(
            due_date=NOW - timedelta(days=30),
            status_category=IssueStatusCategory.DONE,
            status_name="Done",
        )
        assert rule_overdue(issue, CFG, NOW) is None

    def test_no_due_date_is_not_a_finding(self):
        assert rule_overdue(make_issue(due_date=None), CFG, NOW) is None

    def test_a_naive_due_date_does_not_take_the_sync_down(self):
        """Defence in depth behind the client fix.

        The parser is where this is corrected, but `Issue` is constructible by
        anything — a fixture, a future integration, a test. One naive datetime
        must not be able to raise inside a rule and abort a whole project's
        sync, which is exactly what happened: every issue after the offending
        one went unread.
        """
        # DTZ001 is the bug under test, not a slip. Do not "fix" it.
        issue = make_issue(due_date=datetime(2026, 9, 10))  # noqa: DTZ001
        finding = rule_overdue(issue, CFG, NOW)
        assert finding is not None and finding.rule_id == "overdue"

    def test_evaluate_survives_a_naive_due_date(self):
        issues = [make_issue(key="INS-1", due_date=datetime(2026, 9, 10))]  # noqa: DTZ001
        assert evaluate(issues, CFG, NOW)  # must not raise


class TestStaleDetailNamesTheRealStatus:
    """The finding said "In Progress" whatever the status actually was.

    Correct as a category — `Blocked`, `Escalated` and `Waiting for customer` are
    all `indeterminate`, and keying the RULE off the category is right, because
    no two tenants spell these the same way. But the sentence a delivery lead
    reads is not a category. Adding a Blocked status to a real tenant produced:

        INS-22: In Progress with no comments, transitions or updates for 30 days

    on a ticket whose status is Blocked. A lead who opens that ticket and sees
    something the report just contradicted stops trusting the other 33 findings,
    and they are right to.
    """

    def test_it_names_the_status_the_issue_is_actually_in(self):
        issue = make_issue(
            status_name="Blocked",
            status_category=IssueStatusCategory.IN_PROGRESS,
            created_at=NOW - timedelta(days=30),
        )
        finding = rule_stale_in_progress(issue, CFG, NOW)
        assert finding is not None
        assert "Blocked" in finding.detail
        assert "In Progress" not in finding.detail

    def test_an_actual_in_progress_issue_still_reads_naturally(self):
        """The counterweight: a fix that makes the common case worse is not a
        fix."""
        issue = make_issue(
            status_name="In Progress",
            status_category=IssueStatusCategory.IN_PROGRESS,
            created_at=NOW - timedelta(days=30),
        )
        finding = rule_stale_in_progress(issue, CFG, NOW)
        assert finding.detail.startswith("In Progress with no")

    def test_a_tenant_specific_status_name_is_carried_through(self):
        issue = make_issue(
            status_name="Waiting for customer",
            status_category=IssueStatusCategory.IN_PROGRESS,
            created_at=NOW - timedelta(days=30),
        )
        assert "Waiting for customer" in rule_stale_in_progress(issue, CFG, NOW).detail

    def test_the_rule_still_fires_on_category_not_on_name(self):
        """The half that must not change. A status nobody predicted is still
        caught, because the category is what is checked."""
        issue = make_issue(
            status_name="Impediment",
            status_category=IssueStatusCategory.IN_PROGRESS,
            created_at=NOW - timedelta(days=30),
        )
        assert rule_stale_in_progress(issue, CFG, NOW) is not None

        todo = make_issue(
            status_name="In Progress",  # name lies, category does not
            status_category=IssueStatusCategory.TODO,
            created_at=NOW - timedelta(days=30),
        )
        assert rule_stale_in_progress(todo, CFG, NOW) is None


class TestBlockedByStatus:
    """A ticket a human marked Blocked, with no link recorded.

    Found on a live tenant. INS-12's status was `Blocked`; it had no issue
    links, so `rule_blocked_dependency` — which reads links — correctly found
    nothing, and the agent answered "INS-12 is not blocked. This is a definitive
    negative result." Confidently wrong, about a ticket whose own status says
    otherwise, which is the worst answer this system can give.

    A person deliberately moving a ticket to Blocked is a first-class signal and
    arguably the strongest one here: somebody knew enough to record it. The
    engine could not see it, because `statusCategory` flattens `Blocked`,
    `On Hold`, `Impediment` and `In Progress` all to `indeterminate`.

    So the NAME is what carries the signal, and names differ per tenant — which
    is why the set lives in `RiskConfig` and is injected, exactly like
    `reported_link_types`. The rule is still not written against a hardcoded
    name.
    """

    def test_a_blocked_status_is_a_finding_even_with_no_links(self):
        issue = make_issue(status_name="Blocked")
        finding = rule_blocked_status(issue, CFG, NOW)
        assert finding is not None
        assert finding.rule_id == "blocked_status"
        assert "Blocked" in finding.detail

    def test_matching_is_case_and_space_insensitive(self):
        for name in ("blocked", "BLOCKED", " Blocked "):
            assert rule_blocked_status(make_issue(status_name=name), CFG, NOW)

    def test_other_configured_names_are_recognised(self):
        for name in ("On Hold", "Impediment", "Waiting for customer"):
            assert rule_blocked_status(make_issue(status_name=name), CFG, NOW), name

    def test_a_tenant_can_configure_its_own_vocabulary(self):
        cfg = RiskConfig(blocked_status_names=frozenset({"parked"}))
        assert rule_blocked_status(make_issue(status_name="Parked"), cfg, NOW)
        assert rule_blocked_status(make_issue(status_name="Blocked"), cfg, NOW) is None

    def test_an_ordinary_in_progress_issue_is_not_a_finding(self):
        assert rule_blocked_status(make_issue(status_name="In Progress"), CFG, NOW) is None

    def test_a_done_issue_is_never_blocked(self):
        issue = make_issue(
            status_name="Blocked",
            status_category=IssueStatusCategory.DONE,
        )
        assert rule_blocked_status(issue, CFG, NOW) is None

    def test_it_stands_down_when_a_link_already_explains_the_block(self):
        """`rule_blocked_dependency` names the blocker and is strictly more
        useful. Reporting both puts two findings on one problem, and a report
        that double-counts is one a lead learns to discount."""
        issue = make_issue(
            status_name="Blocked",
            links=[
                IssueLink(
                    target_key="INS-5",
                    target_summary="Rotate mutual TLS certificates",
                    target_status_category=IssueStatusCategory.IN_PROGRESS,
                    link_type="blocks",
                    target_blocks_this=True,
                )
            ],
        )
        assert rule_blocked_dependency(issue, CFG, NOW) is not None
        assert rule_blocked_status(issue, CFG, NOW) is None

    def test_a_resolved_blocker_link_does_not_suppress_it(self):
        """The link exists but is finished, so `blocked_dependency` stays quiet.
        The status still says Blocked, and that is now the only signal left."""
        issue = make_issue(
            status_name="Blocked",
            links=[
                IssueLink(
                    target_key="INS-5",
                    target_summary="Done work",
                    target_status_category=IssueStatusCategory.DONE,
                    link_type="blocks",
                    target_blocks_this=True,
                )
            ],
        )
        assert rule_blocked_dependency(issue, CFG, NOW) is None
        assert rule_blocked_status(issue, CFG, NOW) is not None

    def test_the_detail_says_no_blocker_was_recorded(self):
        """The actionable part. The fix is for someone to link the blocker, and
        the finding should say so rather than just restating the status."""
        detail = rule_blocked_status(make_issue(status_name="Blocked"), CFG, NOW).detail
        assert "no blocking issue is linked" in detail.lower()

    def test_it_is_included_in_evaluate(self):
        findings = evaluate([make_issue(status_name="Blocked")], CFG, NOW)
        assert any(f.rule_id == "blocked_status" for f in findings)


class TestBlockedStatusStandDownIsNarrow:
    """The stand-down must key on "a blocker was identified", not "the
    dependency rule said something".

    `rule_blocked_dependency` reports two different things under one rule id.
    HIGH means "blocked by X" — it names what this issue is waiting on, which
    is why the status finding defers to it. MEDIUM means "linked to X", which is
    usually the REVERSE direction: this issue blocks something else. That
    explains nothing about why it cannot move.

    Keyed on the rule firing at all, a ticket a human marked Blocked whose only
    link points downstream was reported as "Linked to 1 unfinished issue" — true,
    pointing the wrong way, and never mentioning that somebody flagged it. The
    explicit human signal was swallowed by a weaker automatic one.
    """

    def _blocked_marked_issue(self, **link_kwargs) -> Issue:
        return make_issue(
            status_name="Blocked",
            links=[
                IssueLink(
                    target_key="INS-40",
                    target_summary="Downstream work",
                    target_status_category=IssueStatusCategory.IN_PROGRESS,
                    link_type="blocks",
                    **link_kwargs,
                )
            ],
        )

    def test_a_downstream_link_does_not_suppress_the_status_finding(self):
        issue = self._blocked_marked_issue(target_blocks_this=False)
        assert rule_blocked_dependency(issue, CFG, NOW).level is RiskLevel.MEDIUM
        assert rule_blocked_status(issue, CFG, NOW) is not None

    def test_an_actual_blocker_still_suppresses_it(self):
        """The half that must not regress — no double-counting when the link
        genuinely explains the block."""
        issue = self._blocked_marked_issue(target_blocks_this=True)
        assert rule_blocked_dependency(issue, CFG, NOW).level is RiskLevel.HIGH
        assert rule_blocked_status(issue, CFG, NOW) is None

    def test_a_resolved_blocker_does_not_suppress_it(self):
        issue = make_issue(
            status_name="Blocked",
            links=[
                IssueLink(
                    target_key="INS-40", target_summary="Finished work",
                    target_status_category=IssueStatusCategory.DONE,
                    link_type="blocks", target_blocks_this=True,
                )
            ],
        )
        assert rule_blocked_status(issue, CFG, NOW) is not None

    def test_a_link_type_the_client_does_not_report_does_not_suppress_it(self):
        """`reported_link_types` excludes "relates to" by default. A link the
        engine has been told to ignore must not silently silence a different
        rule."""
        issue = make_issue(
            status_name="Blocked",
            links=[
                IssueLink(
                    target_key="INS-40", target_summary="Related work",
                    target_status_category=IssueStatusCategory.IN_PROGRESS,
                    link_type="relates_to", target_blocks_this=True,
                )
            ],
        )
        assert rule_blocked_dependency(issue, CFG, NOW) is None
        assert rule_blocked_status(issue, CFG, NOW) is not None
