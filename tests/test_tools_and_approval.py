"""Tests for the tool split and the approval gate.

The guarantee: no code path exists from a model's tool call to a Jira write.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.agent.approval import (
    ApprovalError,
    ApprovalStore,
    ApprovedAction,
    Decision,
    apply_action,
)
from app.agent.tools import (
    ToolRegistry,
    make_flag_tool,
    make_risk_tool,
    make_search_tool,
    risk_tool_for,
)
from app.core.sync_result import CompleteSync, PartialReason, PartialSync
from app.models.domain import (
    Citation,
    Issue,
    IssueStatusCategory,
    Page,
    PageSection,
    ProposedAction,
    RiskFinding,
    RiskLevel,
)
from app.rag.chunking import chunk_page
from app.rag.embedders import FakeEmbedder
from app.rag.retrieval import HybridRetriever

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def finding(key="INS-101", rule_id="stale_in_progress", level=RiskLevel.HIGH):
    return RiskFinding(
        issue_key=key,
        rule_id=rule_id,
        level=level,
        detail="No activity for 23 days.",
        citations=[
            Citation(
                source_type="issue",
                source_id=key,
                url=f"https://x.atlassian.net/browse/{key}",
            )
        ],
    )


def proposal(key="INS-101", with_citations=True):
    return ProposedAction(
        issue_key=key,
        action_type="flag_at_risk",
        payload={"labels_add": ["at-risk"]},
        rationale="Stale for 23 days with no assignee activity.",
        citations=(
            [Citation(source_type="risk_rule", source_id="stale_in_progress", url="#r")]
            if with_citations
            else []
        ),
    )


class RecordingWriter:
    def __init__(self, fail: bool = False):
        self.labels: list[tuple[str, list[str]]] = []
        self.comments: list[tuple[str, str]] = []
        self.fail = fail

    async def add_labels(self, issue_key: str, labels: list[str]) -> None:
        if self.fail:
            raise RuntimeError("Jira 500")
        self.labels.append((issue_key, labels))

    async def add_comment(self, issue_key: str, body: str) -> None:
        if self.fail:
            raise RuntimeError("Jira 500")
        self.comments.append((issue_key, body))


class TestWriteToolsCannotWrite:
    async def test_flag_tool_returns_a_proposal_not_a_write(self):
        result = await make_flag_tool().handler(
            issue_key="INS-101",
            rationale="Stale.",
            evidence_rule_ids=["stale_in_progress"],
        )
        assert result.proposal is not None
        assert "AWAITING HUMAN APPROVAL" in result.content
        assert "nothing has been written" in result.content

    async def test_proposal_without_evidence_is_rejected_at_the_tool(self):
        result = await make_flag_tool().handler(
            issue_key="INS-101", rationale="Feels risky.", evidence_rule_ids=[]
        )
        assert result.is_error
        assert result.proposal is None

    def test_registry_holds_no_writer(self):
        """The security property is an absence, not a check."""
        registry = ToolRegistry().register(make_flag_tool())
        assert not hasattr(registry, "writer")
        assert not hasattr(registry, "jira")

    def test_read_only_schemas_exclude_write_tools(self):
        """Withholding a tool beats instructing the model not to use it."""
        registry = ToolRegistry().register(
            make_risk_tool([finding()]), make_flag_tool()
        )
        names = {t["name"] for t in registry.read_only_schemas()}
        assert names == {"get_risk_findings"}
        assert len(registry.schemas()) == 2


class TestApprovalGate:
    async def test_unapproved_proposal_cannot_reach_the_writer(self):
        store, writer = ApprovalStore(), RecordingWriter()
        with pytest.raises(ApprovalError) as exc:
            await apply_action(proposal(), writer, store)
        assert "must never reach Jira" in str(exc.value)
        assert writer.labels == [] and writer.comments == []

    async def test_approved_action_is_applied(self):
        store, writer = ApprovalStore(), RecordingWriter()
        pid = store.submit(proposal())
        approved = store.approve(pid, actor="vp@acme.com")
        await apply_action(approved, writer, store)

        assert writer.labels == [("INS-101", ["at-risk"])]
        assert "vp@acme.com" in writer.comments[0][1]

    def test_approval_requires_a_named_actor(self):
        store = ApprovalStore()
        pid = store.submit(proposal())
        with pytest.raises(ApprovalError) as exc:
            store.approve(pid, actor="  ")
        assert "named actor" in str(exc.value)

    def test_proposal_without_citations_is_refused_at_intake(self):
        with pytest.raises(ApprovalError) as exc:
            ApprovalStore().submit(proposal(with_citations=False))
        assert "evidence" in str(exc.value)

    def test_a_proposal_cannot_be_approved_twice(self):
        """Re-approving would double-apply the write."""
        store = ApprovalStore()
        pid = store.submit(proposal())
        store.approve(pid, actor="vp@acme.com")
        with pytest.raises(ApprovalError):
            store.approve(pid, actor="vp@acme.com")


class TestAuditTrail:
    def test_rejections_are_recorded(self):
        """'A lead said no' is evidence, and it protects the lead later."""
        store = ApprovalStore()
        pid = store.submit(proposal())
        store.reject(pid, actor="vp@acme.com", note="Known, vendor chasing it.")

        entry = store.audit[0]
        assert entry.decision is Decision.REJECTED
        assert entry.actor == "vp@acme.com"
        assert "vendor chasing" in entry.note

    async def test_audit_is_written_before_the_api_call(self):
        """If the write succeeds and logging dies, the regulator sees a change
        with no approval record. Recording first inverts that risk."""
        store, writer = ApprovalStore(), RecordingWriter(fail=True)
        pid = store.submit(proposal())
        approved = store.approve(pid, actor="vp@acme.com")

        assert len(store.audit) == 1  # recorded at approval, before apply

        with pytest.raises(RuntimeError):
            await apply_action(approved, writer, store)

        assert store.audit[0].applied is False
        assert "Jira 500" in store.audit[0].apply_error

    async def test_successful_apply_is_marked(self):
        store, writer = ApprovalStore(), RecordingWriter()
        pid = store.submit(proposal())
        await apply_action(store.approve(pid, actor="p@acme.com"), writer, store)
        assert store.audit[0].applied is True


class TestReadTools:
    async def test_risk_tool_distinguishes_clean_from_broken(self):
        """'No findings' must not read like 'the check failed'.

        Asserted as a property rather than an exact sentence. The original
        pinned the literal string, so it kept passing while the wording it
        pinned was being misread by the model downstream — the test agreed with
        the code instead of with the requirement. See
        TestAnEmptyResultIsAnAnswerNotAGap.
        """
        clean = await make_risk_tool([]).handler()
        assert not clean.is_error
        assert "no risk findings" in clean.content.lower()
        # Unambiguously a result, not an outage.
        for failure_word in ("failed", "unavailable", "error", "could not"):
            assert failure_word not in clean.content.lower(), failure_word

    async def test_risk_tool_filters_by_issue(self):
        tool = make_risk_tool([finding("INS-101"), finding("INS-102")])
        result = await tool.handler(issue_key="ins-101")
        assert "INS-101" in result.content and "INS-102" not in result.content

    async def test_search_tool_returns_numbered_passages_with_sources(self):
        page = Page(
            id="p1", space_key="ARCH", title="ADR: TLS",
            url="https://x/wiki/p1", version=1, updated_at=NOW,
            sections=[PageSection(heading="Context", level=2, text="Mutual TLS. " * 30)],
        )
        retriever = HybridRetriever(chunk_page(page), FakeEmbedder())
        result = await make_search_tool(retriever).handler(query="mutual TLS")
        assert result.content.startswith("[1]")
        assert "Source: https://" in result.content


class TestRegistryResilience:
    async def test_unknown_tool_returns_an_error_not_a_crash(self):
        """A hallucinated tool name should let the model self-correct."""
        result = await ToolRegistry().register(make_flag_tool()).execute("nope", {})
        assert result.is_error
        assert "propose_flag_at_risk" in result.content

    async def test_bad_arguments_are_surfaced_to_the_model(self):
        registry = ToolRegistry().register(make_flag_tool())
        result = await registry.execute("propose_flag_at_risk", {"wrong": 1})
        assert result.is_error
        assert "Invalid arguments" in result.content


class TestSyncGateReachesTheAgent:
    """ADR-004 said a partial sync cannot produce a report. The risk tool is a
    second consumer of the same findings, and it needs the same guarantee — a
    lead reading "INS-101 is the only thing at risk" cannot tell that two of six
    projects were never read.
    """

    def _complete(self):
        return CompleteSync(
            started_at=NOW,
            issues_seen=4,
            projects_requested=["INS"],
            projects_covered=["INS"],
            findings=[finding()],
        )

    def _partial(self):
        return PartialSync(
            started_at=NOW,
            issues_seen=2,
            projects_requested=["INS", "CLM"],
            partial_findings=[finding()],
            projects_covered=["INS"],
            projects_incomplete=["CLM"],
            reason=PartialReason.PERMISSION_DENIED,
            detail="403 on CLM",
        )

    async def test_complete_sync_serves_its_findings(self):
        result = await risk_tool_for(self._complete()).handler()
        assert "INS-101" in result.content
        assert not result.is_error

    async def test_partial_sync_serves_no_findings_at_all(self):
        """Not "here is what we have, caveat attached". The findings collected
        so far are exactly the ones that read as a complete picture."""
        result = await risk_tool_for(self._partial()).handler()
        assert "INS-101" not in result.content
        assert result.is_error

    async def test_partial_sync_says_what_to_do_about_it(self):
        result = await risk_tool_for(self._partial()).handler()
        assert "Browse Projects" in result.content
        assert "CLM" in result.content

    async def test_partial_sync_tool_still_answers_a_filtered_call(self):
        """The model does not know the sync failed, so it will pass arguments.
        A TypeError here would surface as a tool crash rather than an
        explanation the model can relay."""
        result = await risk_tool_for(self._partial()).handler(
            issue_key="INS-101", level="high"
        )
        assert result.is_error
        assert "INS-101" not in result.content

    def test_both_branches_present_the_same_tool_to_the_model(self):
        """The schema and name must not change with sync health, or the model
        sees a different toolset depending on a Jira permission."""
        complete = risk_tool_for(self._complete())
        partial = risk_tool_for(self._partial())
        assert complete.name == partial.name
        assert complete.input_schema == partial.input_schema
        assert complete.writes is False and partial.writes is False


class TestTheAuditTrailDoesNotClaimAWriteItDidNotMake:
    """`applied: True` meant "the writer returned without error".

    The UI rendered that as ", written to Jira" — while the default writer was
    `LoggingWriter`, which holds no Jira client and logs a line. So the
    Decisions panel stated a change had been made to a tenant that had never
    been contacted. Everything else in this system refuses to assert what it
    cannot show; this one string asserted the opposite.

    The distinction is recorded per entry rather than read from config at
    render time, because an audit row is a durable claim about a past event. A
    row written under a dry run must still say so after a real writer is
    injected — otherwise switching a deployment setting silently rewrites
    history.
    """

    async def test_a_dry_run_writer_is_not_recorded_as_a_jira_write(self):
        store = ApprovalStore()
        pid = store.submit(proposal())
        writer = RecordingWriter()  # no writes_to_jira, so: claims nothing
        await apply_action(store.approve(pid, actor="p@acme.com"), writer, store)

        entry = store.audit[0]
        assert entry.applied is True
        assert entry.written_to_jira is False
        assert entry.to_row()["written_to_jira"] == "False"

    async def test_a_real_writer_is_recorded_as_a_jira_write(self):
        """The counterweight. A test that only ever asserts the cautious value
        passes against code that hardcodes it."""

        class LiveWriter(RecordingWriter):
            writes_to_jira = True

        store = ApprovalStore()
        pid = store.submit(proposal())
        await apply_action(store.approve(pid, actor="p@acme.com"), LiveWriter(), store)
        assert store.audit[0].written_to_jira is True

    async def test_an_undeclared_writer_claims_nothing(self):
        """Default to the claim that is safe to be wrong about. A writer that
        forgets to declare itself under-reports; the reverse invents a write."""

        class Anonymous:
            async def add_labels(self, issue_key, labels): ...
            async def add_comment(self, issue_key, body): ...

        store = ApprovalStore()
        pid = store.submit(proposal())
        await apply_action(store.approve(pid, actor="p@acme.com"), Anonymous(), store)
        assert store.audit[0].written_to_jira is False

    async def test_a_failed_write_claims_nothing_either(self):
        class FailingLiveWriter(RecordingWriter):
            writes_to_jira = True

        store = ApprovalStore()
        pid = store.submit(proposal())
        writer = FailingLiveWriter(fail=True)
        with pytest.raises(RuntimeError):
            await apply_action(store.approve(pid, actor="p@acme.com"), writer, store)

        entry = store.audit[0]
        assert entry.applied is False
        assert entry.written_to_jira is False
        assert entry.apply_error

    def test_a_rejection_never_claims_a_write(self):
        store = ApprovalStore()
        pid = store.submit(proposal())
        store.reject(pid, actor="p@acme.com")
        assert store.audit[0].written_to_jira is False


class TestAnEmptyResultIsAnAnswerNotAGap:
    """"Is INS-12 blocked?" returned "the available content does not contain
    enough information to answer this."

    Every step before the last one was right. The model called
    `get_risk_findings(issue_key="INS-12")`, the rules had genuinely found
    nothing, and the tool said so. Then the answering model read "No risk
    findings for INS-12" as *missing information* and declined under rule 1.

    Absence of evidence read as evidence of absence — inverted. The rules are
    exhaustive over what they check, so "no findings" is not a gap in the
    context, it is the answer, and the answer is no. A delivery lead cannot act
    on "we don't know" when the system does know.

    The fix is in the tool's wording rather than the answering prompt, on
    purpose. The prompt governs refusal on genuinely unanswerable questions —
    a baselined metric at 100% (ADR-006) — and the tool only ever speaks when
    the rules have actually run, so it cannot loosen refusal anywhere else.
    """

    async def test_it_states_the_negative_rather_than_the_absence(self):
        result = await make_risk_tool([]).handler(issue_key="INS-12")
        text = result.content.lower()
        assert "no risk findings" in text
        assert "ins-12" in text
        # The words a model needs to answer the question actually asked.
        assert "not blocked" in text

    async def test_it_says_the_result_is_definitive(self):
        """Without this the model cannot tell "the rules found nothing" from
        "nothing was checked", and declining is the safe reading of the second."""
        result = await make_risk_tool([]).handler(issue_key="INS-12")
        assert "definitive" in result.content.lower()
        assert not result.is_error

    async def test_it_enumerates_what_was_actually_checked(self):
        """A bare "no findings" invites the model to hedge about coverage. The
        rules are a closed set; naming them makes the scope of the negative
        explicit."""
        text = (await make_risk_tool([]).handler(issue_key="INS-12")).content.lower()
        for checked in ("stale", "blocked", "overdue", "sprint"):
            assert checked in text, checked

    async def test_a_level_filter_reads_naturally(self):
        text = (await make_risk_tool([]).handler(level="high")).content
        assert "high" in text
        assert "no risk findings" in text.lower()

    async def test_an_unfiltered_empty_sync_reads_naturally(self):
        text = (await make_risk_tool([]).handler()).content.lower()
        assert "no risk findings" in text

    async def test_a_populated_result_is_unchanged(self):
        """The negative wording must not leak into the positive case."""
        text = (await make_risk_tool([finding()]).handler()).content
        assert "INS-101" in text
        assert "definitive" not in text.lower()
        assert "not blocked" not in text.lower()

    async def test_an_unavailable_sync_still_reads_as_a_failure(self):
        """The one case that genuinely IS missing data. It must not pick up the
        new "we checked and it is fine" phrasing — that would report an unread
        project as good news."""
        from app.agent.tools import make_unavailable_risk_tool

        result = await make_unavailable_risk_tool("Sync incomplete: CLM.").handler()
        assert result.is_error
        assert "not blocked" not in result.content.lower()
        assert "definitive" not in result.content.lower()


class TestTheNegativeDoesNotOverclaim:
    """"Definitive" has to be scoped to what was actually checked.

    The first version of this wording ended "It is not blocked and not at risk."
    On a live tenant that sentence was produced for INS-12, whose status was
    literally `Blocked` — the rules had found nothing because no blocker was
    *linked*, and the wording turned a narrow, true negative into a broad,
    false one.

    A confident wrong answer is the worst output this system can produce, and
    making the negative more assertive is precisely what created the risk. The
    fix is not to hedge it back into uselessness but to say exactly what the
    "no" covers.
    """

    async def test_it_does_not_claim_more_than_the_rules_checked(self):
        text = (await make_risk_tool([]).handler(issue_key="INS-12")).content.lower()
        assert "not at risk" not in text

    async def test_it_scopes_the_negative_to_the_rules(self):
        text = (await make_risk_tool([]).handler(issue_key="INS-12")).content.lower()
        assert "no blocking issue is linked" in text or "linked" in text
        assert "rules" in text

    async def test_it_is_still_a_usable_answer_rather_than_a_hedge(self):
        """The counterweight. Scoping it must not undo the fix — a lead asking
        "is INS-12 blocked?" still needs to be told no."""
        text = (await make_risk_tool([]).handler(issue_key="INS-12")).content.lower()
        assert "not blocked" in text
        assert "definitive" in text
        assert "declin" in text  # still instructs the model not to refuse


class TestTheNegativeIsAnchoredToWhenTheDataWasRead:
    """A definitive "no" from a stale sync is a confident wrong answer.

    Twice on a live tenant: a blocker was added to INS-12, and the service —
    which reads findings once at startup — kept answering "it is not blocked
    (no blocking issue is linked to it and its status is not a blocked one)".
    True of the snapshot it held, false of Jira, and stated without a hint that
    the two might differ.

    The assertive wording is right; anchoring it is what makes it safe. The
    reader needs to see which moment the claim is about.
    """

    def _sync(self, finished_at):
        from app.core.sync_result import CompleteSync

        return CompleteSync(
            started_at=finished_at, finished_at=finished_at, issues_seen=4,
            projects_requested=["INS"], projects_covered=["INS"], findings=[],
        )

    async def test_the_empty_result_names_when_the_sync_ran(self):
        from datetime import datetime, timezone

        at = datetime(2026, 9, 23, 14, 30, tzinfo=timezone.utc)
        result = await risk_tool_for(self._sync(at)).handler(issue_key="INS-12")
        assert "2026-09-23" in result.content
        assert "14:30" in result.content

    async def test_it_says_the_claim_is_as_of_that_moment(self):
        from datetime import datetime, timezone

        result = await risk_tool_for(
            self._sync(datetime(2026, 9, 23, 14, 30, tzinfo=timezone.utc))
        ).handler(issue_key="INS-12")
        assert "as of" in result.content.lower()

    async def test_a_populated_result_is_still_not_cluttered_with_it(self):
        """Findings carry their own evidence and urgency. The timestamp earns
        its place only on the negative, where there is nothing else to check."""
        from datetime import datetime, timezone
        from app.core.sync_result import CompleteSync

        sync = CompleteSync(
            started_at=datetime(2026, 9, 23, 14, 30, tzinfo=timezone.utc),
            issues_seen=4, projects_requested=["INS"], projects_covered=["INS"],
            findings=[finding()],
        )
        assert "as of" not in (await risk_tool_for(sync).handler()).content.lower()
