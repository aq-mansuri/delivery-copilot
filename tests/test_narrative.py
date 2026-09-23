"""Tests for weekly narrative generation.

The guarantee under test is the same one `tests/test_answering.py` pins for
`/ask`, applied to a second consumer: every claim in the narrative must trace
to a specific finding, and a report with nothing to say gets a definitive
answer with no model call at all.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.agent.llm import FakeLLM, text_response
from app.agent.narrative import generate_narrative
from app.core.report import build_report
from app.core.sync_result import CompleteSync
from app.models.domain import Citation, RiskFinding, RiskLevel

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def finding(
    key="INS-1",
    rule_id="stale_in_progress",
    level=RiskLevel.HIGH,
    detail="In Progress with no activity for 21 days.",
    cited=True,
) -> RiskFinding:
    return RiskFinding(
        issue_key=key,
        rule_id=rule_id,
        level=level,
        detail=detail,
        citations=(
            [Citation(source_type="issue", source_id=key, url=f"https://x/browse/{key}")]
            if cited
            else []
        ),
    )


def complete(findings=None, **overrides) -> CompleteSync:
    defaults = dict(
        started_at=NOW,
        issues_seen=50,
        projects_requested=["INS"],
        projects_covered=["INS"],
        findings=list(findings if findings is not None else []),
    )
    defaults.update(overrides)
    return CompleteSync(**defaults)


class TestEmptyReport:
    async def test_no_findings_is_a_definitive_answer(self):
        report = build_report(complete(findings=[]))
        narrative = await generate_narrative(FakeLLM(responses=[]), report)
        assert narrative.answered
        assert "no risk findings" in narrative.text.lower()
        assert "definitive" in narrative.text.lower()

    async def test_no_findings_makes_no_model_call(self):
        """The deterministic branch, not a scripted refusal — the model is
        never asked, the same discipline `answer_question` applies when
        nothing was retrieved."""
        report = build_report(complete(findings=[]))
        llm = FakeLLM(responses=[])
        await generate_narrative(llm, report)
        assert llm.calls == []

    async def test_states_the_scope_that_was_checked(self):
        report = build_report(
            complete(findings=[], projects_covered=["INS", "CLM"], issues_seen=137)
        )
        narrative = await generate_narrative(FakeLLM(responses=[]), report)
        assert "INS" in narrative.text and "CLM" in narrative.text
        assert "137" in narrative.text


class TestGroundedNarrative:
    async def test_covers_every_finding_with_a_resolvable_citation(self):
        report = build_report(
            complete(
                findings=[
                    finding("INS-1"),
                    finding("INS-2", rule_id="blocked_dependency", level=RiskLevel.MEDIUM,
                             detail="Blocked by INS-9, which is still In Progress."),
                ]
            )
        )
        llm = FakeLLM(
            responses=[
                text_response(
                    "High severity: INS-1 has had no activity for 21 days [1].\n\n"
                    "Medium severity: INS-2 is blocked by INS-9, which is still in "
                    "progress [2]."
                )
            ]
        )
        narrative = await generate_narrative(llm, report)
        assert narrative.answered
        labels = {c.label for c in narrative.citations}
        assert labels == {"INS-1 — stale_in_progress", "INS-2 — blocked_dependency"}

    async def test_passage_url_traces_to_the_findings_own_citation(self):
        report = build_report(complete(findings=[finding("INS-1")]))
        llm = FakeLLM(responses=[text_response("INS-1 has had no activity for 21 days [1].")])
        narrative = await generate_narrative(llm, report)
        assert narrative.citations[0].url == "https://x/browse/INS-1"

    async def test_a_finding_with_no_recorded_citation_still_gets_a_passage(self):
        """A rule that has not (yet) attached a citation must not crash the
        narrative — it just has nothing to click through to."""
        report = build_report(complete(findings=[finding("INS-1", cited=False)]))
        llm = FakeLLM(responses=[text_response("INS-1 has had no activity for 21 days [1].")])
        narrative = await generate_narrative(llm, report)
        assert narrative.answered
        assert narrative.citations[0].url == ""

    async def test_uses_the_narrative_system_prompt(self):
        report = build_report(complete(findings=[finding("INS-1")]))
        llm = FakeLLM(responses=[text_response("INS-1 has had no activity for 21 days [1].")])
        await generate_narrative(llm, report)
        assert "weekly delivery-risk narrative" in llm.last_system_prompt


class TestUngroundedNarrativeIsRejected:
    async def test_a_claim_with_no_citation_is_withheld_not_shown(self):
        report = build_report(
            complete(
                findings=[
                    finding("INS-1"),
                    finding("INS-2", level=RiskLevel.MEDIUM),
                ]
            )
        )
        llm = FakeLLM(
            responses=[
                text_response(
                    "INS-1 has had no activity for 21 days [1].\n\n"
                    "INS-2 is also a serious concern that needs immediate "
                    "escalation to the platform team before the next review."
                )
            ]
        )
        narrative = await generate_narrative(llm, report)
        assert narrative.refused
        assert narrative.refusal_reason == "failed_grounding_check"
        assert "attribute" in narrative.text.lower()
