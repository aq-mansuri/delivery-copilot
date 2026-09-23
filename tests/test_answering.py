"""Tests for grounded answering and refusal.

These encode ADR-006: refusal is enforced here, not in retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.answering import (
    REFUSAL_MARKER,
    Passage,
    answer_question,
    build_context,
    check_grounding,
)
from app.agent.llm import FakeLLM, text_response
from app.models.domain import Page, PageSection
from app.rag.chunking import chunk_page
from app.rag.retrieval import ScoredChunk

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def make_hits(n: int = 2) -> list[ScoredChunk]:
    page = Page(
        id="adr-tls",
        space_key="ARCH",
        title="ADR: Claims vendor TLS",
        url="https://acme.atlassian.net/wiki/spaces/ARCH/pages/adr-tls",
        version=1,
        updated_at=NOW,
        sections=[
            PageSection(heading="Context", level=2, text="Mutual TLS is required. " * 20),
            PageSection(heading="Decision", level=2, text="Terminate at gateway. " * 20),
        ],
    )
    chunks = chunk_page(page)
    return [
        ScoredChunk(chunk=c, score=0.03, bm25_rank=i, vector_rank=i)
        for i, c in enumerate(chunks[:n])
    ]


class TestNoContext:
    async def test_empty_retrieval_declines_without_calling_the_model(self):
        """Paying for a confabulation is strictly worse than not asking."""
        llm = FakeLLM(responses=[])
        answer = await answer_question(llm, "anything?", [])
        assert answer.refused
        assert answer.refusal_reason == "no_context_retrieved"
        assert llm.calls == []


class TestModelRefusal:
    async def test_refusal_marker_is_honoured(self):
        llm = FakeLLM(responses=[text_response(REFUSAL_MARKER)])
        answer = await answer_question(llm, "Who approved the exception?", make_hits())
        assert answer.refused
        assert answer.refusal_reason == "model_declined"
        assert REFUSAL_MARKER not in answer.text

    async def test_refusal_text_is_client_readable(self):
        """A lead should never see an internal marker string."""
        llm = FakeLLM(responses=[text_response(REFUSAL_MARKER)])
        answer = await answer_question(llm, "q", make_hits())
        assert "does not contain enough information" in answer.text


class TestGroundingEnforcement:
    async def test_uncited_claim_is_rejected(self):
        """The prompt asks for citations; this is what happens when it doesn't."""
        llm = FakeLLM(
            responses=[
                text_response(
                    "The vendor has confirmed the schema and work resumes Monday."
                )
            ]
        )
        answer = await answer_question(llm, "status?", make_hits())
        assert answer.refused
        assert answer.refusal_reason == "failed_grounding_check"
        assert "uncited claim" in answer.rejected_reason

    async def test_citation_out_of_range_is_rejected(self):
        """Citing [7] when 2 passages were supplied means an invented source."""
        llm = FakeLLM(
            responses=[text_response("Mutual TLS is required by the vendor [7].")]
        )
        answer = await answer_question(llm, "q", make_hits(2))
        assert answer.refused
        assert "non-existent passages [7]" in answer.rejected_reason

    async def test_properly_cited_answer_is_returned(self):
        llm = FakeLLM(
            responses=[
                text_response("Mutual TLS is required on every callback [1].")
            ]
        )
        answer = await answer_question(llm, "q", make_hits(2))
        assert answer.answered
        assert len(answer.citations) == 1
        assert answer.sources()[0]["url"].endswith("adr-tls")

    async def test_multiple_citations_resolve_to_distinct_sources(self):
        llm = FakeLLM(
            responses=[
                text_response(
                    "Mutual TLS is required [1]. It terminates at the gateway [2]."
                )
            ]
        )
        answer = await answer_question(llm, "q", make_hits(2))
        assert answer.answered
        assert len(answer.citations) == 2


class TestGroundingChecker:
    def test_short_connective_sentences_are_exempt(self):
        """Demanding a citation for 'Here is what I found.' just trains the
        model to scatter brackets."""
        report = check_grounding("Here is what I found. TLS is required [1].", 2)
        assert report.is_grounded

    def test_colon_lead_ins_are_exempt(self):
        report = check_grounding("Two items are at risk:\nINS-101 is stale [1].", 2)
        assert report.is_grounded

    def test_long_uncited_sentence_is_caught(self):
        report = check_grounding(
            "The claims vendor has committed to delivering the schema by Friday "
            "and engineering will resume immediately afterwards.",
            2,
        )
        assert not report.is_grounded
        assert len(report.uncited_sentences) == 1


class TestContextBlock:
    def test_passages_are_numbered_from_one(self):
        context = build_context([Passage.from_hit(h) for h in make_hits(2)])
        assert context.startswith("[1]")
        assert "[2]" in context

    def test_source_url_is_included_per_passage(self):
        """A citation the reader cannot follow is not a citation."""
        assert "Source: https://" in build_context(
            [Passage.from_hit(h) for h in make_hits(1)]
        )


class TestPromptContract:
    async def test_system_prompt_forbids_inference(self):
        llm = FakeLLM(responses=[text_response("TLS required [1].")])
        await answer_question(llm, "q", make_hits())
        prompt = llm.last_system_prompt
        assert "Never infer" in prompt
        assert REFUSAL_MARKER in prompt
        assert "presupposes" in prompt

    async def test_temperature_is_zero(self):
        """Same question, same data, same answer. Sampling variance also makes
        eval scores noisy enough to hide regressions."""
        llm = FakeLLM(responses=[text_response("TLS required [1].")])
        await answer_question(llm, "q", make_hits())
        assert llm.calls[0]["temperature"] == 0.0


class TestToolEvidenceIsCitable:
    """Regression guard for a bug the end-to-end demo found and every unit test
    missed: an answer built from tool output cited an unrelated Confluence page,
    because only retrieval chunks were numbered."""

    async def test_tool_output_can_be_cited(self):
        evidence = [
            Passage.from_tool(
                "get_risk_findings",
                "[HIGH] INS-101 (stale_in_progress): No activity for 23 days.",
            )
        ]
        llm = FakeLLM(
            responses=[text_response("INS-101 has had no activity for 23 days [1].")]
        )
        answer = await answer_question(llm, "what is at risk?", evidence)
        assert answer.answered
        assert answer.sources()[0]["label"].startswith("get_risk_findings")
        assert answer.sources()[0]["found_by"] == "tool"

    async def test_tool_and_retrieval_evidence_mix_without_collision(self):
        evidence = [
            Passage.from_tool("get_risk_findings", "[HIGH] INS-101 stale."),
            *[Passage.from_hit(h) for h in make_hits(1)],
        ]
        llm = FakeLLM(
            responses=[text_response("INS-101 is stale [1]. TLS is required [2].")]
        )
        answer = await answer_question(llm, "q", evidence)
        assert answer.answered
        labels = [s["label"] for s in answer.sources()]
        assert labels[0].startswith("get_risk_findings")
        assert "ADR" in labels[1]


class TestAgainstRecordedLiveAnswers:
    """Regression guards built from real Claude output.

    The first live run rejected 3 of 4 answerable questions. Every case here
    failed then and must pass now — and the two fabrication guards at the end
    must still fail, or the fix has simply disabled the check.
    """

    @pytest.mark.parametrize(
        "name",
        ["RELEASE_PLAN", "HOLDING_UP", "ACME_CROSS_SOURCE", "CONTROLS_OUTSTANDING"],
    )
    def test_real_answers_are_accepted(self, name):
        from tests.fixtures import live_answers

        report = check_grounding(getattr(live_answers, name), passage_count=5)
        assert report.is_grounded, f"{name}: {report.uncited_sentences}"

    def test_citation_after_the_full_stop_credits_its_own_sentence(self):
        """The off-by-one that caused every rejection in the first live run."""
        report = check_grounding(
            "The compliance export has no upstream dependencies. [3]", 5
        )
        assert report.is_grounded

    def test_markdown_tables_and_headings_are_not_claims(self):
        text = "### Schedule\n\n| Item | Target |\n|---|---|\n| Integration | Q4 |\n[4]"
        assert check_grounding(text, 5).is_grounded

    def test_fabrication_is_still_rejected(self):
        """The fix must not have quietly disabled the check."""
        report = check_grounding(
            "The vendor confirmed the schema last Thursday and engineering "
            "resumed work immediately afterwards.\n\nDelivery is expected Friday.",
            5,
        )
        assert not report.is_grounded

    def test_invented_passage_number_is_still_rejected(self):
        report = check_grounding("Mutual TLS is required by the vendor [7].", 5)
        assert not report.is_grounded

    def test_strict_mode_is_available_for_evals(self):
        """Per-sentence remains measurable, just not the default."""
        from tests.fixtures import live_answers

        strict = check_grounding(
            live_answers.ACME_CROSS_SOURCE, 5, per_sentence=True
        )
        lenient = check_grounding(live_answers.ACME_CROSS_SOURCE, 5)
        assert lenient.is_grounded
        assert not strict.is_grounded


class TestSecondLiveRun:
    """Fixtures from the run after ADR-008. Answered went 1/6 -> 3/6; these are
    the two that still failed, and they had different causes."""

    def test_short_lead_in_paragraph_is_not_a_claim(self):
        """'Based on the context, two of three items are blocked.' introduces
        cited content. Rejecting it makes the model open with a bracket."""
        from tests.fixtures import live_answers

        report = check_grounding(live_answers.HOLDING_UP_WITH_LEAD_IN, 5)
        assert report.is_grounded, report.uncited_sentences

    def test_long_uncited_opener_is_still_a_claim(self):
        """The lead-in exemption must not become a free pass for a paragraph
        that asserts things."""
        text = (
            "The vendor confirmed the schema on Thursday, engineering resumed "
            "immediately, and the team now expects to close the integration "
            "before the end of the quarter with high confidence.\n\n"
            "Separately, controls remain outstanding [3]."
        )
        assert not check_grounding(text, 5).is_grounded

    def test_lead_in_exemption_requires_later_citations(self):
        """An uncited opener with nothing cited after it is just an uncited
        answer."""
        text = "Based on the context, two of the three items are blocked.\n\nThey remain unresolved and will slip past the quarter end."
        assert not check_grounding(text, 5).is_grounded

    async def test_prose_refusal_is_recorded_as_a_decline_not_a_failure(self):
        """The finding that would have corrupted Day 5's metrics.

        The model declined correctly and explained the false premise, but omitted
        the marker. Filing that as failed_grounding_check counts a correct
        refusal as a bug.
        """
        from tests.fixtures import live_answers

        llm = FakeLLM(
            responses=[text_response(live_answers.PROSE_REFUSAL_NO_MARKER)]
        )
        answer = await answer_question(llm, "Who approved the exception?", make_hits(2))

        assert answer.refused
        assert answer.refusal_reason == "model_declined_without_marker"
        assert not answer.rejected_reason

    async def test_marker_decline_and_prose_decline_are_distinguishable(self):
        """Recorded separately so the eval can see how often the contract is
        missed. A fallback that hides its own use never gets fixed."""
        llm = FakeLLM(responses=[text_response(REFUSAL_MARKER)])
        answer = await answer_question(llm, "q", make_hits(2))
        assert answer.refusal_reason == "model_declined"

    async def test_confabulation_is_not_rescued_by_the_refusal_fallback(self):
        """A fabricated answer must not sneak through by containing a phrase
        that looks like declining."""
        llm = FakeLLM(
            responses=[
                text_response(
                    "The vendor confirmed the schema last Thursday and "
                    "engineering resumed work immediately afterwards."
                )
            ]
        )
        answer = await answer_question(llm, "q", make_hits(2))
        assert answer.refusal_reason == "failed_grounding_check"

    def test_prompt_requires_the_marker_even_when_explaining(self):
        from app.agent.answering import SYSTEM_PROMPT

        prompt = SYSTEM_PROMPT.format(refusal=REFUSAL_MARKER)
        assert "still include the exact refusal line" in prompt


class TestRestraintRules:
    """Added after claim-level judging found 26% overreach across live runs.

    The failures were systematic, not random: strengthening vocabulary
    ("cannot progress" -> "blocked"), conclusions the source does not draw
    ("confirmed breach" for "commitment not met"), and an unsolicited
    recommendations section in every answer.
    """

    def test_prompt_forbids_strengthening_the_source(self):
        from app.agent.answering import SYSTEM_PROMPT

        prompt = SYSTEM_PROMPT.format(refusal=REFUSAL_MARKER)
        assert "cannot progress at present" in prompt
        assert "breach" in prompt

    def test_prompt_forbids_recommendations(self):
        from app.agent.answering import SYSTEM_PROMPT

        prompt = SYSTEM_PROMPT.format(refusal=REFUSAL_MARKER)
        assert "do not recommend actions" in prompt.lower()
        assert "recommended actions" in prompt.lower()

    def test_prompt_names_the_actual_failure_mode(self):
        """Not invention — a helpful assistant being helpful where helpfulness
        is a liability."""
        from app.agent.answering import SYSTEM_PROMPT

        prompt = SYSTEM_PROMPT.format(refusal=REFUSAL_MARKER)
        assert "The most common failure here is not invention" in prompt
