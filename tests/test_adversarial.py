"""Proof that the grounding checks bite.

Live runs report 0 rejections. That is consistent with checks that work and with
checks that never fire; these tests distinguish the two.

Written after noticing that "0 rejected" was being read as a victory condition.
"""

from __future__ import annotations

import pytest

from app.agent.answering import (
    REFUSAL_MARKER,
    Passage,
    answer_question,
    check_grounding,
)
from app.agent.llm import FakeLLM, text_response
from tests.fixtures.adversarial import BAD_ANSWERS


SLA = Passage(
    label="Acme Contract",
    url="https://x/acme",
    text=(
        "Acme commit to a 5 business day turnaround on integration support "
        "requests. This commitment has not been met."
    ),
)


class TestCheckerRejectsBadAnswers:
    @pytest.mark.parametrize(
        "bad", BAD_ANSWERS, ids=[b.reason[:40] for b in BAD_ANSWERS]
    )
    def test_bad_answer_fails_grounding(self, bad):
        report = check_grounding(bad.text, passage_count=bad.passage_count)
        assert not report.is_grounded, (
            f"checker accepted an answer it should reject ({bad.reason})"
        )

    def test_cited_paragraph_does_not_launder_an_uncited_one(self):
        """The subtlest of the set: one real citation next to a fabrication.

        Paragraph-level enforcement means each paragraph is judged on its own,
        which is exactly what stops this.
        """
        text = (
            "INS-101 is stalled pending schema confirmation [1].\n\n"
            "Acme have confirmed they will respond within five working days."
        )
        assert not check_grounding(text, 5).is_grounded

    def test_markdown_structure_does_not_hide_a_claim(self):
        text = (
            "## Summary\n\n| Item | Status |\n|---|---|\n| Schema | Done |\n\n"
            "All blockers are now cleared and the quarter is fully on track."
        )
        assert not check_grounding(text, 5).is_grounded


class TestEndToEndRejection:
    @pytest.mark.parametrize("bad", BAD_ANSWERS, ids=lambda b: b.reason[:30])
    async def test_bad_model_output_never_reaches_the_user(self, bad):
        """The checker rejecting is not enough — the rejection must propagate."""
        llm = FakeLLM(responses=[text_response(bad.text)])
        answer = await answer_question(llm, "q", [SLA] * bad.passage_count)

        assert answer.refused, f"a bad answer was returned to the user ({bad.reason})"
        assert not answer.answered
        # The fabricated text must not be what the user sees.
        assert bad.text not in answer.text

    async def test_refusal_fallback_does_not_rescue_a_fabrication(self):
        """A fabricated answer containing declining-sounding words must still
        be rejected, not reclassified as a polite decline."""
        text = (
            "There is no doubt that Acme confirmed the schema on Thursday and "
            "the team resumed work immediately afterwards."
        )
        llm = FakeLLM(responses=[text_response(text)])
        answer = await answer_question(llm, "q", [SLA, SLA])
        assert answer.refusal_reason == "failed_grounding_check"


class TestGuardsAgainstFutureSimplification:
    """If someone relaxes the checker later, these fail before anything ships."""

    def test_an_answer_with_no_citations_at_all_is_rejected(self):
        assert not check_grounding(
            "The integration is progressing well and should land on time.", 5
        ).is_grounded

    def test_citation_beyond_the_passage_count_is_rejected(self):
        assert not check_grounding("Certificates were issued [9].", 5).is_grounded

    def test_a_properly_cited_answer_is_still_accepted(self):
        """The counterweight: these guards must not be satisfiable by rejecting
        everything."""
        assert check_grounding(
            "Acme committed to a five business day turnaround [1].", 5
        ).is_grounded


class TestRefusalFallbackPrecision:
    """The fallback matched a bare "there is no", so the fabrication "There is
    no doubt that Acme confirmed the schema" was filed as a polite decline.

    Found by an adversarial test, not by reading the regex. refusal_rate is a
    baselined metric; a fabrication counted as a refusal corrupts it.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "There is no doubt that Acme confirmed the schema on Thursday and work resumed.",
            "There is no remaining work on the compliance export, it shipped last week.",
            "The team has no committed date, but delivery is expected before quarter end.",
        ],
    )
    async def test_negation_alone_is_not_a_decline(self, text):
        llm = FakeLLM(responses=[text_response(text)])
        answer = await answer_question(llm, "q", [SLA, SLA])
        assert answer.refusal_reason == "failed_grounding_check"

    @pytest.mark.parametrize(
        "text",
        [
            "The context does not support that premise, so I cannot answer.",
            "The question presupposes an exception that was never raised.",
            "There is no evidence of an approval in the retrieved passages.",
            "The passages do not contain this information at all.",
        ],
    )
    async def test_evidence_referencing_language_is_a_decline(self, text):
        llm = FakeLLM(responses=[text_response(text)])
        answer = await answer_question(llm, "q", [SLA, SLA])
        assert answer.refusal_reason == "model_declined_without_marker"

    def test_pattern_requires_reference_to_the_evidence(self):
        """The rule that keeps this precise: a phrase must be ABOUT the
        sources, not merely contain a negation."""
        from app.agent.answering import _PROSE_REFUSAL

        assert not _PROSE_REFUSAL.search("There is no doubt about it")
        assert _PROSE_REFUSAL.search("The context does not contain this")
