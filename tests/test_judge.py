"""Tests for the groundedness judge."""

from __future__ import annotations

import json

import pytest

from app.agent.answering import Passage
from app.agent.judge import (
    Claim,
    Verdict,
    calibrate,
    extract_claims,
    judge_answer,
    judge_claim,
)
from app.agent.llm import FakeLLM, text_response
from tests.fixtures.calibration_cases import CASES


def verdict_response(verdict: str, evidence: str = "", reason: str = "x"):
    return text_response(
        json.dumps({"verdict": verdict, "evidence": evidence, "reason": reason})
    )


SLA = Passage(
    label="Acme Contract",
    url="https://x/acme",
    text=(
        "Acme commit to a 5 business day turnaround on integration support "
        "requests. This commitment has not been met."
    ),
)


class TestClaimExtraction:
    def test_sentences_inherit_paragraph_citations(self):
        """Paragraph-level enforcement grants this inheritance; the judge exists
        to measure what that grant lets through."""
        text = "There is a clear contractual breach. Acme missed the window. [5]"
        claims = extract_claims(text)
        assert len(claims) == 2
        assert all(c.citation_indices == (5,) for c in claims)

    def test_uncited_paragraphs_are_skipped(self):
        """check_grounding already governs those; judging against no passage is
        meaningless."""
        assert extract_claims("A paragraph with no citation at all here.") == []

    def test_citation_markers_are_stripped_from_claim_text(self):
        claims = extract_claims("Acme missed the agreed window. [5]")
        assert "[5]" not in claims[0].text

    def test_markdown_bullets_are_cleaned(self):
        claims = extract_claims("- Escalate through vendor management. [3]")
        assert claims[0].text.startswith("Escalate")

    def test_fragments_are_ignored(self):
        assert extract_claims("Yes. [1]") == []


class TestJudging:
    async def test_supported_claim_passes(self):
        llm = FakeLLM(responses=[
            verdict_response("supported", "5 business day turnaround")
        ])
        result = await judge_claim(
            llm, Claim("Acme committed to a 5 day turnaround.", (1,)), [SLA]
        )
        assert result.verdict is Verdict.SUPPORTED

    async def test_overreach_is_distinct_from_unsupported(self):
        """Binary supported/unsupported collapses the failure we actually saw."""
        llm = FakeLLM(responses=[verdict_response("overreach", "commitment has not been met")])
        result = await judge_claim(
            llm, Claim("Acme is in breach of contract.", (1,)), [SLA]
        )
        assert result.verdict is Verdict.OVERREACH

    async def test_judge_sees_only_cited_passages(self):
        """Shown the whole answer, a judge rates a confident one more generously."""
        other = Passage(label="Other", url="", text="Unrelated content here.")
        llm = FakeLLM(responses=[verdict_response("supported", "5 business day")])
        await judge_claim(llm, Claim("Acme committed to 5 days.", (1,)), [SLA, other])
        sent = llm.calls[0]["messages"][0]["content"]
        assert "Unrelated content" not in sent

    async def test_fabricated_evidence_downgrades_the_verdict(self):
        """An unverifiable 'supported' is not evidence of support."""
        llm = FakeLLM(responses=[
            verdict_response("supported", "Acme confirmed the schema on Thursday")
        ])
        result = await judge_claim(llm, Claim("Schema confirmed.", (1,)), [SLA])
        assert result.verdict is Verdict.UNJUDGED
        assert "not in the passage" in result.rejected_reason

    async def test_out_of_range_citation_is_unsupported(self):
        llm = FakeLLM(responses=[])
        result = await judge_claim(llm, Claim("Something.", (9,)), [SLA])
        assert result.verdict is Verdict.UNSUPPORTED
        assert llm.calls == []

    async def test_unparseable_judge_output_is_unjudged_not_supported(self):
        """Failing open would inflate every groundedness score."""
        llm = FakeLLM(responses=[text_response("I think it's fine.")])
        result = await judge_claim(llm, Claim("A claim here.", (1,)), [SLA])
        assert result.verdict is Verdict.UNJUDGED


class TestReport:
    async def test_rates_are_computed_per_verdict(self):
        llm = FakeLLM(responses=[
            verdict_response("supported", "5 business day turnaround"),
            verdict_response("overreach", "commitment has not been met"),
        ])
        report = await judge_answer(
            llm,
            "Acme committed to a five day turnaround. There is a clear "
            "contractual breach here. [1]",
            [SLA],
        )
        assert report.total == 2
        assert report.supported_rate == 0.5
        assert report.overreach_rate == 0.5

    async def test_problems_lists_overreach_and_unsupported(self):
        llm = FakeLLM(responses=[
            verdict_response("supported", "5 business day turnaround"),
            verdict_response("overreach", "commitment has not been met"),
        ])
        report = await judge_answer(
            llm,
            "Acme committed to a five day turnaround. There is a clear "
            "contractual breach here. [1]",
            [SLA],
        )
        assert len(report.problems()) == 1


class TestCalibration:
    def test_calibration_set_covers_every_verdict(self):
        expected = {c.expected for c in CASES}
        assert Verdict.SUPPORTED in expected
        assert Verdict.OVERREACH in expected
        assert Verdict.UNSUPPORTED in expected

    def test_every_verdict_has_enough_cases_to_measure(self):
        """A bucket with two cases gives 0%, 50% or 100% — useless resolution.

        Overreach dropped to three after the first calibration relabelled three
        cases as unsupported, so more true overreach examples were added.
        """
        counts = {}
        for case in CASES:
            counts[case.expected] = counts.get(case.expected, 0) + 1
        for verdict in (Verdict.SUPPORTED, Verdict.OVERREACH, Verdict.UNSUPPORTED):
            assert counts.get(verdict, 0) >= 4, f"{verdict.value}: {counts.get(verdict, 0)}"

    async def test_a_lenient_judge_scores_badly_on_calibration(self):
        """The whole point: a judge that always says 'supported' must not look
        good. Otherwise the groundedness number measures leniency."""
        llm = FakeLLM(
            responses=[verdict_response("supported", "") for _ in CASES]
        )
        report = await calibrate(llm, CASES)
        assert report.accuracy < 0.5
        assert report.per_expected()["overreach"] == 0.0


class TestCalibrationMetrics:
    """The first live calibration returned 79% exact agreement with every
    disagreement being overreach vs unsupported — never a bad claim called good.
    Reporting only exact accuracy made a working judge look broken."""

    async def test_label_confusion_does_not_count_as_a_detection_failure(self):
        from app.agent.judge import CalibrationCase, calibrate

        cases = [
            CalibrationCase("A claim about motive here.", "Passage text.", Verdict.OVERREACH),
        ]
        llm = FakeLLM(responses=[verdict_response("unsupported")])
        report = await calibrate(llm, cases)

        assert report.accuracy == 0.0          # labels disagree
        assert report.detection_accuracy == 1.0  # claim still flagged
        assert report.dangerous_misses == []

    async def test_calling_a_bad_claim_supported_is_a_dangerous_miss(self):
        from app.agent.judge import CalibrationCase, calibrate

        cases = [
            CalibrationCase("Acme is in breach.", "A commitment was not met.", Verdict.OVERREACH),
        ]
        llm = FakeLLM(responses=[verdict_response("supported", "A commitment was not met")])
        report = await calibrate(llm, cases)

        assert report.detection_accuracy == 0.0
        assert len(report.dangerous_misses) == 1
        assert "Fix the rubric" in report.summary()

    async def test_lenient_judge_is_caught_by_detection_accuracy(self):
        from app.agent.judge import calibrate

        llm = FakeLLM(responses=[verdict_response("supported") for _ in CASES])
        report = await calibrate(llm, CASES)
        assert report.detection_accuracy < 0.4
        assert len(report.dangerous_misses) >= 8

    def test_rubric_distinguishes_silence_from_understatement(self):
        from app.agent.judge import JUDGE_PROMPT

        assert "silence about a subject is not weak support" in JUDGE_PROMPT
        assert "If you must ADD a subject" in JUDGE_PROMPT


class TestQuoteVerification:
    """The verbatim guard fired on an ellipsis during a live run, silently
    downgrading two valid claims to UNJUDGED. Quoting across a gap is normal;
    inventing content is not."""

    def test_elided_quote_is_accepted(self):
        from app.agent.judge import _quote_is_present

        source = "Compliance export is proceeding. It has no dependencies. It runs in parallel."
        assert _quote_is_present("Compliance export is proceeding... It runs in parallel", source)

    def test_unicode_ellipsis_is_accepted(self):
        from app.agent.judge import _quote_is_present

        source = "Compliance export is proceeding. It runs in parallel."
        assert _quote_is_present("Compliance export is proceeding… runs in parallel", source)

    def test_invented_content_still_fails(self):
        from app.agent.judge import _quote_is_present

        source = "Compliance export is proceeding. It runs in parallel."
        assert not _quote_is_present("Acme confirmed the schema on Thursday", source)

    def test_out_of_order_fragments_fail(self):
        """Fragments must appear in order, or the guard would accept a quote
        assembled from scattered words."""
        from app.agent.judge import _quote_is_present

        source = "Compliance export is proceeding. It runs in parallel."
        assert not _quote_is_present("runs in parallel... export is proceeding", source)

    def test_one_fabricated_fragment_fails_the_whole_quote(self):
        from app.agent.judge import _quote_is_present

        source = "Compliance export is proceeding. It runs in parallel."
        assert not _quote_is_present("Compliance export is proceeding... approved by legal", source)


class TestUnjudgedIsVisible:
    """The summary omitted unjudged, so printed rates summed to 98% and part of
    the measurement was invisible."""

    async def test_summary_reports_unjudged(self):
        llm = FakeLLM(responses=[text_response("not json at all")])
        report = await judge_answer(
            llm, "A claim that should be judged here. [1]", [SLA],
            detect_miscitation=False,
        )
        assert report.unjudged_rate == 1.0
        assert "unjudged" in report.summary()

    async def test_rates_sum_to_one(self):
        llm = FakeLLM(responses=[
            verdict_response("supported", "5 business day turnaround"),
            text_response("garbage"),
        ])
        report = await judge_answer(
            llm,
            "Acme committed to a five day turnaround. Something else entirely "
            "here now. [1]",
            [SLA],
            detect_miscitation=False,
        )
        total = (
            report.supported_rate + report.miscited_rate + report.overreach_rate
            + report.unsupported_rate + report.unjudged_rate
        )
        assert abs(total - 1.0) < 1e-9


class TestConcurrentJudging:
    """Sequential judging was 28s of a 35s live trace. Claims are independent,
    so there was no reason for call N to wait on call N-1."""

    async def test_claims_are_judged_concurrently(self):
        import asyncio
        import time

        class SlowLLM:
            def __init__(self):
                self.calls = 0

            async def complete(self, **kwargs):
                self.calls += 1
                await asyncio.sleep(0.05)
                return verdict_response("supported", "5 business day turnaround")

        answer = " ".join(
            f"Acme committed to a five day turnaround number {i}. [1]"
            for i in range(6)
        )
        llm = SlowLLM()
        start = time.perf_counter()
        report = await judge_answer(llm, answer, [SLA], detect_miscitation=False)
        elapsed = time.perf_counter() - start

        assert report.total == 6
        assert llm.calls == 6
        # Sequential would be ~0.30s; concurrent should be well under half that.
        assert elapsed < 0.15, f"took {elapsed:.3f}s — judging appears sequential"

    async def test_concurrency_is_bounded(self):
        """A 40-claim answer must not open 40 sockets and collect 429s."""
        import asyncio

        peak = 0
        active = 0

        class CountingLLM:
            async def complete(self, **kwargs):
                nonlocal peak, active
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                return verdict_response("supported", "5 business day turnaround")

        answer = " ".join(
            f"Acme committed to a five day turnaround number {i}. [1]"
            for i in range(20)
        )
        await judge_answer(
            CountingLLM(), answer, [SLA], detect_miscitation=False, concurrency=4
        )
        assert peak <= 4

    async def test_report_order_matches_the_answer(self):
        """A reader compares verdicts against the prose; a shuffled list makes
        that impossible."""
        responses = [
            verdict_response("supported", "5 business day turnaround"),
            verdict_response("overreach", "commitment has not been met"),
            verdict_response("supported", "5 business day turnaround"),
        ]
        llm = FakeLLM(responses=list(responses))
        report = await judge_answer(
            llm,
            "Acme committed to a five day turnaround. There is a clear "
            "contractual breach here. The commitment has not been met at all. [1]",
            [SLA],
            detect_miscitation=False,
            concurrency=1,
        )
        assert [j.verdict for j in report.judged] == [
            Verdict.SUPPORTED,
            Verdict.OVERREACH,
            Verdict.SUPPORTED,
        ]

    async def test_empty_answer_makes_no_calls(self):
        llm = FakeLLM(responses=[])
        report = await judge_answer(llm, "No citations in this text at all.", [SLA])
        assert report.total == 0
        assert llm.calls == []


class TestQuoteNormalisation:
    """Six of 100 claims were downgraded to UNJUDGED over markdown punctuation
    and a literal backslash-n. None were fabrications; the effect was to push
    `supported` down and make it look like a judge quality problem."""

    def test_markdown_table_pipes_are_layout_not_content(self):
        from app.agent.judge import _quote_is_present

        source = "| Item | Target |\n|---|---|\n| Partner auth | Not committed |"
        assert _quote_is_present("| Partner auth | Not committed |", source)

    def test_literal_backslash_n_is_tolerated(self):
        from app.agent.judge import _quote_is_present

        source = "Controls not yet satisfied\nCertificates have not been issued"
        assert _quote_is_present(
            "Controls not yet satisfied\\nCertificates have not been issued", source
        )

    def test_words_must_still_appear_in_order(self):
        """Stripping layout must not become stripping content."""
        from app.agent.judge import _quote_is_present

        source = "Controls not yet satisfied. Certificates have not been issued."
        assert not _quote_is_present(
            "Certificates have not been issued. Controls not yet satisfied", source
        )

    def test_a_typo_in_the_quote_still_fails(self):
        from app.agent.judge import _quote_is_present

        assert not _quote_is_present(
            "Controct that must be satisfied", "Controls that must be satisfied"
        )

    def test_fabrication_still_fails(self):
        from app.agent.judge import _quote_is_present

        assert not _quote_is_present(
            "Acme confirmed the schema on Thursday",
            "| Item | Status |\nThe request is unanswered.",
        )
