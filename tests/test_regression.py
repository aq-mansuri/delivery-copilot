"""Tests for the regression suite's own logic.

A suite that cries wolf gets bypassed, and then it protects nothing. These pin
the behaviour that decides whether it is trusted.
"""

from __future__ import annotations

import json

import pytest

from app.core.regression import (
    DEFAULT_THRESHOLDS,
    RegressionResult,
    Threshold,
    load_baseline,
    load_thresholds,
    save_baseline,
)


class TestTolerance:
    def test_drop_within_tolerance_passes(self):
        """Groundedness swings 8 points run to run. Failing on a 4-point drop
        means someone adds --no-verify and the suite protects nothing."""
        passed, _ = Threshold("supported", 0.89, 0.08).check(0.85)
        assert passed

    def test_drop_beyond_tolerance_fails(self):
        passed, message = Threshold("supported", 0.89, 0.08).check(0.75)
        assert not passed
        assert "-14" in message.replace(".0%", "")

    def test_deterministic_metrics_have_no_tolerance(self):
        """Retrieval is deterministic; any movement is a real change."""
        passed, _ = Threshold("recall", 0.625, 0.0).check(0.624)
        assert not passed


class TestDirection:
    def test_improvement_is_not_a_regression(self):
        """A suite that fails on improvements gets ignored like one that fails
        on noise.

        0.01 rather than 0.02: 0.07 - 0.02 lands exactly ON the tolerance, and
        "within tolerance" is inclusive. Picking a boundary value made this fail
        for a reason that had nothing to do with the behaviour under test.
        """
        passed, message = Threshold(
            "overreach", 0.07, 0.05, direction="lower_is_better"
        ).check(0.01)
        assert passed
        assert "re-baselining" in message

    def test_exact_boundary_is_within_tolerance(self):
        """Pinning the inclusive boundary, since a test tripped over it."""
        passed, message = Threshold("recall", 0.60, 0.05).check(0.55)
        assert passed
        assert "within" in message

    def test_lower_is_better_fails_on_increase(self):
        passed, _ = Threshold(
            "overreach", 0.07, 0.05, direction="lower_is_better"
        ).check(0.20)
        assert not passed

    def test_higher_is_better_flags_large_improvement(self):
        passed, message = Threshold("recall", 0.625, 0.05).check(0.90)
        assert passed
        assert "re-baselining" in message


class TestRefusalHasNoTolerance:
    def test_any_refusal_failure_is_a_regression(self):
        """Not drift — a system answering what it cannot answer is broken."""
        threshold = DEFAULT_THRESHOLDS["refusal_rate"]
        assert threshold.tolerance == 0.0
        passed, message = threshold.check(0.99)
        assert not passed
        assert "broken" in message


class TestResult:
    def test_exit_status_reflects_failures(self):
        result = RegressionResult()
        result.add("recall_at_5", 0.625)
        assert result.ok
        result.add("recall_at_5", 0.40)
        assert not result.ok
        assert len(result.failed) == 1

    def test_unknown_metric_is_skipped_not_silently_passed(self):
        result = RegressionResult()
        result.add("some_new_metric", 0.5)
        assert result.ok
        assert result.skipped

    def test_report_warns_against_widening_tolerances(self):
        result = RegressionResult()
        result.add("recall_at_5", 0.30)
        report = result.report()
        assert "never widen a tolerance" in report


class TestBaselineIsTheSourceOfTruth:
    """Written after finding docs/baseline.json claiming recall 56.25% while
    regression.py hardcoded 62.5% — two sources of truth, already disagreeing,
    inside the machinery built to catch exactly that."""

    def test_round_trip(self, tmp_path):
        path = tmp_path / "baseline.json"
        save_baseline({"recall_at_5": 0.625}, path, reason="initial")
        loaded = load_baseline(path)
        assert loaded["metrics"]["recall_at_5"]["value"] == 0.625
        assert loaded["reason"] == "initial"

    def test_file_values_override_the_defaults(self, tmp_path):
        path = tmp_path / "baseline.json"
        save_baseline({"recall_at_5": 0.80}, path, reason="improved retrieval")
        assert load_thresholds(path)["recall_at_5"].baseline == 0.80

    def test_tolerance_is_carried_over_not_reset(self, tmp_path):
        """Tolerance came from measured spread; a new value does not change it."""
        path = tmp_path / "baseline.json"
        save_baseline({"groundedness_supported": 0.93}, path, reason="prompt fix")
        loaded = load_thresholds(path)
        assert loaded["groundedness_supported"].baseline == 0.93
        assert loaded["groundedness_supported"].tolerance == 0.08

    def test_metrics_absent_from_the_file_keep_their_defaults(self, tmp_path):
        """Adding a metric in code must not go unchecked until someone
        re-baselines."""
        path = tmp_path / "baseline.json"
        save_baseline({"recall_at_5": 0.70}, path, reason="x")
        loaded = load_thresholds(path)
        assert "refusal_rate" in loaded
        assert loaded["refusal_rate"].tolerance == 0.0

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        assert load_thresholds(tmp_path / "nope.json") == DEFAULT_THRESHOLDS

    def test_file_warns_about_silencing_failures(self, tmp_path):
        path = tmp_path / "baseline.json"
        save_baseline({"recall_at_5": 1.0}, path, reason="x")
        note = json.loads(path.read_text())["_note"]
        assert "silence a failure" in note
        assert "Never widen a tolerance" in note

    def test_result_records_observed_values_for_rebaselining(self):
        result = RegressionResult()
        result.add("recall_at_5", 0.625)
        assert result.observed == {"recall_at_5": 0.625}


class TestThresholdsAreCalibrated:
    def test_sampled_metrics_have_tolerance(self):
        """Tolerances come from measurement, not taste."""
        assert DEFAULT_THRESHOLDS["groundedness_supported"].tolerance >= 0.05

    def test_deterministic_metrics_do_not(self):
        for key in ("recall_at_5", "cross_source_recall", "identifier_recall"):
            assert DEFAULT_THRESHOLDS[key].tolerance == 0.0

    def test_unsupported_is_lower_is_better(self):
        assert DEFAULT_THRESHOLDS["groundedness_unsupported"].direction == "lower_is_better"


class TestUnits:
    """Cost was printed as "1.6%" for $0.016 because every metric shared a
    percentage formatter."""

    def test_currency_metrics_render_as_currency(self):
        t = Threshold("cost", 0.008, 0.004, unit="usd")
        assert "$0.0080" in t.check(0.008)[1]
        assert "%" not in t.check(0.008)[1]

    def test_ratio_metrics_still_render_as_percentages(self):
        assert "62.5%" in Threshold("recall", 0.625, 0.0).check(0.625)[1]

    def test_delta_uses_the_same_unit(self):
        t = Threshold("cost", 0.008, 0.004, direction="lower_is_better", unit="usd")
        message = t.check(0.016)[1]
        assert "+$0.0080" in message

    def test_unit_survives_a_baseline_round_trip(self, tmp_path):
        path = tmp_path / "b.json"
        save_baseline({"cost_per_answer": 0.009}, path, reason="x")
        assert load_thresholds(path)["cost_per_answer"].unit == "usd"


class TestRebaselineGuard:
    """A run reported "1 REGRESSION(S)" and "Baseline updated" in the same
    output, permanently silencing the failure it had just found."""

    def test_refuses_while_checks_are_failing(self, tmp_path):
        from app.core.regression import RebaselineRefused

        with pytest.raises(RebaselineRefused) as exc:
            save_baseline(
                {"recall_at_5": 0.30},
                tmp_path / "b.json",
                reason="oops",
                failing=["recall_at_5"],
            )
        assert "stops\n  protecting anything" in str(exc.value)

    def test_nothing_is_written_when_refused(self, tmp_path):
        from app.core.regression import RebaselineRefused

        path = tmp_path / "b.json"
        with pytest.raises(RebaselineRefused):
            save_baseline({"x": 0.1}, path, reason="r", failing=["x"])
        assert not path.exists()

    def test_force_allows_an_intended_change(self, tmp_path):
        path = tmp_path / "b.json"
        save_baseline(
            {"recall_at_5": 0.80},
            path,
            reason="new chunker, verified manually",
            failing=["recall_at_5"],
            force=True,
        )
        assert load_thresholds(path)["recall_at_5"].baseline == 0.80

    def test_passing_run_needs_no_force(self, tmp_path):
        path = tmp_path / "b.json"
        save_baseline({"recall_at_5": 0.70}, path, reason="x", failing=[])
        assert path.exists()


class TestCostIsTwoMetrics:
    """One number mixed what the client pays per question with what CI pays per
    commit, and reported the redefinition as a regression."""

    def test_answer_and_eval_costs_are_separate(self):
        assert "cost_per_answer" in DEFAULT_THRESHOLDS
        assert "cost_per_eval" in DEFAULT_THRESHOLDS

    def test_eval_budget_is_larger_than_answer_budget(self):
        assert (
            DEFAULT_THRESHOLDS["cost_per_eval"].baseline
            > DEFAULT_THRESHOLDS["cost_per_answer"].baseline
        )

    def test_both_are_lower_is_better(self):
        for key in ("cost_per_answer", "cost_per_eval"):
            assert DEFAULT_THRESHOLDS[key].direction == "lower_is_better"


class TestSampleSizeAdjustment:
    """A baseline pooled over 3 runs was compared against 1-run checks using a
    tolerance derived from single-run spread. Pass band started at 87.9% while
    single runs had legitimately produced 85%."""

    def test_fewer_runs_widens_the_tolerance(self):
        t = Threshold("supported", 0.959, 0.08, sampled=True, baseline_runs=3)
        assert t.effective_tolerance(3) == pytest.approx(0.08)
        assert t.effective_tolerance(1) == pytest.approx(0.08 * 3 ** 0.5)

    def test_a_normal_single_run_no_longer_fails(self):
        t = Threshold("supported", 0.959, 0.08, sampled=True, baseline_runs=3)
        assert not t.check(0.85, runs=3)[0]
        assert t.check(0.85, runs=1)[0]

    def test_more_runs_does_not_narrow_below_the_recorded_tolerance(self):
        """Tolerance is the observed spread, not a confidence interval to
        shrink at will."""
        t = Threshold("supported", 0.959, 0.08, sampled=True, baseline_runs=3)
        assert t.effective_tolerance(10) == pytest.approx(0.08)

    def test_deterministic_metrics_are_never_adjusted(self):
        t = Threshold("recall", 0.625, 0.0, sampled=False, baseline_runs=3)
        assert t.effective_tolerance(1) == 0.0
        assert not t.check(0.60, runs=1)[0]

    def test_widening_is_stated_in_the_message(self):
        """Silent adjustment is how a suite starts passing things it should
        not, without anyone noticing."""
        t = Threshold("supported", 0.959, 0.08, sampled=True, baseline_runs=3)
        assert "tolerance widened" in t.check(0.90, runs=1)[1]

    def test_baseline_records_its_sample_size(self, tmp_path):
        path = tmp_path / "b.json"
        save_baseline({"groundedness_supported": 0.959}, path, reason="x", runs=3)
        loaded = load_thresholds(path)["groundedness_supported"]
        assert loaded.baseline_runs == 3
        assert loaded.sampled

    def test_result_passes_its_run_count_through(self):
        result = RegressionResult(runs=1)
        result.thresholds["groundedness_supported"] = Threshold(
            "supported", 0.959, 0.08, sampled=True, baseline_runs=3
        )
        result.add("groundedness_supported", 0.85)
        assert result.ok


class TestInstrumentChange:
    """Switching the judge moved supported claims 93.8% -> 83% with no change to
    the system under test. The judge is part of the instrument."""

    def test_same_judge_is_silent(self, tmp_path):
        from app.core.regression import check_instrument

        path = tmp_path / "b.json"
        save_baseline({"x": 1.0}, path, reason="r", judge_model="claude-haiku-4-5")
        assert check_instrument(path, "claude-haiku-4-5") == ""

    def test_different_judge_is_flagged(self, tmp_path):
        from app.core.regression import check_instrument

        path = tmp_path / "b.json"
        save_baseline({"x": 1.0}, path, reason="r", judge_model="claude-sonnet-4-6")
        warning = check_instrument(path, "claude-haiku-4-5", strict=False)
        assert "not comparable" in warning

    def test_strict_mode_raises(self, tmp_path):
        from app.core.regression import InstrumentChanged, check_instrument

        path = tmp_path / "b.json"
        save_baseline({"x": 1.0}, path, reason="r", judge_model="claude-sonnet-4-6")
        with pytest.raises(InstrumentChanged):
            check_instrument(path, "claude-haiku-4-5")

    def test_baseline_records_the_judge(self, tmp_path):
        path = tmp_path / "b.json"
        save_baseline({"x": 1.0}, path, reason="r", judge_model="claude-haiku-4-5")
        assert load_baseline(path)["judge_model"] == "claude-haiku-4-5"

    def test_missing_file_is_not_an_instrument_change(self, tmp_path):
        from app.core.regression import check_instrument

        assert check_instrument(tmp_path / "nope.json", "anything") == ""
