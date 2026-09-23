"""Regression checks against a recorded baseline.

Five days of work is now protected by numbers that only exist if someone
remembers to run three scripts by hand. This turns them into one command that
exits non-zero.

## Noise is a first-class input, not an afterthought

Retrieval is deterministic — any change is real. Groundedness is not: the same
setup produced 85%, 89% and 94% on consecutive runs. A suite that fails on a
4-point drop in a metric that swings 8 points will cry wolf until someone adds
`--no-verify` to their commit hook, and then it protects nothing.

So every metric carries its own tolerance, derived from its observed spread, and
the failure message says which it is. A check nobody trusts is worse than no
check, because it also consumes the attention a real failure would need.

## What is checked

- retrieval recall and per-category recall (deterministic, tight tolerance)
- claim-level groundedness (sampled, wide tolerance, needs --live)
- refusal behaviour on known-unanswerable questions (must stay at 100%)
- cost and latency per question (budget ceilings, not equality)

Refusal has no tolerance. A system that starts answering a question it cannot
answer has not drifted; it has broken.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Threshold:
    """One metric, its expected value, and how much movement is meaningful.

    `tolerance` is the observed run-to-run spread. `direction` says which way
    counts as a regression — recall falling is bad, overreach falling is good,
    and a suite that does not know the difference fails on improvements.
    """

    name: str
    baseline: float
    tolerance: float
    direction: str = "higher_is_better"  # or "lower_is_better"
    note: str = ""
    unit: str = "ratio"  # "ratio" | "usd" | "seconds"
    # Runs the baseline was measured over. Sampled metrics recorded from many
    # runs give a narrower estimate than a single run produces, so comparing
    # the two without adjustment fails healthy runs.
    baseline_runs: int = 1
    # Deterministic metrics need no adjustment however few runs are used.
    sampled: bool = False

    def format(self, value: float) -> str:
        """Render in the metric's own units.

        Cost was being printed as "1.6%" for $0.016 because every metric shared
        a percentage formatter. A number in the wrong unit is not a cosmetic
        problem — it is unreadable to the person deciding whether to act on it.
        """
        if self.unit == "usd":
            return f"${value:.4f}"
        if self.unit == "seconds":
            return f"{value:.1f}s"
        return f"{value:.1%}"

    def effective_tolerance(self, runs: int) -> float:
        """Tolerance adjusted for how many runs the check used.

        Standard error scales as 1/sqrt(n). A baseline pooled over 3 runs
        compared against a 1-run check needs its tolerance widened by
        sqrt(3/1) ≈ 1.73, or a perfectly normal run fails.

        Observed concretely: supported claims baselined at 95.9% over 3 runs
        with an 8-point tolerance derived from single-run spread. That gives a
        pass band starting at 87.9%, while single runs had legitimately
        produced 85%.

        Only sampled metrics are adjusted. Retrieval is deterministic — one run
        is as good as a hundred.
        """
        if not self.sampled or runs >= self.baseline_runs or runs < 1:
            return self.tolerance
        return self.tolerance * math.sqrt(self.baseline_runs / runs)

    def check(self, observed: float, runs: int = 1) -> tuple[bool, str]:
        tolerance = self.effective_tolerance(runs)
        delta = observed - self.baseline

        if self.direction == "higher_is_better":
            regressed = delta < -tolerance
        else:
            regressed = delta > tolerance

        sign = "+" if delta >= 0 else "-"

        widened = (
            f" [tolerance widened from ±{self.format(self.tolerance)} — "
            f"baseline used {self.baseline_runs} runs, this check used {runs}]"
            if tolerance > self.tolerance
            else ""
        )

        if not regressed:
            if abs(delta) > tolerance:
                return True, (
                    f"{self.name}: {self.format(observed)} vs "
                    f"{self.format(self.baseline)} "
                    f"({sign}{self.format(abs(delta))}) — improvement beyond "
                    "tolerance, consider re-baselining"
                )
            return True, (
                f"{self.name}: {self.format(observed)} "
                f"(within ±{self.format(tolerance)}){widened}"
            )

        return False, (
            f"{self.name}: {self.format(observed)} vs baseline "
            f"{self.format(self.baseline)} ({sign}{self.format(abs(delta))}), "
            f"tolerance ±{self.format(tolerance)}{widened}"
            + (f" — {self.note}" if self.note else "")
        )


BASELINE_PATH = Path("docs/regression_baseline.json")

# Defaults, used when no baseline file exists. The file is the source of truth
# once written — these exist so a fresh clone can run the suite and record its
# own numbers rather than failing on a missing file.
#
# Tolerances come from measurement, not from taste:
#   retrieval    deterministic — any movement is real, so 0 tolerance
#   groundedness observed 85/89/94% across three runs → 8 points of swing
#   refusal      a correctness property, not a sampled metric → 0 tolerance
#   cost         a ceiling, not an equality check
DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    "recall_at_5": Threshold(
        "recall@5", 0.625, 0.0,
        note="retrieval is deterministic; any drop is a real regression",
    ),
    "cross_source_recall": Threshold("cross-source recall", 1.0, 0.0),
    "identifier_recall": Threshold(
        "identifier recall", 1.0, 0.0,
        note="BM25 half of the hybrid; a drop usually means the tokenizer",
    ),
    "groundedness_supported": Threshold(
        "supported claims", 0.89, 0.08, sampled=True,
        note="sampled metric; tolerance is the observed run-to-run swing",
    ),
    "groundedness_overreach": Threshold(
        "overreach", 0.07, 0.05, direction="lower_is_better", sampled=True,
    ),
    "groundedness_unsupported": Threshold(
        "unsupported", 0.02, 0.03, direction="lower_is_better", sampled=True,
        note="the dangerous category — claims in no retrieved passage",
    ),
    "refusal_rate": Threshold(
        "refusal on unanswerable", 1.0, 0.0,
        note="not drift — a system answering what it cannot answer is broken",
    ),
    # Two costs, because one number conflated them and reported a redefinition
    # as a regression. Answering is what a client pays per question; evaluating
    # is what CI pays per commit. They move independently and have different
    # budgets.
    "cost_per_answer": Threshold(
        "cost per answer", 0.008, 0.004, direction="lower_is_better",
        unit="usd", sampled=True, note="what the client pays per question",
    ),
    "cost_per_eval": Threshold(
        "cost per evaluated question", 0.030, 0.015,
        direction="lower_is_better", unit="usd", sampled=True,
        note="answering plus claim-level judging; what CI pays per commit",
    ),
}


def load_thresholds(path: str | Path = BASELINE_PATH) -> dict[str, Threshold]:
    """Read thresholds from the baseline file, falling back to the defaults.

    Written after finding `docs/baseline.json` claiming recall 56.25% while
    `regression.py` hardcoded 62.5% — two sources of truth, already
    disagreeing, inside the machinery built to catch exactly that.

    One file, loaded at runtime. Updating a baseline is now an explicit command
    that rewrites it, not an edit buried in a constant somebody forgets exists.
    """
    file = Path(path)
    if not file.exists():
        return dict(DEFAULT_THRESHOLDS)

    data = json.loads(file.read_text())
    thresholds: dict[str, Threshold] = {}
    for key, entry in data.get("metrics", {}).items():
        default = DEFAULT_THRESHOLDS.get(key)
        thresholds[key] = Threshold(
            name=entry.get("name", default.name if default else key),
            baseline=float(entry["value"]),
            tolerance=float(
                entry.get("tolerance", default.tolerance if default else 0.0)
            ),
            direction=entry.get(
                "direction", default.direction if default else "higher_is_better"
            ),
            note=entry.get("note", default.note if default else ""),
            unit=entry.get("unit", default.unit if default else "ratio"),
            baseline_runs=int(entry.get("baseline_runs", 1)),
            sampled=bool(
                entry.get("sampled", default.sampled if default else False)
            ),
        )

    # Defaults for anything the file does not mention, so adding a metric in
    # code does not silently go unchecked until someone re-baselines.
    for key, default in DEFAULT_THRESHOLDS.items():
        thresholds.setdefault(key, default)
    return thresholds


@dataclass
class RegressionResult:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    observed: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, Threshold] = field(default_factory=load_thresholds)
    runs: int = 1

    def add(self, key: str, observed: float) -> None:
        self.observed[key] = observed
        threshold = self.thresholds.get(key)
        if threshold is None:
            self.skipped.append(f"{key}: no threshold defined")
            return
        passed, message = threshold.check(observed, runs=self.runs)
        self.checks.append((key, passed, message))

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]

    @property
    def ok(self) -> bool:
        return not self.failed

    def report(self) -> str:
        lines = []
        for _, passed, message in self.checks:
            lines.append(f"  {'PASS' if passed else 'FAIL'}  {message}")
        for note in self.skipped:
            lines.append(f"  SKIP  {note}")

        if self.ok:
            lines.append(f"\n  {len(self.checks)} checks passed.")
        else:
            lines.append(f"\n  {len(self.failed)} REGRESSION(S):")
            for key, _, message in self.failed:
                lines.append(f"    {message}")
            lines.append(
                "\n  If a failure is an intended change, update the baseline"
                "\n  deliberately — never widen a tolerance to make a failure go"
                "\n  away. A tolerance that absorbs real regressions is how a"
                "\n  suite stops protecting anything."
            )
        return "\n".join(lines)


class InstrumentChanged(RuntimeError):
    """Raised when the baseline was recorded with a different judge model.

    Groundedness is measured BY a model, so the judge is part of the instrument.
    Switching it moved supported claims 93.8% -> 83% with no change to the
    system under test — Haiku simply flags more claims as overreach.

    Comparing across judges reports an instrument change as a regression, which
    is exactly the kind of false alarm that gets a suite ignored.
    """


class RebaselineRefused(RuntimeError):
    """Raised when re-baselining would record a failing value.

    Added after this exact bug: a run reported "1 REGRESSION(S)" and then
    "Baseline updated (8 metrics)" in the same output, permanently silencing the
    failure it had just found. A guard that can be bypassed by the same command
    that detects the problem is not a guard.
    """


def save_baseline(
    observed: dict[str, float],
    path: str | Path = BASELINE_PATH,
    *,
    reason: str = "",
    thresholds: dict[str, Threshold] | None = None,
    failing: list[str] | None = None,
    force: bool = False,
    runs: int = 1,
    judge_model: str = "",
) -> None:
    """Record measured values as the new baseline.

    Tolerance and direction are carried over from the existing thresholds rather
    than reset, because those were derived from measured spread and should not
    change just because the value did.

    `reason` is required in practice, not by the signature: a baseline updated
    without a recorded reason is indistinguishable from one updated to silence a
    failure.
    """
    if failing and not force:
        raise RebaselineRefused(
            "Refusing to re-baseline while checks are failing: "
            f"{', '.join(failing)}.\n"
            "  Fix the regression, or pass --force if the new value is the "
            "intended one\n"
            "  and say so in the reason. Recording a failing value is how a "
            "suite stops\n  protecting anything."
        )

    current = thresholds or load_thresholds(path)
    metrics = {}
    for key, value in observed.items():
        threshold = current.get(key) or DEFAULT_THRESHOLDS.get(key)
        metrics[key] = {
            "value": round(value, 6),
            "name": threshold.name if threshold else key,
            "tolerance": threshold.tolerance if threshold else 0.0,
            "direction": threshold.direction if threshold else "higher_is_better",
            "note": threshold.note if threshold else "",
            "unit": threshold.unit if threshold else "ratio",
            "sampled": threshold.sampled if threshold else False,
            "baseline_runs": runs,
        }

    Path(path).write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                # Which judge produced these numbers. Without it, a judge swap
                # reads as a system regression.
                "judge_model": judge_model,
                "metrics": metrics,
                "_note": (
                    "Source of truth for regression checks. Re-baseline only "
                    "for intended changes and say why in `reason`. A baseline "
                    "updated to silence a failure measures nothing. Never widen "
                    "a tolerance to make a failure go away — tolerances come "
                    "from observed run-to-run spread."
                ),
            },
            indent=2,
        )
    )


def load_baseline(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def check_instrument(
    path: str | Path, judge_model: str, *, strict: bool = True
) -> str:
    """Compare the current judge against the one the baseline was recorded with.

    Returns a warning string, or raises when strict and they differ.
    """
    file = Path(path)
    if not file.exists():
        return ""
    recorded = json.loads(file.read_text()).get("judge_model")
    if not recorded or recorded == judge_model:
        return ""

    message = (
        f"Baseline was recorded with judge {recorded!r}; this run used "
        f"{judge_model!r}.\n"
        "  Groundedness is measured BY a model, so the judge is part of the "
        "instrument\n"
        "  and the numbers are not comparable. Re-baseline deliberately with "
        "--force,\n"
        "  recording the judge change as the reason."
    )
    if strict:
        raise InstrumentChanged(message)
    return message
