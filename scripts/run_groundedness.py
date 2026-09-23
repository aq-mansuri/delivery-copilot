"""Calibrate the judge, then measure claim-level groundedness on live answers.

    python scripts/run_groundedness.py --live --calibrate-only
    python scripts/run_groundedness.py --live
    python scripts/run_groundedness.py --live --runs 3

Calibration runs first by default, because a groundedness score from an
unvalidated judge is a number with no meaning attached.

Use --runs for anything you intend to quote. The judge is not deterministic —
the same 18 calibration cases scored 89% and then 100% on consecutive runs — so
a single figure is a sample. An improvement smaller than the observed spread is
indistinguishable from noise.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.answering import (  # noqa: E402
    REFUSAL_MARKER,
    SYSTEM_PROMPT,
    Passage,
    build_context,
)
from app.agent.judge import Verdict, calibrate, judge_answer  # noqa: E402
from app.rag.chunking import chunk_page  # noqa: E402
from app.rag.decomposition import HeuristicDecomposer, multi_query_search  # noqa: E402
from app.rag.embedders import FakeEmbedder  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from scripts.seed_content import load_offline_corpus  # noqa: E402
from tests.fixtures.calibration_cases import CASES, HELD_OUT  # noqa: E402

QUESTIONS = [
    "What is the current release plan?",
    "What work is holding up the release?",
    "Which controls are still outstanding before go-live?",
    "What is the status of the Acme work and are there contractual risks?",
]


async def measure_once(answerer, judge, retriever, decomposer, *, verbose: bool):
    """One full pass over the question set. Returns per-verdict counts."""
    totals = {v: 0 for v in Verdict}

    for question in QUESTIONS:
        hits, _ = await multi_query_search(retriever, decomposer, question, top_k=5)
        passages = [Passage.from_hit(h) for h in hits]

        response = await answerer.complete(
            system=SYSTEM_PROMPT.format(refusal=REFUSAL_MARKER),
            messages=[
                {
                    "role": "user",
                    "content": f"Context passages:\n\n{build_context(passages)}"
                    f"\n\nQuestion: {question}",
                }
            ],
            max_tokens=1000,
        )
        text = response.text.strip()
        if REFUSAL_MARKER in text:
            if verbose:
                print(f"\n  {question}\n    (declined)")
            continue

        report = await judge_answer(judge, text, passages)
        for judged in report.judged:
            totals[judged.verdict] += 1

        if verbose:
            print(f"\n  {question}")
            print(f"    {report.summary()}")
            for problem in report.problems():
                print(
                    f"\n    [{problem.verdict.value.upper()}] "
                    f"{problem.claim.text[:88]}"
                )
                if problem.reason:
                    print(f"      judge: {problem.reason[:88]}")

    return totals


def rates(totals) -> dict[str, float]:
    total = sum(totals.values())
    if not total:
        return {}
    return {v.value: totals[v] / total for v in Verdict}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model used for judging. Defaults to the cheaper judge model; "
             "pass the answering model to check whether same-model grading "
             "inflates the score.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat and report the spread. One run is a sample, not a measurement.",
    )
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Needs ANTHROPIC_API_KEY in .env")
        return 1

    from app.agent.llm import AnthropicLLM

    from app.agent.llm import DEFAULT_JUDGE_MODEL

    judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
    print(f"  judge model: {judge_model}")
    answerer = AnthropicLLM()
    # A different model, not just a separate instance. Same-model grading
    # correlates errors, and the cheaper model roughly halves cost_per_eval.
    # Always re-run calibration after changing this.
    judge = AnthropicLLM(model=judge_model)

    if not args.skip_calibration:
        print("=" * 74)
        print("CALIBRATING THE JUDGE")
        print("=" * 74)
        report = await calibrate(judge, CASES)
        print("\n" + report.summary())

        for case, got in report.results:
            if case.expected is not got:
                print(f"\n  expected {case.expected.value} -> got {got.value}")
                print(f"    {case.claim}")

        if report.detection_accuracy < 0.9:
            print(
                "\n  Detection below 90%. Fix the rubric before quoting any"
                "\n  groundedness figure."
            )
        print("\n  held-out set (written without checking the judge first):")
        held = await calibrate(judge, HELD_OUT)
        print("  " + held.summary().replace("\n", "\n  "))
        if held.detection_accuracy < report.detection_accuracy - 0.1:
            direction = (
                "over-flagging (supported claims sent to review)"
                if not held.dangerous_misses
                else f"MISSING {len(held.dangerous_misses)} unsupported claim(s)"
            )
            print(
                f"\n  Held-out detection ({held.detection_accuracy:.0%}) is below "
                f"the fitted set ({report.detection_accuracy:.0%})."
                f"\n  The gap is co-adaptation: the fitted set was relabelled "
                "twice to agree"
                "\n  with the judge, so its score partly measures that."
                f"\n\n  Failure mode: {direction}."
            )
            if not held.dangerous_misses:
                print(
                    "  That is the safe direction — a supported claim flagged "
                    "costs a review;"
                    "\n  an unsupported claim missed reaches a board. Quote the "
                    "held-out"
                    "\n  number, not the fitted one."
                )

        if args.calibrate_only:
            return 0

    print("\n" + "=" * 74)
    print("CLAIM-LEVEL GROUNDEDNESS ON LIVE ANSWERS")
    print("=" * 74)

    chunks = [c for page in load_offline_corpus() for c in chunk_page(page)]
    retriever = HybridRetriever(chunks, FakeEmbedder(), candidate_pool=20)
    decomposer = HeuristicDecomposer()

    combined = {v: 0 for v in Verdict}
    per_run: list[dict[str, float]] = []

    for index in range(args.runs):
        if args.runs > 1:
            print(f"\n--- run {index + 1} of {args.runs} ---")
        totals = await measure_once(
            answerer, judge, retriever, decomposer, verbose=(args.runs == 1)
        )
        for verdict, count in totals.items():
            combined[verdict] += count
        run_rates = rates(totals)
        if run_rates:
            per_run.append(run_rates)
            if args.runs > 1:
                print(
                    f"    supported {run_rates['supported']:.0%}   "
                    f"miscited {run_rates['miscited']:.0%}   "
                    f"overreach {run_rates['overreach']:.0%}   "
                    f"unsupported {run_rates['unsupported']:.0%}   "
                    f"unjudged {run_rates['unjudged']:.0%}"
                )

    total_claims = sum(combined.values())
    print("\n" + "=" * 74)
    if total_claims:
        overall = rates(combined)
        print(
            f"  {total_claims} claims across {args.runs} run(s)   "
            f"supported {overall['supported']:.0%}   "
            f"miscited {overall['miscited']:.0%}   "
            f"overreach {overall['overreach']:.0%}   "
            f"unsupported {overall['unsupported']:.0%}   "
            f"unjudged {overall['unjudged']:.0%}"
        )
        traceable = overall["supported"] + overall["miscited"]
        print(f"\n  traceable to the corpus: {traceable:.0%}")

    if len(per_run) > 1:
        print("\n  spread across runs:")
        for metric in ("supported", "overreach", "unsupported"):
            values = [r[metric] for r in per_run]
            print(
                f"    {metric:<12} {min(values):.0%} - {max(values):.0%}   "
                f"(swing {max(values) - min(values):.0%})"
            )
        print(
            "\n    An improvement smaller than the swing is indistinguishable"
            "\n    from noise at this sample size."
        )

    print(
        "\n  MISCITED     true and in the corpus, wrong passage number"
        "\n               -> citation discipline"
        "\n  OVERREACH    stronger than the source ('breach' for 'not met')"
        "\n               -> instruct against strengthening and concluding"
        "\n  UNSUPPORTED  in no retrieved passage. The dangerous one."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))