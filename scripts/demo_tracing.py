"""Show where a question's time and money go.

    python scripts/demo_tracing.py           # scripted, offline
    python scripts/demo_tracing.py --live    # real calls, real numbers

The offline run gives the shape; only --live gives real latency. Both give real
token accounting, because the scripted responses carry usage figures taken from
observed runs.
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
from app.agent.judge import judge_answer  # noqa: E402
from app.agent.llm import FakeLLM, LLMResponse  # noqa: E402
from app.core.tracing import Rates, TracedLLM, Trace, compare  # noqa: E402
from app.rag.chunking import chunk_page  # noqa: E402
from app.rag.decomposition import HeuristicDecomposer, multi_query_search  # noqa: E402
from app.rag.embedders import FakeEmbedder  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from scripts.seed_content import load_offline_corpus  # noqa: E402

QUESTION = "What is the status of the Acme work and are there contractual risks?"

# Token counts observed in live runs, so the offline trace reports realistic
# costs even though its latency is fictional. Updated after the restraint prompt
# shortened answers: input fell 4143 -> 1095. Cost dropped as a side effect of a
# quality fix, and a stale fixture would have kept reporting the old number.
SCRIPTED_ANSWER = LLMResponse(
    text=(
        "INS-101 has been untouched for three weeks pending schema confirmation "
        "from Acme [1]. Two follow-ups remain unanswered [1]. The contract "
        "commits Acme to a five business day turnaround [5]."
    ),
    input_tokens=1095,
    output_tokens=306,
)
SCRIPTED_JUDGE = LLMResponse(
    text='{"verdict":"supported","evidence":"untouched for three weeks","reason":"x"}',
    input_tokens=590,
    output_tokens=54,
)


async def run_once(live: bool, judge_claims: bool) -> Trace:
    trace = Trace(question=QUESTION)

    if live:
        from app.agent.llm import AnthropicLLM

        answerer = TracedLLM(AnthropicLLM(), trace, kind="llm")
        judge = TracedLLM(AnthropicLLM(), trace, kind="judge")
    else:
        answerer = TracedLLM(FakeLLM(responses=[SCRIPTED_ANSWER]), trace, kind="llm")
        judge = TracedLLM(
            FakeLLM(responses=[SCRIPTED_JUDGE] * 40), trace, kind="judge"
        )

    with trace.span("answer_question", "node", question=QUESTION):
        with trace.span("retrieve", "retrieval") as span:
            chunks = [c for page in load_offline_corpus() for c in chunk_page(page)]
            retriever = HybridRetriever(chunks, FakeEmbedder(), candidate_pool=20)
            hits, subqueries = await multi_query_search(
                retriever, HeuristicDecomposer(), QUESTION, top_k=5
            )
            span.attributes["chunks_indexed"] = len(chunks)
            span.attributes["subqueries"] = len(subqueries)
            span.attributes["passages"] = len(hits)

        passages = [Passage.from_hit(h) for h in hits]

        response = await answerer.complete(
            system=SYSTEM_PROMPT.format(refusal=REFUSAL_MARKER),
            messages=[
                {
                    "role": "user",
                    "content": f"Context passages:\n\n{build_context(passages)}"
                    f"\n\nQuestion: {QUESTION}",
                }
            ],
            max_tokens=1000,
        )

        if judge_claims:
            with trace.span("judge_answer", "node"):
                report = await judge_answer(
                    judge, response.text, passages, detect_miscitation=False
                )
                trace.spans[-1].attributes["claims"] = report.total

    return trace


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    if args.live:
        import os

        from dotenv import load_dotenv

        load_dotenv()
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("--live needs ANTHROPIC_API_KEY in .env")
            return 1

    print("=" * 74)
    print("ANSWERING ONLY")
    print("=" * 74)
    answer_trace = await run_once(args.live, judge_claims=False)
    print("\n" + answer_trace.summary())

    print("\n" + "=" * 74)
    print("ANSWERING + CLAIM-LEVEL JUDGING")
    print("=" * 74)
    judged_trace = await run_once(args.live, judge_claims=True)
    print("\n" + judged_trace.summary())

    answer_cost = answer_trace.total_cost
    judge_cost = judged_trace.total_cost - answer_cost
    if answer_cost:
        print(
            f"\n  Judging costs {judge_cost / answer_cost:.1f}x the answer it "
            "grades."
        )
    print(
        "  One judge call per claim. That is what decides whether groundedness"
        "\n  runs per-commit or nightly — and it is invisible until measured."
    )

    if args.runs > 1:
        print("\n" + "=" * 74)
        print(f"SPREAD ACROSS {args.runs} RUNS")
        print("=" * 74)
        traces = [await run_once(args.live, judge_claims=False) for _ in range(args.runs)]
        print("\n" + compare(traces))

    judged_trace.save("docs/trace_sample.json")
    print("\n  full trace written to docs/trace_sample.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
