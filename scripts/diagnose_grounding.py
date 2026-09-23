"""Show why an answer was rejected, sentence by sentence.

    python scripts/diagnose_grounding.py --live
    python scripts/diagnose_grounding.py --live -q "what is at risk right now?"

Written because a live run refused 5 out of 5 scenarios. A refusal rate that
high means one of two things, and they need opposite fixes:

  - the model genuinely is not citing (fix the prompt)
  - the checker is rejecting sentences that are fine (fix the checker)

Guessing which costs an afternoon. This prints the raw answer alongside the
per-sentence verdict, so the cause is visible rather than inferred.

The lesson is the same one as the score floor on Day 3: measure the thing before
changing it. A 100% refusal rate looks like maximum safety and is actually a
system nobody can use — a false negative wearing a safety jacket.
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
    _split_sentences,
    build_context,
    check_grounding,
)
from app.rag.chunking import chunk_page  # noqa: E402
from app.rag.decomposition import HeuristicDecomposer, multi_query_search  # noqa: E402
from app.rag.embedders import FakeEmbedder  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from scripts.seed_content import load_offline_corpus  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is the current release plan?",
    "What work is holding up the release?",
    "Which controls are still outstanding before go-live?",
    "What is the status of the Acme work and are there contractual risks?",
    "Who approved the Acme security exception?",
    "What is our parental leave policy?",
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("-q", "--question", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Needs ANTHROPIC_API_KEY in .env")
        return 1

    from app.agent.llm import AnthropicLLM

    llm = AnthropicLLM()
    chunks = [c for page in load_offline_corpus() for c in chunk_page(page)]
    retriever = HybridRetriever(chunks, FakeEmbedder(), candidate_pool=20)
    decomposer = HeuristicDecomposer()

    questions = args.question or DEFAULT_QUESTIONS
    stats = {"answered": 0, "model_declined": 0, "rejected": 0}

    for question in questions:
        print("\n" + "=" * 74)
        print(question)
        print("=" * 74)

        hits, _ = await multi_query_search(
            retriever, decomposer, question, top_k=args.top_k
        )
        passages = [Passage.from_hit(h) for h in hits]

        print(f"\n  retrieved {len(passages)} passages:")
        for index, passage in enumerate(passages, start=1):
            print(f"    [{index}] {passage.label}")

        if not passages:
            print("\n  (nothing retrieved)")
            continue

        response = await llm.complete(
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
        raw = response.text.strip()

        print("\n  RAW MODEL OUTPUT")
        print("  " + "-" * 70)
        for line in raw.splitlines():
            print(f"  {line}")
        print("  " + "-" * 70)

        if REFUSAL_MARKER in raw:
            print("\n  -> model declined")
            stats["model_declined"] += 1
            continue

        report = check_grounding(raw, passage_count=len(passages))

        # Display the same unit the checker enforces on. Showing sentences
        # while judging paragraphs made the first run's verdicts unreadable —
        # a diagnostic that reports at a different granularity than the thing
        # it diagnoses is worse than none.
        from app.agent.answering import _paragraphs

        print("\n  PER-PARAGRAPH VERDICT (the unit actually enforced)")
        for unit in _paragraphs(raw):
            failing = unit in report.uncited_sentences
            mark = "REJECT" if failing else "  ok  "
            print(f"    [{mark}] {unit[:88].replace(chr(10), ' ')}")

        if report.out_of_range:
            print(f"\n  invented passage numbers: {sorted(report.out_of_range)}")

        if report.is_grounded:
            print("\n  -> ANSWER ACCEPTED")
            stats["answered"] += 1
        else:
            print(
                f"\n  -> REJECTED: {len(report.uncited_sentences)} uncited, "
                f"{len(report.out_of_range)} out of range"
            )
            stats["rejected"] += 1

    total = sum(stats.values())
    print("\n" + "=" * 74)
    print(
        f"answered {stats['answered']}/{total}   "
        f"model declined {stats['model_declined']}/{total}   "
        f"rejected by checker {stats['rejected']}/{total}"
    )
    print(
        "\nIf `rejected by checker` dominates, the checker is too strict and the\n"
        "per-sentence verdicts above show which sentence shapes it is catching.\n"
        "If `model declined` dominates on answerable questions, the prompt or the\n"
        "retrieval is at fault, not the checker."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
