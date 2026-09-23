"""Run adversarial questions against a live model.

    python scripts/check_adversarial.py --live

Every question here is designed to tempt fabrication: the corpus nearly answers
it, or the question presupposes a fact the sources contradict, or it asks for a
specific the sources only gesture at.

The offline suite (tests/test_adversarial.py) proves the CHECKER rejects bad
answers. This proves the SYSTEM does not produce them in the first place, which
is a different and weaker guarantee — but it is the one a client experiences.

A pass here means every adversarial question was declined or rejected. Anything
answered is worth reading in full: the corpus may support it after all, or the
model has found a gap.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.answering import Passage, answer_question  # noqa: E402
from app.rag.chunking import chunk_page  # noqa: E402
from app.rag.decomposition import HeuristicDecomposer, multi_query_search  # noqa: E402
from app.rag.embedders import FakeEmbedder  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from scripts.seed_content import load_offline_corpus  # noqa: E402
from tests.fixtures.adversarial import ADVERSARIAL_QUESTIONS  # noqa: E402


async def probe(llm, retriever, decomposer, case):
    hits, _ = await multi_query_search(
        retriever, decomposer, case.question, top_k=5
    )
    answer = await answer_question(llm, case.question, [Passage.from_hit(h) for h in hits])

    leaked = [
        phrase
        for phrase in case.must_not_contain
        if phrase.lower() in answer.text.lower()
    ]
    return answer, leaked


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", required=True)
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

    print("=" * 74)
    print("ADVERSARIAL PROBES")
    print("=" * 74)

    results = await asyncio.gather(
        *(probe(llm, retriever, decomposer, case) for case in ADVERSARIAL_QUESTIONS)
    )

    held = 0
    for case, (answer, leaked) in zip(ADVERSARIAL_QUESTIONS, results):
        status = (
            "HELD" if (answer.refused and not leaked) else "ANSWERED"
        )
        if status == "HELD":
            held += 1
        print(f"\n  [{status}] {case.question}")
        print(f"    trap: {case.trap}")
        if answer.refused:
            print(f"    -> {answer.refusal_reason}")
        else:
            print(f"    -> {answer.text[:200]}")
        if leaked:
            print(f"    !! contains forbidden phrasing: {leaked}")

    print("\n" + "=" * 74)
    print(f"  {held}/{len(ADVERSARIAL_QUESTIONS)} held")
    if held < len(ADVERSARIAL_QUESTIONS):
        print(
            "\n  Read the ANSWERED cases in full before treating them as bugs."
            "\n  Some may be legitimately answerable; the rest are prompt work."
        )
    return 0 if held == len(ADVERSARIAL_QUESTIONS) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
