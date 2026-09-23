"""Run the retrieval eval and print a scoreable report.

    python scripts/run_eval.py
    python scripts/run_eval.py --k 3 --save baseline.json

Run this BEFORE fixing anything. A system that works with no record of what it
scored beforehand is a system you cannot claim to have improved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunking import chunk_page  # noqa: E402
from app.rag.evaluation import evaluate_retrieval, load_cases  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from scripts.seed_content import load_offline_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cases", default="docs/eval_set.json")
    parser.add_argument("--save", default="")
    parser.add_argument(
        "--decompose", action="store_true",
        help="Split compound questions into sub-queries before retrieving.",
    )
    args = parser.parse_args()

    from tests.fixtures.fake_embedder import FakeEmbedder

    pages = load_offline_corpus()
    chunks = [c for page in pages for c in chunk_page(page)]
    retriever = HybridRetriever(chunks, FakeEmbedder(), candidate_pool=20)
    cases = load_cases(args.cases)

    print(f"corpus: {len(pages)} pages -> {len(chunks)} chunks")
    print(f"cases:  {len(cases)}\n")

    if args.decompose:
        import asyncio

        from app.rag.decomposition import HeuristicDecomposer, multi_query_search

        decomposer = HeuristicDecomposer()

        class DecomposingRetriever:
            """Adapter so the eval harness needs no changes.

            Keeping the harness ignorant of decomposition is what makes the
            before/after numbers comparable — same cases, same metrics, one
            variable changed.
            """

            def search(self, query, top_k=5):
                hits, _ = asyncio.run(
                    multi_query_search(retriever, decomposer, query, top_k=top_k)
                )
                return hits

        report = evaluate_retrieval(DecomposingRetriever(), cases, k=args.k)
    else:
        report = evaluate_retrieval(retriever, cases, k=args.k)
    print(report.summary())

    # Negative cases need their own treatment: "recall" on a case with no
    # relevant pages is meaningless. What matters is whether the system returned
    # nothing — and right now it has no way to.
    negatives = [r for r in report.results if r.case.category == "negative"]
    if negatives:
        declined = sum(1 for r in negatives if not r.retrieved_page_ids)
        print(f"\nnegative cases declined: {declined}/{len(negatives)}")
        for result in negatives:
            state = "declined" if not result.retrieved_page_ids else "ANSWERED"
            print(f"  [{state}] {result.case.question}")
            if result.retrieved_page_ids:
                print(f"           returned {result.retrieved_page_ids[:3]}")

    failures = report.failures()
    if failures:
        print(f"\nfailures ({len(failures)}):")
        for line in failures:
            print(f"  {line}")

    if args.save:
        Path(args.save).write_text(
            json.dumps(
                {
                    "k": report.k,
                    "recall": report.recall_at_k,
                    "mrr": report.mrr,
                    "by_category": report.by_category(),
                    "negatives_declined": (
                        sum(1 for r in negatives if not r.retrieved_page_ids)
                        if negatives else 0
                    ),
                    "negatives_total": len(negatives),
                },
                indent=2,
            )
        )
        print(f"\nsaved to {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
