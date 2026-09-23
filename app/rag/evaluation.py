"""Retrieval evaluation.

Built before any tuning. Without it, "improving retrieval" means changing a
parameter, reading three answers, and forming an impression — which is how you
spend a day making things worse with total confidence.

**Recall@k is the metric that matters here.** If the right chunk isn't in the
retrieved set, no amount of prompting recovers it: the model will answer from
whatever it did get, fluently and wrongly. Precision is recoverable — the model
can ignore an irrelevant chunk. A missing chunk is not.

That asymmetry is the same one the client stated about risk findings, which is
not a coincidence: a false negative is silent, and silent failures are the
expensive kind.

MRR is tracked as a secondary signal. It catches the case where recall looks
fine but the right chunk keeps landing at position five, which degrades answers
once a real context budget forces truncation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.rag.retrieval import HybridRetriever


@dataclass(frozen=True)
class EvalCase:
    """One question and the chunks that must be retrieved to answer it.

    `relevant_page_ids` rather than chunk ids: chunk boundaries shift whenever
    the chunker changes, and an eval set that breaks on every chunking tweak
    stops being used. Page-level ground truth survives.

    `must_not_retrieve` is optional and catches a specific failure — a
    superficially similar page (last quarter's version of the same ADR, another
    team's identically-titled runbook) crowding out the right one.
    """

    question: str
    relevant_page_ids: tuple[str, ...]
    category: str = "general"
    must_not_retrieve: tuple[str, ...] = ()
    note: str = ""


@dataclass
class CaseResult:
    case: EvalCase
    retrieved_page_ids: list[str]
    hit: bool
    reciprocal_rank: float
    contamination: list[str]

    def failure_summary(self) -> str:
        if self.hit and not self.contamination:
            return ""
        parts = []
        if not self.hit:
            parts.append(
                f"missed {list(self.case.relevant_page_ids)}, "
                f"got {self.retrieved_page_ids}"
            )
        if self.contamination:
            parts.append(f"retrieved excluded pages {self.contamination}")
        return f"[{self.case.category}] {self.case.question}: " + "; ".join(parts)


@dataclass
class EvalReport:
    k: int
    results: list[CaseResult]

    @property
    def recall_at_k(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.hit for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    def by_category(self) -> dict[str, float]:
        """Aggregate recall hides which *kind* of question is failing.

        A system at 80% overall might be at 100% on prose questions and 20% on
        identifier lookups — a fixable, specific problem that the single number
        conceals.
        """
        buckets: dict[str, list[bool]] = {}
        for result in self.results:
            buckets.setdefault(result.case.category, []).append(result.hit)
        return {
            category: sum(hits) / len(hits) for category, hits in buckets.items()
        }

    def failures(self) -> list[str]:
        return [s for r in self.results if (s := r.failure_summary())]

    def summary(self) -> str:
        lines = [
            f"recall@{self.k}: {self.recall_at_k:.2%}   MRR: {self.mrr:.3f}   "
            f"({len(self.results)} cases)"
        ]
        for category, recall in sorted(self.by_category().items()):
            lines.append(f"  {category:<20} {recall:.2%}")
        return "\n".join(lines)


def evaluate_retrieval(
    retriever: HybridRetriever, cases: Sequence[EvalCase], *, k: int = 5
) -> EvalReport:
    results: list[CaseResult] = []

    for case in cases:
        hits = retriever.search(case.question, top_k=k)
        page_ids = [hit.chunk.page_id for hit in hits]

        reciprocal_rank = 0.0
        for rank, page_id in enumerate(page_ids):
            if page_id in case.relevant_page_ids:
                reciprocal_rank = 1.0 / (rank + 1)
                break

        results.append(
            CaseResult(
                case=case,
                retrieved_page_ids=page_ids,
                hit=reciprocal_rank > 0,
                reciprocal_rank=reciprocal_rank,
                contamination=[
                    page_id
                    for page_id in page_ids
                    if page_id in case.must_not_retrieve
                ],
            )
        )

    return EvalReport(k=k, results=results)


def load_cases(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text())
    return [
        EvalCase(
            question=item["question"],
            relevant_page_ids=tuple(item["relevant_page_ids"]),
            category=item.get("category", "general"),
            must_not_retrieve=tuple(item.get("must_not_retrieve", [])),
            note=item.get("note", ""),
        )
        for item in data["cases"]
    ]
