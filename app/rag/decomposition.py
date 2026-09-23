"""Query decomposition.

Aimed at one measured weakness: cross-source questions scored 50% at baseline.

    "What is the status of the Acme work and are there contractual risks?"

The answer lives in two pages. The delivery notes carry the status and never
mention the contract; the contract summary carries commercial risk and never
mentions current status. Scored as one query, neither page matches the whole
question well, so both rank below pages that partially match everything.

Splitting the question into two retrievals and fusing the results lets each
sub-query find its own page.

## Two decomposers, deliberately

`HeuristicDecomposer` splits on conjunctions. No model, no cost, deterministic —
which means it can run inside the eval harness and produce a number that is
comparable across runs. It handles the compound-question case, which is most of
the cross-source set.

`LLMDecomposer` handles what the heuristic cannot: implicit multi-part questions
("does the current Jira work match the documented priorities?" is two lookups
with no conjunction to split on). It costs a call per question.

Starting with the heuristic is the point. It is free, it is measurable, and if
it closes the gap then the LLM version is complexity with no evidence behind it.
Reach for the model when the cheap thing has been measured and found wanting —
not before.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.agent.llm import LLM
from app.rag.retrieval import HybridRetriever, ScoredChunk

logger = logging.getLogger(__name__)

MAX_SUBQUERIES = 3

# Conjunctions that usually join two askable halves. "or" is excluded: it
# normally offers alternatives within one question ("blocked or stalled")
# rather than joining two, and splitting on it fragments a single intent.
_SPLIT = re.compile(
    r",?\s+and\s+(?=are|is|was|were|does|do|did|what|which|who|when|how|any|there)",
    re.IGNORECASE,
)

_LEAD_IN = re.compile(
    r"^(?:and\s+)?(?:are there|is there|are|is|what|which|who|does|do|did)\s+",
    re.IGNORECASE,
)


class Decomposer(Protocol):
    async def decompose(self, question: str) -> list[str]: ...


@dataclass
class HeuristicDecomposer:
    """Splits compound questions on conjunctions. No model call."""

    min_part_words: int = 3

    async def decompose(self, question: str) -> list[str]:
        parts = [p.strip(" ?,.") for p in _SPLIT.split(question) if p.strip()]

        # A split that produces a fragment too short to retrieve on is worse
        # than no split — it adds a noise query to the fusion.
        if len(parts) < 2 or any(
            len(p.split()) < self.min_part_words for p in parts
        ):
            return [question]

        # Later parts often lose their subject: "...and are there contractual
        # risks" has no "Acme" in it. Carrying the leading noun phrase forward
        # keeps each sub-query self-contained, which is the whole point.
        subject = _subject_hint(parts[0])
        rebuilt = [parts[0]]
        for part in parts[1:]:
            if subject and subject.lower() not in part.lower():
                part = f"{part} {subject}"
            rebuilt.append(part)

        return rebuilt[:MAX_SUBQUERIES]


def _subject_hint(first_part: str) -> str:
    """Capitalised or distinctive terms from the first clause.

    Crude on purpose. A proper-noun heuristic that catches "Acme" and "O7" is
    enough here, and anything cleverer belongs in the LLM decomposer.
    """
    stripped = _LEAD_IN.sub("", first_part)
    tokens = [t.strip(",.?") for t in stripped.split()]
    proper = [
        t
        for t in tokens
        if (t[:1].isupper() and len(t) > 2) or re.fullmatch(r"[A-Z]+-?\d+", t)
    ]
    return " ".join(proper[:2])


DECOMPOSE_PROMPT = """Split a delivery question into the minimum number of \
independent search queries needed to answer it.

Rules:
1. If one search answers it, return the original question unchanged as the only \
item. Splitting an atomic question makes retrieval worse, not better.
2. Each sub-query must stand alone. Carry the subject forward — "and are there \
risks" is not a usable query; "contractual risks for Acme" is.
3. Never produce more than 3.
4. Do not answer the question or add information that is not in it.

Return ONLY a JSON array of strings."""


@dataclass
class LLMDecomposer:
    """Model-based splitting, for implicit multi-part questions.

    Falls back to the original question on any failure. A decomposer that raises
    takes down a request it was only meant to improve.
    """

    llm: LLM

    async def decompose(self, question: str) -> list[str]:
        import json

        try:
            response = await self.llm.complete(
                system=DECOMPOSE_PROMPT,
                messages=[{"role": "user", "content": question}],
                max_tokens=300,
                temperature=0.0,
            )
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip())
            parts = json.loads(text)
            if not isinstance(parts, list) or not parts:
                return [question]
            cleaned = [str(p).strip() for p in parts if str(p).strip()]
            return cleaned[:MAX_SUBQUERIES] or [question]
        except Exception:
            logger.warning("decomposition failed for %r, using original", question)
            return [question]


async def multi_query_search(
    retriever: HybridRetriever,
    decomposer: Decomposer,
    question: str,
    *,
    top_k: int = 5,
    per_query_k: int = 5,
) -> tuple[list[ScoredChunk], list[str]]:
    """Decompose, retrieve per sub-query, fuse by reciprocal rank.

    RRF again, for the same reason as within a single search: scores from
    different queries are not comparable, ranks are. A chunk that ranks well for
    one sub-query surfaces even if the other sub-query never sees it — which is
    exactly the cross-source case.
    """
    subqueries = await decomposer.decompose(question)

    if len(subqueries) == 1:
        return retriever.search(question, top_k=top_k), subqueries

    fused: dict[str, float] = defaultdict(float)
    best: dict[str, ScoredChunk] = {}

    for subquery in subqueries:
        for rank, hit in enumerate(retriever.search(subquery, top_k=per_query_k)):
            key = hit.chunk.chunk_id
            fused[key] += 1.0 / (60 + rank)
            # Keep the highest-scoring instance so the retrieval reason shown
            # to the user reflects its strongest match, not its last one.
            if key not in best or hit.score > best[key].score:
                best[key] = hit

    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    return [best[key] for key, _ in ordered[:top_k]], subqueries
