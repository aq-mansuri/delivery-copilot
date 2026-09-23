"""Hybrid retrieval: BM25 + dense vectors, fused by reciprocal rank.

Why hybrid rather than embeddings alone. Half the questions in this domain carry
literal identifiers — "what's blocking INS-101", "status of the Acme Claims
contract", "did we sign off on mutual TLS". Dense embeddings are poor at exact
tokens: "INS-101" and "INS-107" sit almost on top of each other in vector space,
and a rare vendor name may barely move the embedding at all. BM25 nails those and
is weak exactly where embeddings are strong — paraphrase, synonymy, "why did we
choose X" against a page that never uses the word "choose".

Why reciprocal rank fusion rather than a weighted score sum. BM25 scores are
unbounded and corpus-dependent; cosine similarity is bounded and isn't. Adding
them requires normalizing two distributions that shift every time the corpus
changes, and the weights need retuning per client. RRF discards the magnitudes
and fuses ranks:

    score(d) = sum over retrievers of 1 / (k + rank(d))

One parameter, k, which damps the influence of top ranks. It is scale-free, so it
survives a corpus change without retuning — the right default when a consultancy
has to make this work across several tenants without babysitting each one.

The embedder is a protocol, not a concrete client. Tests run against a
deterministic fake, so retrieval logic is verifiable without an API key, a
network call, or a bill.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.rag.chunking import Chunk

# Keeps ticket keys (INS-101), versions (1.2.3) and hyphenated terms intact.
# A tokenizer that splits on every hyphen turns INS-101 into "ins" and "101",
# and the identifier queries this index exists to serve stop working.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with what which who whom how why when where do does did our
    we you your this these those""".split()
)


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN.findall(text)
        if token.lower() not in _STOPWORDS
    ]


class Embedder(Protocol):
    """Production implementation wraps a hosted embedding model.

    Deliberately not the Anthropic Messages API — Claude is a generation model;
    embeddings come from a dedicated embedding provider. Keeping this a protocol
    means swapping providers, or moving to a self-hosted model for a client with
    data-residency constraints, touches one class.
    """

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None

    def retrieval_reason(self) -> str:
        """Which retriever found this, for tracing and for debugging bad answers.

        When a client says "why did it cite that page", the answer needs to be
        better than "the search returned it".
        """
        if self.bm25_rank is not None and self.vector_rank is not None:
            return f"keyword #{self.bm25_rank + 1}, semantic #{self.vector_rank + 1}"
        if self.bm25_rank is not None:
            return f"keyword #{self.bm25_rank + 1} only"
        return f"semantic #{self.vector_rank + 1} only"


class BM25Index:
    """Okapi BM25.

    Written out rather than pulled from a library so the parameters are visible
    and tunable per client. In production this is a Postgres `tsvector` query;
    the scoring behaviour is what the tests here pin down.
    """

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self._docs = [tokenize(c.embedding_text()) for c in self.chunks]
        self._lengths = [len(d) for d in self._docs]
        self._avg_length = (sum(self._lengths) / len(self._docs)) if self._docs else 0.0

        self._term_frequencies = [Counter(d) for d in self._docs]
        document_frequency: Counter[str] = Counter()
        for doc in self._docs:
            document_frequency.update(set(doc))

        total = len(self._docs)
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []

        scores: dict[int, float] = defaultdict(float)
        for index, frequencies in enumerate(self._term_frequencies):
            length_norm = (
                self.k1
                * (1 - self.b + self.b * self._lengths[index] / (self._avg_length or 1))
            )
            for term in terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                scores[index] += self._idf.get(term, 0.0) * (
                    tf * (self.k1 + 1) / (tf + length_norm)
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        rrf_k: int = 60,
        candidate_pool: int = 30,
        min_bm25: float = 0.0,
        min_cosine: float = 0.0,
    ):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.rrf_k = rrf_k
        # Relevance floors, applied to RAW scores before fusion — RRF discards
        # magnitudes, so thresholding after fusion is impossible.
        #
        # Calibrated finding, recorded because it is counter-intuitive: these
        # floors CANNOT separate answerable from unanswerable questions on this
        # corpus. Measured ranges overlap (answerable BM25 1.87-7.25, negative
        # 2.54-5.21) because the negatives ask for absent FACTS about PRESENT
        # topics. See ADR-006. Floors remain useful for genuinely off-topic
        # queries; refusal is a generation-time problem.
        self.min_bm25 = min_bm25
        self.min_cosine = min_cosine
        # Each retriever contributes more candidates than the final result size.
        # Fusing two top-5 lists can only ever surface 10 documents; the fusion
        # has nothing to work with.
        self.candidate_pool = candidate_pool

        self.bm25 = BM25Index(self.chunks)
        self.vectors = (
            embedder.embed_documents([c.embedding_text() for c in self.chunks])
            if self.chunks
            else []
        )

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if not self.chunks:
            return []

        bm25_hits = [
            hit
            for hit in self.bm25.search(query, self.candidate_pool)
            if hit[1] >= self.min_bm25
        ]
        bm25_ranks = {index: rank for rank, (index, _) in enumerate(bm25_hits)}

        query_vector = self.embedder.embed_query(query)
        similarities = [
            (index, cosine(query_vector, vector))
            for index, vector in enumerate(self.vectors)
        ]
        similarities = [s for s in similarities if s[1] >= self.min_cosine]
        similarities.sort(key=lambda pair: (-pair[1], pair[0]))
        vector_ranks = {
            index: rank
            for rank, (index, _) in enumerate(similarities[: self.candidate_pool])
        }

        fused: dict[int, float] = defaultdict(float)
        for index, rank in bm25_ranks.items():
            fused[index] += 1.0 / (self.rrf_k + rank)
        for index, rank in vector_ranks.items():
            fused[index] += 1.0 / (self.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return [
            ScoredChunk(
                chunk=self.chunks[index],
                score=score,
                bm25_rank=bm25_ranks.get(index),
                vector_rank=vector_ranks.get(index),
            )
            for index, score in ordered[:top_k]
        ]
