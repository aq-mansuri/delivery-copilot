"""Embedding providers.

Moved here from tests/fixtures. A production module importing from the test tree
is a dependency pointing the wrong way — it means the fake is load-bearing, and
it breaks the moment tests are excluded from a package build.

Claude is a generation model and has no embeddings endpoint, so this is a
separate provider. Anthropic points at Voyage AI; OpenAI, Cohere and
self-hosted sentence-transformers are the other common routes. Keeping the
protocol means swapping provider — or moving to a self-hosted model for a client
with data-residency rules — touches one class and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


VOCABULARY = [
    "tls", "vendor", "claims", "gateway", "compliance", "legal", "signoff",
    "callback", "pilot", "rollout", "migration", "policy", "auth", "partner",
    "architecture", "decision", "timeline", "quarter", "review", "dashboard",
]


class FakeEmbedder:
    """Deterministic bag-of-words embedder for tests and offline runs.

    Real embeddings in a test suite mean an API key, a network call, a bill and
    non-determinism — so retrieval logic becomes untestable in CI, which is
    where it most needs testing.

    Deliberately BAD at exact identifiers: INS-101 is not in the vocabulary and
    contributes nothing. That is what makes the hybrid tests meaningful — if the
    fake were good at everything, a test proving BM25 rescues identifier queries
    would pass for the wrong reason.

    It is also bad at everything outside its 20 words, which is why cosine
    scores come back 0.000 on most real queries. That is a harness artifact, not
    a result. Do not read retrieval quality off a run that uses this.
    """

    def __init__(self, vocabulary: Sequence[str] = VOCABULARY):
        self.vocabulary = list(vocabulary)

    def _vector(self, text: str) -> list[float]:
        from app.rag.retrieval import tokenize

        tokens = tokenize(text)
        counts = [
            float(sum(1 for t in tokens if t == term)) for term in self.vocabulary
        ]
        norm = math.sqrt(sum(c * c for c in counts))
        return [c / norm for c in counts] if norm else counts

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class VoyageEmbedder:
    """Voyage AI embeddings, with batching and an on-disk cache.

    Three things this handles that a naive implementation does not:

    **Batching.** Providers cap inputs per request. 3,000 chunks in one call is
    a 400; 3,000 separate calls is a rate limit and a large bill.

    **Caching.** Re-embedding an unchanged corpus on every restart costs money
    for no benefit. The cache key is a hash of the text plus the model name, so
    changing either invalidates correctly — a cache keyed on text alone silently
    serves vectors from the wrong model after an upgrade, which is very hard to
    diagnose.

    **Asymmetric input types.** Voyage distinguishes `document` from `query`;
    embedding both the same way measurably degrades retrieval. Most providers
    have an equivalent and most people ignore it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = "voyage-3",
        batch_size: int = 128,
        cache_dir: str | Path | None = ".cache/embeddings",
    ):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Voyage API key. Set VOYAGE_API_KEY, or use FakeEmbedder for "
                "offline work."
            )
        self.model = model
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str, input_type: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(
            f"{self.model}:{input_type}:{text}".encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        import httpx

        results: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []

        for index, text in enumerate(texts):
            path = self._cache_path(text, input_type)
            if path and path.exists():
                results[index] = json.loads(path.read_text())
            else:
                pending.append((index, text))

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            resp = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "input": [text for _, text in batch],
                    "model": self.model,
                    "input_type": input_type,
                },
                timeout=60.0,
            )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Voyage {resp.status_code}: {resp.text[:300]}"
                )
            vectors = [item["embedding"] for item in resp.json()["data"]]

            for (index, text), vector in zip(batch, vectors):
                results[index] = vector
                path = self._cache_path(text, input_type)
                if path:
                    path.write_text(json.dumps(vector))

        return [vector for vector in results if vector is not None]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


def default_embedder() -> Embedder:
    """Real provider when a key is present, fake otherwise.

    Falls back loudly rather than silently: a run that quietly used the fake and
    reported good numbers is worse than one that failed.
    """
    if os.getenv("VOYAGE_API_KEY"):
        return VoyageEmbedder()
    logger.warning(
        "VOYAGE_API_KEY not set — using FakeEmbedder. Retrieval scores from "
        "this run are not meaningful."
    )
    return FakeEmbedder()
