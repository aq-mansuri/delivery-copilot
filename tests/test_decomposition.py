"""Tests for query decomposition and multi-query retrieval."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.llm import FakeLLM, text_response
from app.models.domain import Page, PageSection
from app.rag.chunking import chunk_page
from app.rag.decomposition import (
    HeuristicDecomposer,
    LLMDecomposer,
    multi_query_search,
)
from app.rag.embedders import FakeEmbedder
from app.rag.retrieval import HybridRetriever

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def page(pid, title, heading, body):
    return Page(
        id=pid, space_key="ARCH", title=title, url=f"https://x/wiki/{pid}",
        version=1, updated_at=NOW,
        sections=[PageSection(heading=heading, level=2, text=body)],
    )


@pytest.fixture
def retriever():
    pages = [
        page("acme-contract", "Acme Contract Summary", "Commercial risk",
             "The agreement has no service credit mechanism. " * 12),
        page("acme-delivery", "Acme Integration Status", "Where things stand",
             "The integration is not progressing. " * 12),
        page("release-plan", "Q4 Release Plan", "Priorities",
             "Three items carry board commitments. " * 12),
    ]
    return HybridRetriever(
        [c for p in pages for c in chunk_page(p)], FakeEmbedder(), candidate_pool=10
    )


class TestHeuristicSplitting:
    async def test_compound_question_splits_on_and(self):
        parts = await HeuristicDecomposer().decompose(
            "What is the status of the Acme work and are there contractual risks?"
        )
        assert len(parts) == 2

    async def test_subject_is_carried_into_later_parts(self):
        """'and are there contractual risks' has no subject and retrieves noise."""
        parts = await HeuristicDecomposer().decompose(
            "What is the status of the Acme work and are there contractual risks?"
        )
        assert "Acme" in parts[1]

    async def test_atomic_question_is_left_alone(self):
        """Splitting an atomic question makes retrieval worse, not better."""
        q = "What work is holding up the release?"
        assert await HeuristicDecomposer().decompose(q) == [q]

    async def test_or_does_not_split(self):
        """'blocked or stalled' offers alternatives within one intent."""
        q = "Which issues are blocked or stalled?"
        assert await HeuristicDecomposer().decompose(q) == [q]

    async def test_fragment_too_short_aborts_the_split(self):
        """A two-word sub-query adds noise to the fusion."""
        q = "What is at risk and why?"
        assert await HeuristicDecomposer().decompose(q) == [q]


class TestMultiQueryRetrieval:
    async def test_each_subquery_can_surface_its_own_page(self, retriever):
        """The cross-source case: neither page matches the whole question."""
        hits, parts = await multi_query_search(
            retriever,
            HeuristicDecomposer(),
            "What is the status of the Acme work and are there contractual risks?",
            top_k=5,
        )
        assert len(parts) == 2
        assert "acme-contract" in {h.chunk.page_id for h in hits}

    async def test_atomic_question_takes_the_single_query_path(self, retriever):
        hits, parts = await multi_query_search(
            retriever, HeuristicDecomposer(), "What work is holding up the release?"
        )
        assert parts == ["What work is holding up the release?"]
        assert hits

    async def test_results_are_deduplicated_across_subqueries(self, retriever):
        hits, _ = await multi_query_search(
            retriever,
            HeuristicDecomposer(),
            "What is the Acme status and are there Acme risks documented?",
            top_k=5,
        )
        ids = [h.chunk.chunk_id for h in hits]
        assert len(ids) == len(set(ids))


class TestLLMDecomposer:
    async def test_parses_a_json_array(self):
        llm = FakeLLM(responses=[
            text_response('["Acme delivery status", "Acme contractual risks"]')
        ])
        parts = await LLMDecomposer(llm).decompose("status and risks for Acme?")
        assert parts == ["Acme delivery status", "Acme contractual risks"]

    async def test_markdown_fences_tolerated(self):
        llm = FakeLLM(responses=[text_response('```json\n["a query here"]\n```')])
        assert await LLMDecomposer(llm).decompose("q") == ["a query here"]

    async def test_failure_falls_back_to_the_original(self):
        """A decomposer that raises takes down a request it only meant to
        improve."""
        llm = FakeLLM(responses=[text_response("I think you want two searches.")])
        assert await LLMDecomposer(llm).decompose("original q") == ["original q"]

    async def test_more_than_three_parts_are_capped(self):
        llm = FakeLLM(responses=[text_response('["a a a","b b b","c c c","d d d"]')])
        assert len(await LLMDecomposer(llm).decompose("q")) == 3
