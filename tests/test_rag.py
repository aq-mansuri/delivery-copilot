"""Tests for chunking, hybrid retrieval, and the eval harness."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.models.domain import Page, PageSection
from app.rag.chunking import chunk_page
from app.rag.evaluation import EvalCase, evaluate_retrieval
from app.rag.retrieval import BM25Index, HybridRetriever, tokenize
from tests.fixtures.fake_embedder import FakeEmbedder

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def make_page(sections, page_id="p1", title="ADR: Claims vendor integration"):
    return Page(
        id=page_id,
        space_key="ARCH",
        title=title,
        url=f"https://acme.atlassian.net/wiki/spaces/ARCH/pages/{page_id}",
        version=1,
        updated_at=NOW,
        labels=["architecture"],
        sections=[PageSection(**s) for s in sections],
    )


class TestChunking:
    def test_heading_ancestry_is_preserved(self):
        """'Timeline' alone retrieves from every page that has a Timeline."""
        page = make_page([
            {"heading": "Context", "level": 2, "text": "A" * 300},
            {"heading": "Timeline", "level": 3, "text": "B" * 300},
        ])
        chunks = chunk_page(page)
        timeline = [c for c in chunks if c.heading_path[-1] == "Timeline"][0]
        assert timeline.heading_path == ("Context", "Timeline")

    def test_embedding_text_carries_the_breadcrumb(self):
        """A body-only embedding loses the subject entirely."""
        page = make_page([{"heading": "Decision", "level": 2, "text": "C" * 300}])
        chunk = chunk_page(page)[0]
        assert chunk.embedding_text().startswith(
            "ADR: Claims vendor integration > Decision"
        )

    def test_small_sections_merge_forward(self):
        """Fragments match everything weakly and answer nothing."""
        page = make_page([
            {"heading": "Status", "level": 2, "text": "Accepted."},
            {"heading": "Context", "level": 2, "text": "D" * 400},
        ])
        chunks = chunk_page(page)
        assert len(chunks) == 1
        assert "Accepted." in chunks[0].text

    def test_oversized_sections_split_at_paragraphs(self):
        page = make_page([
            {
                "heading": "Long",
                "level": 2,
                "text": "\n\n".join(["E" * 600 for _ in range(6)]),
            }
        ])
        chunks = chunk_page(page, max_chars=1400)
        assert len(chunks) > 1
        assert all(len(c.text) <= 1400 for c in chunks)

    def test_sentences_are_never_split(self):
        sentence = "The vendor requires mutual TLS on every callback endpoint. "
        page = make_page([{"heading": "Long", "level": 2, "text": sentence * 60}])
        chunks = chunk_page(page, max_chars=900)
        for chunk in chunks:
            assert chunk.text.rstrip().endswith((".", "!", "?"))

    def test_chunk_ids_are_stable_and_unique(self):
        page = make_page([
            {"heading": "A", "level": 2, "text": "F" * 400},
            {"heading": "B", "level": 2, "text": "G" * 400},
        ])
        first = chunk_page(page)
        second = chunk_page(page)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert len({c.chunk_id for c in first}) == len(first)

    def test_citation_label_is_human_readable(self):
        page = make_page([{"heading": "Decision", "level": 2, "text": "H" * 400}])
        assert chunk_page(page)[0].citation_label() == (
            "ADR: Claims vendor integration — Decision"
        )


class TestTokenizer:
    def test_ticket_keys_survive_tokenization(self):
        """Splitting on hyphens turns INS-101 into 'ins' and '101'."""
        assert "ins-101" in tokenize("What is blocking INS-101?")

    def test_stopwords_are_removed(self):
        assert tokenize("what is the status of") == ["status"]

    def test_versions_stay_intact(self):
        assert "1.2.3" in tokenize("upgrade to 1.2.3")


class TestBM25:
    def test_rare_terms_outrank_common_ones(self):
        page_a = make_page(
            [{"heading": "A", "level": 2, "text": "vendor " * 40}], page_id="a"
        )
        page_b = make_page(
            [{"heading": "B", "level": 2, "text": "mutual TLS handshake " * 10}],
            page_id="b",
        )
        chunks = chunk_page(page_a) + chunk_page(page_b)
        index = BM25Index(chunks)
        top = index.search("mutual TLS", top_k=1)[0][0]
        assert chunks[top].page_id == "b"

    def test_empty_query_returns_nothing(self):
        chunks = chunk_page(make_page([{"heading": "A", "level": 2, "text": "x" * 300}]))
        assert BM25Index(chunks).search("the of and", top_k=5) == []


class TestHybridRetrieval:
    @pytest.fixture
    def retriever(self):
        pages = [
            make_page(
                [{"heading": "Context", "level": 2, "text":
                  "The claims vendor requires mutual TLS. " * 12}],
                page_id="tls",
                title="ADR: Claims vendor TLS",
            ),
            make_page(
                [{"heading": "Blocker", "level": 2, "text":
                  "INS-101 is waiting on partner auth rollout. " * 12}],
                page_id="ticket",
                title="Sprint 13 notes",
            ),
            make_page(
                [{"heading": "Compliance", "level": 2, "text":
                  "Legal signoff is required before any policy migration. " * 12}],
                page_id="legal",
                title="Compliance checklist",
            ),
        ]
        chunks = [c for p in pages for c in chunk_page(p)]
        return HybridRetriever(chunks, FakeEmbedder(), candidate_pool=10)

    def test_identifier_query_is_rescued_by_keyword_search(self, retriever):
        """The fake embedder cannot see INS-101 at all — this is BM25's job.

        This is the whole argument for hybrid, in one test.
        """
        hits = retriever.search("what is blocking INS-101", top_k=2)
        assert hits[0].chunk.page_id == "ticket"
        assert hits[0].bm25_rank is not None

    def test_semantic_query_works_without_exact_terms(self, retriever):
        hits = retriever.search("compliance signoff policy migration", top_k=2)
        assert hits[0].chunk.page_id == "legal"

    def test_agreement_between_retrievers_outranks_either_alone(self, retriever):
        hits = retriever.search("claims vendor mutual TLS", top_k=3)
        best = hits[0]
        assert best.chunk.page_id == "tls"
        assert best.bm25_rank is not None and best.vector_rank is not None

    def test_retrieval_reason_is_explainable(self, retriever):
        """'Why did it cite that page' needs a better answer than 'search did'."""
        reason = retriever.search("mutual TLS", top_k=1)[0].retrieval_reason()
        assert "keyword" in reason or "semantic" in reason

    def test_empty_corpus_returns_empty(self):
        assert HybridRetriever([], FakeEmbedder()).search("anything") == []

    def test_candidate_pool_exceeds_top_k(self, retriever):
        """Fusing two top-5 lists can only ever surface 10 documents."""
        assert retriever.candidate_pool > 5


class TestEvalHarness:
    @pytest.fixture
    def retriever(self):
        pages = [
            make_page(
                [{"heading": "Context", "level": 2, "text":
                  "Mutual TLS is required by the claims vendor. " * 12}],
                page_id="tls", title="ADR: TLS",
            ),
            make_page(
                [{"heading": "Notes", "level": 2, "text":
                  "Legal signoff pending for policy migration. " * 12}],
                page_id="legal", title="Compliance",
            ),
        ]
        return HybridRetriever(
            [c for p in pages for c in chunk_page(p)], FakeEmbedder()
        )

    def test_recall_and_mrr_are_computed(self, retriever):
        report = evaluate_retrieval(
            retriever,
            [
                EvalCase("mutual TLS claims vendor", ("tls",), category="semantic"),
                EvalCase("legal signoff migration", ("legal",), category="semantic"),
            ],
            k=3,
        )
        assert report.recall_at_k == 1.0
        assert report.mrr > 0.9

    def test_misses_are_reported_with_detail(self, retriever):
        report = evaluate_retrieval(
            retriever, [EvalCase("quarterly budget forecast", ("nonexistent",))], k=3
        )
        assert report.recall_at_k == 0.0
        assert "missed" in report.failures()[0]

    def test_category_breakdown_exposes_uneven_performance(self, retriever):
        """80% overall can be 100% prose and 20% identifiers."""
        report = evaluate_retrieval(
            retriever,
            [
                EvalCase("mutual TLS", ("tls",), category="semantic"),
                EvalCase("budget forecast", ("missing",), category="identifier"),
            ],
            k=3,
        )
        breakdown = report.by_category()
        assert breakdown["semantic"] == 1.0
        assert breakdown["identifier"] == 0.0

    def test_contamination_is_flagged(self, retriever):
        """A near-duplicate page crowding out the right one."""
        report = evaluate_retrieval(
            retriever,
            [EvalCase("mutual TLS", ("tls",), must_not_retrieve=("legal",))],
            k=5,
        )
        assert report.results[0].contamination == ["legal"]

    def test_summary_is_readable(self, retriever):
        report = evaluate_retrieval(
            retriever, [EvalCase("mutual TLS", ("tls",))], k=3
        )
        assert "recall@3" in report.summary()
