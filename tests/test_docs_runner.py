"""Tests for the Confluence documentation load.

Mirrors `sync_runner.run_sync`'s shape deliberately: a per-space loop, one
space's failure does not abort the others, and the caller gets enough detail
to tell "fully covered" from "some spaces unreadable" apart.

The one thing this does NOT mirror is `SyncResult`'s complete/partial type
split. A partial risk sync is dangerous — it reads as complete and
understates risk. A partial documentation corpus is not: `/ask` already has a
safe answer for "not enough context" (ADR-006's refusal), so a smaller real
corpus just means more refusals, not a false confident claim. Encoding that
distinction in the return shape (a `DocsLoad` with a completeness flag,
not two incompatible types) is deliberate, not an oversight.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.docs_runner import DocsLoad, run_docs_load
from app.integrations.atlassian.confluence import ConfluenceError
from app.models.domain import Page

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def page(page_id: str, space_key: str, title: str = "Untitled") -> Page:
    return Page(
        id=page_id, space_key=space_key, title=title,
        url=f"https://x/wiki/{page_id}", version=1, updated_at=NOW,
    )


class StubConfluenceClient:
    """A `ConfluenceClient`-shaped stub: no HTTP, no respx, just canned pages
    or a raised error per space — the same substitution `run_sync` accepts a
    real-vs-fake `JiraClient` for."""

    def __init__(self, by_space: dict[str, list[Page] | Exception]):
        self.by_space = by_space
        self.calls: list[tuple[str, list[str] | None]] = []

    async def iter_pages(self, space_key: str, *, labels=None):
        self.calls.append((space_key, labels))
        result = self.by_space.get(space_key, [])
        if isinstance(result, Exception):
            raise result
        for p in result:
            yield p


class TestAllSpacesSucceed:
    async def test_pages_from_every_space_are_collected(self):
        client = StubConfluenceClient({
            "ARCH": [page("1", "ARCH"), page("2", "ARCH")],
            "COMP": [page("3", "COMP")],
        })
        load = await run_docs_load(client, ["ARCH", "COMP"])
        assert {p.id for p in load.pages} == {"1", "2", "3"}
        assert load.spaces_covered == ["ARCH", "COMP"]
        assert load.spaces_incomplete == []
        assert load.is_complete
        assert not load.is_empty

    async def test_labels_are_passed_through_to_the_client(self):
        client = StubConfluenceClient({"ARCH": []})
        await run_docs_load(client, ["ARCH"], labels=["architecture", "compliance"])
        assert client.calls == [("ARCH", ["architecture", "compliance"])]


class TestOneSpaceFails:
    async def test_a_failed_space_does_not_abort_the_others(self):
        """Same discipline as `run_sync`: project four 403s, one through three
        are still fully read."""
        client = StubConfluenceClient({
            "ARCH": [page("1", "ARCH")],
            "COMP": ConfluenceError("403 on COMP"),
        })
        load = await run_docs_load(client, ["ARCH", "COMP"])
        assert [p.id for p in load.pages] == ["1"]
        assert load.spaces_covered == ["ARCH"]
        assert load.spaces_incomplete == ["COMP"]
        assert not load.is_complete
        assert not load.is_empty

    async def test_pages_already_read_from_the_failed_space_are_kept(self):
        """`iter_pages` is a generator — a failure on page three must not
        discard the two pages already yielded before it. Confirmed by a stub
        that yields, then raises, mid-stream."""

        class HalfwayClient:
            async def iter_pages(self, space_key, *, labels=None):
                yield page("1", space_key)
                yield page("2", space_key)
                raise ConfluenceError("rate limited")

        load = await run_docs_load(HalfwayClient(), ["ARCH"])
        assert [p.id for p in load.pages] == ["1", "2"]
        assert load.spaces_incomplete == ["ARCH"]


class TestAllSpacesFail:
    async def test_result_is_empty_not_partial_with_zero_pages(self):
        client = StubConfluenceClient({
            "ARCH": ConfluenceError("down"),
            "COMP": ConfluenceError("down"),
        })
        load = await run_docs_load(client, ["ARCH", "COMP"])
        assert load.is_empty
        assert not load.is_complete
        assert load.spaces_covered == []


class TestEmptySpacesAreNotFailure:
    def test_a_space_with_no_matching_pages_is_still_covered(self):
        """Zero pages because a label filter matched nothing is not the same
        as a space that could not be read — the first is a legitimate
        (if uninteresting) result, the second is a gap `/health` must report."""
        load_sync = DocsLoad(pages=[], spaces_covered=["ARCH"], spaces_incomplete=[])
        assert load_sync.is_complete
        assert load_sync.is_empty  # empty AND complete: nothing to index, nothing broken
