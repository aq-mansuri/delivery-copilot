"""Runs a documentation load across Confluence spaces.

Same shape as `sync_runner.run_sync` for the same reason: one space's
failure must not abort the others, and the caller needs enough detail to
tell "fully covered" from "some spaces unreadable" apart rather than a bare
list of pages that could have come from either.

## Why there is no `CompleteDocsLoad` / `PartialDocsLoad` split

`SyncResult` splits into two incompatible types (ADR-004) because a partial
risk sync is dangerous in a specific way: it reads as complete and
understates risk — a lead asks "is INS-12 blocked?" and gets a confident
"no" from a snapshot that never read the project holding the blocker.

A partial documentation corpus does not create that failure. `/ask` already
has a safe, tested answer for "not enough context": it declines (ADR-006).
A smaller real corpus just produces more refusals, not a false confident
claim — refusal is the safe outcome here, not a symptom to route around. So
`DocsLoad` carries `is_complete` as a fact to report, not a gate that blocks
construction.

Total failure is the one case treated like Jira's: `is_empty` distinguishes
it, and the caller (`app.api.main.load_live_docs`) refuses to index nothing
as though it were the bundled sample corpus — the same "never mixed" rule
Jira's failed-sync path already follows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from app.integrations.atlassian.confluence import ConfluenceError
from app.models.domain import Page

logger = logging.getLogger(__name__)


class PageSource(Protocol):
    """What `run_docs_load` needs from a Confluence client — real or stub."""

    def iter_pages(self, space_key: str, *, labels: list[str] | None = None): ...


@dataclass(frozen=True)
class DocsLoad:
    pages: list[Page] = field(default_factory=list)
    spaces_covered: list[str] = field(default_factory=list)
    spaces_incomplete: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.spaces_incomplete

    @property
    def is_empty(self) -> bool:
        return not self.pages


async def run_docs_load(
    client: PageSource,
    space_keys: list[str],
    *,
    labels: list[str] | None = None,
) -> DocsLoad:
    """Read every configured space, keeping whatever each one yielded before
    it failed.

    `iter_pages` is a generator, not a batch call — a space that raises on
    page three must not discard the two pages already read from it. The loop
    below relies on that: pages accumulate as they arrive, and only the
    failure is recorded separately.
    """
    pages: list[Page] = []
    covered: list[str] = []
    incomplete: list[str] = []

    for space_key in space_keys:
        try:
            async for p in client.iter_pages(space_key, labels=labels):
                pages.append(p)
            covered.append(space_key)
        except ConfluenceError as exc:
            logger.warning("confluence space %s incomplete: %s", space_key, exc)
            incomplete.append(space_key)

    return DocsLoad(pages=pages, spaces_covered=covered, spaces_incomplete=incomplete)
