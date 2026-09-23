"""Turning pages into retrievable chunks.

The failure this module exists to prevent: a chunk that is correct but
unusable. "We terminate TLS at the gateway" retrieved on its own tells the
reader nothing about which integration, which decision, or when. The heading
path is what makes a retrieved fragment answerable, so every chunk carries it.

Sizing rules, in order of priority:

1. Never split mid-sentence. A severed sentence embeds badly and reads worse in
   a citation the client will click through to.
2. Prefer section boundaries, then paragraph boundaries, then sentences.
3. Merge sections too small to stand alone, rather than emitting fragments that
   match everything weakly.

No overlap between chunks. Overlap is the usual reflex, but it inflates the
index, double-counts in scoring, and mostly compensates for splitting at
arbitrary offsets — which the rules above already avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.domain import Page, PageSection

# Roughly 350 words. Small enough that a retrieved chunk is mostly signal,
# large enough that a decision and its rationale usually stay together.
DEFAULT_MAX_CHARS = 2000
DEFAULT_MIN_CHARS = 200

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit with enough context to be answerable alone."""

    chunk_id: str
    page_id: str
    page_title: str
    page_url: str
    heading_path: tuple[str, ...]
    text: str
    # Set when a chunk merges forward across a heading boundary. Without it the
    # citation names the FIRST heading while the chunk also carries the second
    # section's text — so a reader clicking through to verify a claim lands on
    # the wrong section and concludes the system is fabricating.
    end_heading: str | None = None
    space_key: str = ""
    labels: tuple[str, ...] = field(default_factory=tuple)

    def embedding_text(self) -> str:
        """What actually gets embedded.

        The heading path is prepended rather than stored beside the text: an
        embedding of the body alone loses the subject, so "Context" sections
        from thirty different ADRs all land in the same neighbourhood.
        """
        breadcrumb = " > ".join((self.page_title,) + self.heading_path)
        return f"{breadcrumb}\n\n{self.text}"

    def citation_label(self) -> str:
        if not self.heading_path:
            return self.page_title
        start = self.heading_path[-1]
        if self.end_heading and self.end_heading != start:
            return f"{self.page_title} — {start} → {self.end_heading}"
        return f"{self.page_title} — {start}"


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split a too-long block at the least damaging boundary available."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        if not paragraph.strip():
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph.strip())
            continue

        # Paragraph itself is too long — fall back to sentences.
        current = ""
        for sentence in _SENTENCE_END.split(paragraph):
            candidate = f"{current} {sentence}".strip()
            if len(candidate) > max_chars and current:
                pieces.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current.strip():
            pieces.append(current.strip())

    # Recombine adjacent pieces that fit together, so we don't emit a run of
    # tiny fragments just because the paragraph breaks fell awkwardly.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) + 2 <= max_chars:
            merged[-1] = f"{merged[-1]}\n\n{piece}"
        else:
            merged.append(piece)
    return merged


def _heading_path(sections: list[PageSection], index: int) -> tuple[str, ...]:
    """Walk backwards to build the full heading ancestry for a section.

    An h3 under an h2 should carry both. Without the ancestry, "Timeline"
    retrieves identically from every page that has a Timeline section.
    """
    section = sections[index]
    if section.heading is None:
        return ()

    path = [section.heading]
    current_level = section.level
    for previous in reversed(sections[:index]):
        if previous.heading and previous.level < current_level:
            path.append(previous.heading)
            current_level = previous.level
            if current_level <= 1:
                break
    return tuple(reversed(path))


def chunk_page(
    page: Page,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending_text = ""
    pending_path: tuple[str, ...] = ()

    def emit(
        text: str, path: tuple[str, ...], end_heading: str | None = None
    ) -> None:
        if not text.strip():
            return
        for piece in _split_oversized(text.strip(), max_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.id}#{len(chunks)}",
                    page_id=page.id,
                    page_title=page.title,
                    page_url=page.url,
                    heading_path=path,
                    text=piece,
                    space_key=page.space_key,
                    labels=tuple(page.labels),
                    end_heading=end_heading,
                )
            )

    for index, section in enumerate(page.sections):
        path = _heading_path(page.sections, index)
        body = section.text.strip()
        if not body:
            continue

        if pending_text:
            # Previous section was too small to stand alone. Merge forward,
            # keeping the earlier section's path so the chunk still reads as
            # belonging somewhere specific.
            combined = f"{pending_text}\n\n{body}"
            if len(combined) <= max_chars:
                # The merged chunk spans two headings; record both so the
                # citation can say so instead of silently naming only the first.
                emit(
                    combined,
                    pending_path or path,
                    end_heading=section.heading if pending_path else None,
                )
                pending_text, pending_path = "", ()
                continue
            emit(pending_text, pending_path)
            pending_text, pending_path = "", ()

        if len(body) < min_chars:
            pending_text, pending_path = body, path
            continue

        emit(body, path)

    if pending_text:
        emit(pending_text, pending_path)

    return chunks
