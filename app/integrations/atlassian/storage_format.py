"""Parser for Confluence storage format.

Storage format is not HTML. It is XHTML with Confluence's own namespaced
elements mixed in — `ac:structured-macro`, `ac:plain-text-body`, `ri:attachment`
— and a naive tag strip produces text full of macro parameter values, table of
contents placeholders, and Jira issue keys pulled out of context.

Two decisions here, both aimed at Day 3 retrieval quality:

1. **Output is heading-delimited sections, not one text blob.** Headings are the
   document's own structure, which makes them the least arbitrary chunk boundary
   available. Splitting on character counts severs sentences from the heading
   that gives them meaning.

2. **Navigational macros are dropped; prose macros are kept.** A `toc` macro
   contributes nothing but retrieves for everything. An `info` or `note` panel
   is usually where the actual caveat lives — exactly the content someone needs
   when asking why a decision was made.

Written against stdlib HTMLParser rather than lxml: no compiled dependency, and
the input is trusted tenant content rather than adversarial markup.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from app.models.domain import PageSection

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Macros that carry prose worth retrieving.
_PROSE_MACROS = {"info", "note", "warning", "tip", "panel", "expand", "excerpt"}

# Macros that are navigation or rendering furniture. Their text is either
# duplicated elsewhere or meaningless out of context.
_SKIP_MACROS = {
    "toc",
    "children",
    "pagetree",
    "recently-updated",
    "contentbylabel",
    "include",
    "excerpt-include",
    "anchor",
    "gallery",
    "attachments",
}

# Block-level elements that should produce a line break in the extracted text.
_BLOCK_TAGS = {
    "p", "div", "li", "tr", "br", "blockquote",
    "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
}

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class _StorageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[PageSection] = []

        self._heading: str | None = None
        self._level: int = 0
        self._buffer: list[str] = []

        self._in_heading = False
        self._heading_buffer: list[str] = []

        # Macro handling. Depth counter rather than a boolean: macros nest, and
        # a boolean gets cleared by the inner macro's close tag.
        self._skip_depth = 0
        self._macro_stack: list[str] = []
        self._in_macro_parameter = False

    # -- section boundaries -------------------------------------------------

    def _flush_section(self) -> None:
        text = self._clean("".join(self._buffer))
        if self._heading is not None or text:
            section = PageSection(
                heading=self._heading, level=self._level, text=text
            )
            if not section.is_empty() or section.heading:
                self.sections.append(section)
        self._buffer = []

    @staticmethod
    def _clean(raw: str) -> str:
        lines = [_WHITESPACE.sub(" ", line).strip() for line in raw.split("\n")]
        return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()

    # -- HTMLParser hooks ---------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)

        if tag == "ac:structured-macro":
            name = (attributes.get("ac:name") or "").lower()
            self._macro_stack.append(name)
            if name in _SKIP_MACROS or (
                name not in _PROSE_MACROS and name != "code"
            ):
                # Unknown macros are skipped rather than included. Confluence
                # tenants install dozens of third-party macros whose rendered
                # output is markup, not prose; including them by default
                # poisons retrieval with noise.
                self._skip_depth += 1
            return

        if tag == "ac:parameter":
            # Parameter values are config ("border=true", "colour=red"), never
            # content.
            self._in_macro_parameter = True
            return

        if self._skip_depth or self._in_macro_parameter:
            return

        if tag in _HEADING_TAGS:
            self._flush_section()
            self._in_heading = True
            self._heading_buffer = []
            self._level = int(tag[1])
            return

        if tag == "ri:attachment":
            filename = attributes.get("ri:filename")
            if filename:
                self._buffer.append(f"[attachment: {filename}]")
            return

        if tag in _BLOCK_TAGS:
            self._buffer.append("\n")
        elif tag in {"td", "th"}:
            # Cells joined with a separator so a row stays one readable line
            # instead of collapsing into an ambiguous run of words.
            self._buffer.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "ac:structured-macro":
            name = self._macro_stack.pop() if self._macro_stack else ""
            if name in _SKIP_MACROS or (
                name not in _PROSE_MACROS and name != "code"
            ):
                self._skip_depth = max(0, self._skip_depth - 1)
            return

        if tag == "ac:parameter":
            self._in_macro_parameter = False
            return

        if self._skip_depth:
            return

        if tag in _HEADING_TAGS:
            self._heading = self._clean("".join(self._heading_buffer)) or None
            self._in_heading = False
            return

        if tag in _BLOCK_TAGS:
            self._buffer.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._in_macro_parameter:
            return
        if self._in_heading:
            self._heading_buffer.append(data)
        else:
            self._buffer.append(data)

    def unknown_decl(self, data: str) -> None:
        """Code-block bodies arrive as CDATA, which HTMLParser reports here.

        Without this, every code block silently vanishes — and in an
        architecture space, the code block is often the decision.
        """
        if self._skip_depth or self._in_macro_parameter:
            return
        if data.startswith("CDATA["):
            self._buffer.append("\n" + data[len("CDATA["):] + "\n")

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._flush_section()


def parse_storage(storage: str) -> list[PageSection]:
    """Turn Confluence storage format into heading-delimited sections."""
    if not storage:
        return []
    parser = _StorageParser()
    parser.feed(storage)
    parser.close()
    return [s for s in parser.sections if not s.is_empty() or s.heading]
