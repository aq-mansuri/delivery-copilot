"""Grounded answering.

This module discharges ADR-006: refusal is a generation-time obligation, because
retrieval cannot distinguish an absent fact from a present topic.

The load-bearing idea is that **the prompt asks, the parser enforces**. A prompt
saying "only use the provided context" is a request the model usually honours.
That is not good enough for a regulated insurer, so every citation the model
emits is checked against the chunks actually retrieved, and an answer that makes
claims without resolvable citations is rejected by code rather than trusted.

Three failure modes are handled separately, because they need different
responses:

- **No context retrieved** — decline without calling the model at all. Cheaper,
  faster, and impossible to get wrong.
- **Context retrieved but it does not answer** — the model declines. This is the
  case retrieval cannot detect, and the one the eval set's negatives target.
- **Model answers but cites nothing, or cites out of range** — code rejects the
  answer. This is confabulation caught after the fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agent.llm import LLM
from app.rag.retrieval import ScoredChunk


@dataclass(frozen=True)
class Passage:
    """One numbered piece of evidence the answer may be built from.

    Introduced because retrieval chunks were not the only evidence source. Tool
    results — risk findings especially — are evidence too, and citing them
    against a retrieval chunk produced answers whose [1] pointed at an unrelated
    Confluence page.

    A uniform passage type means the numbering the model sees and the sources
    resolved afterwards come from the same list, which is the only way the
    mapping can be correct by construction rather than by luck.
    """

    label: str
    url: str
    text: str
    found_by: str = ""

    @classmethod
    def from_hit(cls, hit: ScoredChunk) -> "Passage":
        return cls(
            label=hit.chunk.citation_label(),
            url=hit.chunk.page_url,
            text=hit.chunk.text,
            found_by=hit.retrieval_reason(),
        )

    @classmethod
    def from_tool(
        cls, tool_name: str, content: str, url: str = "", label: str = ""
    ) -> "Passage":
        """Evidence produced by a tool.

        `label` is what the reader sees beside the citation, so it comes from
        the tool rather than being derived from its name. The derived version
        said "(computed from Jira)" for every tool, which was true of the risk
        engine and false of the other two — a proposal the model had just
        drafted appeared in the evidence rail as though Jira had reported it.
        """
        return cls(
            label=label or tool_name,
            url=url,
            text=content,
            found_by="tool",
        )

# The model is told to cite as [1], [2]. Anything else is a parse failure, not
# a style difference — a free-form citation cannot be checked against a source.
_CITATION = re.compile(r"\[(\d+)\]")

REFUSAL_MARKER = "INSUFFICIENT_EVIDENCE"

# A lead-in longer than this is making its own claims, not introducing others'.
# Calibrated from real output: observed framing lines run 10-18 words.
LEAD_IN_MAX_WORDS = 25

# Phrases that mean "I am declining", used only as a fallback when the model
# explains a false premise well but forgets the marker. Detected declines are
# recorded distinctly from marker declines so the eval can see how often the
# contract is being missed — a fallback that hides its own use is a fallback
# that never gets fixed.
#
# Every phrase must refer to the EVIDENCE, not merely contain a negation. An
# earlier version matched a bare "there is no", which fired on the fabrication
# "There is no doubt that Acme confirmed the schema on Thursday" and filed it as
# a polite decline. The user never saw it either way, but refusal_rate is a
# baselined metric and a fabrication counted as a refusal corrupts it.
#
# Found by an adversarial test, not by reading the regex.
_PROSE_REFUSAL = re.compile(
    r"(?:"
    r"presupposes"
    r"|(?:context|passages?|sources?|documentation|material)\s+(?:do(?:es)?\s+not|"
    r"provide[sd]?\s+no|contain[s]?\s+no)"
    r"|do(?:es)?\s+not\s+(?:support|contain|address|mention)\s+(?:this|that|the)"
    r"|no\s+(?:evidence|record|mention|information)\s+(?:of|for|in|that)"
    r"|not\s+enough\s+information"
    r"|cannot\s+(?:find|determine|answer)"
    r"|no\s+such\s+(?:exception|record|item|approval)\s"
    r")",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You answer delivery questions for an insurance company using \
ONLY the numbered context passages provided.

Rules, in priority order:

1. If the context does not contain the information needed, reply with exactly \
this and nothing else:
{refusal}

2. Never infer, estimate, or extrapolate. If the question asks who approved \
something and no approval is recorded, that is INSUFFICIENT_EVIDENCE — not an \
invitation to name whoever seems most likely.

3. Every factual claim must carry a citation in square brackets referencing the \
passage it came from, like [1] or [2][3]. Cite at least once per paragraph, and \
prefer citing each sentence that states a fact. A paragraph with no citation \
will be rejected.

4. If the question presupposes something the context does not support, explain \
which premise fails AND still include the exact refusal line from rule 1. The \
explanation is welcome; the marker is what tells the system a decline happened, \
and without it a correct refusal is recorded as a failure.

5. Use the source's own strength of language. If a passage says work "cannot \
progress at present", do not write "blocked". If it says a commitment "has not \
been met", do not write "breach" — that is a legal conclusion the source does \
not draw. Never upgrade "untouched" to "abandoned", "pending" to "stalled", or \
a stated fact into a characterisation.

6. Report what the sources say. Do not draw conclusions from them, do not \
explain why something happened unless the source explains it, and do not \
recommend actions. No "recommended actions" section. A delivery lead decides \
what to do; your job is to give them an accurate picture to decide from.

7. Be concise. This is read by delivery leads before an executive review.

You are not being asked to be helpful at the cost of being wrong. Declining is \
a correct answer and is preferred over a plausible one.

The most common failure here is not invention. It is a helpful assistant being \
helpful — sharpening language, drawing the obvious conclusion, suggesting the \
next step. In a regulated insurer those additions become the reader's problem \
when a board asks where a claim came from. Restraint is the requirement."""


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    citations: tuple[Passage, ...] = ()
    refused: bool = False
    refusal_reason: str = ""
    rejected_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def answered(self) -> bool:
        return not self.refused and not self.rejected_reason

    def sources(self) -> list[dict[str, str]]:
        return [
            {"label": p.label, "url": p.url, "found_by": p.found_by}
            for p in self.citations
        ]


@dataclass
class GroundingReport:
    """Per-answer grounding detail, for the Day 5 evals and for tracing."""

    cited_indices: set[int] = field(default_factory=set)
    out_of_range: set[int] = field(default_factory=set)
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return not self.out_of_range and not self.uncited_sentences


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def build_context(passages: list[Passage]) -> str:
    """Number the passages so citations are checkable.

    The source label and URL go in the block too. Without them the model can
    cite [2] but has no way to name what [2] is, and a citation the reader
    cannot follow is not a citation.
    """
    blocks = []
    for index, passage in enumerate(passages, start=1):
        source = f"Source: {passage.url}\n" if passage.url else ""
        blocks.append(f"[{index}] {passage.label}\n{source}{passage.text}")
    return "\n\n---\n\n".join(blocks)


# Markdown the model emits as structure, not as claims. Headings, table rows,
# rules and bare list markers carry no assertion of their own — the claim is in
# the cell or the bullet text, which is checked separately.
_STRUCTURAL = re.compile(
    r"^(?:#{1,6}\s|\|.*\||-{3,}$|\*{2,}$|\d+\.$|[-*+]$|\*\*[^*]+:\*\*$)"
)


def _split_sentences(text: str) -> list[str]:
    r"""Split into sentences, keeping a trailing citation with its own sentence.

    The naive `split on (?<=[.!?])\s+` is wrong here, and it was wrong in a way
    that made the whole grounding check reject good answers.

    Models commonly write the citation AFTER the full stop:

        ...runs in parallel. [3]

    Split naively and "[3]" begins the NEXT sentence. Sentence N loses the
    citation it earned; sentence N+1 inherits one it did not. Every rejection in
    the first live run was this single off-by-one — the checker was not too
    strict, it was misattributing.

    So after splitting, a fragment that opens with citations has them pulled
    back onto the preceding sentence.
    """
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    merged: list[str] = []
    for fragment in raw:
        leading = re.match(r"^((?:\[\d+\]\s*)+)(.*)$", fragment, re.DOTALL)
        if leading and merged:
            merged[-1] = f"{merged[-1]} {leading.group(1).strip()}"
            remainder = leading.group(2).strip()
            if remainder:
                merged.append(remainder)
        else:
            merged.append(fragment)
    return merged


def _claim_lines(sentence: str) -> list[str]:
    """Lines within a sentence that actually assert something.

    A "sentence" spanning a markdown table or heading block is not one claim; it
    is structure plus claims. Checking the whole blob for a bracket passes
    tables with one stray citation and fails headings that need none.
    """
    return [
        line.strip()
        for line in sentence.splitlines()
        if line.strip() and not _STRUCTURAL.match(line.strip())
    ]


def check_grounding(
    answer_text: str, passage_count: int, *, per_sentence: bool = False
) -> GroundingReport:
    """Verify every claim traces to a real passage.

    Two distinct failures. An out-of-range citation ([7] when five passages were
    supplied) means the model invented a source — the more alarming of the two.
    An uncited claim means an assertion with no traceable origin — the more
    common.

    ## Granularity, and why the default is the paragraph

    `per_sentence=True` demands a citation on every sentence. Measured against a
    real model it rejected 3 of 4 answerable questions, because models cite at
    the end of a paragraph: three sentences of connected reasoning, then [3].
    That is a normal, readable style, and rejecting it produced a system with a
    100% refusal rate — which is not safety, it is a false negative wearing a
    safety jacket.

    The default therefore requires at least one resolvable citation per
    paragraph that makes claims.

    Be honest about what this buys and what it costs. A paragraph carrying [3]
    is traceable: a lead can click through and check it. But a paragraph with
    one real citation could carry three sentences the source does not support,
    and this check would pass it. The stricter mode is retained for evals, where
    the tradeoff is worth measuring rather than assuming.
    """
    report = GroundingReport()

    for match in _CITATION.finditer(answer_text):
        index = int(match.group(1))
        report.cited_indices.add(index)
        if index < 1 or index > passage_count:
            report.out_of_range.add(index)

    units = (
        _split_sentences(answer_text)
        if per_sentence
        else _paragraphs(answer_text)
    )

    for position, unit in enumerate(units):
        if _CITATION.search(unit):
            continue

        # A short uncited paragraph followed by cited content is a lead-in:
        # "Based on the context, two of the three items are blocked." Everything
        # after it carries the evidence. Rejecting it makes the model open with
        # a bracket, which reads worse and grounds nothing extra.
        if (
            not per_sentence
            and len(unit.split()) <= LEAD_IN_MAX_WORDS
            and any(_CITATION.search(later) for later in units[position + 1 :])
        ):
            continue

        claims = _claim_lines(unit)
        if not claims:
            # Pure structure: a heading, a rule, a table row, a list marker.
            continue

        prose = " ".join(claims)
        if len(prose.split()) <= 6:
            continue
        if prose.rstrip().endswith(":"):
            # A lead-in whose claim lives in the lines that follow.
            continue

        report.uncited_sentences.append(unit)

    return report


async def answer_question(
    llm: LLM,
    question: str,
    evidence: list[ScoredChunk] | list[Passage],
    *,
    system_prompt: str | None = None,
    max_tokens: int = 1000,
) -> GroundedAnswer:
    passages: list[Passage] = [
        item if isinstance(item, Passage) else Passage.from_hit(item)
        for item in evidence
    ]

    if not passages:
        # No model call. Nothing retrieved means nothing to ground an answer in,
        # and asking anyway is paying for a confabulation.
        return GroundedAnswer(
            text="I could not find anything in Jira or Confluence that addresses "
            "this.",
            refused=True,
            refusal_reason="no_context_retrieved",
        )

    system = (system_prompt or SYSTEM_PROMPT).format(refusal=REFUSAL_MARKER)
    context = build_context(passages)

    response = await llm.complete(
        system=system,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\n"
                f"Question: {question}",
            }
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    text = response.text.strip()

    if REFUSAL_MARKER in text:
        return GroundedAnswer(
            text="The available Jira and Confluence content does not contain "
            "enough information to answer this.",
            refused=True,
            refusal_reason="model_declined",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    report = check_grounding(text, passage_count=len(passages))

    # Fallback: the model declined in prose without the marker. Observed live —
    # it named the false premise, which is better than the protocol required,
    # then fell through to the grounding check and was logged as a failure.
    # A correct decline recorded as a grounding failure corrupts the refusal
    # metrics this project is about to start measuring.
    if not report.is_grounded and _PROSE_REFUSAL.search(text):
        return GroundedAnswer(
            text=text,
            citations=tuple(
                passages[i - 1]
                for i in sorted(report.cited_indices)
                if 1 <= i <= len(passages)
            ),
            refused=True,
            refusal_reason="model_declined_without_marker",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    if not report.is_grounded:
        # Rejected rather than returned with a warning. A warned answer still
        # gets read, and the person reading it is about to brief a board.
        detail = []
        if report.out_of_range:
            detail.append(f"cited non-existent passages {sorted(report.out_of_range)}")
        if report.uncited_sentences:
            detail.append(f"{len(report.uncited_sentences)} uncited claim(s)")
        return GroundedAnswer(
            text="I could not produce an answer I can fully attribute to the "
            "source material.",
            refused=True,
            refusal_reason="failed_grounding_check",
            rejected_reason="; ".join(detail),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    cited = tuple(
        passages[index - 1]
        for index in sorted(report.cited_indices)
        if 1 <= index <= len(passages)
    )

    return GroundedAnswer(
        text=text,
        citations=cited,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
