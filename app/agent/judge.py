"""Claim-level groundedness, judged by a second model.

`check_grounding` verifies that a citation *resolves*. It cannot verify that the
cited passage *supports* the claim. Live output showed the gap plainly:

    "Formally log the SLA breach, as the contract mechanism exists to apply
     pressure. [5]"

Passage [5] states the SLA terms. It recommends nothing. The citation is valid
and the claim is not supported by it.

    "There is a clear contractual breach."

The source says a commitment was not met. "Breach" is a legal conclusion. At a
regulated insurer, a tool asserting contractual breach in a document that reaches
a board creates a problem that belongs to the VP, not to us.

## Three verdicts, not two

Binary supported/unsupported collapses the distinction that matters here.

- `SUPPORTED` — the passage states this.
- `OVERREACH` — the passage is related and the claim goes beyond it:
  recommendations, legal conclusions, causal attribution, forecasts. This is the
  failure actually observed, and it is invisible to a binary rubric because a
  judge asked "is this supported?" about a plausible inference tends to say yes.
- `UNSUPPORTED` — the passage does not address the claim at all.

Overreach is the interesting category for this client because it is where a
fluent, useful-sounding answer becomes a liability.

## Judging the judge

A model grading a model is not evidence until the grader is itself measured.
`calibrate()` runs the judge against claims with known verdicts and reports
accuracy per category. Judges are systematically lenient — a judge that scores
90% groundedness while agreeing with only 60% of known-bad cases is producing a
number that means nothing.

Run calibration before quoting any groundedness figure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.agent.answering import Passage, _CITATION, _paragraphs
from app.agent.llm import LLM

logger = logging.getLogger(__name__)


# Independent judge calls run concurrently, bounded so a 40-claim answer does
# not open 40 sockets and collect 429s. Measured: 9 sequential calls took 28s of
# a 35s trace. That is the difference between an eval that runs per-commit and
# one that quietly becomes a nightly job — and a nightly safety net catches
# regressions a day late.
MAX_CONCURRENT_JUDGES = 6


# Below this, a fragment is a heading, a list label or a stub rather than an
# assertion. Calibrated against real output: "Acme missed the window" (4 words)
# and "Escalate through vendor management" (4) are both real claims and were
# being silently dropped at 5 — which would have quietly excluded short
# recommendations, exactly the overreach this judge exists to catch.
MIN_CLAIM_WORDS = 4


class Verdict(str, Enum):
    SUPPORTED = "supported"
    OVERREACH = "overreach"
    # Supported by the corpus, but not by the passage the answer cited. A
    # citation-discipline problem, not a truthfulness one.
    MISCITED = "miscited"
    # Not supported by ANY retrieved passage. The dangerous category.
    UNSUPPORTED = "unsupported"
    UNJUDGED = "unjudged"


# Verdicts that route a claim to human review.
PROBLEM_VERDICTS = frozenset(
    {Verdict.OVERREACH, Verdict.MISCITED, Verdict.UNSUPPORTED}
)


JUDGE_PROMPT = """You check whether a single claim is supported by a single \
source passage. You are not being asked whether the claim is true, sensible, or \
useful — only whether this passage states it.

Verdicts:

- "supported": the passage states this claim, or states it in different words.
- "overreach": the claim restates something the passage DOES say, but with added \
strength, conclusion or judgement. The subject matter is the same; the assertion \
is stronger. Example: the passage says "this commitment has not been met" — the \
claim "Acme is in breach of contract" is overreach, because breach is a legal \
conclusion drawn from the same fact.
- "unsupported": the passage is silent on what the claim asserts. This includes \
claims about motive, causation, future outcomes or recommended actions when the \
passage states none of those — silence about a subject is not weak support for \
it. Example: a passage recording that a test was not scheduled says nothing \
about WHY, so "it was skipped due to schedule pressure" is unsupported, not \
overreach.

The boundary: if you must delete part of the claim to reach what the passage \
says, it is overreach. If you must ADD a subject the passage never raises \
(motive, cause, prediction, recommendation), it is unsupported.

`evidence` must be a short quote copied EXACTLY from the passage, character for \
character. For "unsupported", use an empty string.

Default to "overreach" when genuinely torn between supported and overreach. A \
false alarm costs a review; a missed overreach reaches a board.

Return ONLY a JSON object:
{"verdict": "...", "evidence": "...", "reason": "one short sentence"}"""


@dataclass(frozen=True)
class Claim:
    text: str
    citation_indices: tuple[int, ...]


@dataclass(frozen=True)
class JudgedClaim:
    claim: Claim
    verdict: Verdict
    evidence: str = ""
    reason: str = ""
    rejected_reason: str = ""


@dataclass
class GroundednessReport:
    judged: list[JudgedClaim] = field(default_factory=list)

    def _count(self, verdict: Verdict) -> int:
        return sum(1 for j in self.judged if j.verdict is verdict)

    @property
    def total(self) -> int:
        return len(self.judged)

    @property
    def supported_rate(self) -> float:
        return self._count(Verdict.SUPPORTED) / self.total if self.total else 0.0

    @property
    def overreach_rate(self) -> float:
        return self._count(Verdict.OVERREACH) / self.total if self.total else 0.0

    @property
    def miscited_rate(self) -> float:
        return self._count(Verdict.MISCITED) / self.total if self.total else 0.0

    @property
    def unsupported_rate(self) -> float:
        return self._count(Verdict.UNSUPPORTED) / self.total if self.total else 0.0

    @property
    def unjudged_rate(self) -> float:
        """Claims the judge could not rule on.

        Reported explicitly. It was omitted from the summary line at first, so
        the printed percentages did not sum to 100 and part of the measurement
        was invisible. A metric you cannot see is a metric you cannot fix.
        """
        return self._count(Verdict.UNJUDGED) / self.total if self.total else 0.0

    @property
    def traceable_rate(self) -> float:
        """Supported plus miscited: the claim is in the corpus somewhere.

        The complement is what actually worries a regulated client — claims the
        retrieved material does not contain at all.
        """
        if not self.total:
            return 0.0
        return (
            self._count(Verdict.SUPPORTED) + self._count(Verdict.MISCITED)
        ) / self.total

    def problems(self) -> list[JudgedClaim]:
        return [j for j in self.judged if j.verdict in PROBLEM_VERDICTS]

    def summary(self) -> str:
        if not self.total:
            return "no claims judged"
        return (
            f"{self.total} claims   "
            f"supported {self.supported_rate:.0%}   "
            f"miscited {self.miscited_rate:.0%}   "
            f"overreach {self.overreach_rate:.0%}   "
            f"unsupported {self.unsupported_rate:.0%}   "
            f"unjudged {self.unjudged_rate:.0%}"
        )


def extract_claims(answer_text: str) -> list[Claim]:
    """Split an answer into cited claims.

    Works on sentences rather than paragraphs — the point of this check is to
    find the unsupported sentence inside an otherwise well-cited paragraph,
    which is exactly what the paragraph-level check cannot see.

    Uncited sentences are skipped: `check_grounding` already governs those, and
    judging a claim against no passage is meaningless.
    """
    claims: list[Claim] = []

    for paragraph in _paragraphs(answer_text):
        paragraph_citations = tuple(
            int(m.group(1)) for m in _CITATION.finditer(paragraph)
        )
        for raw in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = raw.strip()
            if not sentence:
                continue

            own = tuple(int(m.group(1)) for m in _CITATION.finditer(sentence))
            text = _CITATION.sub("", sentence).strip(" -*|").strip()
            if len(text.split()) < MIN_CLAIM_WORDS:
                continue

            # A sentence with no bracket of its own inherits the paragraph's
            # citations. That inheritance is precisely what paragraph-level
            # enforcement grants, so this measures what that grant permits.
            indices = own or paragraph_citations
            if not indices:
                continue
            claims.append(Claim(text=text, citation_indices=tuple(sorted(set(indices)))))

    return claims


def _normalize(text: str) -> str:
    """Normalise for quote comparison.

    Beyond whitespace, two things had to be handled because they produced false
    fabrication reports on real judge output:

    **A literal backslash-n.** The judge sometimes emits "\\n" inside the JSON
    string rather than a real newline. It is quoting correctly; the escape
    survived one encoding layer too many.

    **Markdown table punctuation.** Quoting across a table renders as
    "planned. | Compliance", where the pipe is layout, not content.

    Six of one hundred claims were downgraded to UNJUDGED over these, which
    pushed `supported` down and looked like a judge quality problem. Stripping
    layout characters keeps the guard honest: every WORD must still appear, in
    order.
    """
    text = text.replace("\\n", " ").replace("\\t", " ")
    text = re.sub(r"[|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_is_present(quote: str, source: str) -> bool:
    """Check a quote against the source, allowing elision.

    A judge quoting across a gap writes "first part... second part". Exact
    matching rejects that, which is normal quoting being treated as fabrication.
    Observed live: two claims were silently downgraded to UNJUDGED over an
    ellipsis.

    Each fragment must still appear verbatim and in order, so the guard still
    catches an invented quote — the relaxation is about the join, not the
    content.
    """
    haystack = _normalize(source)
    fragments = [
        f for f in re.split(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*", quote) if f.strip()
    ]
    if not fragments:
        return False

    cursor = 0
    for fragment in fragments:
        position = haystack.find(_normalize(fragment), cursor)
        if position == -1:
            return False
        cursor = position + len(_normalize(fragment))
    return True


def _parse(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


async def judge_claim(
    llm: LLM, claim: Claim, passages: list[Passage]
) -> JudgedClaim:
    """Judge one claim against only the passages it cites.

    The judge never sees the rest of the answer. Shown the whole thing, a judge
    rates a confident, well-structured answer more generously — the halo effect
    is well documented and this is the cheapest defence against it.
    """
    cited = [
        passages[i - 1] for i in claim.citation_indices if 1 <= i <= len(passages)
    ]
    if not cited:
        return JudgedClaim(
            claim=claim,
            verdict=Verdict.UNSUPPORTED,
            rejected_reason="citation index out of range",
        )

    source = "\n\n".join(f"{p.label}\n{p.text}" for p in cited)

    response = await llm.complete(
        system=JUDGE_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"PASSAGE:\n{source}\n\nCLAIM:\n{claim.text}",
            }
        ],
        max_tokens=300,
        temperature=0.0,
    )

    parsed = _parse(response.text)
    if parsed is None:
        return JudgedClaim(
            claim=claim,
            verdict=Verdict.UNJUDGED,
            rejected_reason="judge returned unparseable output",
        )

    try:
        verdict = Verdict(str(parsed.get("verdict", "")).strip().lower())
    except ValueError:
        return JudgedClaim(
            claim=claim,
            verdict=Verdict.UNJUDGED,
            rejected_reason=f"unknown verdict {parsed.get('verdict')!r}",
        )

    evidence = str(parsed.get("evidence", "")).strip()
    reason = str(parsed.get("reason", "")).strip()

    # Same enforcement as blocker classification: a quote that is not in the
    # passage means the justification was invented, so the verdict resting on it
    # cannot be trusted either. Downgraded rather than discarded — an
    # unverifiable "supported" is not evidence of support.
    if verdict is Verdict.SUPPORTED and evidence:
        if not _quote_is_present(evidence, source):
            logger.warning("judge fabricated evidence: %r", evidence[:60])
            return JudgedClaim(
                claim=claim,
                verdict=Verdict.UNJUDGED,
                reason=reason,
                rejected_reason="judge's supporting quote is not in the passage",
            )

    return JudgedClaim(
        claim=claim, verdict=verdict, evidence=evidence, reason=reason
    )


async def judge_answer(
    llm: LLM,
    answer_text: str,
    passages: list[Passage],
    *,
    detect_miscitation: bool = True,
    concurrency: int | None = None,
) -> GroundednessReport:
    """Judge every cited claim, then separate miscitation from fabrication.

    Measured live, roughly 40% of claims came back not-supported. Inspecting them
    showed three different failures wearing one label:

      - the claim is true and in the corpus, but cites the wrong passage number
      - the claim is stronger than any source ("confirmed breach")
      - the claim is nowhere in the corpus

    Those need three different fixes — citation discipline, an instruction
    against drawing conclusions, and genuine concern respectively — so a single
    number cannot tell you what to do. A second pass against ALL passages
    separates the first from the third.

    Costs one extra judge call per failing claim, which is cheap because it only
    runs on failures.

    Claims are judged concurrently. They are independent — no claim's verdict
    depends on another's — and running them in sequence made judging 80% of a
    live trace's wall time.
    """
    claims = extract_claims(answer_text)
    if not claims:
        return GroundednessReport()

    semaphore = asyncio.Semaphore(concurrency or MAX_CONCURRENT_JUDGES)

    async def judge_one(claim: Claim) -> JudgedClaim:
        async with semaphore:
            judged = await judge_claim(llm, claim, passages)

            if (
                detect_miscitation
                and judged.verdict is Verdict.UNSUPPORTED
                and len(passages) > len(claim.citation_indices)
            ):
                everything = Claim(
                    text=claim.text,
                    citation_indices=tuple(range(1, len(passages) + 1)),
                )
                elsewhere = await judge_claim(llm, everything, passages)
                if elsewhere.verdict is Verdict.SUPPORTED:
                    return JudgedClaim(
                        claim=claim,
                        verdict=Verdict.MISCITED,
                        evidence=elsewhere.evidence,
                        reason=(
                            "Supported by the retrieved corpus but not by the "
                            f"cited passage(s) {list(claim.citation_indices)}."
                        ),
                    )
            return judged

    # gather preserves input order, so report order still matches the answer.
    # Ordering matters here: a reader compares the verdict list against the
    # prose, and a shuffled list makes that impossible.
    judged = await asyncio.gather(*(judge_one(c) for c in claims))
    return GroundednessReport(judged=list(judged))


# ----------------------------------------------------------- judging the judge


@dataclass(frozen=True)
class CalibrationCase:
    claim: str
    passage_text: str
    expected: Verdict
    note: str = ""


@dataclass
class CalibrationReport:
    results: list[tuple[CalibrationCase, Verdict]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Exact label agreement. Useful, but not the operative number."""
        if not self.results:
            return 0.0
        return sum(1 for case, got in self.results if case.expected is got) / len(
            self.results
        )

    @property
    def detection_accuracy(self) -> float:
        """Did the judge flag every claim a source does not support?

        OVERREACH and UNSUPPORTED both land in `problems()`, so both route a
        claim to review. Confusing one for the other changes the wording of a
        report; confusing either for SUPPORTED lets an unsupported claim reach a
        board.

        Calibration showed 79% exact agreement and 100% detection — every
        disagreement was overreach vs unsupported, never a bad claim called
        good. Reporting only exact accuracy would have made a working judge look
        broken.
        """
        if not self.results:
            return 0.0
        return sum(
            1
            for case, got in self.results
            if (case.expected in PROBLEM_VERDICTS) == (got in PROBLEM_VERDICTS)
        ) / len(self.results)

    @property
    def dangerous_misses(self) -> list[tuple["CalibrationCase", Verdict]]:
        """Unsupported claims the judge called supported. The only failures that
        matter operationally."""
        return [
            (case, got)
            for case, got in self.results
            if case.expected in PROBLEM_VERDICTS and got not in PROBLEM_VERDICTS
        ]

    def per_expected(self) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for case, got in self.results:
            buckets.setdefault(case.expected.value, []).append(case.expected is got)
        return {k: sum(v) / len(v) for k, v in buckets.items()}

    def summary(self) -> str:
        lines = [
            f"exact label agreement:  {self.accuracy:.0%} ({len(self.results)} cases)",
            f"problem detection:      {self.detection_accuracy:.0%}  "
            "<- the operative number",
            "",
        ]
        for expected, rate in sorted(self.per_expected().items()):
            lines.append(f"  expected {expected:<12} agreed {rate:.0%}")

        misses = self.dangerous_misses
        if misses:
            lines.append(
                f"\n  {len(misses)} DANGEROUS MISS(ES) — judge called an "
                "unsupported claim supported:"
            )
            for case, got in misses:
                lines.append(f"    {case.claim[:70]}  -> {got.value}")
            lines.append(
                "\n  Fix the rubric before quoting a groundedness score. A lenient "
                "judge\n  produces a number that measures itself."
            )
        else:
            lines.append(
                "\n  No dangerous misses: every unsupported claim was flagged. "
                "Label\n  disagreements between `overreach` and `unsupported` "
                "change report\n  wording, not whether a claim reaches review."
            )
        return "\n".join(lines)


async def calibrate(
    llm: LLM, cases: list[CalibrationCase]
) -> CalibrationReport:
    report = CalibrationReport()
    for case in cases:
        passage = Passage(label="calibration", url="", text=case.passage_text)
        judged = await judge_claim(
            llm, Claim(text=case.claim, citation_indices=(1,)), [passage]
        )
        report.results.append((case, judged.verdict))
    return report
