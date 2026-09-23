"""Blocker classification.

The only place in this system where the model does the finding rather than the
explaining — and it earns the exception. Deciding that

    "Legal review is still outstanding — no ETA given."

is a compliance blocker rather than a vendor one requires reading English.
Keyword rules ("legal" → LEGAL_COMPLIANCE) collapse on the first comment that
says "waiting on our legal counsel to hear back from the vendor's legal team",
which is genuinely ambiguous and needs judgement.

Note what is still deterministic: **whether** an issue is blocked comes from the
changelog and links (Day 2's rules). Only the **category** of a blocker already
established is classified here. The model is not being asked to find risk.

## The enforcement, and why it matters more here than in answering

A classifier that returns a category is unverifiable — the output is one enum
value and there is nothing to check it against. So the model must also return a
verbatim quote from the comment, and code verifies that quote actually appears
in the source text. A hallucinated justification becomes a detectable
hallucination rather than a silent misclassification.

This is the same "prompt asks, parser enforces" pattern as `answering.py`, and it
is the reason confidence scores from a model are not used as the safeguard.
Self-reported confidence is a number the model chooses; a quote is a claim about
the world that can be falsified.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.agent.llm import LLM
from app.models.domain import BlockerCategory, Issue

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You classify the cause of a blocked delivery item for an \
insurance company.

Categories:
- vendor: waiting on an external supplier or third-party service
- legal_compliance: waiting on legal review, compliance sign-off, or a regulator
- cross_team: waiting on another internal team inside the same company
- technical: a technical obstacle with no external dependency
- unknown: the text does not say, or is too ambiguous to call

Rules:
1. Choose `unknown` whenever the text does not clearly indicate a cause. Guessing \
is worse than declining — a wrong category sends a delivery lead to chase the \
wrong person.
2. `evidence` must be a short quote copied EXACTLY from the text you were given, \
character for character. Do not paraphrase, summarize, or tidy it up. If no \
sentence supports your choice, use `unknown` with an empty evidence string.
3. Distinguish internal teams from external vendors carefully. "Platform team" is \
cross_team. "Acme" is vendor.

Return ONLY a JSON object, no preamble and no markdown fences:
{"category": "...", "evidence": "...", "reasoning": "one short sentence"}"""


@dataclass(frozen=True)
class BlockerClassification:
    issue_key: str
    category: BlockerCategory
    evidence: str = ""
    reasoning: str = ""
    rejected_reason: str = ""

    @property
    def is_usable(self) -> bool:
        return (
            self.category is not BlockerCategory.UNKNOWN and not self.rejected_reason
        )


def _normalize(text: str) -> str:
    """Whitespace-insensitive comparison for the evidence check.

    Models reliably normalize whitespace when quoting — a newline becomes a
    space, a double space becomes single. Rejecting on that would fail honest
    quotes and teach nothing. Word changes still fail, which is the point.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_json(text: str) -> dict | None:
    """Parse the model's JSON, tolerating fences it was told not to emit.

    Being strict here means a well-classified comment gets thrown away over a
    formatting slip. Being lenient about *format* while strict about *content*
    is the right split.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
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


def blocker_text(issue: Issue) -> str:
    """The text a classification may be drawn from.

    Comments only, newest first — the summary describes the work, not why it is
    stuck, and including it invites the model to classify the task rather than
    the blocker.
    """
    comments = sorted(issue.comments, key=lambda c: c.created_at, reverse=True)
    return "\n\n".join(c.body.strip() for c in comments if c.body.strip())


async def classify_blocker(
    llm: LLM, issue: Issue, *, max_tokens: int = 300
) -> BlockerClassification:
    text = blocker_text(issue)

    if not text:
        # No model call. Nothing to read means nothing to classify, and asking
        # anyway produces a confident category drawn from the issue key.
        return BlockerClassification(
            issue_key=issue.key,
            category=BlockerCategory.UNKNOWN,
            reasoning="No comments on this issue.",
        )

    response = await llm.complete(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Issue {issue.key} comments:\n\n{text}"}],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    parsed = _extract_json(response.text)
    if parsed is None:
        return BlockerClassification(
            issue_key=issue.key,
            category=BlockerCategory.UNKNOWN,
            rejected_reason="model did not return parseable JSON",
        )

    raw_category = str(parsed.get("category", "")).strip().lower()
    try:
        category = BlockerCategory(raw_category)
    except ValueError:
        # The model invented a category outside the enum. Common enough to
        # handle explicitly rather than let an exception escape.
        return BlockerClassification(
            issue_key=issue.key,
            category=BlockerCategory.UNKNOWN,
            rejected_reason=f"unknown category {raw_category!r}",
        )

    evidence = str(parsed.get("evidence", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()

    if category is BlockerCategory.UNKNOWN:
        return BlockerClassification(
            issue_key=issue.key,
            category=category,
            reasoning=reasoning,
        )

    if not evidence:
        return BlockerClassification(
            issue_key=issue.key,
            category=BlockerCategory.UNKNOWN,
            reasoning=reasoning,
            rejected_reason="classification supplied no supporting quote",
        )

    if _normalize(evidence) not in _normalize(text):
        # The quote is not in the source. Downgraded to UNKNOWN rather than
        # trusted — a fabricated justification means the category behind it
        # cannot be relied on either.
        logger.warning(
            "fabricated evidence for %s: %r not found in comments",
            issue.key,
            evidence[:80],
        )
        return BlockerClassification(
            issue_key=issue.key,
            category=BlockerCategory.UNKNOWN,
            reasoning=reasoning,
            rejected_reason="supporting quote does not appear in the source text",
        )

    return BlockerClassification(
        issue_key=issue.key,
        category=category,
        evidence=evidence,
        reasoning=reasoning,
    )


async def classify_all(
    llm: LLM, issues: list[Issue], *, only_blocked: bool = True
) -> list[BlockerClassification]:
    """Classify a set of issues.

    `only_blocked` keeps the model off issues the deterministic rules did not
    flag. Classifying all 3,000 open issues would cost real money to categorize
    blockers that do not exist — the rules decide *what* to look at, the model
    decides *what kind*.
    """
    selected = [
        issue
        for issue in issues
        if not only_blocked
        or any(link.target_blocks_this for link in issue.links)
        or issue.comments
    ]
    return [await classify_blocker(llm, issue) for issue in selected]
