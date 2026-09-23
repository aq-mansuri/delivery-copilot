"""Does this question ask for a change, or ask about the state of things?

The answer decides one thing only: whether the write tools are put in front of
the model at all. It is deliberately not a model call, and the reason is the
rule this project is built on — if a thing can be computed, computing it wins.

## Why a keyword gate is enough here, when it would not be elsewhere

Blocker classification (`classification.py`) needs a model because the cost of
being wrong is a delivery lead chasing the wrong team. This gate has no such
cost in either direction:

- **Missed request.** Write tools are withheld, the agent answers the question
  as asked, and the lead rephrases. A wasted round-trip.
- **Spurious detection.** A write tool is *offered*. The model still has to
  choose it, anything it produces is a `ProposedAction`, and that still needs a
  named human. The floor is a proposal somebody rejects — noise, not risk.

Nothing here can write to Jira, because `ToolRegistry` holds nothing that can
(ADR-002). This gate only adjusts how much the model is offered.

## It matches the actions that exist, not the idea of change

"Close INS-101" is unmistakably a request, and it returns False, because there
is no `close_issue` proposal for the model to make. Detecting it would offer
tools that cannot serve it, and the user would get a confused refusal instead
of a plain answer about the issue. The gate is kept in step with
`apply_action`'s branches on purpose: add an action there, add its phrasing
here, and add a case to `tests/test_intent.py`.
"""

from __future__ import annotations

import re

# Opening a sentence with one of these makes it a question *about* an action,
# not a request for one. "Why was INS-101 flagged as at risk?" is a question
# with a verb in it. Modals are excluded on purpose: "Can you flag INS-101?"
# and "Should we flag INS-101?" are requests wearing a question mark.
_INTERROGATIVE = re.compile(r"^(who|whom|whose|what|which|why|when|where|how)\b")

_AT_RISK = r"at[-\s]?risk"

# Each pattern corresponds to an action `apply_action` can actually perform.
_WRITE_PATTERNS = (
    # flag_at_risk. `flag` and `escalate` carry the request on their own;
    # `\bflag\b` does not match "flagged", which is what keeps "issues flagged
    # as at risk" out.
    re.compile(r"\bflag\b"),
    re.compile(r"\bescalate\b"),
    re.compile(rf"\b(mark|label|tag|raise)\b.{{0,40}}\b{_AT_RISK}\b"),
    # add_comment.
    re.compile(r"\bcomment\s+on\b"),
    re.compile(r"\b(add|leave|post|write)\b.{0,25}\b(comment|note)\b"),
)


def write_intent(question: str) -> bool:
    """True when the text asks for a change this system can propose."""
    text = question.strip().lower()
    if not text:
        return False
    if _INTERROGATIVE.match(text):
        return False
    return any(pattern.search(text) for pattern in _WRITE_PATTERNS)
