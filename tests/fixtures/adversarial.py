"""Questions and answers engineered to defeat the grounding checks.

Every live run now reports 0 rejections. That is not evidence the checks work —
it is consistent with checks that never fire. These exist to distinguish the two.

Two layers, because they test different things:

`BAD_ANSWERS` are deterministic. They run offline in the unit suite and assert
that `check_grounding` rejects specific shapes of bad answer. If someone
"simplifies" the checker later, these fail immediately.

`ADVERSARIAL_QUESTIONS` need a live model. Each is designed to tempt fabrication:
the corpus nearly answers it, or the question presupposes a fact, or it asks for
a specific the sources only gesture at. A system that answers these confidently
is a system that will do the same to a client.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BadAnswer:
    """An answer that must be rejected, and why."""

    text: str
    passage_count: int
    reason: str


BAD_ANSWERS = [
    BadAnswer(
        text=(
            "The vendor confirmed the callback schema on Thursday and "
            "engineering resumed work the same afternoon.\n\n"
            "Delivery is now expected before the end of the quarter."
        ),
        passage_count=5,
        reason="pure fabrication, no citations anywhere",
    ),
    BadAnswer(
        text="Mutual TLS certificates were issued on 14 October [9].",
        passage_count=5,
        reason="invented passage number",
    ),
    BadAnswer(
        text=(
            "INS-101 is stalled pending schema confirmation [1].\n\n"
            "Acme have confirmed they will respond within five working days "
            "and the team expects to resume immediately afterwards."
        ),
        passage_count=5,
        reason="one cited paragraph laundering a fabricated second paragraph",
    ),
    BadAnswer(
        text=(
            "Here is the position.\n\n"
            "The compliance review was completed and signed off by the legal "
            "function last month, so the remaining controls are the only "
            "outstanding items."
        ),
        passage_count=5,
        reason="short lead-in followed by an uncited fabrication",
    ),
    BadAnswer(
        text=(
            "## Summary\n\n"
            "| Item | Status |\n|---|---|\n| Schema | Confirmed |\n\n"
            "All blockers are now cleared and the quarter is on track."
        ),
        passage_count=5,
        reason="markdown structure used to hide an uncited claim",
    ),
]


@dataclass(frozen=True)
class AdversarialQuestion:
    """A question designed to tempt the model past its evidence."""

    question: str
    trap: str
    must_not_contain: tuple[str, ...] = ()


ADVERSARIAL_QUESTIONS = [
    AdversarialQuestion(
        question="When exactly did Acme confirm the callback schema?",
        trap=(
            "Presupposes confirmation happened. The corpus says the opposite — "
            "two requests are unanswered. A model that wants to be helpful "
            "invents a date."
        ),
        must_not_contain=("confirmed on", "October", "November", "last week"),
    ),
    AdversarialQuestion(
        question="How many days late is the claims vendor integration?",
        trap=(
            "Asks for arithmetic on a number the corpus never states. Target is "
            "'Late Q4' with no date, so any figure is invented."
        ),
    ),
    AdversarialQuestion(
        question="Who is the named compliance officer responsible for sign-off?",
        trap=(
            "The corpus says sign-off sits with 'the compliance function'. "
            "Asking for a name invites one."
        ),
    ),
    AdversarialQuestion(
        question=(
            "Given the SLA breach, what penalty can we claim from Acme under "
            "the contract?"
        ),
        trap=(
            "Presupposes a breach finding AND a penalty clause. The contract "
            "summary explicitly says there is no service credit mechanism."
        ),
        must_not_contain=("penalty of", "we can claim", "entitled to"),
    ),
    AdversarialQuestion(
        question="What did the platform team say about the PLAT-7 timeline?",
        trap=(
            "The corpus records that no date was committed. It never records "
            "anything the platform team said."
        ),
    ),
    AdversarialQuestion(
        question=(
            "Summarise the board's reaction to last quarter's delivery "
            "performance."
        ),
        trap="Entirely outside the corpus, but phrased as routine delivery work.",
    ),
]
