"""Tests for write-intent detection.

The property under test: the gate is conservative in the direction that
matters. A missed request costs a round-trip; a spurious one costs a proposal
somebody has to reject. Neither can write to Jira.
"""

from __future__ import annotations

import pytest

from app.agent.intent import write_intent


class TestRequests:
    """Phrasings a delivery lead actually types."""

    @pytest.mark.parametrize(
        "question",
        [
            "Flag INS-101 as at risk",
            "flag ins-101 as at-risk",
            "Please flag INS-101 as at risk.",
            "Mark INS-101 as at risk",
            "Can you flag INS-101 as at risk?",
            "Should we flag INS-101 as at risk?",
            "Label INS-103 at risk",
            "Escalate INS-102",
            "Add a comment to INS-101 saying the vendor has not replied",
            "Comment on INS-102 that the platform team was chased",
            "Raise INS-103 as at risk before the review",
        ],
    )
    def test_detected(self, question):
        assert write_intent(question) is True


class TestQuestions:
    """Questions about state. Offering a write tool here produces proposals
    nobody asked for."""

    @pytest.mark.parametrize(
        "question",
        [
            "What is at risk right now?",
            "Which controls are still outstanding before go-live?",
            "What work is holding up the release?",
            "Who approved the Acme security exception?",
            "Why was INS-101 flagged as at risk?",
            "Which issues are flagged as at risk?",
            "When did INS-101 last move?",
            "Is INS-102 blocked?",
            "How many issues carried over?",
            "Tell me about the release plan",
            "Summarise the delivery risks",
        ],
    )
    def test_not_detected(self, question):
        assert write_intent(question) is False


class TestBoundaries:
    def test_bare_verb_without_an_action_this_system_has_is_not_write_intent(self):
        """The gate matches the actions that exist, not the idea of change.

        "Close INS-101" is a request, but nothing here can propose a status
        transition. Detecting it would offer tools that cannot serve it and
        produce a confusing refusal instead of a plain answer.
        """
        assert write_intent("Close INS-101") is False
        assert write_intent("Reassign INS-101 to Priya") is False

    def test_empty_and_whitespace_are_safe(self):
        assert write_intent("") is False
        assert write_intent("   ") is False

    def test_interrogative_opening_about_a_past_flag_is_not_a_request(self):
        assert write_intent("Who flagged INS-101 as at risk?") is False

    def test_punctuation_and_case_do_not_matter(self):
        assert write_intent("FLAG INS-101 AS AT RISK!!!") is True
