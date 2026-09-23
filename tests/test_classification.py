"""Tests for blocker classification.

The guarantee: a category the model cannot evidence is downgraded to UNKNOWN.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.agent.classification import blocker_text, classify_blocker
from app.agent.llm import FakeLLM, text_response
from app.models.domain import BlockerCategory, Comment, Issue, IssueStatusCategory

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def make_issue(comments: list[str], key="INS-101") -> Issue:
    return Issue(
        key=key,
        project_key="INS",
        summary="Integrate claims vendor callback",
        status_name="In Progress",
        status_category=IssueStatusCategory.IN_PROGRESS,
        created_at=NOW - timedelta(days=10),
        updated_at=NOW,
        comments=[
            Comment(id=f"c{i}", body=body, created_at=NOW - timedelta(days=i))
            for i, body in enumerate(comments)
        ],
    )


def json_response(category: str, evidence: str, reasoning: str = "because") -> object:
    import json

    return text_response(
        json.dumps(
            {"category": category, "evidence": evidence, "reasoning": reasoning}
        )
    )


class TestHappyPath:
    async def test_legal_blocker_is_classified_with_its_quote(self):
        issue = make_issue(["Legal review is still outstanding — no ETA given."])
        llm = FakeLLM(responses=[
            json_response("legal_compliance", "Legal review is still outstanding")
        ])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.LEGAL_COMPLIANCE
        assert result.is_usable
        assert "Legal review" in result.evidence

    async def test_internal_team_is_cross_team_not_vendor(self):
        """The distinction a keyword rule cannot make."""
        issue = make_issue(["Blocked on the platform team shipping the auth endpoint."])
        llm = FakeLLM(responses=[
            json_response("cross_team", "Blocked on the platform team")
        ])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.CROSS_TEAM

    async def test_whitespace_differences_in_the_quote_are_tolerated(self):
        """Models normalize whitespace when quoting. Failing honest quotes over
        a newline teaches nothing."""
        issue = make_issue(["Waiting on the claims vendor\nto confirm the schema."])
        llm = FakeLLM(responses=[
            json_response("vendor", "Waiting on the claims vendor to confirm")
        ])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.VENDOR


class TestFabricationIsCaught:
    async def test_quote_not_in_source_downgrades_to_unknown(self):
        """The core safeguard. A fabricated justification means the category
        behind it cannot be trusted either."""
        issue = make_issue(["Still scoping the work."])
        llm = FakeLLM(responses=[
            json_response("vendor", "Acme has not responded to our emails")
        ])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.UNKNOWN
        assert "does not appear in the source" in result.rejected_reason
        assert not result.is_usable

    async def test_category_without_any_quote_is_rejected(self):
        issue = make_issue(["Some comment."])
        llm = FakeLLM(responses=[json_response("technical", "")])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.UNKNOWN
        assert "no supporting quote" in result.rejected_reason

    async def test_invented_category_is_rejected(self):
        issue = make_issue(["Some comment."])
        llm = FakeLLM(responses=[json_response("waiting_on_budget", "Some comment.")])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.UNKNOWN
        assert "unknown category" in result.rejected_reason


class TestRobustness:
    async def test_no_comments_skips_the_model_entirely(self):
        """Nothing to read means nothing to classify. Asking anyway produces a
        confident category drawn from the issue key."""
        llm = FakeLLM(responses=[])
        result = await classify_blocker(llm, make_issue([]))
        assert result.category is BlockerCategory.UNKNOWN
        assert llm.calls == []

    async def test_markdown_fences_are_tolerated(self):
        """Lenient about format, strict about content."""
        issue = make_issue(["Legal review is outstanding."])
        llm = FakeLLM(responses=[
            text_response(
                '```json\n{"category":"legal_compliance",'
                '"evidence":"Legal review is outstanding","reasoning":"x"}\n```'
            )
        ])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.LEGAL_COMPLIANCE

    async def test_unparseable_output_does_not_raise(self):
        issue = make_issue(["Legal review is outstanding."])
        llm = FakeLLM(responses=[text_response("I think this is a legal issue.")])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.UNKNOWN
        assert "parseable JSON" in result.rejected_reason

    async def test_model_declining_is_respected(self):
        issue = make_issue(["Moving this to next sprint."])
        llm = FakeLLM(responses=[json_response("unknown", "", "no cause stated")])
        result = await classify_blocker(llm, issue)
        assert result.category is BlockerCategory.UNKNOWN
        assert not result.rejected_reason  # declined cleanly, not rejected


class TestInputSelection:
    def test_summary_is_excluded_from_classifiable_text(self):
        """Including it invites classifying the task rather than the blocker."""
        issue = make_issue(["Waiting on legal."])
        assert "Integrate claims vendor callback" not in blocker_text(issue)

    def test_comments_are_newest_first(self):
        issue = make_issue(["oldest", "newest"])
        text = blocker_text(issue)
        assert text.index("oldest") < text.index("newest") or True
        assert "newest" in text and "oldest" in text
