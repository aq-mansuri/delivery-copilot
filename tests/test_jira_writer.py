"""Contract tests for the only code that can change a client's Jira.

Everything else in this project is careful about what it *says*. This is the one
module careful about what it *does*, and it is reached only through
`apply_action`, which needs an `ApprovedAction`, which needs a named human.

The tests assert request shapes rather than hitting a tenant, because the things
that break here — an ADF body Jira silently rejects, a `set` where an `add`
belonged — are invisible until they have already overwritten someone's data.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.integrations.atlassian.client import AtlassianError
from app.integrations.atlassian.writer import JiraIssueWriter

BASE_URL = "https://acme.atlassian.net"
ISSUE_URL = f"{BASE_URL}/rest/api/3/issue/INS-1"


@pytest.fixture
def writer() -> JiraIssueWriter:
    return JiraIssueWriter(
        base_url=BASE_URL, email="svc@acme.com", api_token="token"
    )


class TestItDeclaresItself:
    def test_it_says_it_writes_to_jira(self, writer):
        """Read by `apply_action` and printed on every audit row. A real writer
        that failed to declare itself would record real changes as a dry run."""
        assert writer.writes_to_jira is True


class TestLabels:
    @respx.mock
    async def test_labels_are_added_not_assigned(self, writer):
        """`update.labels[].add`, never `fields.labels`.

        Setting `fields` replaces the whole array, so flagging one issue
        at-risk would silently delete every other label a team had put on it —
        a destructive edit that looks like a successful one, on a ticket
        somebody approved a *label addition* for.
        """
        route = respx.put(ISSUE_URL).mock(return_value=httpx.Response(204))
        await writer.add_labels("INS-1", ["at-risk"])

        body = route.calls.last.request.read().decode()
        assert '"update"' in body
        assert '"add": "at-risk"' in body.replace('"add":"at-risk"', '"add": "at-risk"')
        assert '"fields"' not in body

    @respx.mock
    async def test_every_label_is_sent(self, writer):
        route = respx.put(ISSUE_URL).mock(return_value=httpx.Response(204))
        await writer.add_labels("INS-1", ["at-risk", "delivery-review"])
        body = route.calls.last.request.read().decode()
        assert "at-risk" in body and "delivery-review" in body

    @respx.mock
    async def test_an_empty_label_list_makes_no_call(self, writer):
        """A no-op request still consumes rate limit and still appears in a
        client's audit log as a modification."""
        route = respx.put(ISSUE_URL).mock(return_value=httpx.Response(204))
        await writer.add_labels("INS-1", [])
        assert not route.called


class TestComments:
    @respx.mock
    async def test_the_comment_body_is_adf(self, writer):
        """Jira Cloud's v3 comment endpoint takes Atlassian Document Format.
        Posting a plain string 400s, and the message names `body` without
        saying why."""
        route = respx.post(f"{ISSUE_URL}/comment").mock(
            return_value=httpx.Response(201, json={"id": "10001"})
        )
        await writer.add_comment("INS-1", "Flagged by Delivery Copilot.")

        payload = route.calls.last.request.read().decode()
        assert '"type": "doc"' in payload or '"type":"doc"' in payload
        assert '"version": 1' in payload or '"version":1' in payload
        assert "Flagged by Delivery Copilot." in payload

    @respx.mock
    async def test_paragraphs_survive_as_separate_blocks(self, writer):
        route = respx.post(f"{ISSUE_URL}/comment").mock(
            return_value=httpx.Response(201, json={"id": "1"})
        )
        await writer.add_comment("INS-1", "First line.\n\nSecond line.")
        payload = route.calls.last.request.read().decode()
        assert payload.count('"paragraph"') == 2

    @respx.mock
    async def test_an_empty_comment_makes_no_call(self, writer):
        route = respx.post(f"{ISSUE_URL}/comment").mock(
            return_value=httpx.Response(201, json={"id": "1"})
        )
        await writer.add_comment("INS-1", "   ")
        assert not route.called


class TestErrorsSayWhatToDo:
    @respx.mock
    async def test_a_404_names_the_issue(self, writer):
        respx.put(ISSUE_URL).mock(
            return_value=httpx.Response(
                404, json={"errorMessages": ["Issue does not exist."], "errors": {}}
            )
        )
        with pytest.raises(AtlassianError) as exc:
            await writer.add_labels("INS-1", ["at-risk"])
        assert "INS-1" in str(exc.value)

    @respx.mock
    async def test_a_403_names_the_missing_permission(self, writer):
        respx.put(ISSUE_URL).mock(
            return_value=httpx.Response(
                403, json={"errorMessages": ["You do not have permission"], "errors": {}}
            )
        )
        with pytest.raises(AtlassianError) as exc:
            await writer.add_labels("INS-1", ["at-risk"])
        assert "Edit Issues" in str(exc.value)

    @respx.mock
    async def test_a_400_surfaces_jiras_own_field_errors(self, writer):
        """`raise_for_status()` throws the body away, and the body is the only
        place Jira says which field it objected to."""
        respx.put(ISSUE_URL).mock(
            return_value=httpx.Response(
                400,
                json={"errorMessages": [], "errors": {"labels": "cannot contain spaces"}},
            )
        )
        with pytest.raises(AtlassianError) as exc:
            await writer.add_labels("INS-1", ["at risk"])
        assert "cannot contain spaces" in str(exc.value)

    @respx.mock
    async def test_a_failed_write_is_raised_not_swallowed(self, writer):
        """`apply_action` records the error against the audit row and re-raises.
        A writer that returned quietly would produce an audit trail claiming a
        change that never landed."""
        respx.post(f"{ISSUE_URL}/comment").mock(
            return_value=httpx.Response(500, text="upstream boom")
        )
        with pytest.raises(AtlassianError):
            await writer.add_comment("INS-1", "hello")


class TestItIsOnlyReachableThroughApproval:
    async def test_the_registry_still_holds_no_writer(self):
        """The security property this whole design rests on. Adding a real
        writer must not put one anywhere the model can reach."""
        from app.agent.tools import ToolRegistry

        registry = ToolRegistry()
        assert not hasattr(registry, "writer")
        assert not any(
            "writer" in name or "jira" in name.lower()
            for name in vars(registry)
        )

    async def test_apply_action_still_refuses_an_unapproved_proposal(self, writer):
        from app.agent.approval import ApprovalError, ApprovalStore, apply_action
        from app.models.domain import Citation, ProposedAction

        proposal = ProposedAction(
            issue_key="INS-1", action_type="flag_at_risk",
            payload={"labels_add": ["at-risk"]}, rationale="stale",
            citations=[Citation(source_type="risk_rule", source_id="r", url="#r")],
        )
        with pytest.raises(ApprovalError):
            await apply_action(proposal, writer, ApprovalStore())  # type: ignore[arg-type]
