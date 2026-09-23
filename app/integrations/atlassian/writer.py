"""The only code in this project that changes a client's Jira.

Deliberately its own module, and deliberately not part of `JiraClient`. The
reader is constructed everywhere — the sync, the connection check, three
scripts. If it also carried `add_labels`, every one of those call sites would be
holding something that can mutate a tenant, and the guarantee in ADR-002 would
be a convention instead of a fact.

`ToolRegistry` still holds nothing that can write. Nothing the model can reach
imports this module. The one path in is `apply_action`, which takes an
`ApprovedAction`, which `ApprovalStore.approve` will not produce without a named
human.

## It is not constructed unless someone asks for it

`build_services` injects this only when `JIRA_ALLOW_WRITES` is set *and* Jira is
configured. The default stays `LoggingWriter`, because a service that can write
to a client's tenant the moment it boots is a service that writes to a client's
tenant by accident.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.integrations.atlassian.client import AtlassianError, _explain

logger = logging.getLogger(__name__)


def to_adf(text: str) -> dict[str, Any]:
    """Wrap plain text as an Atlassian Document Format document.

    `/rest/api/3/issue/{key}/comment` takes ADF, not a string. Posting a string
    returns a 400 naming `body` and nothing else, which reads like the comment
    was malformed rather than the wrong shape entirely — the v2 endpoint took a
    string, so every example older than a couple of years is wrong here.

    Blank-line-separated blocks become separate paragraphs. Joining them into
    one would run the rationale into the attribution line, which is the part a
    reader scans for.
    """
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": paragraph}],
            }
            for paragraph in paragraphs
        ],
    }


class JiraIssueWriter:
    """Applies an approved change to a real tenant.

    Implements the `IssueWriter` protocol in `app/agent/approval.py`.
    """

    #: What the audit trail records. True here, and only here.
    writes_to_jira = True

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=(email, api_token),
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _check(self, resp: httpx.Response, issue_key: str, what: str) -> None:
        """Turn a failure into a message an operator can act on.

        No `raise_for_status()`. Atlassian puts the actual cause in
        `errorMessages[]` and a field-level `errors{}` object, and the helper
        discards the body — leaving a bare status code for a problem Jira has
        already described in a sentence.

        Errors are raised rather than returned. `apply_action` records the
        failure against the audit row before re-raising, so a write that did not
        land never leaves a row claiming it did.
        """
        if resp.status_code < 400:
            return

        detail = _explain(resp)
        if resp.status_code == 404:
            raise AtlassianError(
                f"Cannot {what} {issue_key}: Jira says it does not exist, or the "
                f"service account cannot see it. Check the key and the account's "
                f"Browse Projects permission. ({detail})"
            )
        if resp.status_code == 403:
            raise AtlassianError(
                f"Cannot {what} {issue_key}: the service account lacks Edit "
                f"Issues on that project. Fixable in Jira project settings. "
                f"({detail})"
            )
        if resp.status_code == 401:
            raise AtlassianError(
                f"Cannot {what} {issue_key}: Jira auth failed — check "
                f"ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN. ({detail})"
            )
        raise AtlassianError(
            f"Cannot {what} {issue_key}: Jira returned {resp.status_code}. {detail}"
        )

    async def add_labels(self, issue_key: str, labels: list[str]) -> None:
        """Add labels, leaving the existing ones alone.

        `update.labels[].add`, never `fields.labels`. Sending `fields` replaces
        the entire array — so flagging one issue at-risk would delete every
        label a team had put on it, silently, while returning 204. The approval
        the human gave was for *adding* a label; a destructive edit is not the
        smaller-scoped version of that.
        """
        if not labels:
            # Nothing to do. A no-op PUT still spends rate limit and still shows
            # up in the client's own issue history as a modification.
            return

        resp = await self._client.put(
            f"/rest/api/3/issue/{issue_key}",
            json={"update": {"labels": [{"add": label} for label in labels]}},
        )
        self._check(resp, issue_key, "label")
        logger.info("added labels %s to %s", labels, issue_key)

    async def add_comment(self, issue_key: str, body: str) -> None:
        if not body.strip():
            return

        resp = await self._client.post(
            f"/rest/api/3/issue/{issue_key}/comment",
            json={"body": to_adf(body)},
        )
        self._check(resp, issue_key, "comment on")
        logger.info("commented on %s", issue_key)
