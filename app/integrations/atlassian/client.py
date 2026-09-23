"""Jira Cloud client.

The unglamorous layer, and the one that decides whether this project survives
contact with a real tenant. Three things it has to get right:

1. Pagination — Jira caps page size and lies about it. Never trust maxResults.
2. Rate limits — 429s come with Retry-After. Honour it; don't invent your own.
3. Changelog — NOT returned by default. Stale detection is impossible without
   it, so `expand=changelog` is non-optional here.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx

from app.models.domain import (
    ChangelogEntry,
    Comment,
    Issue,
    IssueLink,
    IssueStatusCategory,
)

logger = logging.getLogger(__name__)

_STATUS_CATEGORY_MAP = {
    "new": IssueStatusCategory.TODO,
    "indeterminate": IssueStatusCategory.IN_PROGRESS,
    "done": IssueStatusCategory.DONE,
}


_BLOCKED_BY_PHRASES = frozenset(
    {"is blocked by", "depends on", "is caused by", "is prevented by"}
)

_LINK_TYPE_ALIASES = {
    "blocks": "blocks",
    "dependency": "depends_on",
    "depends": "depends_on",
    "relates": "relates_to",
    "duplicate": "duplicates",
    "cloners": "clones",
}


def _normalize_link_type(type_name: str) -> str:
    key = (type_name or "").strip().lower()
    for alias, normalized in _LINK_TYPE_ALIASES.items():
        if alias in key:
            return normalized
    return key.replace(" ", "_") or "unknown"


def _parse_links(
    raw_links: list[dict[str, Any]], blocked_by_phrases: frozenset[str]
) -> list[IssueLink]:
    """Resolve link direction once, here, and never re-derive it downstream.

    Jira semantics, stated precisely because this is the easiest thing in the
    whole integration to invert:

    In issue A's payload, a link entry carrying `outwardIssue: B` means the
    relationship reads A -> B using `type.outward` ("blocks"). An entry carrying
    `inwardIssue: C` means A's relationship to C reads with `type.inward`
    ("is blocked by") — so C is the thing holding A up.

    Direction is therefore decided by which key is present, combined with the
    applicable description. Phrase matching is used rather than type names
    because tenants create custom link types freely; the phrase set is injectable
    so a client with unusual wording is a config change, not a code change.
    """
    links: list[IssueLink] = []

    for raw in raw_links or []:
        link_type = raw.get("type") or {}

        if "inwardIssue" in raw:
            target = raw["inwardIssue"]
            description = (link_type.get("inward") or "").strip().lower()
        elif "outwardIssue" in raw:
            target = raw["outwardIssue"]
            description = (link_type.get("outward") or "").strip().lower()
        else:
            continue

        target_fields = target.get("fields") or {}
        category_key = (
            (target_fields.get("status") or {}).get("statusCategory", {}).get("key")
        )

        links.append(
            IssueLink(
                target_key=target["key"],
                target_summary=target_fields.get("summary"),
                target_status_category=(
                    _STATUS_CATEGORY_MAP.get(category_key) if category_key else None
                ),
                link_type=_normalize_link_type(link_type.get("name", "")),
                target_blocks_this=description in blocked_by_phrases,
            )
        )

    return links


class AtlassianError(RuntimeError):
    pass


def _explain(resp: httpx.Response) -> str:
    """Jira's own account of what was wrong with the request.

    Messages arrive under `errorMessages` for query-level faults and under
    `errors` (field -> reason) for parameter-level ones; a single request can
    populate either, so both are read. Gateways and proxies answer with HTML,
    hence the non-JSON fallback — callers only catch AtlassianError, and an
    httpx error escaping here would bypass every handler upstream.
    """
    try:
        body = resp.json()
    except ValueError:
        detail = (resp.text or "").strip()
        return detail[:200] if detail else f"HTTP {resp.status_code} with no details"

    if not isinstance(body, dict):
        return f"HTTP {resp.status_code}: {body}"

    parts = [str(m) for m in body.get("errorMessages") or []]
    parts += [f"{k}: {v}" for k, v in (body.get("errors") or {}).items()]
    return " / ".join(parts) or f"HTTP {resp.status_code} with no details"


class JiraClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        *,
        target_release_field: str,
        max_retries: int = 5,
        page_size: int = 100,
        blocked_by_phrases: frozenset[str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Custom field IDs differ per tenant. Hardcoding `customfield_10014`
        # works in your sandbox and breaks at the client. Inject it.
        self.target_release_field = target_release_field
        self.max_retries = max_retries
        self.page_size = page_size
        self.blocked_by_phrases = blocked_by_phrases or _BLOCKED_BY_PHRASES
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=(email, api_token),
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with exponential backoff that respects Retry-After."""
        delay = 1.0
        for attempt in range(self.max_retries):
            resp = await self._client.get(path, params=params)

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay))
                logger.warning(
                    "jira rate limited, sleeping %.1fs (attempt %d)", wait, attempt + 1
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code >= 500:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code == 401:
                raise AtlassianError("Jira auth failed — check email/API token.")
            if resp.status_code == 403:
                raise AtlassianError(
                    "Jira returned 403. The service account likely lacks Browse "
                    "Projects permission on one of the six projects."
                )
            # A 400 is always a query we constructed, and Jira names the exact
            # defect — an unbounded JQL, a custom field ID that does not exist
            # on this tenant. raise_for_status() discards the body and leaves a
            # URL-encoded traceback, which is how an operator ends up reading
            # percent-escapes instead of a one-line fix. Not retried: a
            # malformed query is malformed on every attempt.
            if resp.status_code == 400:
                raise AtlassianError(f"Jira rejected the request: {_explain(resp)}")
            resp.raise_for_status()
            return resp.json()

        raise AtlassianError(f"Jira unavailable after {self.max_retries} retries")

    # Fields must be requested explicitly. The /search/jql endpoint returns only
    # `id` by default — a silent behaviour change from the old /search, which
    # returned a navigable set. Omit this and every mapping fails with a
    # KeyError on 'fields' that looks like a parser bug.
    def _field_list(self) -> list[str]:
        return [
            "summary", "status", "assignee", "created", "updated",
            "duedate", "project", "comment", "issuelinks",
            self.target_release_field,
        ]

    async def search_issues(self, jql: str) -> AsyncIterator[Issue]:
        """Page through a JQL search, yielding normalized Issues.

        Uses /rest/api/3/search/jql. The older /rest/api/3/search with
        `startAt`/`total` is deprecated: offset paging over a live index can
        skip or duplicate rows when issues are edited mid-sync, and `total` was
        expensive enough that Atlassian removed it rather than keep computing
        it. There is now a separate approximate-count endpoint for the cases
        that genuinely need a number.

        The practical consequence is that Jira and Confluence now paginate the
        same way — follow the server's cursor until it stops giving you one.
        Neither exposes a total, so the token is the only terminator.

        A generator, not a list: 3,000 issues with full changelogs is a few
        hundred MB of JSON if you materialize it all.
        """
        params: dict[str, Any] = {
            "jql": jql,
            "maxResults": self.page_size,
            "expand": "changelog",
            "fields": ",".join(self._field_list()),
        }

        while True:
            payload = await self._get("/rest/api/3/search/jql", params)

            raw_issues = payload.get("issues", [])
            for raw in raw_issues:
                try:
                    yield self._to_domain(raw)
                except Exception:
                    # One malformed issue must not kill a 3,000-issue sync.
                    logger.exception("skipping unmappable issue %s", raw.get("key"))

            # Two terminators, and both are checked. `isLast` is the documented
            # signal, but a final page can arrive with isLast absent and no
            # token — trusting either alone risks an extra request or, worse,
            # an infinite loop re-sending a stale cursor.
            next_token = payload.get("nextPageToken")
            if payload.get("isLast") or not next_token:
                break

            params = {**params, "nextPageToken": next_token}

    def _to_domain(self, raw: dict[str, Any]) -> Issue:
        f = raw["fields"]
        status = f["status"]
        category_key = status.get("statusCategory", {}).get("key", "new")

        comments = [
            Comment(
                id=c["id"],
                body=_flatten_adf(c.get("body")),
                author_id=(c.get("author") or {}).get("accountId"),
                created_at=_parse_dt(c["created"]),
            )
            for c in (f.get("comment") or {}).get("comments", [])
        ]

        changelog = [
            ChangelogEntry(
                field=item["field"],
                from_value=item.get("fromString"),
                to_value=item.get("toString"),
                at=_parse_dt(history["created"]),
                author_id=(history.get("author") or {}).get("accountId"),
            )
            for history in (raw.get("changelog") or {}).get("histories", [])
            for item in history.get("items", [])
        ]

        return Issue(
            key=raw["key"],
            sprint_name=_current_sprint(changelog),
            project_key=f["project"]["key"],
            summary=f["summary"],
            status_name=status["name"],
            status_category=_STATUS_CATEGORY_MAP.get(
                category_key, IssueStatusCategory.TODO
            ),
            assignee_id=(f.get("assignee") or {}).get("accountId"),
            target_release=_coerce_target_release(f.get(self.target_release_field)),
            created_at=_parse_dt(f["created"]),
            updated_at=_parse_dt(f["updated"]),
            due_date=_parse_due_date(f["duedate"]) if f.get("duedate") else None,
            comments=comments,
            changelog=changelog,
            links=_parse_links(f.get("issuelinks"), self.blocked_by_phrases),
        )


def _current_sprint(changelog: list[ChangelogEntry]) -> str | None:
    """Which sprint the issue is in now, read from its move history.

    Derived from the changelog rather than from the Sprint custom field on
    purpose. That field's id is per-tenant — `customfield_10020` here and
    something else elsewhere — so reading it means one more id to resolve,
    inject and eventually get wrong on a client site, for a value the changelog
    already carries. The carryover rule reads these same entries.

    Sorted by timestamp rather than taking the last element: Jira returns
    histories newest-first on some tenants and oldest-first on others, so
    trusting list order reads the wrong end on half of them.

    An issue removed from every sprint has a final move `to` an empty string.
    That is "no sprint", not a sprint named "" — and a finding reading
    "currently: " is worse than one reading "no sprint".
    """
    moves = [e for e in changelog if e.field.lower() in {"sprint", "sprints"}]
    if not moves:
        return None
    latest = max(moves, key=lambda e: e.at)
    name = (latest.to_value or "").strip()
    return name or None


def _coerce_target_release(value: Any) -> str | None:
    """Teams populate this field three different ways. Normalize, don't guess.

    Empty string is treated as absent — 40% of tickets, and a blank is not a
    commitment.
    """
    if value is None:
        return None
    if isinstance(value, dict):  # select-list custom field
        value = value.get("value") or value.get("name")
    if isinstance(value, list):
        value = value[0] if value else None
        if isinstance(value, dict):
            value = value.get("value") or value.get("name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _flatten_adf(body: Any) -> str:
    """Jira Cloud returns comments as Atlassian Document Format, not text."""
    if isinstance(body, str):
        return body
    if not isinstance(body, dict):
        return ""
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text":
                out.append(node.get("text", ""))
            for child in node.get("content", []) or []:
                walk(child)

    walk(body)
    return " ".join(out).strip()


def _parse_dt(value: str) -> datetime:
    """Parse a Jira timestamp, always offset-aware.

    The `tzinfo` fallback is not defensive padding. Jira returns every
    timestamp with an offset except one — `duedate`, which is a bare calendar
    day — so `fromisoformat` produced offset-aware datetimes for years and a
    naive one for that single field. Downstream, `Issue.due_date >= now` raised
    TypeError inside a risk rule and aborted the sync for the whole project:
    every issue after the first one with a due date went unread, and the
    operator saw "1 of 1 projects were not fully read" with no cause attached.

    A naive datetime crossing this boundary is the bug. Stopping it here means
    no rule, report or comparison written later has to remember.
    """
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_due_date(value: str) -> datetime:
    """Jira's `duedate`, as the end of the day it names.

    `duedate` is a calendar day ("2026-09-30"), not an instant. A deadline of
    the 30th is met by work finished on the 30th, so parsing it to midnight
    reports the issue as overdue from the first second of the day it is due —
    a day early, on every overdue issue, on a number a delivery lead checks
    against their own board.

    Full timestamps are left alone: if a tenant ever returns one here, it is
    already an instant and reinterpreting it would be the same mistake in
    reverse.
    """
    parsed = _parse_dt(value)
    is_date_only = len(value.strip()) == 10  # "YYYY-MM-DD"
    if not is_date_only:
        return parsed
    return parsed.replace(hour=23, minute=59, second=59)
