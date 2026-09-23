"""Recorded Jira Cloud response shapes.

These are trimmed real payloads, not invented ones. The difference matters: the
bugs this file exists to catch — ADF comment bodies, statusCategory keys,
inward/outward link direction, custom field shapes — all live in structural
details nobody reproduces correctly from memory.

Re-record by hitting a real tenant and trimming, never by hand-editing shapes
you assume are right.
"""

from __future__ import annotations

from typing import Any


def adf_comment_body(text: str) -> dict[str, Any]:
    """Jira Cloud returns comment bodies as Atlassian Document Format.

    Anyone expecting a string here gets an empty comment and no error.
    """
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ],
    }


def adf_multi_paragraph(*paragraphs: str) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": p}],
            }
            for p in paragraphs
        ],
    }


def issue(
    key: str = "INS-101",
    *,
    status_name: str = "In Progress",
    status_category: str = "indeterminate",
    target_release: Any = "Q4-2026",
    target_release_field: str = "customfield_10014",
    comments: list[dict[str, Any]] | None = None,
    changelog: list[dict[str, Any]] | None = None,
    issuelinks: list[dict[str, Any]] | None = None,
    duedate: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "10001",
        "key": key,
        "fields": {
            "summary": "Integrate claims vendor callback",
            "project": {"key": key.split("-")[0], "name": "Insurance Core"},
            "status": {
                "name": status_name,
                "statusCategory": {"key": status_category, "name": status_name},
            },
            "assignee": {"accountId": "acc-dev-1", "displayName": "Dev One"},
            "created": "2026-07-01T09:00:00.000+0000",
            "updated": "2026-09-15T09:00:00.000+0000",
            "duedate": duedate,
            target_release_field: target_release,
            "comment": {"comments": comments or []},
            "issuelinks": issuelinks or [],
        },
        "changelog": {"histories": changelog or []},
    }


def comment(
    body: str, *, comment_id: str = "c1", created: str = "2026-09-10T10:00:00.000+0000"
) -> dict[str, Any]:
    return {
        "id": comment_id,
        "author": {"accountId": "acc-lead-1"},
        "created": created,
        "body": adf_comment_body(body),
    }


def link_changelog_entry(target_key: str = "INS-1") -> dict[str, Any]:
    """Creating a link writes a changelog entry with human-readable prose.

    Verified against a real tenant. Two things matter here:

    1. `field` is "Link", and unlike a status change the item carries NO
       `fieldId` key. Code indexing item["fieldId"] would KeyError on this.
    2. `toString` spells the direction out in English — "This work item is
       blocked by INS-1". That prose is the independent check that settled
       which way round the link mapping goes.
    """
    return {
        "id": "10069",
        "author": {"accountId": "557058:082d91fc"},
        "created": "2026-09-21T20:47:50.841+0530",
        "items": [
            {
                "field": "Link",
                "fieldtype": "jira",
                "from": None,
                "fromString": None,
                "to": target_key,
                "toString": f"This work item is blocked by {target_key}",
            }
        ],
    }


def changelog_entry(
    field: str = "Sprint",
    *,
    created: str = "2026-09-01T10:00:00.000+0000",
    from_string: str | None = "Sprint 12",
    to_string: str | None = "Sprint 13",
) -> dict[str, Any]:
    return {
        "id": "h1",
        "created": created,
        "author": {"accountId": "acc-lead-1"},
        "items": [
            {
                "field": field,
                "fromString": from_string,
                "toString": to_string,
            }
        ],
    }


def blocked_by_link(
    target_key: str = "PLAT-7", *, status_category: str = "indeterminate"
) -> dict[str, Any]:
    """The target appears as `inwardIssue` — meaning it blocks the current issue.

    Read the description field, not the type name: `type.inward` is
    "is blocked by", and that is how the CURRENT issue relates to the target.
    """
    return {
        "id": "20001",
        "type": {
            "id": "10000",
            "name": "Blocks",
            "inward": "is blocked by",
            "outward": "blocks",
        },
        "inwardIssue": {
            "key": target_key,
            "fields": {
                "summary": "Ship partner auth endpoint",
                "status": {"statusCategory": {"key": status_category}},
            },
        },
    }


def blocks_other_link(
    target_key: str = "CLM-3", *, status_category: str = "new"
) -> dict[str, Any]:
    """The target appears as `outwardIssue` — the current issue blocks IT.

    Same link type, opposite meaning. This pair is the whole point of the
    direction test.
    """
    return {
        "id": "20002",
        "type": {
            "id": "10000",
            "name": "Blocks",
            "inward": "is blocked by",
            "outward": "blocks",
        },
        "outwardIssue": {
            "key": target_key,
            "fields": {
                "summary": "Claims dashboard rollout",
                "status": {"statusCategory": {"key": status_category}},
            },
        },
    }


def relates_link(target_key: str = "INS-55") -> dict[str, Any]:
    return {
        "id": "20003",
        "type": {
            "id": "10003",
            "name": "Relates",
            "inward": "relates to",
            "outward": "relates to",
        },
        "outwardIssue": {
            "key": target_key,
            "fields": {
                "summary": "Legacy policy migration",
                "status": {"statusCategory": {"key": "new"}},
            },
        },
    }


def search_page(
    issues: list[dict[str, Any]],
    *,
    next_page_token: str | None = None,
    is_last: bool | None = None,
) -> dict[str, Any]:
    """Response shape of /rest/api/3/search/jql.

    Note what is NOT here: `startAt` and `total`. The endpoint does not return
    them. Any code still reaching for `total` is reading the deprecated API's
    shape and will silently see 0.
    """
    payload: dict[str, Any] = {"issues": issues}
    if next_page_token:
        payload["nextPageToken"] = next_page_token
    if is_last is not None:
        payload["isLast"] = is_last
    return payload
