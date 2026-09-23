"""Confluence Cloud client.

Two differences from the Jira client that are worth naming, because assuming
one Atlassian product behaves like another is how this integration breaks:

**Pagination is cursor-based, not offset-based.** Confluence v2 returns an
opaque `_links.next` and there is no `total`. The Jira pattern — increment
`startAt`, compare against `total` — has no equivalent here. Following the
server's cursor is also more correct under concurrent edits, where offset paging
can skip or duplicate rows.

**Scope is a filter, not a fetch-everything.** The client has a few thousand
pages and roughly 200 that matter: architecture decisions, vendor contracts,
compliance sign-offs. Embedding the rest costs money and actively degrades
retrieval, because meeting notes and onboarding checklists compete with the
pages that hold real answers. Selection happens here, by label or space, not
after embedding.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from app.integrations.atlassian.storage_format import parse_storage
from app.models.domain import Page

logger = logging.getLogger(__name__)


class ConfluenceError(RuntimeError):
    pass


class ConfluenceClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        *,
        max_retries: int = 5,
        page_size: int = 50,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.page_size = page_size
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=(email, api_token),
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        delay = 1.0
        for attempt in range(self.max_retries):
            resp = await self._client.get(path, params=params)

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay))
                logger.warning(
                    "confluence rate limited, sleeping %.1fs (attempt %d)",
                    wait,
                    attempt + 1,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code >= 500:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code == 401:
                raise ConfluenceError(
                    "Confluence auth failed — check email/API token."
                )
            if resp.status_code == 403:
                raise ConfluenceError(
                    "Confluence returned 403. The service account likely lacks "
                    "View permission on the requested space."
                )
            resp.raise_for_status()
            return resp.json()

        raise ConfluenceError(
            f"Confluence unavailable after {self.max_retries} retries"
        )

    async def iter_pages(
        self,
        space_key: str,
        *,
        labels: list[str] | None = None,
    ) -> AsyncIterator[Page]:
        """Yield pages in a space, optionally restricted to labels.

        `body-format=storage` is required. Without it the body comes back as
        rendered HTML with macros already expanded into markup, which is worse
        to parse and loses the macro identity the parser uses to decide what to
        keep.
        """
        params: dict[str, Any] = {
            "space-key": space_key,
            "body-format": "storage",
            "limit": self.page_size,
        }
        if labels:
            params["label"] = ",".join(labels)

        path = "/wiki/api/v2/pages"

        while True:
            payload = await self._get(path, params)

            for raw in payload.get("results", []):
                try:
                    yield self._to_domain(raw, space_key)
                except Exception:
                    logger.exception(
                        "skipping unmappable page %s", raw.get("id")
                    )

            next_link = (payload.get("_links") or {}).get("next")
            if not next_link:
                break

            # The cursor link is returned as a path with its own query string.
            # Re-sending our original params alongside it double-applies filters
            # and, on some tenants, resets the cursor — an infinite loop that
            # only shows up past the first page.
            path = _relative_path(next_link)
            params = None

    def _to_domain(self, raw: dict[str, Any], space_key: str) -> Page:
        storage = ((raw.get("body") or {}).get("storage") or {}).get("value", "")
        version = raw.get("version") or {}
        links = raw.get("_links") or {}

        return Page(
            id=str(raw["id"]),
            space_key=raw.get("spaceId") and space_key or space_key,
            title=raw.get("title", ""),
            url=f"{self.base_url}/wiki{links.get('webui', '')}",
            version=int(version.get("number", 1)),
            updated_at=_parse_dt(version.get("createdAt")),
            labels=[
                label.get("name", "")
                for label in ((raw.get("labels") or {}).get("results") or [])
            ],
            sections=parse_storage(storage),
        )


def _relative_path(link: str) -> str:
    """Confluence returns `next` sometimes absolute, sometimes relative."""
    parsed = urlparse(link)
    path = parsed.path
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


def _parse_dt(value: str | None) -> datetime:
    if not value:
        raise ConfluenceError("page version is missing a timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
