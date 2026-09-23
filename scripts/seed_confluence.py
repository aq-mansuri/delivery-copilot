"""Push the seed corpus to a real Confluence space.

    python scripts/seed_confluence.py --space ARCH --dry-run
    python scripts/seed_confluence.py --space ARCH

Content comes from scripts/seed_content.py, the same source the offline corpus
uses — so what you evaluate locally and what lives in the tenant cannot drift.

Note the page ids here are OUR slugs, not Confluence's. Confluence assigns
numeric ids on creation, so the script writes a mapping file
(docs/page_id_map.json) translating slug -> real id. Your eval set references
slugs; the mapping is applied at load time. This keeps the eval set readable and
portable across tenants, which matters the first time you rebuild a sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seed_content import PAGES  # noqa: E402


async def create_page(
    client: httpx.AsyncClient, space_id: str, title: str, storage: str
) -> str:
    resp = await client.post(
        "/wiki/api/v2/pages",
        json={
            "spaceId": space_id,
            "status": "current",
            "title": title,
            "body": {"representation": "storage", "value": storage},
        },
    )
    if resp.status_code >= 400:
        raise SystemExit(f"Create failed ({resp.status_code}): {resp.text}")
    return str(resp.json()["id"])


async def add_labels(client: httpx.AsyncClient, page_id: str, labels: tuple[str, ...]):
    # v2 has no label-write endpoint yet; v1 is still the supported path.
    for label in labels:
        await client.post(
            f"/wiki/rest/api/content/{page_id}/label",
            json=[{"prefix": "global", "name": label}],
        )


async def resolve_space_id(client: httpx.AsyncClient, space_key: str) -> str:
    resp = await client.get("/wiki/api/v2/spaces", params={"keys": space_key})
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise SystemExit(
            f"Space {space_key} not found, or the account cannot see it."
        )
    return str(results[0]["id"])


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="Space key, e.g. ARCH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"Would create {len(PAGES)} pages in {args.space}:\n")
        for page in PAGES:
            body = page.storage.strip()
            print(f"  {page.page_id:<24} {page.title}")
            print(f"  {'':<24} labels={list(page.labels)}  {len(body)} chars storage")
        print("\nRe-run without --dry-run to create these.")
        return 0

    load_dotenv()
    base_url = os.getenv("ATLASSIAN_BASE_URL")
    email = os.getenv("ATLASSIAN_EMAIL")
    token = os.getenv("ATLASSIAN_API_TOKEN")
    if not all([base_url, email, token]):
        print("Missing credentials. Copy .env.example to .env and fill it in.")
        return 1

    client = httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        auth=(email, token),
        timeout=30.0,
        headers={"Content-Type": "application/json"},
    )

    mapping: dict[str, str] = {}
    try:
        space_id = await resolve_space_id(client, args.space)
        print(f"Space {args.space} -> id {space_id}\n")

        for page in PAGES:
            real_id = await create_page(
                client, space_id, page.title, page.storage.strip()
            )
            await add_labels(client, real_id, page.labels)
            mapping[page.page_id] = real_id
            print(f"  created {page.page_id:<24} -> {real_id}  {page.title}")

        out = Path("docs/page_id_map.json")
        out.write_text(json.dumps(mapping, indent=2))
        print(f"\nWrote slug -> id mapping to {out}")
        print("\nNext: python scripts/run_eval.py")
    finally:
        await client.aclose()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
