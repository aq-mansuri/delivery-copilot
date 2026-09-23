"""List custom fields on the tenant so you can find the right id.

    python scripts/find_field.py                 # all custom fields
    python scripts/find_field.py --name release  # filter by name

Field ids are tenant-specific. `customfield_10014` on one site is a completely
different field on another, and Jira rejects a search that names a field id it
does not recognise — so a wrong id here surfaces as a 400 on every query rather
than as a missing value.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="", help="case-insensitive substring")
    args = parser.parse_args()

    load_dotenv()
    base_url = os.getenv("ATLASSIAN_BASE_URL")
    email = os.getenv("ATLASSIAN_EMAIL")
    token = os.getenv("ATLASSIAN_API_TOKEN")
    configured = os.getenv("JIRA_TARGET_RELEASE_FIELD", "")

    if not all([base_url, email, token]):
        print("Missing credentials in .env")
        return 1

    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"), auth=(email, token), timeout=30.0
    ) as client:
        resp = await client.get("/rest/api/3/field")
        if resp.status_code >= 400:
            print(f"Failed ({resp.status_code}): {resp.text[:300]}")
            return 1

        fields = resp.json()
        custom = [f for f in fields if f.get("custom")]

        if args.name:
            needle = args.name.lower()
            custom = [f for f in custom if needle in f.get("name", "").lower()]

        if not custom:
            print(f"No custom fields matching {args.name!r}.")
            print("\nThe field may not exist yet. Create it in Jira:")
            print("  Settings -> Issues -> Custom fields -> Create field")
            print("  Then add it to the project's screens, or it will be")
            print("  invisible on create and on search.")
            return 1

        print(f"\n{len(custom)} custom field(s):\n")
        for f in sorted(custom, key=lambda x: x.get("name", "")):
            field_id = f.get("id", "")
            kind = (f.get("schema") or {}).get("type", "?")
            marker = "  <-- currently in .env" if field_id == configured else ""
            print(f"  {field_id:<24} {f.get('name', ''):<32} [{kind}]{marker}")

        # Verify the configured id is real — a stale id is a silent 400 factory.
        if configured:
            known = {f.get("id") for f in fields}
            if configured not in known:
                print(
                    f"\nWARNING: JIRA_TARGET_RELEASE_FIELD={configured} does not "
                    "exist on this tenant."
                )
                print("  Jira rejects any search naming an unknown field id, which")
                print("  is very likely the 400 you are seeing.")
            else:
                print(f"\nJIRA_TARGET_RELEASE_FIELD={configured} exists. Good.")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
