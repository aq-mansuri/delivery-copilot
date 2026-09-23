"""Smoke test: prove credentials and field IDs are right before building on them.

Run this the moment your sandbox exists. Every hour spent debugging the risk
engine against data that never loaded is an hour wasted.

    python scripts/check_jira_connection.py            # every project
    python scripts/check_jira_connection.py INS CLM    # named projects
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from app.integrations.atlassian.client import AtlassianError, JiraClient


def _jql(project_keys: list[str]) -> str:
    """A bounded query. /search/jql rejects an unrestricted one outright with
    "Unbounded JQL queries are not allowed here" — a bare `ORDER BY created
    DESC` is a 400, not a cheap way to ask for everything. `project IS NOT
    EMPTY` is the restriction that costs nothing and still spans the tenant,
    which is what a connection check wants.
    """
    if not project_keys:
        return "project IS NOT EMPTY ORDER BY created DESC"
    keys = ", ".join(f'"{k}"' for k in project_keys)
    return f"project IN ({keys}) ORDER BY created DESC"


async def main(project_keys: list[str]) -> int:
    load_dotenv()

    required = ["ATLASSIAN_BASE_URL", "ATLASSIAN_EMAIL", "ATLASSIAN_API_TOKEN"]
    if missing := [k for k in required if not os.getenv(k)]:
        print(f"Missing env vars: {', '.join(missing)}")
        print("Copy .env.example to .env and fill it in.")
        return 1

    client = JiraClient(
        base_url=os.environ["ATLASSIAN_BASE_URL"],
        email=os.environ["ATLASSIAN_EMAIL"],
        api_token=os.environ["ATLASSIAN_API_TOKEN"],
        target_release_field=os.environ["JIRA_TARGET_RELEASE_FIELD"],
    )

    jql = _jql(project_keys)
    print(f"JQL: {jql}\n")

    try:
        count = 0
        with_release = 0
        with_changelog = 0

        async for issue in client.search_issues(jql):
            count += 1
            with_release += bool(issue.target_release)
            with_changelog += bool(issue.changelog)
            if count <= 3:
                print(
                    f"  {issue.key:<10} {issue.status_category.value:<12} "
                    f"release={issue.target_release or '(blank)'}  "
                    f"last_activity={issue.last_activity_at().date()}"
                )
            if count >= 50:
                break

        print(f"\nFetched {count} issues.")
        print(f"  Target Release populated: {with_release}/{count}")
        print(f"  Changelog present:        {with_changelog}/{count}")

        if count and not with_changelog:
            print("\nWARNING: no changelogs came back. Stale detection cannot work.")
            print("Check that expand=changelog is reaching the API.")
        if count and not with_release:
            print("\nWARNING: Target Release is blank on every issue.")
            print("Your JIRA_TARGET_RELEASE_FIELD is probably the wrong custom field ID.")
            print("Run: GET /rest/api/3/field and find the right one.")

    except AtlassianError as exc:
        print(f"Connection failed: {exc}")
        return 1
    finally:
        await client.aclose()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
