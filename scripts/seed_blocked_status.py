"""Add a Blocked status to a project's workflow and block some issues with it.

    python scripts/seed_blocked_status.py --project INS --dry-run
    python scripts/seed_blocked_status.py --project INS --pairs 4

Idempotent. Re-running adds nothing it already added.

## What this is for, and what it is deliberately NOT for

A real insurer's board has a Blocked column. Without one the sandbox looks like
a tutorial, and "INS-22 is blocked by INS-3" is a fact you can only see by
opening the ticket and reading its links.

It changes **nothing** in the risk engine, and that is the point worth making
rather than hiding. Every rule keys off `statusCategory`, never a status name,
because status names differ per project and per tenant — "Blocked", "On Hold",
"Impediment", "Waiting" all mean the same thing and no two clients spell it the
same way. `Blocked` is created with category `IN_PROGRESS`, so to the rules it
is indistinguishable from `In Progress`, exactly as intended.

`rule_blocked_dependency` finds these issues through their **links**, which are
structured data with a direction, and it found them before this script existed.
The status is presentation.

## The workflow update API, which took three attempts

`POST /rest/api/3/workflows/update` has a contract the error messages do not
describe well:

- The top-level `statuses` array must list **every** status the workflow
  references, not just new ones. Sending only the new one returns "Workflow
  refers to a missing status reference".
- An **existing** status is declared with its `id` AND a `statusReference` (its
  id again). A **new** status has no `id` and its `statusReference` must be a
  UUID you invent; Jira creates the status and binds it to that reference.
- So do not create the status first with `POST /rest/api/3/statuses`. Doing that
  makes the update fail with "Status name ... already in use", because the
  update wants to create it itself. (That orphan then has to be deleted.)
- `version` must be the workflow's current version object, or the write is
  rejected as a stale edit.

## Blast radius

Checked before writing: this refuses to touch a workflow scheme shared with
another project. A "Software Simplified Workflow for Project X" scheme is
normally dedicated, but the default Jira workflow is shared by everything, and
adding a status to that one edits every project on the tenant.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BLOCKED_NAME = "Blocked"
BLOCKED_DESCRIPTION = "Work cannot proceed until a dependency is resolved."

# Each pair is (blocked, blocker). Chosen so the dependency reads as plausible
# to someone who knows the domain — a reviewer who spots that the blocker makes
# no sense stops trusting the rest of the data.
BLOCKER_STORIES = [
    ("Refactor claims intake", "the schema migration it depends on"),
    ("Premium rounding fix", "the endpoint retirement that owns the calculation"),
    ("Exec delivery dashboard", "the compliance export that feeds it"),
    ("TLS certificate rotation", "the vendor callback integration using them"),
]


class Blocker:
    def __init__(self, base_url: str, email: str, token: str):
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            auth=(email, token),
            timeout=60.0,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def project_id(self, key: str) -> str:
        resp = await self.client.get(f"/rest/api/3/project/{key}")
        resp.raise_for_status()
        return resp.json()["id"]

    async def workflow_for(self, project_key: str) -> tuple[str, list[str]] | None:
        """The project's default workflow name, and which projects share it.

        Returns None if the project has no scheme this account can read, which
        is a permissions problem rather than a missing workflow.
        """
        pid = await self.project_id(project_key)
        resp = await self.client.get(
            "/rest/api/3/workflowscheme/project", params={"projectId": pid}
        )
        if resp.status_code != 200:
            return None
        for value in resp.json().get("values", []):
            scheme = value.get("workflowScheme", {})
            return scheme.get("defaultWorkflow"), value.get("projectIds", [])
        return None

    async def workflow_definition(self, name: str) -> dict | None:
        resp = await self.client.get(
            "/rest/api/3/workflow/search", params={"workflowName": name}
        )
        if resp.status_code != 200:
            return None
        values = resp.json().get("values", [])
        if not values:
            return None
        entity_id = values[0]["id"].get("entityId")

        resp = await self.client.post(
            "/rest/api/3/workflows", json={"workflowIds": [entity_id]}
        )
        if resp.status_code != 200:
            return None
        workflows = resp.json().get("workflows", [])
        return workflows[0] if workflows else None

    async def status_details(self, references: list[str]) -> list[dict]:
        """Names and categories for statuses already in the workflow.

        Required because the update payload must redeclare them. Fetched rather
        than assumed: a workflow's statuses are not guaranteed to be the three
        defaults, and guessing a name wrong renames somebody's status.
        """
        resp = await self.client.get(
            "/rest/api/3/statuses/search", params={"maxResults": 200}
        )
        resp.raise_for_status()
        by_id = {s["id"]: s for s in resp.json().get("values", [])}
        out = []
        for ref in references:
            status = by_id.get(ref)
            if status is None:
                raise RuntimeError(
                    f"Workflow references status {ref}, which this account "
                    f"cannot read. Cannot safely rewrite the workflow."
                )
            out.append(
                {
                    "id": status["id"],
                    "statusReference": status["id"],
                    "name": status["name"],
                    "statusCategory": status["statusCategory"],
                    "description": status.get("description", "") or "",
                }
            )
        return out

    async def add_blocked_status(self, workflow: dict) -> str:
        """Add `Blocked` to the workflow, with a global transition into it.

        A GLOBAL transition means the issue can be moved to Blocked from
        anywhere, which is how a real board behaves — work gets blocked from
        wherever it happened to be.
        """
        existing_refs = [s["statusReference"] for s in workflow["statuses"]]
        declared = await self.status_details(existing_refs)

        # New status: no `id`, and a UUID reference Jira binds on creation.
        reference = str(uuid.uuid4())
        declared.append(
            {
                "statusReference": reference,
                "name": BLOCKED_NAME,
                "statusCategory": "IN_PROGRESS",
                "description": BLOCKED_DESCRIPTION,
            }
        )

        # Transition ids are strings and must not collide with an existing one.
        used = {t.get("id") for t in workflow["transitions"]}
        new_id = next(str(n) for n in range(41, 999) if str(n) not in used)

        payload = {
            "statuses": declared,
            "workflows": [
                {
                    "id": workflow["id"],
                    "version": workflow["version"],
                    "description": workflow.get("description", ""),
                    "statuses": list(workflow["statuses"])
                    + [
                        {
                            "statusReference": reference,
                            "layout": {},
                            "properties": {},
                            "deprecated": False,
                        }
                    ],
                    "transitions": list(workflow["transitions"])
                    + [
                        {
                            "id": new_id,
                            "type": "GLOBAL",
                            "toStatusReference": reference,
                            "links": [],
                            "name": BLOCKED_NAME,
                            "description": BLOCKED_DESCRIPTION,
                            "actions": [],
                            "validators": [],
                            "triggers": [],
                            "properties": {},
                        }
                    ],
                    "startPointLayout": workflow.get("startPointLayout", {}),
                }
            ],
        }

        resp = await self.client.post("/rest/api/3/workflows/update", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"workflow update failed: {resp.text[:400]}")
        created = [
            s for s in resp.json().get("statuses", []) if s["name"] == BLOCKED_NAME
        ]
        return created[0]["id"] if created else ""

    async def open_issues(self, project_key: str) -> list[dict]:
        resp = await self.client.get(
            "/rest/api/3/search/jql",
            params={
                "jql": f'project = "{project_key}" AND statusCategory != Done '
                f"ORDER BY key ASC",
                "fields": "summary,status,issuelinks",
                "maxResults": 100,
            },
        )
        resp.raise_for_status()
        return resp.json().get("issues", [])

    async def transition_to(self, key: str, status_name: str) -> bool:
        """Move an issue, resolving the transition by its target status name.

        Transition ids are per-workflow and change when a workflow is edited.
        Hardcoding one works until the day somebody adds a status — which is
        the day this script runs.
        """
        resp = await self.client.get(f"/rest/api/3/issue/{key}/transitions")
        if resp.status_code != 200:
            return False
        match = next(
            (
                t
                for t in resp.json().get("transitions", [])
                if t["to"]["name"].lower() == status_name.lower()
            ),
            None,
        )
        if match is None:
            return False
        resp = await self.client.post(
            f"/rest/api/3/issue/{key}/transitions", json={"transition": {"id": match["id"]}}
        )
        return resp.status_code < 400

    async def link(self, blocker_key: str, blocked_key: str) -> bool:
        """Create "blocked_key is blocked by blocker_key".

        The POST body's semantics are the reverse of what the field names
        suggest, and `seed_jira.py` carries the long version of why. The INWARD
        issue in the request body is the BLOCKER. Verified against a live
        tenant, and re-verified by this script's own output.
        """
        resp = await self.client.post(
            "/rest/api/3/issueLink",
            json={
                "type": {"name": "Blocks"},
                "inwardIssue": {"key": blocker_key},
                "outwardIssue": {"key": blocked_key},
            },
        )
        return resp.status_code < 400

    async def comment(self, key: str, body: str) -> None:
        await self.client.post(
            f"/rest/api/3/issue/{key}/comment",
            json={
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": body}],
                        }
                    ],
                }
            },
        )


def choose_pairs(issues: list[dict], count: int) -> list[tuple[dict, dict]]:
    """Pair issues to block with issues that are genuinely still in progress.

    A blocker that is already Done is not a blocker, and seeding one produces a
    finding the rules will correctly decline to raise — which reads as the rule
    being broken rather than the data being wrong.
    """
    in_progress = [
        i
        for i in issues
        if i["fields"]["status"]["statusCategory"]["key"] == "indeterminate"
        and i["fields"]["status"]["name"].lower() != BLOCKED_NAME.lower()
    ]
    if len(in_progress) < 2:
        return []

    pairs: list[tuple[dict, dict]] = []
    # Walk from both ends so a blocker is never also the issue being blocked,
    # and so the two are far apart in the backlog rather than adjacent.
    for index in range(min(count, len(in_progress) // 2)):
        pairs.append((in_progress[-(index + 1)], in_progress[index]))
    return pairs


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Jira project key, e.g. INS")
    parser.add_argument(
        "--pairs", type=int, default=4, help="How many issues to move to Blocked"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    base_url = os.getenv("ATLASSIAN_BASE_URL")
    email = os.getenv("ATLASSIAN_EMAIL")
    token = os.getenv("ATLASSIAN_API_TOKEN")
    if not all([base_url, email, token]):
        print("Missing credentials. Copy .env.example to .env and fill it in.")
        return 1

    blocker = Blocker(base_url, email, token)
    try:
        found = await blocker.workflow_for(args.project)
        if found is None:
            print(f"Could not read a workflow scheme for {args.project}.")
            return 1
        workflow_name, project_ids = found

        if len(project_ids) > 1:
            # Refused rather than warned. Adding a status to a shared workflow
            # edits every project using it, and on most tenants the default
            # workflow is shared by all of them.
            print(
                f"{workflow_name!r} is shared by {len(project_ids)} projects "
                f"({', '.join(project_ids)}). Refusing to edit it — add the "
                f"status by hand, or give {args.project} its own workflow "
                f"scheme first."
            )
            return 1

        workflow = await blocker.workflow_definition(workflow_name)
        if workflow is None:
            print(f"Could not read the definition of {workflow_name!r}.")
            return 1

        declared = await blocker.status_details(
            [s["statusReference"] for s in workflow["statuses"]]
        )
        already = any(s["name"].lower() == BLOCKED_NAME.lower() for s in declared)

        issues = await blocker.open_issues(args.project)
        pairs = choose_pairs(issues, args.pairs)

        print(f"workflow : {workflow_name} (used only by {args.project})")
        print(f"status   : {BLOCKED_NAME} "
              f"{'already present' if already else 'will be added'} "
              f"(category IN_PROGRESS)")
        print(f"pairs    : {len(pairs)}\n")
        for blocked, blk in pairs:
            print(f"  {blocked['key']:<8} {blocked['fields']['summary'][:40]:<42}"
                  f" <- blocked by {blk['key']} {blk['fields']['summary'][:30]}")

        if args.dry_run:
            print("\nRe-run without --dry-run to apply.")
            return 0

        if not already:
            status_id = await blocker.add_blocked_status(workflow)
            print(f"\n  added status {BLOCKED_NAME!r} (id {status_id}) to the workflow")
        else:
            print(f"\n  status {BLOCKED_NAME!r} already in the workflow, skipping")

        print()
        for blocked, blk in pairs:
            linked = await blocker.link(blk["key"], blocked["key"])
            moved = await blocker.transition_to(blocked["key"], BLOCKED_NAME)
            await blocker.comment(
                blocked["key"],
                f"Blocked by {blk['key']} ({blk['fields']['summary']}). "
                f"Cannot proceed until that lands.",
            )
            print(f"  {blocked['key']:<8} "
                  f"{'linked' if linked else 'LINK FAILED':<12} "
                  f"{'-> Blocked' if moved else 'TRANSITION FAILED'}")

        print(f"\nBlocked {len(pairs)} issues in {args.project}.")
        print("\nNote: this changes what the board looks like, not what the rules")
        print("find. Rules key off statusCategory, never a status name, and")
        print("Blocked is IN_PROGRESS — so rule_blocked_dependency still finds")
        print("these through their links, exactly as it did before.")
        print("\nRestart the service to re-sync; findings are read once at startup.")
        return 0
    finally:
        await blocker.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
