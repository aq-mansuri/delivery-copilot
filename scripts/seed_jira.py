"""Seed a Jira Cloud sandbox with realistically broken delivery data.

    python scripts/seed_jira.py --project INS --dry-run
    python scripts/seed_jira.py --project INS

## The limitation you need to know before you start

**Jira will not let you backdate `created`, comments, or changelog entries.**
Everything this script creates is timestamped now. Stale detection needs tickets
that have been idle for weeks, and no amount of API cleverness produces that on a
fresh tenant.

Two ways people get this wrong. Some seed the data and wait a fortnight. Others
quietly relax the thresholds to 2 days so something fires, then ship thresholds
tuned to fake data.

The right answer was designed in on Day 1: every rule takes `now` as a parameter.
Seed today, then evaluate with a clock 30 days ahead:

    from datetime import datetime, timedelta, timezone
    future = datetime.now(timezone.utc) + timedelta(days=30)
    findings = evaluate(issues, cfg, future)

The rules are pure functions of (issue, config, now), so this is not a hack —
it's the same property that makes them unit-testable. Worth saying out loud in an
interview: injectable time is what lets you test time-dependent logic at all.

## What gets seeded

- Issues that look healthy but have no activity (stale, once you shift the clock)
- ~40% with Target Release blank, matching the client's real data quality
- Cross-project blocking links in both directions
- Overdue due dates, which CAN be backdated — `duedate` is a plain date field
- Comments carrying blocker language for the classifier on Day 4

## What it cannot seed

Sprint carryover needs a scrum board and real sprint transitions. Creating those
via API requires the Agile API plus a board id, and the changelog entries still
timestamp now. Easiest path is to move a few issues between sprints manually in
the UI — three minutes, and you get genuine changelog entries.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

random.seed(42)  # reproducible sandboxes; a rerun should not reshuffle the data

SUMMARIES = [
    "Integrate claims vendor callback endpoint",
    "Enable partner auth on claims portal",
    "Migrate legacy policy records to new schema",
    "Add audit logging to approval workflow",
    "Rotate mutual TLS certificates",
    "Build exec delivery dashboard",
    "Fix premium calculation rounding",
    "Upgrade document storage service",
    "Implement compliance export for regulator",
    "Refactor claims intake validation",
    "Add rate limiting to vendor webhook",
    "Backfill missing policy identifiers",
    "Split claims notification worker",
    "Retire the v1 quotes endpoint",
    "Add idempotency keys to payment callbacks",
    "Reconcile broker commission ledger",
    "Harden PII redaction in support exports",
    "Move renewal batch off the legacy scheduler",
    "Add SLA monitoring to the underwriting queue",
    "Correct VAT handling on mid-term adjustments",
    "Replace nightly CSV feed with an event stream",
    "Document the claims escalation runbook",
    "Cache policyholder lookups at the edge",
    "Remove hardcoded broker identifiers",
]

BLOCKER_COMMENTS = [
    "Waiting on the claims vendor to confirm the callback schema.",
    "Legal review is still outstanding — no ETA given.",
    "Blocked on the platform team shipping the auth endpoint.",
    "Compliance asked for another sign-off round before we proceed.",
    "Vendor contract renewal has to close before we can integrate.",
]

HEALTHY_COMMENTS = [
    "PR is open and under review.",
    "Deployed to staging, running validation.",
    "Schema agreed, implementation underway.",
]


class Seeder:
    def __init__(self, base_url: str, email: str, token: str, field: str):
        self.field = field
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            auth=(email, token),
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def issue_type_id(self, project_key: str) -> str:
        """Type ids differ per project and per tenant. Never hardcode them."""
        resp = await self.client.get(
            "/rest/api/3/issue/createmeta",
            params={"projectKeys": project_key, "expand": "projects.issuetypes"},
        )
        resp.raise_for_status()
        projects = resp.json().get("projects", [])
        if not projects:
            raise SystemExit(
                f"Project {project_key} not found, or the account cannot create "
                "issues in it."
            )
        types = projects[0]["issuetypes"]
        for candidate in types:
            if candidate["name"].lower() in {"task", "story"}:
                return candidate["id"]
        return types[0]["id"]

    async def create_issue(
        self,
        project_key: str,
        type_id: str,
        summary: str,
        *,
        target_release: str | None,
        due: date | None,
    ) -> str:
        fields: dict = {
            "project": {"key": project_key},
            "issuetype": {"id": type_id},
            "summary": summary,
        }
        if target_release:
            fields[self.field] = target_release
        if due:
            # `duedate` is a plain date, so this one CAN be backdated — which is
            # why the overdue rule is the only one testable without a clock shift.
            fields["duedate"] = due.isoformat()

        resp = await self.client.post("/rest/api/3/issue", json={"fields": fields})
        if resp.status_code >= 400:
            raise SystemExit(
                f"Create failed ({resp.status_code}): {resp.text}\n"
                f"If this mentions '{self.field}', your custom field id is wrong "
                "or the field is not on the create screen for this project."
            )
        return resp.json()["key"]

    async def transition_to_in_progress(self, key: str) -> None:
        resp = await self.client.get(f"/rest/api/3/issue/{key}/transitions")
        resp.raise_for_status()
        for transition in resp.json().get("transitions", []):
            category = (
                transition.get("to", {})
                .get("statusCategory", {})
                .get("key")
            )
            if category == "indeterminate":
                await self.client.post(
                    f"/rest/api/3/issue/{key}/transitions",
                    json={"transition": {"id": transition["id"]}},
                )
                return

    async def add_comment(self, key: str, body: str) -> None:
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

    async def scrum_board_id(self, project_key: str) -> int | None:
        """Find the project's scrum board, if it has one.

        Board ids are per-tenant and not derivable from the project key, and a
        team-managed (next-gen) project may have no board the Agile API will
        admit to. Returns None rather than raising: sprint data is an extra,
        and failing the whole seed because a board is missing would be the
        wrong trade.
        """
        resp = await self.client.get(
            "/rest/agile/1.0/board",
            params={"projectKeyOrId": project_key, "type": "scrum", "maxResults": 50},
        )
        if resp.status_code != 200:
            return None
        boards = resp.json().get("values", [])
        return boards[0]["id"] if boards else None

    async def create_sprint(self, board_id: int, name: str) -> int | None:
        resp = await self.client.post(
            "/rest/agile/1.0/sprint",
            json={"name": name, "originBoardId": board_id},
        )
        if resp.status_code >= 400:
            print(f"  ! could not create sprint {name!r}: "
                  f"{resp.status_code} {resp.text[:120]}")
            return None
        return resp.json()["id"]

    async def move_to_sprint(self, sprint_id: int, keys: list[str]) -> bool:
        """Move issues into a sprint.

        This is the only way to produce genuine carryover data. Jira will not
        let you backdate a changelog entry, but it *will* record a real Sprint
        field change every time an issue moves — and `rule_sprint_carryover`
        counts exactly those entries. Walk an issue through three sprints and
        the rule sees three moves, because three moves actually happened.
        """
        resp = await self.client.post(
            f"/rest/agile/1.0/sprint/{sprint_id}/issue", json={"issues": keys}
        )
        if resp.status_code >= 400:
            print(f"  ! could not move {keys} into sprint {sprint_id}: "
                  f"{resp.status_code} {resp.text[:120]}")
            return False
        return True

    async def link(self, blocker_key: str, blocked_key: str) -> None:
        """Create "blocked_key is blocked by blocker_key".

        The POST body's semantics are the reverse of what the field names
        suggest, and this was wrong in an earlier version. Verified against a
        real tenant: posting outwardIssue=A, inwardIssue=B produces a link that
        Jira's own changelog describes, on A, as "This work item is blocked by
        B". So the INWARD issue in the request body is the BLOCKER.

        Read direction (see `_parse_links` in the client) is the opposite
        convention: when fetching issue X, a link entry carrying `inwardIssue`
        means that issue blocks X. Both are now correct; they simply do not
        mirror each other, which is exactly why this needed checking against a
        live instance rather than reasoning.
        """
        await self.client.post(
            "/rest/api/3/issueLink",
            json={
                "type": {"name": "Blocks"},
                "inwardIssue": {"key": blocker_key},
                "outwardIssue": {"key": blocked_key},
            },
        )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Jira project key, e.g. INS")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--sprints",
        type=int,
        default=0,
        help=(
            "Create N sprints on the project's scrum board and walk a third of "
            "the issues through all of them, producing genuine carryover "
            "changelog entries. Needs N>=2 for rule_sprint_carryover to fire."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    base_url = os.getenv("ATLASSIAN_BASE_URL")
    email = os.getenv("ATLASSIAN_EMAIL")
    token = os.getenv("ATLASSIAN_API_TOKEN")
    field = os.getenv("JIRA_TARGET_RELEASE_FIELD", "customfield_10014")

    if not args.dry_run and not all([base_url, email, token]):
        print("Missing credentials. Copy .env.example to .env and fill it in.")
        return 1

    plan = []
    for index in range(args.count):
        summary = SUMMARIES[index % len(SUMMARIES)]
        # 40% blank, matching the client's stated data quality. Seeding clean
        # data produces a system that works beautifully on data you will never
        # be given.
        has_release = random.random() > 0.4
        plan.append(
            {
                "summary": f"{summary} ({index + 1})",
                "target_release": "Q4-2026" if has_release else None,
                "in_progress": index % 4 != 3,
                "comment": (
                    random.choice(BLOCKER_COMMENTS)
                    if index % 3 == 0
                    else random.choice(HEALTHY_COMMENTS)
                    if index % 4 == 1
                    else None
                ),
                "overdue": index % 5 == 0,
            }
        )

    if args.dry_run:
        print(f"Would create {len(plan)} issues in {args.project}:\n")
        for item in plan:
            print(
                f"  {item['summary'][:48]:<50} "
                f"release={item['target_release'] or '(blank)':<9} "
                f"{'in-progress' if item['in_progress'] else 'todo':<12}"
                f"{' overdue' if item['overdue'] else ''}"
                f"{' +comment' if item['comment'] else ''}"
            )
        blank = sum(1 for i in plan if not i["target_release"])
        overdue = sum(1 for i in plan if i["overdue"])
        print(f"\n  Target Release blank: {blank}/{len(plan)} "
              f"({blank / len(plan):.0%})")
        print(f"  Overdue:              {overdue}/{len(plan)}")
        if args.sprints >= 2:
            carried = max(1, len(plan) // 3)
            print(f"  Sprints:              {args.sprints}, walking {carried} "
                  f"issue(s) through all of them")
            print(f"                        -> {args.sprints} Sprint changelog "
                  f"entries each, so rule_sprint_carryover fires")
        elif args.sprints == 1:
            print("  Sprints:              1 — no carryover; needs at least 2")
        else:
            print("  Sprints:              none (pass --sprints 3 for carryover)")
        print("\nRe-run without --dry-run to create these.")
        return 0

    seeder = Seeder(base_url, email, token, field)
    try:
        type_id = await seeder.issue_type_id(args.project)
        print(f"Using issue type id {type_id}\n")

        created: list[str] = []
        for item in plan:
            key = await seeder.create_issue(
                args.project,
                type_id,
                item["summary"],
                target_release=item["target_release"],
                due=(date.today() - timedelta(days=9)) if item["overdue"] else None,
            )
            created.append(key)
            if item["in_progress"]:
                await seeder.transition_to_in_progress(key)
            if item["comment"]:
                await seeder.add_comment(key, item["comment"])
            print(f"  created {key}  {item['summary'][:44]}")

        # A couple of real blocking links, both directions represented.
        if len(created) >= 4:
            await seeder.link(created[1], created[0])
            await seeder.link(created[3], created[2])
            print(f"\n  linked {created[1]} blocks {created[0]}")
            print(f"  linked {created[3]} blocks {created[2]}")

        # Sprint carryover, the one rule that cannot be faked. Jira refuses to
        # backdate a changelog, but every real sprint move writes a real entry,
        # and the rule counts entries. So walk a subset of issues through
        # several sprints and the carryover is genuine rather than simulated.
        if args.sprints >= 2:
            board_id = await seeder.scrum_board_id(args.project)
            if board_id is None:
                print(f"\n  ! {args.project} has no scrum board the Agile API "
                      f"will report. Skipping sprints; everything else is "
                      f"seeded. Create a scrum board in Jira and re-run with "
                      f"--sprints to add carryover data.")
            else:
                print(f"\n  using scrum board {board_id}")
                carried = created[: max(1, len(created) // 3)]
                moved = 0
                for number in range(1, args.sprints + 1):
                    sprint_id = await seeder.create_sprint(
                        board_id, f"{args.project} Sprint {number}"
                    )
                    if sprint_id is None:
                        continue
                    if await seeder.move_to_sprint(sprint_id, carried):
                        moved += 1
                        print(f"  sprint {number}: moved "
                              f"{', '.join(carried)}")
                if moved >= 2:
                    print(f"\n  {len(carried)} issue(s) now carry {moved} real "
                          f"sprint moves — rule_sprint_carryover will fire.")

        print(f"\nSeeded {len(created)} issues into {args.project}.")
        print("\nNext:")
        print("  1. python scripts/check_jira_connection.py")
        print("  2. Restart the service so it re-syncs — findings are read once,")
        print("     at startup.")
        print("  3. Stale detection still needs a shifted clock. Jira will not")
        print("     backdate `created`, so evaluate ahead rather than lowering")
        print("     thresholds to fit fake data:")
        print("       future = datetime.now(timezone.utc) + timedelta(days=30)")
        print("       findings = evaluate(issues, cfg, future)")
        print("       evaluate(issues, cfg, future)")

    finally:
        await seeder.aclose()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
