---
name: atlassian-integration
description: Use when writing or debugging code that calls the Jira Cloud or Confluence Cloud REST APIs — searching issues, reading changelogs or issue links, paginating results, resolving custom field ids, parsing Confluence storage format, or diagnosing a 400/401/403 from either product. Covers the deprecated /rest/api/3/search endpoint, cursor pagination differences between the two products, link direction semantics, and which fields cannot be trusted. Trigger on Jira, Confluence, Atlassian, JQL, issuelinks, changelog, customfield, storage format, ADF, or an Atlassian HTTP error.
---

# Atlassian integration

Hard-won specifics. Each of these cost real debugging time and none are obvious
from the docs alone. Verified against a live tenant unless marked otherwise.

## Searching issues

`/rest/api/3/search` is **deprecated**. Use `/rest/api/3/search/jql`.

Differences that break naive ports:

- Pagination is `nextPageToken` + `isLast`, not `startAt` + `total`. There is no
  `total` — a separate approximate-count endpoint exists for that.
- **Check both terminators.** A final page can arrive with `isLast` absent and no
  token. Trusting `isLast` alone re-sends a stale cursor forever.
- **`fields` must be requested explicitly.** The endpoint returns only `id` by
  default. Omit it and every mapping fails with `KeyError: 'fields'`, which looks
  like a parser bug and is not.
- `expand=changelog` still works here (verified live).
- The full `jql` must be re-sent alongside the cursor. Unlike Confluence.

## Fields you cannot trust

**`fields.updated`** is bumped by automation, bulk edits and field syncs. A dead
ticket shows a fresh timestamp. Compute activity from the max of created, comment
timestamps and changelog entry timestamps instead.

**Status names** differ per project; `statusCategory.key` (`new` /
`indeterminate` / `done`) is stable. Never write a rule against a status name.

**Custom field ids** are per-tenant. `customfield_10014` on one site is a
different field elsewhere, and naming an unknown id makes Jira 400 the whole
search. Resolve via `GET /rest/api/3/field` and inject the id.

## Issue link direction

The easiest thing in the whole integration to invert, and inverting it is
silent.

**Reading** issue X: a link entry carrying `inwardIssue` means that issue
**blocks X** (the type's `inward` description reads "is blocked by"). An entry
carrying `outwardIssue` means **X blocks it**.

**Writing** via `POST /rest/api/3/issueLink`: the semantics are the reverse of
what the field names suggest. Posting `outwardIssue: A, inwardIssue: B` produces
a link Jira describes, on A, as "This work item is blocked by B".

Both directions verified against a live tenant. The independent check is the
changelog, which spells it out in English: `toString` reads
`"This work item is blocked by INS-1"`.

Match link types by the `inward`/`outward` **description phrase**, not the type
name — tenants create custom link types freely.

## Changelog

Top-level on the issue: `changelog.histories[].items[]`, each with `field`,
`fromString`, `toString`.

**Status items carry `fieldId`; Link items do not.** `item["fieldId"]` KeyErrors
on any linked issue. Use `item["field"]`.

Link creation writes a changelog entry, so it counts as activity.

## Confluence

- Paginates by **cursor** (`_links.next`), not offset. No `total`.
- **Do not re-send the original params with the cursor.** It double-applies the
  filter and on some tenants resets paging — an infinite loop that only appears
  past page one.
- `next` is absolute on some tenants, relative on others.
- Request `body-format=storage`. Rendered HTML loses macro identity.
- Storage format is XHTML plus `ac:`-namespaced elements, not HTML. Drop
  navigational macros (`toc`, `children`, `pagetree`); keep prose macros (`info`,
  `note`, `warning`, `expand`).
- **Code blocks arrive as CDATA**, reported by `HTMLParser.unknown_decl`, not
  `handle_data`. Without a handler they vanish silently — and in an architecture
  space the code block is often the decision.
- Labels have no v2 write endpoint; use v1 `/wiki/rest/api/content/{id}/label`.

## Errors

Never call `raise_for_status()` on an Atlassian 4xx. The body carries
`errorMessages[]` and a field-level `errors{}` object, and the helper throws it
away, leaving a bare status code.

| Code | Usual cause |
|---|---|
| 400 on search | Unknown custom field id in `fields`, or an unsupported param |
| 401 | Bad email/token pair |
| 403 | Service account lacks Browse Projects (Jira) or View (Confluence) |
| 429 | Honour `Retry-After`; do not substitute your own backoff |

Diagnose a 400 by bisecting parameters rather than guessing — see
`scripts/diagnose_search.py`.

## Adding a status to a workflow

`POST /rest/api/3/workflows/update`. Three attempts, and the error messages
describe none of this:

- The top-level `statuses` array must list **every** status the workflow
  references, not only the new one. Send just the new one and you get "Workflow
  refers to a missing status reference".
- An **existing** status is declared with both `id` and `statusReference` (its
  id again). A **new** status has no `id`, and its `statusReference` must be a
  UUID you invent — Jira creates the status and binds it to that reference.
  Numeric references for new statuses fail with "The reference X is not a UUID".
- Do **not** create the status first with `POST /rest/api/3/statuses`. The
  update wants to create it, and a pre-existing one fails the whole call with
  "Status name ... already in use". Delete the orphan
  (`DELETE /rest/api/3/statuses?id=`) and let the update own it.
- `version` must be the workflow's current version object or the write is
  rejected as a stale edit. Read it from `POST /rest/api/3/workflows`
  (`{"workflowIds": [entityId]}`), not from `/workflow/search`, which does not
  carry it.
- Transition ids are strings, workflow-local, and must not collide. Resolve a
  transition to use by matching `to.name`, never by hardcoding an id — ids shift
  the moment anyone edits the workflow.

**Check the blast radius first.** `GET /rest/api/3/workflowscheme/project`
returns `projectIds`. A "Software Simplified Workflow for Project X" scheme is
normally dedicated, but the default Jira workflow is shared by every project on
the tenant, and adding a status to that edits all of them.

A new status's `statusCategory` is what the rules see. `Blocked`, `Escalated`,
`On Hold` and `Waiting for customer` are all `IN_PROGRESS`, so adding one
changes the board and changes nothing in a rule keyed off the category — which
is the intended outcome, not a gap.

## Seeding a test tenant

Jira will not let you backdate `created`, comments or changelog entries.
Stale-detection data therefore cannot be created directly.

Do **not** lower thresholds to make rules fire on fresh data — that ships
thresholds tuned to fake data. Instead, pass a shifted clock, which the rules
accept as a parameter for exactly this reason:

    future = datetime.now(timezone.utc) + timedelta(days=30)
    findings = evaluate(issues, cfg, future)

`duedate` is a plain date field and **can** be backdated.

Sprint carryover needs real sprint transitions on a scrum board; move a few
issues by hand in the UI.
