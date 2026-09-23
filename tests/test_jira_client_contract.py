"""Contract tests for the Jira client.

These assert against recorded response *shapes* rather than a live tenant. What
they're really protecting: every one of these behaviours is invisible in a happy-
path demo and breaks on first contact with a real customer instance.

The direction tests are the most valuable in the file. An inverted link mapping
reports a genuinely blocked ticket as low-severity context — the false negative
the client explicitly cannot absorb — and it does so silently, with green tests,
until someone manually checks a ticket.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.integrations.atlassian.client import AtlassianError, JiraClient
from app.models.domain import IssueStatusCategory
from tests.fixtures import jira_payloads as fx

BASE_URL = "https://acme.atlassian.net"
SEARCH_URL = f"{BASE_URL}/rest/api/3/search/jql"
FIELD = "customfield_10014"


@pytest.fixture
def client() -> JiraClient:
    return JiraClient(
        base_url=BASE_URL,
        email="svc@acme.com",
        api_token="token",
        target_release_field=FIELD,
        page_size=2,
    )


async def collect(client: JiraClient, jql: str = "project = INS"):
    return [issue async for issue in client.search_issues(jql)]


class TestPagination:
    @respx.mock
    async def test_follows_the_next_page_token(self, client):
        """Three issues, page size two. The cursor must be sent on request two."""
        respx.get(SEARCH_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=fx.search_page(
                        [fx.issue("INS-1"), fx.issue("INS-2")],
                        next_page_token="CAEaAggD",
                        is_last=False,
                    ),
                ),
                httpx.Response(
                    200, json=fx.search_page([fx.issue("INS-3")], is_last=True)
                ),
            ]
        )

        issues = await collect(client)

        assert [i.key for i in issues] == ["INS-1", "INS-2", "INS-3"]
        assert len(respx.calls) == 2
        assert respx.calls[1].request.url.params["nextPageToken"] == "CAEaAggD"

    @respx.mock
    async def test_jql_is_resent_with_the_cursor(self, client):
        """Unlike Confluence, this endpoint wants the full query alongside the
        token. Dropping it returns an unfiltered result set."""
        respx.get(SEARCH_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=fx.search_page([fx.issue("INS-1")], next_page_token="tok"),
                ),
                httpx.Response(200, json=fx.search_page([fx.issue("INS-2")])),
            ]
        )

        await collect(client, jql="project = INS")

        second = respx.calls[1].request.url
        assert second.params["nextPageToken"] == "tok"
        assert second.params["jql"] == "project = INS"

    @respx.mock
    async def test_absent_token_terminates_even_without_is_last(self, client):
        """A final page can arrive with no token and no isLast. Trusting only
        isLast would re-send a stale cursor forever."""
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([fx.issue("INS-1")]))
        )

        issues = await collect(client)
        assert [i.key for i in issues] == ["INS-1"]
        assert len(respx.calls) == 1

    @respx.mock
    async def test_is_last_stops_paging_even_if_a_token_is_present(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page(
                    [fx.issue("INS-1")], next_page_token="stale", is_last=True
                ),
            )
        )

        issues = await collect(client)
        assert len(issues) == 1
        assert len(respx.calls) == 1

    @respx.mock
    async def test_fields_are_requested_explicitly(self, client):
        """/search/jql returns only `id` by default — a silent change from the
        old endpoint. Without this, every mapping KeyErrors on 'fields'."""
        route = respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([]))
        )
        await collect(client)
        requested = route.calls[0].request.url.params["fields"]
        for field in ["summary", "status", "issuelinks", FIELD]:
            assert field in requested

    @respx.mock
    async def test_changelog_expansion_is_always_requested(self, client):
        """Without expand=changelog, stale detection silently sees nothing."""
        route = respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([]))
        )
        await collect(client)
        assert route.calls[0].request.url.params["expand"] == "changelog"

    @respx.mock
    async def test_deprecated_offset_endpoint_is_not_used(self, client):
        """Regression guard against reverting to /rest/api/3/search."""
        route = respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([]))
        )
        await collect(client)
        url = str(route.calls[0].request.url)
        assert "/search/jql" in url
        assert "startAt" not in url


class TestRateLimiting:
    @respx.mock
    async def test_429_is_retried_honouring_retry_after(self, client, monkeypatch):
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr("app.integrations.atlassian.client.asyncio.sleep", fake_sleep)

        respx.get(SEARCH_URL).side_effect = [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json=fx.search_page([fx.issue("INS-1")])),
        ]

        issues = await collect(client)

        assert [i.key for i in issues] == ["INS-1"]
        assert slept == [7.0], "must use the server's Retry-After, not our own backoff"

    @respx.mock
    async def test_gives_up_after_max_retries(self, client, monkeypatch):
        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr("app.integrations.atlassian.client.asyncio.sleep", fake_sleep)
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(429))

        with pytest.raises(AtlassianError):
            await collect(client)


class TestErrorSurfacing:
    @respx.mock
    async def test_403_names_the_likely_cause(self, client):
        """The operator fix is a project permission. Say so."""
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(403))

        with pytest.raises(AtlassianError) as exc:
            await collect(client)
        assert "Browse Projects" in str(exc.value)

    @respx.mock
    async def test_401_is_distinguished_from_403(self, client):
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(401))

        with pytest.raises(AtlassianError) as exc:
            await collect(client)
        assert "auth failed" in str(exc.value).lower()


class TestResilience:
    @respx.mock
    async def test_one_malformed_issue_does_not_kill_the_page(self, client):
        """3,000-issue sync must survive a single unmappable record."""
        broken = {"key": "INS-BAD"}  # no fields at all
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page([fx.issue("INS-1"), broken, fx.issue("INS-3")]),
            )
        )

        issues = await collect(client)
        assert [i.key for i in issues] == ["INS-1", "INS-3"]


class TestAdfFlattening:
    @respx.mock
    async def test_comment_bodies_are_extracted_from_adf(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page(
                    [fx.issue("INS-1", comments=[fx.comment("Vendor confirmed schema")])]
                ),
            )
        )

        issues = await collect(client)
        assert issues[0].comments[0].body == "Vendor confirmed schema"

    @respx.mock
    async def test_multi_paragraph_bodies_are_joined(self, client):
        raw = fx.issue("INS-1", comments=[fx.comment("placeholder")])
        raw["fields"]["comment"]["comments"][0]["body"] = fx.adf_multi_paragraph(
            "Legal is reviewing.", "No ETA yet."
        )
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([raw]))
        )

        issues = await collect(client)
        assert issues[0].comments[0].body == "Legal is reviewing. No ETA yet."


class TestLinkDirection:
    """The highest-value tests in this file.

    Same link type, opposite meaning depending on which key holds the target.
    Invert this and blocked work reports as context.
    """

    @respx.mock
    async def test_inward_issue_means_the_target_blocks_us(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page(
                    [fx.issue("INS-1", issuelinks=[fx.blocked_by_link("PLAT-7")])]
                ),
            )
        )

        link = (await collect(client))[0].links[0]
        assert link.target_key == "PLAT-7"
        assert link.target_blocks_this is True
        assert link.link_type == "blocks"

    @respx.mock
    async def test_outward_issue_means_we_block_the_target(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page(
                    [fx.issue("INS-1", issuelinks=[fx.blocks_other_link("CLM-3")])]
                ),
            )
        )

        link = (await collect(client))[0].links[0]
        assert link.target_key == "CLM-3"
        assert link.target_blocks_this is False

    @respx.mock
    async def test_both_directions_on_one_issue_stay_distinct(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page(
                    [
                        fx.issue(
                            "INS-1",
                            issuelinks=[
                                fx.blocked_by_link("PLAT-7"),
                                fx.blocks_other_link("CLM-3"),
                                fx.relates_link("INS-55"),
                            ],
                        )
                    ]
                ),
            )
        )

        links = {l.target_key: l for l in (await collect(client))[0].links}
        assert links["PLAT-7"].target_blocks_this is True
        assert links["CLM-3"].target_blocks_this is False
        assert links["INS-55"].link_type == "relates_to"

    @respx.mock
    async def test_link_target_status_category_is_mapped(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page(
                    [
                        fx.issue(
                            "INS-1",
                            issuelinks=[
                                fx.blocked_by_link("PLAT-7", status_category="done")
                            ],
                        )
                    ]
                ),
            )
        )

        link = (await collect(client))[0].links[0]
        assert link.target_status_category is IssueStatusCategory.DONE
        assert link.is_resolved() is True


class TestTargetReleaseCoercion:
    @respx.mock
    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("Q4-2026", "Q4-2026"),
            ("  Q4-2026  ", "Q4-2026"),
            ("", None),
            (None, None),
            ({"value": "Q1-2027"}, "Q1-2027"),
            ([{"value": "Q2-2027"}], "Q2-2027"),
            ([], None),
        ],
    )
    async def test_field_shapes_are_normalized(self, client, raw_value, expected):
        """Teams populate this three different ways; blank means absent."""
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page([fx.issue("INS-1", target_release=raw_value)]),
            )
        )

        assert (await collect(client))[0].target_release == expected


class TestChangelogMapping:
    @respx.mock
    async def test_sprint_moves_are_preserved_for_carryover_counting(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page(
                    [
                        fx.issue(
                            "INS-1",
                            changelog=[
                                fx.changelog_entry("Sprint"),
                                fx.changelog_entry(
                                    "Sprint", created="2026-09-08T10:00:00.000+0000"
                                ),
                            ],
                        )
                    ]
                ),
            )
        )

        issue = (await collect(client))[0]
        assert len([e for e in issue.changelog if e.field == "Sprint"]) == 2

    @respx.mock
    async def test_last_activity_uses_changelog_not_updated_field(self, client):
        """ADR-003, enforced at the integration boundary."""
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json=fx.search_page([fx.issue("INS-1", changelog=[])])
            )
        )

        issue = (await collect(client))[0]
        # `updated` is 2026-09-15; real last activity is creation on 2026-07-01.
        assert issue.last_activity_at().date().isoformat() == "2026-07-01"


class TestAgainstRealTenantPayloads:
    """Shapes copied verbatim from a live Atlassian Cloud instance.

    Every other fixture in this file was written from documentation and memory.
    These were recorded, which makes them the only fixtures that can disprove an
    assumption rather than restate one.
    """

    @respx.mock
    async def test_real_link_payload_maps_to_blocked_by(self, client):
        """The mapping that severity depends on, checked against real data.

        Jira's own changelog on the same issue reads "This work item is blocked
        by INS-1", which independently confirms the direction.
        """
        real_link = {
            "id": "10000",
            "type": {
                "id": "10000",
                "name": "Blocks",
                "inward": "is blocked by",
                "outward": "blocks",
                "self": "https://x.atlassian.net/rest/api/3/issueLinkType/10000",
            },
            "inwardIssue": {
                "id": "10044",
                "key": "INS-1",
                "self": "https://x.atlassian.net/rest/api/3/issue/10044",
                "fields": {
                    "summary": "Integrate claims vendor callback endpoint (1)",
                    "status": {
                        "name": "In Progress",
                        "statusCategory": {"id": 4, "key": "indeterminate"},
                    },
                    "priority": {"name": "Medium", "id": "3"},
                    "issuetype": {"name": "Story", "subtask": False},
                },
            },
        }
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json=fx.search_page([fx.issue("INS-2", issuelinks=[real_link])])
            )
        )

        link = (await collect(client))[0].links[0]
        assert link.target_key == "INS-1"
        assert link.target_blocks_this is True
        assert link.link_type == "blocks"
        assert link.target_status_category is IssueStatusCategory.IN_PROGRESS

    @respx.mock
    async def test_link_changelog_item_has_no_fieldId(self, client):
        """Real finding: status items carry `fieldId`, Link items do not.

        Indexing item["fieldId"] would KeyError on any linked issue.
        """
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.search_page(
                    [fx.issue("INS-2", changelog=[fx.link_changelog_entry("INS-1")])]
                ),
            )
        )

        entry = (await collect(client))[0].changelog[0]
        assert entry.field == "Link"
        assert entry.to_value == "This work item is blocked by INS-1"

    @respx.mock
    async def test_extra_keys_in_real_payloads_are_ignored(self, client):
        """Real responses carry self, iconUrl, avatarUrls, colorName and more.

        A superset is harmless, but only if nothing iterates fields blindly.
        """
        raw = fx.issue("INS-1")
        raw["fields"]["status"]["iconUrl"] = "https://x/icon.png"
        raw["fields"]["status"]["statusCategory"]["colorName"] = "yellow"
        raw["fields"]["comment"]["total"] = 0
        raw["self"] = "https://x.atlassian.net/rest/api/3/issue/10055"

        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([raw]))
        )
        assert (await collect(client))[0].key == "INS-1"


class TestDueDateIsDateOnly:
    """`duedate` is the one field Jira returns as a calendar day.

    Every other timestamp comes back as `2026-09-15T09:00:00.000+0000`, so the
    shared parser produced offset-aware datetimes and nobody noticed it could
    also produce a naive one. The first real tenant with a due date raised
    TypeError inside a risk rule and aborted the whole project's sync — every
    issue after it went unread, and the operator saw "1 of 1 projects were not
    fully read" with no clue why.
    """

    @respx.mock
    async def test_a_date_only_due_date_is_timezone_aware(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json=fx.search_page([fx.issue(duedate="2026-09-30")])
            )
        )
        issue = (await collect(client))[0]
        assert issue.due_date is not None
        assert issue.due_date.tzinfo is not None

    @respx.mock
    async def test_the_due_date_can_be_compared_against_now(self, client):
        """The assertion the missing test would have made. Comparing a parsed
        issue against the clock is what every rule does."""
        from datetime import datetime, timezone

        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json=fx.search_page([fx.issue(duedate="2026-09-30")])
            )
        )
        issue = (await collect(client))[0]
        assert issue.due_date < datetime(2099, 1, 1, tzinfo=timezone.utc)

    @respx.mock
    async def test_a_due_date_lands_at_the_end_of_its_day(self, client):
        """A deadline of the 30th is met by work finished on the 30th. Parsing
        it to midnight makes the issue overdue from the first second of the day
        it is due — a day early, on a number a client will check."""
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200, json=fx.search_page([fx.issue(duedate="2026-09-30")])
            )
        )
        issue = (await collect(client))[0]
        assert (issue.due_date.year, issue.due_date.month, issue.due_date.day) == (
            2026, 9, 30,
        )
        assert issue.due_date.hour == 23

    @respx.mock
    async def test_a_full_timestamp_is_still_parsed_unchanged(self, client):
        """The fix must not reinterpret the fields that were always correct."""
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([fx.issue()]))
        )
        issue = (await collect(client))[0]
        assert issue.updated_at.tzinfo is not None
        assert (issue.updated_at.hour, issue.updated_at.minute) == (9, 0)

    @respx.mock
    async def test_no_due_date_stays_none(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([fx.issue()]))
        )
        assert (await collect(client))[0].due_date is None


class TestSprintName:
    """`sprint_name` was never populated, so every carryover finding read
    "Carried over 3 times between sprints (currently: no sprint)" — the count
    correct, the one piece of context a lead needs missing.

    It is derived from the changelog rather than from the Sprint custom field.
    The field id is per-tenant (customfield_10020 here, something else
    elsewhere), so reading it means another id to resolve, inject and get wrong.
    The changelog is already fetched for the carryover rule and already says
    where the issue landed.
    """

    @respx.mock
    async def test_the_latest_sprint_move_names_the_current_sprint(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([
                fx.issue(changelog=[
                    fx.changelog_entry("Sprint", created="2026-09-01T09:00:00.000+0000",
                                       from_string="", to_string="INS Sprint 1"),
                    fx.changelog_entry("Sprint", created="2026-09-14T09:00:00.000+0000",
                                       from_string="INS Sprint 1", to_string="INS Sprint 2"),
                ])
            ]))
        )
        assert (await collect(client))[0].sprint_name == "INS Sprint 2"

    @respx.mock
    async def test_the_latest_move_wins_regardless_of_payload_order(self, client):
        """Jira returns histories newest-first on some tenants and oldest-first
        on others. Taking the last element of the list rather than the latest
        timestamp reads the wrong one on half of them."""
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([
                fx.issue(changelog=[
                    fx.changelog_entry("Sprint", created="2026-09-14T09:00:00.000+0000",
                                       from_string="INS Sprint 1", to_string="INS Sprint 2"),
                    fx.changelog_entry("Sprint", created="2026-09-01T09:00:00.000+0000",
                                       from_string="", to_string="INS Sprint 1"),
                ])
            ]))
        )
        assert (await collect(client))[0].sprint_name == "INS Sprint 2"

    @respx.mock
    async def test_an_issue_removed_from_every_sprint_has_no_sprint(self, client):
        """Jira records removal as a move to an empty string. Reporting that as
        a sprint named "" is worse than reporting none."""
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([
                fx.issue(changelog=[
                    fx.changelog_entry("Sprint", created="2026-09-01T09:00:00.000+0000",
                                       from_string="", to_string="INS Sprint 1"),
                    fx.changelog_entry("Sprint", created="2026-09-14T09:00:00.000+0000",
                                       from_string="INS Sprint 1", to_string=""),
                ])
            ]))
        )
        assert (await collect(client))[0].sprint_name is None

    @respx.mock
    async def test_an_issue_that_never_moved_has_no_sprint(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([fx.issue()]))
        )
        assert (await collect(client))[0].sprint_name is None

    @respx.mock
    async def test_other_changelog_fields_are_ignored(self, client):
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json=fx.search_page([
                fx.issue(changelog=[
                    fx.changelog_entry("status", created="2026-09-14T09:00:00.000+0000",
                                       from_string="To Do", to_string="In Progress"),
                ])
            ]))
        )
        assert (await collect(client))[0].sprint_name is None
