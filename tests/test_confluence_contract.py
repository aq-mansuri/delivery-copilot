"""Contract tests for the Confluence client and storage-format parser.

The cursor-pagination tests matter most. Offset paging bugs show up on page one
in any demo; cursor bugs show up on page two, against a real space, usually as
an infinite loop.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.integrations.atlassian.confluence import ConfluenceClient, ConfluenceError
from app.integrations.atlassian.storage_format import parse_storage
from tests.fixtures import confluence_payloads as fx

BASE_URL = "https://acme.atlassian.net"
PAGES_URL = f"{BASE_URL}/wiki/api/v2/pages"


@pytest.fixture
def client() -> ConfluenceClient:
    return ConfluenceClient(
        base_url=BASE_URL, email="svc@acme.com", api_token="token", page_size=2
    )


async def collect(client: ConfluenceClient, space="ARCH", labels=None):
    return [page async for page in client.iter_pages(space, labels=labels)]


class TestCursorPagination:
    @respx.mock
    async def test_follows_the_server_cursor(self, client):
        """No `total` field exists here — the cursor is the only terminator."""
        respx.get(PAGES_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=fx.page_list(
                        [fx.page("1"), fx.page("2")],
                        next_link="/wiki/api/v2/pages?cursor=abc123",
                    ),
                ),
                httpx.Response(200, json=fx.page_list([fx.page("3")])),
            ]
        )

        pages = await collect(client)

        assert [p.id for p in pages] == ["1", "2", "3"]
        assert len(respx.calls) == 2
        assert "cursor=abc123" in str(respx.calls[1].request.url)

    @respx.mock
    async def test_absent_next_link_terminates(self, client):
        route = respx.get(PAGES_URL).mock(
            return_value=httpx.Response(200, json=fx.page_list([fx.page("1")]))
        )
        pages = await collect(client)
        assert len(pages) == 1
        assert route.call_count == 1

    @respx.mock
    async def test_original_filters_are_not_resent_with_the_cursor(self, client):
        """Re-sending `space-key` alongside the cursor double-applies the filter
        and on some tenants resets paging — an infinite loop past page one."""
        respx.get(PAGES_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=fx.page_list(
                        [fx.page("1")],
                        next_link="/wiki/api/v2/pages?cursor=abc123",
                    ),
                ),
                httpx.Response(200, json=fx.page_list([fx.page("2")])),
            ]
        )

        await collect(client)

        second_request = respx.calls[1].request
        assert "cursor=abc123" in str(second_request.url)
        assert "space-key" not in str(second_request.url)

    @respx.mock
    async def test_absolute_next_links_are_handled(self, client):
        """Confluence returns `next` absolute on some tenants, relative on others."""
        respx.get(PAGES_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=fx.page_list(
                        [fx.page("1")],
                        next_link=f"{BASE_URL}/wiki/api/v2/pages?cursor=xyz",
                    ),
                ),
                httpx.Response(200, json=fx.page_list([fx.page("2")])),
            ]
        )

        pages = await collect(client)
        assert [p.id for p in pages] == ["1", "2"]


class TestRequestShape:
    @respx.mock
    async def test_storage_body_format_is_requested(self, client):
        """Rendered HTML loses macro identity, which the parser needs."""
        route = respx.get(PAGES_URL).mock(
            return_value=httpx.Response(200, json=fx.page_list([]))
        )
        await collect(client)
        assert route.calls[0].request.url.params["body-format"] == "storage"

    @respx.mock
    async def test_labels_narrow_the_fetch(self, client):
        """~200 of a few thousand pages matter. Filter before embedding, not after."""
        route = respx.get(PAGES_URL).mock(
            return_value=httpx.Response(200, json=fx.page_list([]))
        )
        await collect(client, labels=["architecture", "compliance"])
        assert route.calls[0].request.url.params["label"] == "architecture,compliance"


class TestErrorHandling:
    @respx.mock
    async def test_403_names_the_space_permission(self, client):
        respx.get(PAGES_URL).mock(return_value=httpx.Response(403))
        with pytest.raises(ConfluenceError) as exc:
            await collect(client)
        assert "View permission" in str(exc.value)

    @respx.mock
    async def test_429_honours_retry_after(self, client, monkeypatch):
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(
            "app.integrations.atlassian.confluence.asyncio.sleep", fake_sleep
        )
        respx.get(PAGES_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "5"}),
                httpx.Response(200, json=fx.page_list([fx.page("1")])),
            ]
        )

        pages = await collect(client)
        assert len(pages) == 1
        assert slept == [5.0]

    @respx.mock
    async def test_one_bad_page_does_not_kill_the_space(self, client):
        broken = {"id": "999"}  # no version block
        respx.get(PAGES_URL).mock(
            return_value=httpx.Response(
                200, json=fx.page_list([fx.page("1"), broken, fx.page("3")])
            )
        )
        pages = await collect(client)
        assert [p.id for p in pages] == ["1", "3"]


class TestStorageFormatParsing:
    def test_sections_split_on_headings(self):
        sections = parse_storage(fx.STORAGE_ARCHITECTURE)
        headings = [s.heading for s in sections]
        assert headings == [None, "Context", "Decision", "Rollout table"]
        assert sections[0].level == 0
        assert sections[3].level == 3

    def test_navigational_macros_are_dropped(self):
        """A TOC macro retrieves for everything and answers nothing."""
        text = "\n".join(s.text for s in parse_storage(fx.STORAGE_ARCHITECTURE))
        assert "maxLevel" not in text
        assert "3" not in text.split("Q4")[0].replace("Q4 2026", "")

    def test_prose_macros_are_kept(self):
        """The info panel is usually where the real caveat lives."""
        sections = {s.heading: s.text for s in parse_storage(fx.STORAGE_ARCHITECTURE)}
        assert "Compliance sign-off is required" in sections["Context"]

    def test_code_block_cdata_is_extracted(self):
        """CDATA bodies vanish silently without an explicit handler."""
        sections = {s.heading: s.text for s in parse_storage(fx.STORAGE_ARCHITECTURE)}
        assert "curl --cert client.pem" in sections["Decision"]

    def test_macro_parameters_are_never_content(self):
        text = "\n".join(s.text for s in parse_storage(fx.STORAGE_ARCHITECTURE))
        assert "bash" not in text

    def test_unknown_third_party_macros_are_skipped(self):
        sections = parse_storage(fx.STORAGE_THIRD_PARTY_MACRO)
        text = "\n".join(s.text for s in sections)
        assert "Real prose worth keeping" in text
        assert "colour" not in text

    def test_table_cells_stay_on_one_line_per_row(self):
        sections = {s.heading: s.text for s in parse_storage(fx.STORAGE_ARCHITECTURE)}
        assert "| Pilot | Q4 2026" in sections["Rollout table"]

    def test_attachments_are_named_not_dropped(self):
        text = "\n".join(s.text for s in parse_storage(fx.STORAGE_ARCHITECTURE))
        assert "[attachment: sequence.png]" in text

    def test_page_without_headings_yields_one_section(self):
        sections = parse_storage(fx.STORAGE_NO_HEADINGS)
        assert len(sections) == 1
        assert sections[0].heading is None
        assert "Single paragraph page." in sections[0].text
        assert "Second paragraph." in sections[0].text

    def test_empty_storage_yields_nothing(self):
        assert parse_storage("") == []


class TestPageMapping:
    @respx.mock
    async def test_url_is_absolute_and_browsable(self, client):
        """Citations must be clickable — a relative path is not a citation."""
        respx.get(PAGES_URL).mock(
            return_value=httpx.Response(200, json=fx.page_list([fx.page("1")]))
        )
        page = (await collect(client))[0]
        assert page.url.startswith("https://")
        assert "/spaces/ARCH/pages/12345/ADR" in page.url

    @respx.mock
    async def test_version_and_labels_are_captured(self, client):
        respx.get(PAGES_URL).mock(
            return_value=httpx.Response(
                200,
                json=fx.page_list(
                    [fx.page("1", version=7, labels=["architecture", "vendor"])]
                ),
            )
        )
        page = (await collect(client))[0]
        assert page.version == 7
        assert page.labels == ["architecture", "vendor"]
