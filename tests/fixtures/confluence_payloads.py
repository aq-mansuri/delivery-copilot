"""Recorded Confluence Cloud v2 response shapes."""

from __future__ import annotations

from typing import Any

STORAGE_ARCHITECTURE = """
<p>Decision record for the claims vendor integration.</p>
<ac:structured-macro ac:name="toc">
  <ac:parameter ac:name="maxLevel">3</ac:parameter>
</ac:structured-macro>
<h2>Context</h2>
<p>The vendor requires mutual TLS for all callback endpoints.</p>
<ac:structured-macro ac:name="info">
  <ac:rich-text-body><p>Compliance sign-off is required before go-live.</p></ac:rich-text-body>
</ac:structured-macro>
<h2>Decision</h2>
<p>We terminate TLS at the gateway.</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">bash</ac:parameter>
  <ac:plain-text-body><![CDATA[curl --cert client.pem https://vendor/callback]]></ac:plain-text-body>
</ac:structured-macro>
<h3>Rollout table</h3>
<table><tbody>
  <tr><th>Phase</th><th>Date</th></tr>
  <tr><td>Pilot</td><td>Q4 2026</td></tr>
</tbody></table>
<p>See <ac:image><ri:attachment ri:filename="sequence.png" /></ac:image> for the flow.</p>
"""

STORAGE_NO_HEADINGS = "<p>Single paragraph page.</p><p>Second paragraph.</p>"

STORAGE_THIRD_PARTY_MACRO = """
<h2>Status</h2>
<p>Real prose worth keeping.</p>
<ac:structured-macro ac:name="some-vendor-chart">
  <ac:parameter ac:name="config">{"colour":"red","border":true}</ac:parameter>
</ac:structured-macro>
"""


def page(
    page_id: str = "12345",
    *,
    title: str = "ADR: Claims vendor integration",
    storage: str = STORAGE_ARCHITECTURE,
    version: int = 4,
    created_at: str = "2026-08-20T11:30:00.000Z",
    labels: list[str] | None = None,
    webui: str = "/spaces/ARCH/pages/12345/ADR",
) -> dict[str, Any]:
    return {
        "id": page_id,
        "title": title,
        "spaceId": "9001",
        "version": {"number": version, "createdAt": created_at},
        "body": {"storage": {"value": storage, "representation": "storage"}},
        "labels": {
            "results": [{"name": name} for name in (labels or ["architecture"])]
        },
        "_links": {"webui": webui},
    }


def page_list(
    pages: list[dict[str, Any]], *, next_link: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"results": pages, "_links": {}}
    if next_link:
        payload["_links"]["next"] = next_link
    return payload
