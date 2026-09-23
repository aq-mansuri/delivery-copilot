"""Confluence corpus for the sandbox.

Moved here from `scripts/` because the running service imports it: the API
builds its index from `load_offline_corpus()`. A service depending on something
in `scripts/` means the container has to ship `scripts/`, and a reader cannot
tell which scripts are load-bearing and which are one-off tools.

Same dependency-direction problem as `demo.py` importing `FakeEmbedder` from
`tests/`. `scripts/` holds things you run; `app/` holds things that run.

One source of truth for two consumers: `seed_confluence.py` pushes these to a
real tenant, and `load_offline_corpus()` builds Page objects directly so the
eval harness runs with no network at all.

## The content is engineered against the eval set

Every page is written so its eval case tests what it claims to. Three rules
followed throughout:

**Semantic targets never contain the query's words.** The release plan does not
say "holding up"; it says "cannot progress" and "stalled". If the target page
used the query's phrasing, a semantic case would be a keyword case wearing a
semantic label, and it would pass without the embeddings doing anything.

**Negative cases have topically adjacent bait.** "Who approved the Acme security
exception?" presupposes an exception that does not exist — but there IS an Acme
contract page and there IS a security page, both of which retrieval will want to
return. A negative case with no nearby content passes trivially and proves
nothing.

**Cross-source cases genuinely split the answer.** The O7 page names the blocking
issues but not the schedule impact; the release plan carries the dates but not
the issue keys. Neither page alone answers the question, so retrieving one and
stopping produces an answer that reads complete and is not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedPage:
    page_id: str
    title: str
    labels: tuple[str, ...]
    storage: str


PAGES: list[SeedPage] = [
    SeedPage(
        page_id="release-plan",
        title="Q4 2026 Release Plan",
        labels=("delivery", "release"),
        storage="""
<p>Scope and sequencing for the Q4 release across the insurance core teams.</p>
<ac:structured-macro ac:name="toc"><ac:parameter ac:name="maxLevel">2</ac:parameter></ac:structured-macro>
<h2>Priorities</h2>
<p>Three items carry board commitments this quarter, in order: the claims vendor
integration, the compliance export for the regulator, and partner authentication
on the claims portal. Everything else is discretionary and may slip without
escalation.</p>
<h2>Sequencing</h2>
<p>Partner authentication must land before the claims portal work can proceed,
because the portal consumes the auth endpoint owned by the platform team. The
compliance export has no upstream dependencies and can run in parallel.</p>
<h2>Current state</h2>
<p>Two of the three committed items cannot progress at present. The claims vendor
integration is stalled pending an external confirmation, and partner
authentication is held outside this group. The compliance export is proceeding
as planned.</p>
<h2>Schedule</h2>
<table><tbody>
<tr><th>Item</th><th>Target</th><th>Confidence</th></tr>
<tr><td>Claims vendor integration</td><td>Late Q4</td><td>Low</td></tr>
<tr><td>Compliance export</td><td>Mid Q4</td><td>High</td></tr>
<tr><td>Partner authentication</td><td>Not committed</td><td>Unknown</td></tr>
</tbody></table>
""",
    ),
    SeedPage(
        page_id="epic-o7",
        title="Epic O7: Claims Vendor Integration",
        labels=("delivery", "epic"),
        storage="""
<p>O7 covers the end-to-end integration with the Acme claims processing platform.</p>
<h2>Scope</h2>
<p>Callback endpoint, mutual TLS termination, claims intake validation, and the
audit trail required for regulated record-keeping.</p>
<h2>Issues under this epic</h2>
<table><tbody>
<tr><th>Issue</th><th>Summary</th><th>State</th></tr>
<tr><td>INS-101</td><td>Integrate claims vendor callback endpoint</td><td>Impeded</td></tr>
<tr><td>INS-102</td><td>Enable partner auth on claims portal</td><td>Impeded</td></tr>
<tr><td>INS-107</td><td>Implement compliance export for regulator</td><td>Proceeding</td></tr>
</tbody></table>
<h2>Items unable to proceed</h2>
<p>INS-101 has been unable to advance since the vendor was asked to confirm the
callback schema. No response has been received. INS-102 is owned here but
depends on work held by the platform team.</p>
<ac:structured-macro ac:name="note"><ac:rich-text-body>
<p>INS-107 is the only item under O7 currently moving without external
constraint.</p>
</ac:rich-text-body></ac:structured-macro>
""",
    ),
    SeedPage(
        page_id="sprint-notes",
        title="Sprint 14 Delivery Notes",
        labels=("delivery",),
        storage="""
<h2>Summary</h2>
<p>Sprint 14 closed with two committed items unfinished.</p>
<h2>Items that did not move</h2>
<p>INS-101 has sat untouched for three weeks. The team is awaiting schema
confirmation from Acme and has had no reply to two follow-ups. The assignee has
moved onto other work in the meantime, so the board shows the issue as active
when in practice nothing is happening.</p>
<p>INS-102 cannot advance until the platform team ships the partner auth
endpoint tracked as PLAT-7. No committed date has been given for that work.</p>
<h2>Carried forward</h2>
<p>INS-103, the legacy policy record migration, moved to Sprint 15. This is its
third carry-over. Scoping remains incomplete.</p>
""",
    ),
    SeedPage(
        page_id="dependencies",
        title="Cross-Team Ownership and Hand-offs",
        labels=("delivery", "architecture"),
        storage="""
<p>Work items in the insurance core stream that require delivery by groups
outside this reporting line.</p>
<h2>Platform engineering</h2>
<p>The partner authentication endpoint (PLAT-7) is owned by platform
engineering. Claims portal work consumes it and cannot be completed ahead of it.
Platform have not committed to a date and are not represented in this delivery
review.</p>
<h2>External vendors</h2>
<p>Acme owns the callback schema definition. Our integration work is paused
until they confirm it. Escalation path runs through the vendor management
function rather than engineering.</p>
<h2>Legal and compliance</h2>
<p>Sign-off on the regulated data flow sits with the compliance function.
Historical turnaround on similar reviews has been three weeks, and the review
has not yet been requested for this integration.</p>
""",
    ),
    SeedPage(
        page_id="security-requirements",
        title="Controls Required Before Claims Integration Go-Live",
        labels=("architecture", "compliance"),
        storage="""
<p>Controls that must be satisfied before the claims vendor integration may
process live policyholder data.</p>
<h2>Controls not yet satisfied</h2>
<p>Mutual TLS is specified but certificates have not been issued for the
production gateway. Penetration testing of the callback endpoint has not been
scheduled. The audit trail for approval actions is designed but not implemented.</p>
<h2>Controls satisfied</h2>
<p>Data residency review is complete. Encryption at rest is inherited from the
existing document storage platform.</p>
<h2>Exceptions process</h2>
<p>Any control not satisfied at go-live requires a documented exception approved
by the compliance function. No exceptions have been raised for this
integration.</p>
""",
    ),
    SeedPage(
        page_id="acme-contract",
        title="Acme Claims Platform — Contract Summary",
        labels=("vendor", "commercial"),
        storage="""
<h2>Term</h2>
<p>Three-year agreement with Acme Claims Technologies, renewing annually
thereafter. Current term ends 31 March 2027.</p>
<h2>Service levels</h2>
<p>Acme commit to a 5 business day turnaround on integration support requests.
This commitment has not been met during the current integration; two schema
confirmation requests remain unanswered beyond the agreed window.</p>
<h2>Commercial risk</h2>
<p>The agreement contains no service credit mechanism for integration support
delays, so the SLA breach carries no financial remedy. Renewal negotiation is
the only available leverage and does not open until January.</p>
<h2>Delivery risk</h2>
<p>Our Q4 commitment depends on vendor responsiveness that we cannot contractually
compel.</p>
""",
    ),
    SeedPage(
        page_id="meridian-contract",
        title="Meridian Document Storage — Contract Summary",
        labels=("vendor", "commercial"),
        storage="""
<h2>Term</h2>
<p>Rolling annual agreement with Meridian Data Services for document storage and
retrieval. Auto-renews each July unless cancelled with 90 days notice.</p>
<h2>Service levels</h2>
<p>99.9% availability with service credits for breaches. No breaches recorded in
the past four quarters.</p>
<h2>Status</h2>
<p>Stable. No open commercial issues. The storage upgrade tracked as INS-108 is
in scope of the existing agreement and requires no contract change.</p>
""",
    ),
    SeedPage(
        page_id="acme-delivery-notes",
        title="Acme Integration — Delivery Status",
        labels=("delivery", "vendor"),
        storage="""
<h2>Where things stand</h2>
<p>The integration is not progressing. The callback schema question was raised
with Acme three weeks ago and has not been answered. Engineering work is
complete up to the point where the schema is required.</p>
<h2>What we are doing about it</h2>
<p>Two follow-ups have been sent through the support channel. Vendor management
have been asked to escalate. No response has been received to either.</p>
<h2>Impact</h2>
<p>If confirmation does not arrive within two weeks, the late Q4 target is not
achievable and the commitment will need to be restated to the board.</p>
""",
    ),
]


def load_offline_corpus():
    """Build Page objects directly, no Atlassian required.

    Lets the eval harness run today rather than after the tenant is provisioned —
    the same reason `demo.py` exists.
    """
    from datetime import datetime, timedelta, timezone

    from app.integrations.atlassian.storage_format import parse_storage
    from app.models.domain import Page

    now = datetime.now(timezone.utc)
    return [
        Page(
            id=seed.page_id,
            space_key="ARCH",
            title=seed.title,
            url=f"https://acme.atlassian.net/wiki/spaces/ARCH/pages/{seed.page_id}",
            version=1,
            updated_at=now - timedelta(days=index * 3),
            labels=list(seed.labels),
            sections=parse_storage(seed.storage),
        )
        for index, seed in enumerate(PAGES)
    ]
