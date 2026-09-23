"""End-to-end demo against synthetic data. No Atlassian, no API keys, no network.

    python scripts/demo.py

Exists for two reasons. First, so the pipeline is runnable and inspectable from
day one rather than only after a tenant is provisioned. Second, because a demo
that needs live credentials is a demo that fails in front of a client when their
VPN blocks you — having an offline path is a delivery habit, not a toy.

The synthetic data deliberately includes the failure modes the rules exist to
catch: a ticket that looks healthy but is dead, a blank Target Release, a real
cross-team blocker, and a sync that comes up short.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Run as `python scripts/demo.py` from the project root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.report import UnreportableSyncError, build_report
from app.core.sandbox import sandbox_issues
from app.core.sync_result import CompleteSync, PartialReason, PartialSync
from app.models.domain import (
    Page,
    PageSection,
)
from app.rag.chunking import chunk_page
from app.rag.retrieval import HybridRetriever
from app.risk.rules import RiskConfig, evaluate

NOW = datetime.now(timezone.utc)
CFG = RiskConfig(base_url="https://acme.atlassian.net")


def build_pages() -> list[Page]:
    return [
        Page(
            id="adr-tls",
            space_key="ARCH",
            title="ADR: Claims vendor TLS",
            url="https://acme.atlassian.net/wiki/spaces/ARCH/pages/adr-tls",
            version=3,
            updated_at=NOW - timedelta(days=30),
            labels=["architecture"],
            sections=[
                PageSection(
                    heading="Context",
                    level=2,
                    text=(
                        "The claims vendor requires mutual TLS on every callback "
                        "endpoint. Compliance sign-off is required before go-live, "
                        "and legal review has historically taken three weeks."
                    ),
                ),
                PageSection(
                    heading="Decision",
                    level=2,
                    text=(
                        "We terminate TLS at the gateway rather than in each "
                        "service. Certificates rotate quarterly via the platform "
                        "team's existing process."
                    ),
                ),
            ],
        ),
        Page(
            id="sprint-14",
            space_key="DEL",
            title="Sprint 14 notes",
            url="https://acme.atlassian.net/wiki/spaces/DEL/pages/sprint-14",
            version=1,
            updated_at=NOW - timedelta(days=3),
            labels=["delivery"],
            sections=[
                PageSection(
                    heading="Blockers",
                    level=2,
                    text=(
                        "INS-102 is waiting on the platform team to ship the "
                        "partner auth endpoint (PLAT-7). No committed date yet."
                    ),
                )
            ],
        ),
    ]


def main() -> int:
    issues = sandbox_issues(NOW)

    print("=" * 72)
    print("RISK ENGINE (deterministic — no model involved)")
    print("=" * 72)

    findings = evaluate(issues, CFG, NOW)
    for finding in sorted(findings, key=lambda f: (f.level.value, f.issue_key)):
        print(f"\n[{finding.level.value.upper():<6}] {finding.issue_key}  ({finding.rule_id})")
        print(f"         {finding.detail}")
        for citation in finding.citations:
            print(f"         -> {citation.url}")

    flagged = {f.issue_key for f in findings}
    clean = [i.key for i in issues if i.key not in flagged]
    print(f"\nNo findings for: {', '.join(clean) or '(none)'}")

    print("\n" + "=" * 72)
    print("SYNC GATE")
    print("=" * 72)

    complete = CompleteSync(
        started_at=NOW,
        issues_seen=len(issues),
        projects_requested=["INS"],
        findings=findings,
        projects_covered=["INS"],
    )
    report = build_report(complete)
    print(f"\nComplete sync -> report built: {report.total_findings()} findings")
    print(f"  rule counts: {report.rule_counts}")

    partial = PartialSync(
        started_at=NOW,
        issues_seen=2,
        projects_requested=["INS", "CLM"],
        partial_findings=findings[:1],
        projects_covered=["INS"],
        projects_incomplete=["CLM"],
        reason=PartialReason.PERMISSION_DENIED,
        detail="403 on project CLM",
    )
    try:
        build_report(partial)
        print("\nBUG: partial sync produced a report")
    except UnreportableSyncError as exc:
        print(f"\nPartial sync -> blocked, as designed:\n  {exc}")
    print(f"  operator sees: {partial.operator_message()}")

    print("\n" + "=" * 72)
    print("RETRIEVAL")
    print("=" * 72)

    try:
        from tests.fixtures.fake_embedder import FakeEmbedder
    except ImportError:
        print("\n(skipped — tests/fixtures/fake_embedder.py not found)")
        return 0

    chunks = [c for page in build_pages() for c in chunk_page(page)]
    retriever = HybridRetriever(chunks, FakeEmbedder(), candidate_pool=10)

    for query in [
        "what is blocking INS-102",
        "why do we terminate TLS at the gateway",
        "who signs off before go live",
    ]:
        print(f"\n  ? {query}")
        for hit in retriever.search(query, top_k=2):
            print(f"      {hit.chunk.citation_label()}  [{hit.retrieval_reason()}]")
            print(f"      {hit.chunk.text[:90]}...")

    print("\n" + "=" * 72)
    print("Note: the model has not been called once. Everything above is rules")
    print("and search. The LLM's job starts on Day 4 — explaining these findings,")
    print("not finding them.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
