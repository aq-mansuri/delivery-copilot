"""Tests for the HTTP service.

Everything runs against FakeLLM — no API key, no network, no bill.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.agent.approval import ApprovalStore
from app.agent.llm import FakeLLM, text_response, tool_response
from app.api.main import LoggingWriter, Services, build_registry, create_app
from app.core.sync_result import CompleteSync, PartialReason, PartialSync
from app.models.domain import (
    Citation,
    Page,
    PageSection,
    ProposedAction,
    RiskFinding,
    RiskLevel,
)
from app.rag.chunking import chunk_page
from app.rag.decomposition import HeuristicDecomposer
from app.rag.embedders import FakeEmbedder
from app.rag.retrieval import HybridRetriever

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def risk_finding(key="INS-101") -> RiskFinding:
    return RiskFinding(
        issue_key=key, rule_id="stale_in_progress", level=RiskLevel.HIGH,
        detail="In Progress with no activity for 23 days.",
        citations=[Citation(source_type="issue", source_id=key, url=f"https://x/{key}")],
    )


def complete_sync(findings=None) -> CompleteSync:
    return CompleteSync(
        started_at=NOW, issues_seen=4,
        projects_requested=["INS"], projects_covered=["INS"],
        findings=list(findings if findings is not None else [risk_finding()]),
    )


def partial_sync() -> PartialSync:
    return PartialSync(
        started_at=NOW, issues_seen=2,
        projects_requested=["INS", "CLM"], projects_covered=["INS"],
        partial_findings=[risk_finding()], projects_incomplete=["CLM"],
        reason=PartialReason.PERMISSION_DENIED, detail="403 on CLM",
    )


def make_services(responses=None, *, sync=None) -> Services:
    page = Page(
        id="controls", space_key="ARCH",
        title="Controls Required Before Go-Live",
        url="https://x/wiki/controls", version=1, updated_at=NOW,
        sections=[
            PageSection(
                heading="Controls not yet satisfied", level=2,
                text="Mutual TLS certificates have not been issued. " * 12,
            )
        ],
    )
    retriever = HybridRetriever(chunk_page(page), FakeEmbedder(), candidate_pool=10)
    sync = sync or complete_sync()
    return Services(
        llm=FakeLLM(responses=list(responses or [])),
        retriever=retriever,
        decomposer=HeuristicDecomposer(),
        store=ApprovalStore(),
        registry=build_registry(retriever, sync),
        sync=sync,
        writer=LoggingWriter(),
    )


def client(services) -> TestClient:
    return TestClient(create_app(services))


def events(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    out = []
    name = None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: "):]
        elif line.startswith("data: ") and name:
            out.append((name, json.loads(line[len("data: "):])))
    return out


def proposal(key="INS-101") -> ProposedAction:
    return ProposedAction(
        issue_key=key, action_type="flag_at_risk",
        payload={"labels_add": ["at-risk"]},
        rationale="No activity for 23 days.",
        citations=[Citation(source_type="risk_rule", source_id="stale", url="#r")],
    )


class TestHealth:
    def test_reports_index_size(self):
        with client(make_services()) as c:
            body = c.get("/health").json()
        # "ok" or "degraded" depending on whether a key is present in the
        # environment running the tests; both are healthy for indexing.
        assert body["status"] in {"ok", "degraded"}
        assert body["chunks_indexed"] > 0


class TestAskStream:
    """Scripting note: /ask runs the agent graph, so a no-tool question costs
    two model calls — one reasoning turn, then the answer node, which re-asks
    rather than reusing the reasoning text so everything user-facing passes the
    grounding check."""

    def test_stages_arrive_in_order(self):
        services = make_services([
            text_response("I have what I need."),
            text_response("Certificates have not been issued [1]."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "which controls are outstanding?"})

        stages = [d["stage"] for name, d in events(response) if name == "stage"]
        assert stages == ["retrieving", "reasoning", "generating", "validating"]

    def test_passages_are_sent_before_the_answer(self):
        """The reader must be able to tell 'no sources' from 'sources but no
        answer'."""
        services = make_services([
            text_response("I have what I need."),
            text_response("Certificates have not been issued [1]."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "which controls?"})

        names = [name for name, _ in events(response)]
        assert names.index("passages") < names.index("answer")

    def test_answer_carries_sources(self):
        services = make_services([
            text_response("I have what I need."),
            text_response("Certificates have not been issued [1]."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "which controls?"})

        answer = next(d for name, d in events(response) if name == "answer")
        assert answer["sources"]
        assert answer["sources"][0]["url"].startswith("https://")

    def test_ungrounded_answer_is_never_streamed_as_an_answer(self):
        """The whole reason this streams stages rather than tokens: a rejected
        claim must never appear on screen."""
        services = make_services([
            text_response("I have what I need."),
            text_response("The vendor confirmed the schema on Thursday afternoon."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "what did the vendor say?"})

        names = [name for name, _ in events(response)]
        assert "answer" not in names
        assert "refusal" in names
        assert "Thursday" not in response.text

    def test_done_event_carries_cost_and_latency(self):
        services = make_services([
            text_response("I have what I need."),
            text_response("Certificates have not been issued [1]."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "which controls?"})

        done = next(d for name, d in events(response) if name == "done")
        assert "trace_id" in done and "cost_usd" in done and "ms" in done

    def test_model_failure_becomes_an_error_event(self):
        """A dead SSE connection leaves the page spinning with no explanation."""
        services = make_services([])  # FakeLLM will raise
        with client(services) as c:
            response = c.post("/ask", json={"question": "which controls?"})

        assert any(name == "error" for name, _ in events(response))

    def test_short_question_is_rejected(self):
        with client(make_services()) as c:
            assert c.post("/ask", json={"question": "hi"}).status_code == 422


class TestAgentIsReachableFromTheUI:
    """The regression this file exists to prevent.

    /ask was a pure retrieval path with no tool loop: `make_flag_tool` appeared
    nowhere in app/, `build_services` constructed no registry, and findings were
    an empty list. "Flag INS-101 as at risk" was answered as a question about
    INS-101 and "Waiting on you" stayed empty, because nothing in that path
    could propose anything.
    """

    def test_the_toolset_is_actually_wired(self):
        from app.api.main import build_services

        tools = set(build_services().registry.tools)
        assert tools == {
            "search_documentation",
            "get_risk_findings",
            "propose_flag_at_risk",
        }

    def test_the_service_boots_with_findings(self):
        """`findings=[]` made the risk tool answer "nothing is at risk" with
        total confidence — indistinguishable from good news."""
        from app.api.main import build_services

        services = build_services()
        assert services.sync.findings
        assert {f.issue_key for f in services.sync.findings} >= {"INS-101"}

    def test_the_risk_engine_answers_through_the_endpoint(self):
        services = make_services([
            tool_response("get_risk_findings", {}),
            text_response("Summarising."),
            text_response("INS-101 has had no activity for 23 days [1]."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "what is at risk?"})

        tools = [d["name"] for name, d in events(response) if name == "tool"]
        assert tools == ["get_risk_findings"]

    def test_a_write_request_produces_a_proposal_event(self):
        services = make_services([
            tool_response("propose_flag_at_risk", {
                "issue_key": "INS-101",
                "rationale": "No activity for 23 days while In Progress.",
                "evidence_rule_ids": ["stale_in_progress"],
            }),
            text_response("Proposed."),
            text_response("I have proposed flagging INS-101 as at-risk [1]."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "Flag INS-101 as at risk"})
            proposals = c.get("/proposals").json()["pending"]

        event = next(d for name, d in events(response) if name == "proposal")
        assert event["issue_key"] == "INS-101"
        assert event["rationale"]
        assert [p["id"] for p in proposals] == [event["id"]]

    def test_the_proposal_is_pending_not_applied(self):
        """The whole point of the gate. It reaches the UI unapproved."""
        services = make_services([
            tool_response("propose_flag_at_risk", {
                "issue_key": "INS-101",
                "rationale": "No activity for 23 days.",
                "evidence_rule_ids": ["stale_in_progress"],
            }),
            text_response("Proposed."),
            text_response("I have proposed flagging INS-101 [1]."),
        ])
        with client(services) as c:
            c.post("/ask", json={"question": "Flag INS-101 as at risk"})

        assert services.writer.calls == []
        assert services.store.audit == []
        assert len(services.store.pending) == 1

    def test_write_tools_are_withheld_from_a_question(self):
        """A question about risk must not be able to propose a change, however
        the model is prompted."""
        services = make_services([
            text_response("I have what I need."),
            text_response("Certificates have not been issued [1]."),
        ])
        with client(services) as c:
            c.post("/ask", json={"question": "what is at risk?"})

        assert "propose_flag_at_risk" not in services.llm.calls[0]["tools"]

    def test_write_tools_are_offered_to_a_request(self):
        services = make_services([
            text_response("I have what I need."),
            text_response("Certificates have not been issued [1]."),
        ])
        with client(services) as c:
            c.post("/ask", json={"question": "Flag INS-101 as at risk"})

        assert "propose_flag_at_risk" in services.llm.calls[0]["tools"]


class TestEvidenceNumbering:
    """A citation that resolves to the wrong source is worse than no citation:
    it is checkable, and it checks out wrong."""

    def test_passages_are_resent_to_match_the_answer(self):
        """After a tool round the answer node puts tool output at [1]. A UI
        holding the retrieval list would light a Confluence page when the reader
        clicks the citation that means the risk engine."""
        services = make_services([
            tool_response("get_risk_findings", {}),
            text_response("Summarising."),
            text_response("INS-101 has had no activity for 23 days [1]."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "what is at risk?"})

        sent = [d for name, d in events(response) if name == "passages"]
        assert [d["final"] for d in sent] == [False, True]
        assert sent[-1]["passages"][0]["found_by"] == "tool"

    def test_final_passages_agree_with_the_answer_sources(self):
        services = make_services([
            tool_response("get_risk_findings", {}),
            text_response("Summarising."),
            text_response("INS-101 has had no activity for 23 days [1]."),
        ])
        with client(services) as c:
            response = c.post("/ask", json={"question": "what is at risk?"})

        final = [d for name, d in events(response) if name == "passages"][-1]
        answer = next(d for name, d in events(response) if name == "answer")
        assert final["passages"][0]["label"] == answer["sources"][0]["label"]


class TestPartialSyncReachesTheUI:
    def test_health_reports_a_partial_sync_and_what_to_do(self):
        with client(make_services(sync=partial_sync())) as c:
            body = c.get("/health").json()
        assert body["sync_status"] == "partial"
        # None, not 0. Zero findings is good news; an unread project is not.
        assert body["risk_findings"] is None
        assert "Browse Projects" in body["sync_message"]

    def test_health_reports_a_complete_sync(self):
        with client(make_services()) as c:
            body = c.get("/health").json()
        assert body["sync_status"] == "complete"
        assert body["risk_findings"] == 1
        assert body["sync_message"] == ""

    def test_the_risk_tool_serves_nothing_from_a_partial_sync(self):
        services = make_services([
            tool_response("get_risk_findings", {}),
            text_response("Summarising."),
            text_response("Risk findings are unavailable for this sync [1]."),
        ], sync=partial_sync())
        with client(services) as c:
            response = c.post("/ask", json={"question": "what is at risk?"})

        # The tool result goes back to the model, never to the page.
        assert "INS-101" not in response.text
        tool_result = services.llm.calls[1]["messages"][-1]["content"][0]
        assert tool_result["is_error"] is True
        assert "INS-101" not in tool_result["content"]


class TestReport:
    """The weekly narrative: rules assemble it (`build_report`), the model
    writes it up (`generate_narrative`). ADR-004's gate applies to this
    endpoint exactly as it applies to the risk tool — a third consumer of the
    same `SyncResult`, not a second implementation of the guarantee."""

    def test_returns_a_narrative_grounded_in_the_findings(self):
        services = make_services(
            [text_response("INS-101 has had no activity for 23 days [1].")],
            sync=complete_sync([risk_finding("INS-101")]),
        )
        with client(services) as c:
            response = c.post("/report")

        assert response.status_code == 200
        body = response.json()
        assert body["narrative_answered"] is True
        assert "INS-101" in body["narrative"]
        assert body["issues_seen"] == 4
        assert body["sections"] == [
            {
                "level": "high",
                "findings": [
                    {
                        "issue_key": "INS-101",
                        "rule_id": "stale_in_progress",
                        "detail": "In Progress with no activity for 23 days.",
                    }
                ],
            }
        ]
        assert body["sources"][0]["url"] == "https://x/INS-101"

    def test_sources_are_the_full_numbered_list_not_only_cited_ones(self):
        """The same bug ADR-009 fixed for `/ask`'s evidence rail: a bracket in
        the text has to resolve against every passage the model was given, not
        just the ones it happened to cite — otherwise [2] in the narrative can
        point at the wrong finding, or at nothing, once one finding goes
        uncited."""
        findings = [
            risk_finding("INS-101"),
            risk_finding("INS-102"),
        ]
        services = make_services(
            # Only INS-101 is cited; INS-102 is covered in the report but the
            # model leaves it uncited in this run.
            [text_response("INS-101 has had no activity for 23 days [1].")],
            sync=complete_sync(findings),
        )
        with client(services) as c:
            body = c.post("/report").json()

        assert body["sources"] == [
            {"index": 1, "label": "INS-101 — stale_in_progress", "url": "https://x/INS-101"},
            {"index": 2, "label": "INS-102 — stale_in_progress", "url": "https://x/INS-102"},
        ]

    def test_a_partial_sync_cannot_produce_a_report(self):
        """The same guarantee `tests/test_sync_gate.py` pins for `build_report`
        directly, now pinned at the HTTP boundary — and the model is never
        called, because the gate sits upstream of it."""
        services = make_services(sync=partial_sync())
        with client(services) as c:
            response = c.post("/report")

        assert response.status_code == 409
        assert "CLM" in response.json()["detail"]
        assert services.llm.calls == []

    def test_no_findings_is_reported_with_no_model_call(self):
        services = make_services([], sync=complete_sync([]))
        with client(services) as c:
            body = c.post("/report").json()

        assert body["narrative_answered"] is True
        assert "no risk findings" in body["narrative"].lower()
        assert body["sections"] == []
        assert services.llm.calls == []

    def test_an_ungrounded_narrative_is_reported_as_unanswered_not_hidden(self):
        services = make_services(
            [
                text_response(
                    "INS-101 has had no activity for 23 days [1].\n\n"
                    "It is also the single largest risk to the release and "
                    "should be escalated to the steering committee immediately."
                )
            ],
            sync=complete_sync([risk_finding("INS-101")]),
        )
        with client(services) as c:
            body = c.post("/report").json()

        assert body["narrative_answered"] is False
        assert body["refusal_reason"] == "failed_grounding_check"
        # The structured findings are still there even though the prose was
        # withheld — a rejected narrative is not a rejected report.
        assert body["sections"][0]["findings"][0]["issue_key"] == "INS-101"


class TestApproval:
    def test_approval_requires_a_named_actor(self):
        """An audit trail showing 'approved by (unknown)' is not an audit trail."""
        services = make_services()
        pid = services.store.submit(proposal())
        with client(services) as c:
            response = c.post(f"/proposals/{pid}/approve", json={})
        assert response.status_code == 400
        assert "X-Actor" in response.json()["detail"]

    def test_approval_applies_and_records(self):
        services = make_services()
        pid = services.store.submit(proposal())
        with client(services) as c:
            response = c.post(
                f"/proposals/{pid}/approve",
                json={"note": "agreed"},
                headers={"X-Actor": "vp@acme.com"},
            )
        assert response.status_code == 200
        assert services.writer.calls
        assert services.store.audit[0].actor == "vp@acme.com"

    def test_rejection_is_recorded_without_writing(self):
        services = make_services()
        pid = services.store.submit(proposal())
        with client(services) as c:
            c.post(
                f"/proposals/{pid}/reject",
                json={"note": "known, vendor chasing"},
                headers={"X-Actor": "vp@acme.com"},
            )
        assert services.writer.calls == []
        assert services.store.audit[0].decision.value == "rejected"

    def test_double_approval_is_a_conflict_not_a_second_write(self):
        services = make_services()
        pid = services.store.submit(proposal())
        with client(services) as c:
            headers = {"X-Actor": "vp@acme.com"}
            assert c.post(f"/proposals/{pid}/approve", json={}, headers=headers).status_code == 200
            second = c.post(f"/proposals/{pid}/approve", json={}, headers=headers)
        assert second.status_code == 409
        assert len(services.writer.calls) == 2  # labels + comment, from the first only

    def test_unknown_proposal_is_a_conflict(self):
        with client(make_services()) as c:
            response = c.post(
                "/proposals/prop_9999/approve",
                json={},
                headers={"X-Actor": "p@acme.com"},
            )
        assert response.status_code == 409


class TestProposalListing:
    def test_pending_proposals_expose_their_evidence(self):
        """A lead approving a change must see what it rests on."""
        services = make_services()
        services.store.submit(proposal())
        with client(services) as c:
            body = c.get("/proposals").json()
        assert body["pending"][0]["citations"]
        assert body["pending"][0]["rationale"]

    def test_audit_is_visible(self):
        services = make_services()
        pid = services.store.submit(proposal())
        services.store.reject(pid, actor="p@acme.com")
        with client(services) as c:
            body = c.get("/proposals").json()
        assert body["audit"][0]["decision"] == "rejected"


class TestDefaultWriterIsSafe:
    def test_default_writer_does_not_touch_jira(self, clean_env):
        """A service that can write to a client tenant the moment it boots is a
        service that writes to a client tenant by accident.

        `clean_env`, not decoration: `build_services()` reads real settings,
        and a machine with `JIRA_ALLOW_WRITES=1` in its own `.env` — this one,
        once real writes were verified against a live tenant — makes this
        assertion false for a reason that has nothing to do with the code."""
        from app.api.main import build_services

        assert isinstance(build_services().writer, LoggingWriter)


class TestUI:
    """The UI is a built React bundle, so these assert it is mounted and that
    the API routes still win — not its contents, which live in frontend/src."""

    def test_built_ui_is_served_at_root(self):
        with client(make_services()) as c:
            response = c.get("/")
        assert response.status_code == 200
        assert "Delivery Copilot" in response.text

    def test_api_routes_are_not_shadowed_by_the_static_mount(self):
        """Mounting StaticFiles at "/" swallows every route registered after
        it. This is the test that catches a reordering."""
        with client(make_services()) as c:
            assert c.get("/health").json()["chunks_indexed"] > 0
            assert "pending" in c.get("/proposals").json()


@pytest.fixture
def no_api_key(monkeypatch):
    """An unconfigured environment, whether or not the machine has a .env.

    Deleting the variable is not enough: `settings()` calls `load_dotenv()`,
    which reads it straight back from the developer's .env. So dotenv is
    neutered for the duration. Patching the module function is enough because
    config.py imports it inside the function body.
    """
    import dotenv

    from app.core import config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    config.settings.cache_clear()
    yield
    config.settings.cache_clear()


@pytest.fixture
def with_api_key(monkeypatch):
    from app.core import config

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    config.settings.cache_clear()
    yield
    config.settings.cache_clear()


class TestConfigErrors:
    """The service told a user "FakeLLM ran out of scripted responses" when the
    real problem was a missing API key — an internal detail that says nothing
    about what to do.

    These fixtures pin the environment in both directions, so the result does
    not depend on whether the machine running them has a .env.
    """

    def test_missing_key_produces_an_actionable_message(self, no_api_key):
        services = make_services([])  # FakeLLM with nothing scripted
        with client(services) as c:
            response = c.post("/ask", json={"question": "which controls?"})

        error = next(d for name, d in events(response) if name == "error")
        assert error["code"] == "missing_api_key"
        assert "ANTHROPIC_API_KEY" in error["message"]
        assert "scripted" not in error["message"]

    def test_health_reports_missing_config(self, no_api_key):
        with client(make_services()) as c:
            body = c.get("/health").json()
        assert body["can_answer"] is False
        assert body["status"] == "degraded"
        assert "ANTHROPIC_API_KEY" in body["missing_config"]

    def test_health_is_ok_when_configured(self, with_api_key):
        """The counterweight: a test that only ever asserts the unhappy path
        can pass against a service that is always broken."""
        with client(make_services()) as c:
            body = c.get("/health").json()
        assert body["can_answer"] is True
        assert body["status"] == "ok"
        assert body["missing_config"] == []


class TestTheUIDoesNotClaimAWriteItDidNotMake:
    def test_health_reports_that_the_default_writer_is_a_dry_run(self):
        with client(make_services()) as c:
            body = c.get("/health").json()
        assert body["writes_to_jira"] is False

    def test_the_default_writer_declares_itself_a_dry_run(self):
        assert LoggingWriter().writes_to_jira is False

    def test_an_approved_change_is_not_recorded_as_written(self):
        """The bug in one assertion: approving through the shipped default
        produced an audit row the UI rendered as ", written to Jira"."""
        services = make_services()
        pid = services.store.submit(proposal())
        with client(services) as c:
            c.post(
                f"/proposals/{pid}/approve",
                json={},
                headers={"X-Actor": "vp@acme.com"},
            )
            row = c.get("/proposals").json()["audit"][0]

        assert row["applied"] == "True"
        assert row["written_to_jira"] == "False"

    def test_a_real_writer_is_reported_as_one(self):
        class LiveWriter(LoggingWriter):
            writes_to_jira = True

        services = make_services()
        services.writer = LiveWriter()
        pid = services.store.submit(proposal())
        with client(services) as c:
            assert c.get("/health").json()["writes_to_jira"] is True
            c.post(
                f"/proposals/{pid}/approve",
                json={},
                headers={"X-Actor": "vp@acme.com"},
            )
            row = c.get("/proposals").json()["audit"][0]
        assert row["written_to_jira"] == "True"


class TestTheExamplesMatchTheLoadedData:
    """The suggestion chips were hardcoded to INS-101, a sandbox fixture key.

    Pointed at a real tenant the demo's most important button asked the agent
    to flag an issue that does not exist — so the first click in front of an
    interviewer produced a confused non-answer about a missing ticket.
    """

    def test_health_names_an_issue_that_is_actually_at_risk(self):
        with client(make_services()) as c:
            body = c.get("/health").json()
        assert body["top_risk_issue"] == "INS-101"

    def test_it_prefers_the_highest_severity_finding(self):
        sync = complete_sync([
            risk_finding("INS-9"),
            risk_finding("INS-4"),
        ])
        sync = sync.model_copy(update={
            "findings": [
                sync.findings[0].model_copy(update={"level": RiskLevel.MEDIUM}),
                sync.findings[1],
            ]
        })
        with client(make_services(sync=sync)) as c:
            assert c.get("/health").json()["top_risk_issue"] == "INS-4"

    def test_no_findings_means_no_suggestion_rather_than_a_made_up_key(self):
        with client(make_services(sync=complete_sync([]))) as c:
            assert c.get("/health").json()["top_risk_issue"] is None

    def test_a_partial_sync_suggests_nothing(self):
        """The findings exist under `partial_findings`, and reaching for them
        here would leak exactly what the sync gate withholds."""
        with client(make_services(sync=partial_sync())) as c:
            assert c.get("/health").json()["top_risk_issue"] is None


class TestHealthDescribesBothHalvesOfTheData:
    """Risk findings and documentation come from different places, and with a
    real tenant configured they can disagree about which world they are in:
    Jira issues are the client's, Confluence answers are the bundled sample
    corpus. Either alone is obvious. Together they read as one live system, and
    "which controls are outstanding before go-live?" returns a confident answer
    about a gateway the client does not own.
    """

    def test_documentation_source_is_reported(self):
        with client(make_services()) as c:
            body = c.get("/health").json()
        assert body["docs_source"] == "bundled_corpus"
        assert body["docs_message"] == ""

    def test_a_real_confluence_load_is_reported(self):
        services = make_services()
        services.docs_source = "confluence"
        with client(services) as c:
            body = c.get("/health").json()
        assert body["docs_source"] == "confluence"

    def test_partial_confluence_coverage_is_reported(self):
        """One space unreachable does not read the same as full coverage —
        the same distinction `sync_message` already makes for Jira."""
        services = make_services()
        services.docs_source = "confluence_partial"
        services.docs_message = "Could not read COMP; some documentation may be missing."
        with client(services) as c:
            body = c.get("/health").json()
        assert body["docs_source"] == "confluence_partial"
        assert "COMP" in body["docs_message"]

    def test_total_confluence_failure_is_reported_distinctly_from_bundled_corpus(self):
        """A tenant that asked for real Confluence and got nothing must not
        read the same as one that never configured it — the first is a gap to
        fix, the second is the documented default."""
        services = make_services()
        services.docs_source = "confluence_unavailable"
        with client(services) as c:
            body = c.get("/health").json()
        assert body["docs_source"] == "confluence_unavailable"
        assert body["docs_source"] != "bundled_corpus"


@pytest.fixture
def clean_env(monkeypatch):
    """An environment this test controls completely.

    `delenv` is not enough: `settings()` calls `load_dotenv()`, which reads the
    variable straight back off the developer's own .env — so a test asserting
    "writes are off" passes on a machine without one and fails on the machine
    that actually has Jira configured. The same trap is documented on
    `no_api_key` below; it caught this file twice.
    """
    import dotenv

    from app.core import config

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for name in (
        "ATLASSIAN_BASE_URL", "ATLASSIAN_EMAIL", "ATLASSIAN_API_TOKEN",
        "JIRA_PROJECT_KEYS", "JIRA_ALLOW_WRITES",
        "CONFLUENCE_SPACE_KEYS", "CONFLUENCE_LABELS",
    ):
        monkeypatch.delenv(name, raising=False)
    config.settings.cache_clear()
    yield monkeypatch
    config.settings.cache_clear()


class TestConfluenceIsOptIn:
    """Same shape as `TestRealWritesAreOptInOnly`: shared Atlassian credentials
    are necessary but never sufficient on their own. `JIRA_PROJECT_KEYS` scopes
    which Jira projects are read; `CONFLUENCE_SPACE_KEYS` is the same kind of
    explicit scope for documentation, and the two are independent — a tenant
    read for Jira must not silently start reading Confluence too."""

    def test_credentials_alone_do_not_enable_confluence(self, clean_env):
        from app.core import config

        for key, value in {
            "ATLASSIAN_BASE_URL": "https://acme.atlassian.net",
            "ATLASSIAN_EMAIL": "svc@acme.com",
            "ATLASSIAN_API_TOKEN": "tok",
        }.items():
            clean_env.setenv(key, value)
        config.settings.cache_clear()
        assert config.settings().has_confluence is False

    def test_space_keys_alone_do_not_enable_confluence(self, clean_env):
        from app.core import config

        clean_env.setenv("CONFLUENCE_SPACE_KEYS", "ARCH")
        config.settings.cache_clear()
        assert config.settings().has_confluence is False

    def test_credentials_and_space_keys_together_enable_confluence(self, clean_env):
        from app.core import config

        for key, value in {
            "ATLASSIAN_BASE_URL": "https://acme.atlassian.net",
            "ATLASSIAN_EMAIL": "svc@acme.com",
            "ATLASSIAN_API_TOKEN": "tok",
            "CONFLUENCE_SPACE_KEYS": "ARCH",
        }.items():
            clean_env.setenv(key, value)
        config.settings.cache_clear()
        assert config.settings().has_confluence is True

    def test_jira_and_confluence_are_independently_scoped(self, clean_env):
        """The dangerous near-miss in the other direction: a tenant configured
        to read Jira must not automatically start reading Confluence, and
        vice versa."""
        from app.core import config

        for key, value in {
            "ATLASSIAN_BASE_URL": "https://acme.atlassian.net",
            "ATLASSIAN_EMAIL": "svc@acme.com",
            "ATLASSIAN_API_TOKEN": "tok",
            "JIRA_PROJECT_KEYS": "INS",
        }.items():
            clean_env.setenv(key, value)
        config.settings.cache_clear()
        assert config.settings().has_jira is True
        assert config.settings().has_confluence is False

    def test_space_keys_are_parsed_like_project_keys(self, clean_env):
        from app.core import config

        clean_env.setenv("CONFLUENCE_SPACE_KEYS", " arch, compliance ,,")
        config.settings.cache_clear()
        assert config.settings().confluence_space_keys == ("ARCH", "COMPLIANCE")

    def test_labels_are_parsed_without_case_folding(self, clean_env):
        """Unlike space keys, Confluence labels are lowercase-with-hyphens by
        convention and the API's label filter is case-sensitive — uppercasing
        them the way project keys are would silently match nothing."""
        from app.core import config

        clean_env.setenv("CONFLUENCE_LABELS", " architecture, Compliance-Sign-Off ,,")
        config.settings.cache_clear()
        assert config.settings().confluence_labels == (
            "architecture", "Compliance-Sign-Off",
        )

    def test_labels_are_optional(self, clean_env):
        from app.core import config

        clean_env.setenv("CONFLUENCE_SPACE_KEYS", "ARCH")
        config.settings.cache_clear()
        assert config.settings().confluence_labels == ()


class TestRealWritesAreOptInOnly:
    """A service that can write to a client's tenant the moment it boots is a
    service that writes to a client's tenant by accident."""

    def test_the_default_build_cannot_write(self, clean_env):
        from app.api.main import build_services

        assert build_services().writer.writes_to_jira is False

    def test_credentials_alone_do_not_enable_writes(self, clean_env):
        """The dangerous near-miss. A tenant configured for *reading* must not
        silently become a tenant configured for writing."""
        from app.core import config

        for key, value in {
            "ATLASSIAN_BASE_URL": "https://acme.atlassian.net",
            "ATLASSIAN_EMAIL": "svc@acme.com",
            "ATLASSIAN_API_TOKEN": "tok",
            "JIRA_PROJECT_KEYS": "INS",
        }.items():
            clean_env.setenv(key, value)
        config.settings.cache_clear()
        assert config.settings().has_jira is True
        assert config.settings().jira_allow_writes is False
        assert config.settings().can_write_to_jira is False

    def test_the_flag_alone_does_not_enable_writes(self, clean_env):
        """Without credentials there is nothing to write with, and constructing
        a writer anyway would fail at the first approval rather than at boot."""
        from app.core import config

        clean_env.setenv("JIRA_ALLOW_WRITES", "true")
        config.settings.cache_clear()
        assert config.settings().can_write_to_jira is False

    def test_both_together_enable_writes(self, clean_env):
        from app.core import config

        for key, value in {
            "ATLASSIAN_BASE_URL": "https://acme.atlassian.net",
            "ATLASSIAN_EMAIL": "svc@acme.com",
            "ATLASSIAN_API_TOKEN": "tok",
            "JIRA_PROJECT_KEYS": "INS",
            "JIRA_ALLOW_WRITES": "true",
        }.items():
            clean_env.setenv(key, value)
        config.settings.cache_clear()
        assert config.settings().can_write_to_jira is True

    def test_only_an_explicit_truthy_value_counts(self, clean_env):
        """"false", "0" and "" must not read as permission. An env var that is
        present-but-off is the most common way a flag like this gets turned on
        by accident."""
        from app.core import config

        for raw in ("", "false", "False", "0", "no", "off"):
            clean_env.setenv("JIRA_ALLOW_WRITES", raw)
            config.settings.cache_clear()
            assert config.settings().jira_allow_writes is False, raw
        for raw in ("true", "TRUE", "1", "yes", "on"):
            clean_env.setenv("JIRA_ALLOW_WRITES", raw)
            config.settings.cache_clear()
            assert config.settings().jira_allow_writes is True, raw


class TestTheDemoClockIsVisible:
    """Stale detection cannot be seeded. Jira refuses to backdate `created`,
    comments or changelog entries, so a freshly seeded tenant has nothing that
    has sat still for two weeks — and `stale_in_progress`, the rule the client
    actually asked for, reports nothing on live data.

    The wrong fix is lowering `stale_days_high` until something fires, which
    ships thresholds tuned to fake data and then argues them with the client in
    week three. The rules take `now` as a parameter precisely so the clock can
    move instead.

    Moving it is only honest if it is impossible to miss, so it is off by
    default, reported by /health, and stated in the UI.
    """

    def test_it_is_off_by_default(self, clean_env):
        from app.core import config

        assert config.settings().risk_clock_offset_days == 0

    def test_health_reports_the_offset(self):
        with client(make_services()) as c:
            assert c.get("/health").json()["clock_offset_days"] == 0

    def test_a_negative_offset_is_refused(self, clean_env):
        """Evaluating in the past hides real findings. Only looking forward has
        a defensible reason."""
        from app.core import config

        clean_env.setenv("RISK_CLOCK_OFFSET_DAYS", "-10")
        config.settings.cache_clear()
        assert config.settings().risk_clock_offset_days == 0

    def test_a_non_numeric_offset_is_refused(self, clean_env):
        from app.core import config

        clean_env.setenv("RISK_CLOCK_OFFSET_DAYS", "soon")
        config.settings.cache_clear()
        assert config.settings().risk_clock_offset_days == 0

    def test_the_offset_moves_the_evaluation_clock(self, clean_env):
        """The property that matters: the same issues produce stale findings
        under a shifted clock and none under the real one."""
        from datetime import datetime, timedelta, timezone

        from app.core.sandbox import sandbox_config, sandbox_issues
        from app.risk.rules import evaluate

        now = datetime(2026, 9, 15, tzinfo=timezone.utc)
        issues = [
            i for i in sandbox_issues(now) if i.key == "INS-104"
        ]  # healthy today
        assert not [
            f for f in evaluate(issues, sandbox_config(), now)
            if f.rule_id == "stale_in_progress"
        ]
        assert [
            f for f in evaluate(issues, sandbox_config(), now + timedelta(days=30))
            if f.rule_id == "stale_in_progress"
        ]


class TestResync:
    """Findings were read once, at startup, and never again.

    On a live tenant that produced two confidently wrong answers in a row: a
    blocker was linked to INS-12 in Jira and the service kept replying "it is
    not blocked", because its snapshot predated the change. Restarting the
    process was the only remedy, which is not a thing you do mid-demo and not a
    thing a delivery lead can do at all.

    `clean_env` is not decoration. `resync` branches on `settings().has_jira`,
    which is True on any machine with a real .env — the first version of these
    tests read the live tenant, passed here, and would have failed in CI. Two of
    them passed for the wrong reason even then, because `load_live_findings`
    rebuilds the registry on failure as well as on success. `tests/conftest.py`
    now refuses outbound sockets so this cannot recur silently.
    """

    def test_health_reports_when_the_data_was_read(self):
        with client(make_services()) as c:
            body = c.get("/health").json()
        assert body["synced_at"]
        assert body["sync_age_seconds"] >= 0

    def test_resync_refreshes_the_findings(self, clean_env):
        services = make_services(sync=complete_sync([]))
        with client(services) as c:
            assert c.get("/health").json()["risk_findings"] == 0
            response = c.post("/sync")
            assert response.status_code == 200
            assert c.get("/health").json()["risk_findings"] > 0

    def test_resync_reports_what_it_found(self, clean_env):
        with client(make_services(sync=complete_sync([]))) as c:
            body = c.post("/sync").json()
        assert body["status"] == "complete"
        assert body["risk_findings"] > 0
        assert body["synced_at"]

    def test_resync_rebuilds_the_toolset(self, clean_env):
        """`make_risk_tool` closes over the findings it serves. Refreshing the
        sync without rebuilding the registry leaves the agent answering from the
        old list — the same bug, one layer down and harder to see."""
        services = make_services(sync=complete_sync([]))
        with client(services) as c:
            before = services.registry.tools["get_risk_findings"]
            c.post("/sync")
            assert services.registry.tools["get_risk_findings"] is not before

    def test_resync_does_not_disturb_pending_approvals(self, clean_env):
        """A proposal is a decision waiting on a person. Re-reading Jira must
        not discard one, or a lead loses work by pressing refresh."""
        services = make_services(sync=complete_sync([]))
        pid = services.store.submit(proposal())
        with client(services) as c:
            c.post("/sync")
            assert pid in services.store.pending

    def test_a_failed_resync_keeps_the_previous_findings(self):
        """Serving nothing because a refresh failed is worse than serving the
        last known good picture and saying how old it is."""
        services = make_services()
        original = services.sync

        def boom(_services):
            raise RuntimeError("jira unreachable")

        import app.api.main as main

        with client(services) as c:
            main._RESYNC = boom
            try:
                response = c.post("/sync")
            finally:
                main._RESYNC = main.load_live_findings
        assert response.status_code == 503
        assert "jira unreachable" in response.json()["detail"]
        assert services.sync is original


class TestConfluenceLoadNeverFallsBackToTheBundledCorpus:
    """The load-bearing guarantee of wiring Confluence in at all: a live load
    that fails completely must serve NO documentation rather than silently
    keep answering from the bundled sample pages. Mixing the two is worse
    than either alone — a client reading their own live Jira findings next to
    invented Confluence answers has no way to tell the difference, and it is
    the harder failure to notice precisely because the Jira half is real."""

    async def test_total_failure_empties_the_index_rather_than_keeping_it(
        self, clean_env, monkeypatch
    ):
        from app.core import config

        for key, value in {
            "ATLASSIAN_BASE_URL": "https://acme.atlassian.net",
            "ATLASSIAN_EMAIL": "svc@acme.com",
            "ATLASSIAN_API_TOKEN": "tok",
            "CONFLUENCE_SPACE_KEYS": "ARCH",
        }.items():
            clean_env.setenv(key, value)
        config.settings.cache_clear()

        import app.api.main as main
        from app.integrations.atlassian.confluence import ConfluenceError

        class BoomingClient:
            def __init__(self, **kwargs):
                pass

            async def iter_pages(self, space_key, *, labels=None):
                raise ConfluenceError("403 on ARCH")
                yield  # pragma: no cover - makes this an async generator

            async def aclose(self):
                pass

        monkeypatch.setattr(
            "app.integrations.atlassian.confluence.ConfluenceClient", BoomingClient
        )

        services = make_services()
        # The bundled corpus's own chunk, still present before the load —
        # this is what must NOT still be there afterward.
        assert services.retriever.chunks

        await main.load_live_docs(services)

        assert services.docs_source == "confluence_unavailable"
        assert services.retriever.chunks == []
        assert services.retriever.search("Mutual TLS") == []
