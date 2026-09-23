"""HTTP service.

## One endpoint, the whole system

`/ask` runs the agent graph. Not a retrieval-and-answer shortcut beside it —
the graph, the same one `scripts/demo_agent.py` drives and `tests/test_graph.py`
pins.

It was the shortcut for a while, and the failure that produced is worth keeping
written down. The UI called a straight retrieve → answer → validate path with no
tool loop, so "Flag INS-101 as at risk" was answered as though it were a
question *about* INS-101: five passages retrieved, a grounded paragraph
returned, and "Waiting on you" empty, because nothing in that path can propose
anything. Every piece of the approval story existed, was tested, and was
reachable only from a script. A guarantee you cannot demonstrate in the product
is a guarantee the client has no reason to believe.

## What the graph gets that the shortcut had

Query decomposition, which is a measured retrieval gain rather than a
preference (`run_eval.py --decompose`). It moved into the graph's retrieve node
rather than being dropped, so routing through the agent costs nothing that was
previously being paid for.

## Write intent

`write_intent()` decides whether the write tools are offered. It is keyword
matching, deliberately: the model still chooses whether to use what it is
offered, anything it produces is a `ProposedAction`, and that still needs a
named human. See `app/agent/intent.py` for why neither direction of error can
reach Jira.

## The streaming problem, and why this streams stages rather than tokens

Grounding is validated on the complete answer. An answer that streams token by
token and is then rejected has already been read — and in a regulated insurer,
showing an unvalidated claim to a delivery lead is the exact failure the last
two days were spent preventing. "It appeared and then vanished" is worse than a
six-second wait, because the reader remembers what they saw.

So the stream carries *progress*, not prose: retrieving, passages found,
reasoning, the tools as they run, generating, validating, then the answer in one
piece — or the refusal. The user sees the system working and never sees a claim
that failed its check.

The obvious refinement is paragraph-level streaming, since grounding is enforced
per paragraph: emit each paragraph as it completes and passes. That is a real
improvement and it needs a streaming LLM client, which the protocol does not
have yet. Noted rather than half-built.

## Approval

`ApprovalStore` has been a test fixture for two days. Here it becomes something
a person clicks. The actor is taken from a header rather than invented — an
audit trail showing "approved by (unknown)" is not an audit trail, and the store
refuses it anyway.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.answering import Passage
from app.agent.approval import (
    ApprovalError,
    ApprovalStore,
    IssueWriter,
    apply_action,
)
from app.agent.graph import AgentDeps, stream_agent
from app.agent.intent import write_intent
from app.agent.llm import LLM, AnthropicLLM, default_llm
from app.agent.narrative import finding_passage, generate_narrative
from app.agent.tools import (
    ToolRegistry,
    make_flag_tool,
    make_search_tool,
    risk_tool_for,
)
from app.core.config import settings
from app.core.report import build_report
from app.core.sandbox import sandbox_sync
from app.core.sync_result import CompleteSync, PartialReason, PartialSync, SyncResult
from app.core.sync_runner import run_sync
from app.core.tracing import Trace, TracedLLM
from app.rag.chunking import chunk_page
from app.rag.decomposition import Decomposer, HeuristicDecomposer
from app.rag.embedders import default_embedder
from app.rag.retrieval import HybridRetriever
from app.risk.rules import RiskConfig

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class ApprovalRequest(BaseModel):
    note: str = ""


@dataclass
class Services:
    """Everything a request needs, built once at startup.

    The retriever indexes the corpus in its constructor — a few hundred
    milliseconds and, with a real embedder, a bill. Doing that per request would
    make the service unusable and expensive at the same time.

    `sync` is held rather than a bare findings list so the type carries its own
    provenance: `risk_tool_for` reads it, and a `PartialSync` produces a risk
    tool that reports nothing rather than reporting a fraction as though it were
    everything (ADR-004).
    """

    llm: LLM
    retriever: HybridRetriever
    decomposer: Decomposer
    store: ApprovalStore
    registry: ToolRegistry
    sync: SyncResult
    writer: IssueWriter | None = None
    # "sandbox" or "jira". Surfaced on /health because a lead looking at
    # INS-101 needs to know whether it is their tenant or the demo one.
    findings_source: str = "sandbox"
    # "bundled_corpus", "confluence", "confluence_partial" or
    # "confluence_unavailable" — see `load_live_docs`. Independent of
    # `findings_source`: a tenant read for Jira does not imply Confluence was
    # ever configured, let alone that it loaded cleanly.
    docs_source: str = "bundled_corpus"
    docs_message: str = ""

    @property
    def synced_at(self) -> datetime:
        """When the findings currently being served were read.

        Taken off the sync itself rather than stored separately, so it cannot
        drift from the data it describes.
        """
        return self.sync.finished_at


class LoggingWriter:
    """Stand-in for a real Jira client.

    Deliberately the default. A service that can write to a client's tenant the
    moment it boots is a service that writes to a client's tenant by accident;
    the real writer is injected explicitly at deploy time.

    It says so out loud, and that is the part that was missing. The audit trail
    recorded `applied: True` — meaning the writer returned without error — and
    the Decisions panel rendered that as ", written to Jira" against a tenant
    this process had never contacted. Every other claim in this system is
    checked against its source; that one was not checked against anything.
    """

    #: Read by `apply_action` and reported on every audit row.
    writes_to_jira = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def add_labels(self, issue_key: str, labels: list[str]) -> None:
        logger.info("WOULD add labels %s to %s", labels, issue_key)
        self.calls.append(f"labels {labels} -> {issue_key}")

    async def add_comment(self, issue_key: str, body: str) -> None:
        logger.info("WOULD comment on %s", issue_key)
        self.calls.append(f"comment -> {issue_key}")


def build_registry(retriever: HybridRetriever, sync: SyncResult) -> ToolRegistry:
    """The toolset, assembled in one place.

    Rebuilt whenever the sync is replaced, because `make_risk_tool` closes over
    the findings it serves. A tool holding a mutable list that something else
    refills is the kind of shared state that makes "why did it say nothing was
    at risk" unanswerable an hour before a demo.

    The write tool is always registered and separately withheld — see
    `AgentDeps.schemas`. Registering it conditionally would mean two toolsets to
    reason about instead of one gate.
    """
    return ToolRegistry().register(
        make_search_tool(retriever),
        risk_tool_for(sync),
        make_flag_tool(),
    )


def _build_writer(config) -> IssueWriter:
    """The writer, which is a dry run unless two separate switches are thrown.

    `can_write_to_jira` requires credentials AND `JIRA_ALLOW_WRITES`. Reading a
    client's tenant is how a pilot begins, and the day someone adds project keys
    must not be the day the service gains the ability to edit their tickets.

    Constructed here rather than reached for inside `apply_action`, so there is
    exactly one place in the codebase that decides whether this deployment can
    change anything — and `/health` reports what it decided.
    """
    if not config.can_write_to_jira:
        return LoggingWriter()

    from app.integrations.atlassian.writer import JiraIssueWriter

    logger.warning(
        "JIRA_ALLOW_WRITES is set — approved actions will be written to %s. "
        "Every write still requires a named human via the approval gate.",
        config.atlassian_base_url,
    )
    return JiraIssueWriter(
        base_url=config.atlassian_base_url,
        email=config.atlassian_email,
        api_token=config.atlassian_api_token,
    )


def build_services() -> Services:
    from app.rag.corpus import load_offline_corpus

    config = settings()
    chunks = [c for page in load_offline_corpus() for c in chunk_page(page)]
    retriever = HybridRetriever(chunks, default_embedder(), candidate_pool=20)

    # The sandbox tenant, unless startup replaces it with a real sync. Findings
    # were an empty list here for a while, which made the risk tool answer
    # "nothing is at risk" with complete confidence — indistinguishable from
    # good news, and the most dangerous empty default in the system.
    sync = sandbox_sync()

    return Services(
        llm=AnthropicLLM() if config.has_llm else default_llm(),
        retriever=retriever,
        decomposer=HeuristicDecomposer(),
        store=ApprovalStore(),
        registry=build_registry(retriever, sync),
        sync=sync,
        writer=_build_writer(config),
        findings_source="sandbox",
        docs_source="bundled_corpus",
    )


async def load_live_findings(services: Services) -> None:
    """Replace the sandbox findings with a real Jira sync.

    Note what does NOT happen on failure: a fall back to the sandbox. Serving
    invented issues to someone who believes they are looking at their own tenant
    is worse than serving nothing, and it is the failure they would be least
    able to detect. A failed sync becomes a `PartialSync`, which `risk_tool_for`
    turns into a risk tool that reports no findings and says why.

    Run `scripts/check_jira_connection.py` first against a new tenant. The two
    things that fail silently — changelogs not arriving, a wrong custom field id
    — are both invisible from here.
    """
    from app.integrations.atlassian.client import JiraClient

    config = settings()
    started_at = datetime.now(timezone.utc)
    projects = list(config.jira_project_keys)

    # See `_offset` in config.py. Zero unless someone deliberately set it.
    as_of = started_at + timedelta(days=config.risk_clock_offset_days)
    if config.risk_clock_offset_days:
        logger.warning(
            "RISK_CLOCK_OFFSET_DAYS=%d — risk rules are evaluated as of %s, not "
            "today. Findings are dated ahead.",
            config.risk_clock_offset_days,
            as_of.date(),
        )

    client = JiraClient(
        base_url=config.atlassian_base_url,
        email=config.atlassian_email,
        api_token=config.atlassian_api_token,
        target_release_field=config.jira_target_release_field,
    )
    try:
        sync: SyncResult = await run_sync(
            client,
            projects,
            RiskConfig(base_url=config.atlassian_base_url),
            now=as_of,
        )
    except Exception as exc:
        # run_sync handles per-project failures itself; reaching here means the
        # run did not start — bad credentials, wrong base URL, no network.
        logger.exception("jira sync failed at startup")
        sync = PartialSync(
            started_at=started_at,
            issues_seen=0,
            projects_requested=projects,
            partial_findings=[],
            projects_covered=[],
            projects_incomplete=projects,
            reason=PartialReason.UPSTREAM_ERROR,
            detail=str(exc),
        )
    finally:
        await client.aclose()

    services.sync = sync
    services.findings_source = "jira"
    services.registry = build_registry(services.retriever, sync)

    if isinstance(sync, CompleteSync):
        logger.info(
            "jira sync complete: %d findings from %d issues across %s",
            len(sync.findings),
            sync.issues_seen,
            ", ".join(sync.projects_covered) or "no projects",
        )
    else:
        logger.warning("%s", sync.operator_message())


async def load_live_docs(services: Services) -> None:
    """Replace the bundled sample corpus with real Confluence pages.

    Same rule as `load_live_findings`, applied to the other half of the data:
    a failed live load must not fall back to the bundled corpus. A client
    reading their own live Jira findings next to invented Confluence answers
    is the same failure as the reverse — indistinguishable from the real
    thing to whoever is reading it, and the harder of the two to notice,
    because the Jira half being real is what makes the whole page look
    trustworthy.

    Unlike Jira, a partial result still gets indexed rather than discarded.
    `run_docs_load`'s docstring has the reasoning: a smaller real corpus is
    not dangerous the way a smaller real risk-finding set is, because `/ask`
    already declines safely on missing context (ADR-006). Only total
    failure — no space readable at all — is treated like Jira's: no chunks
    indexed, not a silent revert to the sample pages.
    """
    from app.core.docs_runner import run_docs_load
    from app.integrations.atlassian.confluence import ConfluenceClient

    config = settings()
    client = ConfluenceClient(
        base_url=config.atlassian_base_url,
        email=config.atlassian_email,
        api_token=config.atlassian_api_token,
    )
    try:
        load = await run_docs_load(
            client,
            list(config.confluence_space_keys),
            labels=list(config.confluence_labels) or None,
        )
    except Exception as exc:
        # run_docs_load handles per-space failures itself; reaching here means
        # the run did not start at all — bad credentials, wrong base URL.
        logger.exception("confluence load failed at startup")
        services.retriever = HybridRetriever([], default_embedder(), candidate_pool=20)
        services.docs_source = "confluence_unavailable"
        services.docs_message = str(exc)
        services.registry = build_registry(services.retriever, services.sync)
        return
    finally:
        await client.aclose()

    if load.is_empty:
        services.retriever = HybridRetriever([], default_embedder(), candidate_pool=20)
        services.docs_source = "confluence_unavailable"
        services.docs_message = (
            f"Could not read any pages from "
            f"{', '.join(config.confluence_space_keys) or 'the configured spaces'}."
        )
        services.registry = build_registry(services.retriever, services.sync)
        logger.warning("confluence load produced no pages: %s", services.docs_message)
        return

    chunks = [c for page in load.pages for c in chunk_page(page)]
    services.retriever = HybridRetriever(chunks, default_embedder(), candidate_pool=20)
    services.docs_source = "confluence" if load.is_complete else "confluence_partial"
    services.docs_message = (
        ""
        if load.is_complete
        else (
            f"Could not read {', '.join(load.spaces_incomplete)}; some "
            f"documentation may be missing."
        )
    )
    services.registry = build_registry(services.retriever, services.sync)

    if load.is_complete:
        logger.info(
            "confluence load complete: %d pages, %d chunks across %s",
            len(load.pages), len(chunks), ", ".join(load.spaces_covered) or "no spaces",
        )
    else:
        logger.warning("%s", services.docs_message)


async def resync(services: Services) -> None:
    """Re-read findings into an existing `Services`.

    The gap this closes: findings were read once at startup and never again, so
    a change made in Jira during a session was invisible until the process was
    restarted. On a live tenant that produced two confidently wrong answers in
    a row — a blocker was linked to an issue and the service kept saying it was
    not blocked, because its snapshot predated the link.

    Mutates in place rather than rebuilding `Services`, because the
    `ApprovalStore` must survive: a pending proposal is a decision waiting on a
    person, and losing one to a refresh means a lead loses work by pressing a
    button.

    The previous sync is kept if this fails. Serving nothing because a refresh
    failed is worse than serving the last known picture and saying how old it
    is — which `/health` and the risk tool both now do.
    """
    if settings().has_jira:
        await load_live_findings(services)
        return

    services.sync = sandbox_sync()
    services.findings_source = "sandbox"
    # Rebuilt, not mutated. `make_risk_tool` closes over the findings it serves,
    # so refreshing the sync without rebuilding the registry leaves the agent
    # answering from the old list — the same bug, one layer down.
    services.registry = build_registry(services.retriever, services.sync)


# Indirection so a test can make a refresh fail without a live tenant. Assigned
# rather than imported at the call site, which is what makes it patchable.
_RESYNC = resync


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _passages_payload(passages: list[Passage], *, final: bool) -> dict[str, Any]:
    """The evidence rail's contents.

    Sent twice per run and that is the point. The first send is the retrieval
    result, so a reader can see what the system read before it reads the answer.
    The second is what the answer's citations actually index into — after a tool
    round those differ, because the answer node puts tool output at [1]. A UI
    that kept the first list would light the wrong source when someone clicked a
    citation, which is worse than showing none.
    """
    return {
        "final": final,
        "passages": [
            {
                "index": index,
                "label": passage.label,
                "url": passage.url,
                "found_by": passage.found_by,
            }
            for index, passage in enumerate(passages, start=1)
        ],
    }


_LEVEL_ORDER = {"high": 0, "medium": 1, "low": 2}


def _top_risk_issue(sync: SyncResult) -> str | None:
    """The worst finding's issue key, or None.

    Reads `CompleteSync.findings` only. A `PartialSync` keeps its findings under
    `partial_findings` precisely so code reaching for `.findings` cannot pick
    them up, and surfacing one here would put an issue key on screen drawn from
    a sync the rest of the system refuses to report from.
    """
    if not isinstance(sync, CompleteSync) or not sync.findings:
        return None
    worst = min(sync.findings, key=lambda f: _LEVEL_ORDER.get(f.level.value, 9))
    return worst.issue_key


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        if services is None and settings().has_jira:
            await load_live_findings(app.state.services)
        if services is None and settings().has_confluence:
            await load_live_docs(app.state.services)
        yield

    app = FastAPI(title="Delivery Copilot", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        svc: Services = app.state.services
        config = settings()
        complete = isinstance(svc.sync, CompleteSync)
        return {
            "status": "ok" if config.has_llm else "degraded",
            "can_answer": config.has_llm,
            "missing_config": config.missing_for_live(),
            "chunks_indexed": len(svc.retriever.chunks),
            "findings_source": svc.findings_source,
            # "bundled_corpus" unless CONFLUENCE_SPACE_KEYS is set — the two
            # data sources are configured, and can fail, independently.
            # Reported rather than left to be discovered mid-demo, the same
            # reason `sync_message` exists for the Jira half.
            "docs_source": svc.docs_source,
            "docs_message": svc.docs_message,
            # Non-zero means findings were evaluated against a future date, so
            # that stale detection has something to find on a sandbox no older
            # than the day it was seeded. Reported because a risk report
            # dated ahead of today must never be mistaken for one dated now.
            "clock_offset_days": config.risk_clock_offset_days,
            "sync_status": svc.sync.status.value,
            # None, not 0, when the sync is partial. Zero findings is good news;
            # "we could not read two of your projects" is not, and the two must
            # never render the same way.
            "risk_findings": len(svc.sync.findings) if complete else None,
            # The issue the UI offers as its worked example. Taken from the
            # findings rather than hardcoded: the chips said "Flag INS-101",
            # a sandbox fixture key, so against a real tenant the demo's most
            # important button asked the agent to flag a ticket that does not
            # exist. None when there is nothing to point at — an invented key
            # is what this is fixing.
            "top_risk_issue": _top_risk_issue(svc.sync),
            "sync_message": "" if complete else svc.sync.operator_message(),
            "synced_at": svc.synced_at.isoformat(),
            # How stale the answers are. A confident "not blocked" from a sync
            # taken an hour ago is a confident wrong answer waiting to happen.
            "sync_age_seconds": max(
                0,
                round(
                    (datetime.now(timezone.utc) - svc.synced_at).total_seconds()
                ),
            ),
            "pending_proposals": len(svc.store.pending),
            "audit_entries": len(svc.store.audit),
            # So the UI can say "recorded" rather than "written to Jira" when
            # the configured writer is a dry run.
            "writes_to_jira": bool(getattr(svc.writer, "writes_to_jira", False)),
        }

    @app.post("/ask")
    async def ask(
        request: AskRequest, http: Request, x_actor: str = Header(default="")
    ) -> StreamingResponse:
        svc: Services = app.state.services

        async def stream() -> AsyncIterator[str]:
            trace = Trace(question=request.question)
            allow_writes = write_intent(request.question)

            deps = AgentDeps(
                TracedLLM(svc.llm, trace),
                svc.retriever,
                svc.registry,
                allow_writes=allow_writes,
                decomposer=svc.decomposer,
                top_k=request.top_k,
            )

            # Stashed from the reason node and emitted when the tools node
            # reports. Emitting on intent instead would show a tool call that
            # the round limit can still prevent from ever running.
            requested: list[dict[str, Any]] = []

            try:
                yield sse(
                    "stage",
                    {"stage": "retrieving", "writes_offered": allow_writes},
                )

                async for node, update in stream_agent(
                    deps, request.question, actor=x_actor.strip()
                ):
                    if await http.is_disconnected():
                        # Stop before the next model call. Without this a closed
                        # tab still pays for an answer nobody will read.
                        logger.info("client disconnected mid-run")
                        return

                    if node == "retrieve":
                        yield sse(
                            "passages",
                            {
                                "subqueries": update.get("subqueries") or [],
                                **_passages_payload(
                                    [
                                        Passage.from_hit(hit)
                                        for hit in update.get("hits") or []
                                    ],
                                    final=False,
                                ),
                            },
                        )
                        yield sse("stage", {"stage": "reasoning"})

                    elif node == "reason":
                        requested = list(update.get("pending_calls") or [])
                        if not requested:
                            yield sse("stage", {"stage": "generating"})

                    elif node == "tools":
                        for call in requested:
                            yield sse(
                                "tool",
                                {"name": call["name"], "input": call.get("input") or {}},
                            )
                        requested = []

                        for proposal in update.get("proposals") or []:
                            # Submitted here rather than inside the graph. The
                            # graph must not hold the approval store: a node
                            # that can enqueue an approvable action is one
                            # refactor away from being a node that applies one.
                            try:
                                proposal_id = svc.store.submit(proposal)
                            except ApprovalError as exc:
                                logger.warning("proposal refused at intake: %s", exc)
                                continue
                            yield sse(
                                "proposal",
                                {
                                    "id": proposal_id,
                                    "issue_key": proposal.issue_key,
                                    "action": proposal.action_type,
                                    "rationale": proposal.rationale,
                                },
                            )
                        yield sse("stage", {"stage": "reasoning"})

                    elif node == "answer":
                        yield sse(
                            "passages",
                            _passages_payload(
                                update.get("evidence") or [], final=True
                            ),
                        )
                        yield sse("stage", {"stage": "validating"})

                        answer = update["answer"]
                        if answer.refused:
                            yield sse(
                                "refusal",
                                {
                                    "text": answer.text,
                                    "reason": answer.refusal_reason,
                                    "detail": answer.rejected_reason,
                                },
                            )
                        else:
                            yield sse(
                                "answer",
                                {"text": answer.text, "sources": answer.sources()},
                            )

                yield sse(
                    "done",
                    {
                        "trace_id": trace.trace_id,
                        "ms": round(trace.wall_ms),
                        "cost_usd": round(trace.total_cost, 6),
                        "input_tokens": trace.total_input_tokens,
                        "output_tokens": trace.total_output_tokens,
                    },
                )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Errors go down the stream as an event. An SSE connection that
                # dies mid-flight leaves the page spinning with no explanation.
                logger.exception("ask failed")
                if not settings().has_llm:
                    # Without this the user saw "FakeLLM ran out of scripted
                    # responses", an internal test-harness detail that says
                    # nothing about the actual problem.
                    yield sse(
                        "error",
                        {
                            "message": "No ANTHROPIC_API_KEY is configured, so "
                            "the service cannot answer questions. Add it to "
                            ".env and restart.",
                            "code": "missing_api_key",
                        },
                    )
                else:
                    yield sse("error", {"message": str(exc)})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # nginx buffers SSE into uselessness
            },
        )

    @app.post("/sync")
    async def trigger_sync() -> dict[str, Any]:
        """Re-read Jira now.

        Explicit rather than automatic on every question. Re-syncing per request
        would make each answer wait on a full project read and would hammer a
        client's tenant; re-syncing on a timer hides the staleness this exists
        to expose. A person asks, and `/health` shows them why they should.
        """
        svc: Services = app.state.services
        try:
            await _RESYNC(svc)
        except Exception as exc:
            logger.exception("resync failed")
            raise HTTPException(
                status_code=503,
                detail=f"Could not re-read Jira: {exc}. The previous findings "
                f"are still being served — check /health for how old they are.",
            ) from exc

        complete = isinstance(svc.sync, CompleteSync)
        return {
            "status": svc.sync.status.value,
            "findings_source": svc.findings_source,
            "risk_findings": len(svc.sync.findings) if complete else None,
            "issues_seen": svc.sync.issues_seen,
            "synced_at": svc.synced_at.isoformat(),
            "message": "" if complete else svc.sync.operator_message(),
        }

    @app.post("/report")
    async def report() -> dict[str, Any]:
        """The weekly narrative: rules assemble it, the model writes it up.

        `build_report` is the same function `scripts/demo.py` calls and
        `tests/test_sync_gate.py` pins — one report generator, gated the same
        way regardless of caller. A `PartialSync` is refused here rather than
        producing a narrative over `partial_findings`: a fluent paragraph about
        the projects that WERE read is exactly the "looks finished, silently
        omits the rest" failure ADR-004 exists to prevent.

        POST, not GET, because it costs an LLM call — the same reasoning as
        `/sync`: explicit and re-runnable on demand, not something a page load
        pays for by accident.
        """
        svc: Services = app.state.services
        if not isinstance(svc.sync, CompleteSync):
            raise HTTPException(status_code=409, detail=svc.sync.operator_message())

        delivery_report = build_report(svc.sync)
        narrative = await generate_narrative(svc.llm, delivery_report)

        # The full numbered list, not `narrative.citations` — that tuple holds
        # only the findings the model actually cited, so a bracket the model
        # left out of it would still be [4] in the text but missing from a
        # list built by filtering. Same discipline the graph's `evidence`
        # publishes for `/ask`: the reader gets the list the brackets index
        # into, not the subset that happened to get used.
        all_findings = [f for section in delivery_report.sections for f in section.findings]
        passages = [finding_passage(f) for f in all_findings]

        return {
            "generated_at": svc.synced_at.isoformat(),
            "findings_source": svc.findings_source,
            "issues_seen": delivery_report.issues_seen,
            "projects_covered": delivery_report.projects_covered,
            "rule_counts": delivery_report.rule_counts,
            "sections": [
                {
                    "level": section.level.value,
                    "findings": [
                        {
                            "issue_key": f.issue_key,
                            "rule_id": f.rule_id,
                            "detail": f.detail,
                        }
                        for f in section.findings
                    ],
                }
                for section in delivery_report.sections
            ],
            "narrative": narrative.text,
            "narrative_answered": narrative.answered,
            "refusal_reason": narrative.refusal_reason,
            "sources": [
                {"index": index, "label": p.label, "url": p.url}
                for index, p in enumerate(passages, start=1)
            ],
        }

    @app.get("/proposals")
    async def list_proposals() -> dict[str, Any]:
        svc: Services = app.state.services
        return {
            "pending": [
                {
                    "id": proposal_id,
                    "issue_key": p.issue_key,
                    "action": p.action_type,
                    "rationale": p.rationale,
                    "citations": [
                        {"source": c.source_id, "url": c.url} for c in p.citations
                    ],
                }
                for proposal_id, p in svc.store.pending.items()
            ],
            "audit": [entry.to_row() for entry in svc.store.audit[-20:]],
        }

    @app.post("/proposals/{proposal_id}/approve")
    async def approve(
        proposal_id: str,
        body: ApprovalRequest,
        x_actor: str = Header(default=""),
    ) -> dict[str, Any]:
        svc: Services = app.state.services
        if not x_actor.strip():
            # Rejected here as well as in the store, so the caller gets a 400
            # naming the header rather than a 500 from an ApprovalError.
            raise HTTPException(
                status_code=400,
                detail="X-Actor header is required. Every approval must name a "
                "person; the audit trail has to show who made each change.",
            )
        try:
            approved = svc.store.approve(proposal_id, actor=x_actor, note=body.note)
            await apply_action(approved, svc.writer, svc.store)
        except ApprovalError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return {
            "status": "applied",
            "proposal_id": proposal_id,
            "approved_by": x_actor,
        }

    @app.post("/proposals/{proposal_id}/reject")
    async def reject(
        proposal_id: str,
        body: ApprovalRequest,
        x_actor: str = Header(default=""),
    ) -> dict[str, Any]:
        svc: Services = app.state.services
        if not x_actor.strip():
            raise HTTPException(status_code=400, detail="X-Actor header is required.")
        try:
            svc.store.reject(proposal_id, actor=x_actor, note=body.note)
        except ApprovalError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "rejected", "proposal_id": proposal_id}

    # Mounted last so the API routes above win. Vite builds into this folder,
    # so `npm run build` then `uvicorn` is the whole deploy — no copy step to
    # forget. html=True serves index.html at "/" and for unknown paths, which
    # is what a client-side router needs.
    static = Path(__file__).parent / "static"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    else:
        logger.warning(
            "No built UI at %s — run `npm run build` in frontend/. The API "
            "works without it.",
            static,
        )

    return app


app = create_app()
