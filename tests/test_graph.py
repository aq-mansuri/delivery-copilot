"""Tests for the agent graph.

The property under test: control flow belongs to us. The model picks tools; it
does not decide whether to keep going.

Note on scripting: a run makes one LLM call per `reason` pass PLUS one final
call in the `answer` node. The answer node deliberately re-asks rather than
reusing the reasoning turn's text, so everything user-facing goes through the
grounding check. Script accordingly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agent.graph import MAX_TOOL_ROUNDS, AgentDeps, run_agent, stream_agent
from app.agent.llm import FakeLLM, text_response, tool_response
from app.agent.tools import ToolRegistry, make_flag_tool, make_risk_tool, make_search_tool
from app.models.domain import Citation, Page, PageSection, RiskFinding, RiskLevel
from app.rag.chunking import chunk_page
from app.rag.decomposition import HeuristicDecomposer
from app.rag.embedders import FakeEmbedder
from app.rag.retrieval import HybridRetriever

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)


def make_retriever():
    page = Page(
        id="adr-tls", space_key="ARCH", title="ADR: Claims vendor TLS",
        url="https://x/wiki/adr-tls", version=1, updated_at=NOW,
        sections=[
            PageSection(heading="Context", level=2, text="Mutual TLS required. " * 25),
            PageSection(heading="Decision", level=2, text="Terminate at gateway. " * 25),
        ],
    )
    return HybridRetriever(chunk_page(page), FakeEmbedder())


def finding(key="INS-101"):
    return RiskFinding(
        issue_key=key, rule_id="stale_in_progress", level=RiskLevel.HIGH,
        detail="No activity for 23 days.",
        citations=[Citation(source_type="issue", source_id=key, url=f"https://x/{key}")],
    )


def make_deps(llm, *, allow_writes=False, decomposer=None):
    registry = ToolRegistry().register(
        make_search_tool(make_retriever()),
        make_risk_tool([finding()]),
        make_flag_tool(),
    )
    return AgentDeps(
        llm,
        make_retriever(),
        registry,
        allow_writes=allow_writes,
        decomposer=decomposer,
    )


class TestTermination:
    async def test_no_tool_calls_goes_straight_to_answer(self):
        llm = FakeLLM(responses=[
            text_response("thinking"),
            text_response("Mutual TLS is required [1]."),
        ])
        state = await run_agent(make_deps(llm), "why mutual TLS?")
        assert state["answer"].answered
        assert state["tool_rounds"] == 0

    async def test_tool_round_limit_is_enforced(self):
        """A model that keeps asking for tools must still terminate.

        This is the property a free-running loop does not have.
        """
        llm = FakeLLM(responses=[
            tool_response("get_risk_findings", {}),
            tool_response("get_risk_findings", {}),
            tool_response("get_risk_findings", {}),
            tool_response("get_risk_findings", {}),
            tool_response("get_risk_findings", {}),
            text_response("Mutual TLS is required [1]."),
        ])
        state = await run_agent(make_deps(llm), "what is at risk?")
        assert state["tool_rounds"] <= MAX_TOOL_ROUNDS
        assert state["answer"] is not None

    async def test_every_run_produces_an_answer_object(self):
        llm = FakeLLM(responses=[
            tool_response("get_risk_findings", {}),
            text_response("done"),
            text_response("INS-101 has been stale for 23 days [1]."),
        ])
        state = await run_agent(make_deps(llm), "what is at risk?")
        assert state["answer"] is not None
        assert state["answer"].answered


class TestToolFlow:
    async def test_tool_result_is_returned_to_the_model(self):
        llm = FakeLLM(responses=[
            tool_response("get_risk_findings", {"issue_key": "INS-101"}),
            text_response("done"),
            text_response("INS-101 has been stale for 23 days [1]."),
        ])
        await run_agent(make_deps(llm), "status of INS-101?")

        # Second call must carry the assistant tool_use turn AND the result.
        second = llm.calls[1]["messages"]
        roles = [m["role"] for m in second]
        assert "assistant" in roles
        blocks = [b for m in second if isinstance(m["content"], list)
                  for b in m["content"]]
        assert any(b.get("type") == "tool_use" for b in blocks)
        assert any(b.get("type") == "tool_result" for b in blocks)

    async def test_unknown_tool_does_not_crash_the_run(self):
        """A hallucinated tool name should let the model self-correct."""
        llm = FakeLLM(responses=[
            tool_response("no_such_tool", {}),
            text_response("done"),
            text_response("Mutual TLS is required [1]."),
        ])
        state = await run_agent(make_deps(llm), "q")
        assert state["answer"] is not None

    async def test_proposals_are_collected_not_applied(self):
        llm = FakeLLM(responses=[
            tool_response("propose_flag_at_risk", {
                "issue_key": "INS-101",
                "rationale": "Stale for 23 days.",
                "evidence_rule_ids": ["stale_in_progress"],
            }),
            text_response("done"),
            text_response("I have proposed flagging INS-101 [1]."),
        ])
        state = await run_agent(make_deps(llm, allow_writes=True), "flag it")
        assert len(state["proposals"]) == 1
        assert state["proposals"][0].approved_by is None


class TestToolExposure:
    async def test_write_tools_are_withheld_by_default(self):
        """Not offering a tool beats instructing the model not to use it."""
        llm = FakeLLM(responses=[
            text_response("thinking"),
            text_response("Mutual TLS is required [1]."),
        ])
        await run_agent(make_deps(llm), "q")
        assert "propose_flag_at_risk" not in llm.calls[0]["tools"]
        assert "get_risk_findings" in llm.calls[0]["tools"]

    async def test_write_tools_appear_only_when_explicitly_enabled(self):
        llm = FakeLLM(responses=[
            text_response("thinking"),
            text_response("Mutual TLS is required [1]."),
        ])
        await run_agent(make_deps(llm, allow_writes=True), "q")
        assert "propose_flag_at_risk" in llm.calls[0]["tools"]


class TestGrounding:
    async def test_final_answer_still_passes_the_grounding_check(self):
        """Post-tool answers are where a model most often summarizes without
        citing. The answer node does not trust the reasoning turn."""
        llm = FakeLLM(responses=[
            tool_response("get_risk_findings", {}),
            text_response("done"),
            text_response("The vendor confirmed everything and work resumes."),
        ])
        state = await run_agent(make_deps(llm), "status?")
        assert state["answer"].refused
        assert state["answer"].refusal_reason == "failed_grounding_check"


class TestAccounting:
    async def test_tokens_accumulate_across_the_run(self):
        llm = FakeLLM(responses=[
            tool_response("get_risk_findings", {}),
            text_response("done"),
            text_response("INS-101 is stale [1].", input_tokens=500, output_tokens=40),
        ])
        state = await run_agent(make_deps(llm), "q")
        assert state["input_tokens"] >= 500


class TestRetrievedContextReachesTheModel:
    """The retrieve node existed to save a round-trip and did not.

    It stored hits in state, the reason node sent only the question, and the
    model's first turn had no context at all. Its own docstring claimed the
    opposite. The cost was one wasted `search_documentation` call on every run,
    re-fetching what had already been fetched — invisible offline, a third of
    the latency and bill on a live one.
    """

    async def test_first_turn_carries_the_retrieved_passages(self):
        llm = FakeLLM(responses=[
            text_response("thinking"),
            text_response("Mutual TLS is required [1]."),
        ])
        await run_agent(make_deps(llm), "why mutual TLS?")

        first = llm.calls[0]["messages"]
        assert len(first) == 1, "the model's first turn must be a single user turn"
        content = first[0]["content"]
        assert "Mutual TLS required" in content, "retrieved text is missing"
        assert "[1]" in content, "passages must be numbered for citation"
        assert "why mutual TLS?" in content, "the question itself is missing"

    async def test_passage_numbering_matches_the_answer_node(self):
        """The model sees [1]..[n] twice — once reasoning, once answering. If
        the two lists differ, a citation resolves to the wrong source."""
        llm = FakeLLM(responses=[
            text_response("thinking"),
            text_response("Mutual TLS is required [1]."),
        ])
        state = await run_agent(make_deps(llm), "why mutual TLS?")

        reasoning_context = llm.calls[0]["messages"][0]["content"]
        answering_context = llm.calls[1]["messages"][0]["content"]
        first_label = state["evidence"][0].label
        assert first_label in reasoning_context
        assert first_label in answering_context

    async def test_decomposer_is_used_when_supplied(self):
        """Query decomposition is a measured retrieval gain (run_eval.py
        --decompose). Routing the UI through the graph must not silently drop
        it."""
        llm = FakeLLM(responses=[
            text_response("thinking"),
            text_response("Mutual TLS is required [1]."),
        ])
        state = await run_agent(
            make_deps(llm, decomposer=HeuristicDecomposer()),
            "why mutual TLS and what does the gateway terminate?",
        )
        assert state["subqueries"]
        assert state["hits"]

    async def test_no_decomposer_still_retrieves(self):
        llm = FakeLLM(responses=[
            text_response("thinking"),
            text_response("Mutual TLS is required [1]."),
        ])
        state = await run_agent(make_deps(llm), "why mutual TLS?")
        assert state["hits"]
        assert state["subqueries"] == []


class TestEvidenceIsReported:
    """Whatever the answer cited must be recoverable, or the UI lights up the
    wrong source when a reader clicks [1]."""

    async def test_evidence_is_exposed_in_the_final_state(self):
        llm = FakeLLM(responses=[
            text_response("thinking"),
            text_response("Mutual TLS is required [1]."),
        ])
        state = await run_agent(make_deps(llm), "q")
        assert state["evidence"]
        assert state["evidence"][0].label

    async def test_tool_output_is_evidence_one_when_a_tool_ran(self):
        """The answer node puts tool results first. A UI numbering from `hits`
        alone shows a Confluence page as [1] while the answer means the risk
        engine."""
        llm = FakeLLM(responses=[
            tool_response("get_risk_findings", {}),
            text_response("done"),
            text_response("INS-101 has been stale for 23 days [1]."),
        ])
        state = await run_agent(make_deps(llm), "what is at risk?")
        assert state["evidence"][0].found_by == "tool"
        assert state["answer"].citations[0].found_by == "tool"


class TestStreaming:
    """The API streams the graph rather than awaiting it, so a reader watches
    the tool call happen instead of a spinner."""

    async def test_every_node_is_reported_in_order(self):
        llm = FakeLLM(responses=[
            tool_response("get_risk_findings", {}),
            text_response("done"),
            text_response("INS-101 has been stale for 23 days [1]."),
        ])
        seen = []
        final = None
        async for node, payload in stream_agent(make_deps(llm), "what is at risk?"):
            if node == "final":
                final = payload
            else:
                seen.append(node)

        assert seen == ["retrieve", "reason", "tools", "reason", "answer"]
        assert final["answer"].answered

    async def test_final_state_matches_an_awaited_run(self):
        """Streaming must not be a second, subtly different code path."""
        script = lambda: [
            tool_response("get_risk_findings", {}),
            text_response("done"),
            text_response("INS-101 has been stale for 23 days [1]."),
        ]
        awaited = await run_agent(make_deps(FakeLLM(responses=script())), "q")

        streamed = None
        async for node, payload in stream_agent(
            make_deps(FakeLLM(responses=script())), "q"
        ):
            if node == "final":
                streamed = payload

        assert streamed["answer"].text == awaited["answer"].text
        assert len(streamed["proposals"]) == len(awaited["proposals"])
        assert streamed["tool_rounds"] == awaited["tool_rounds"]

    async def test_proposals_surface_on_the_tools_update(self):
        """The API submits proposals to the approval store as they appear. If
        they only arrived in the final state, "waiting on you" would fill in
        after the answer rather than as the agent works."""
        llm = FakeLLM(responses=[
            tool_response("propose_flag_at_risk", {
                "issue_key": "INS-101",
                "rationale": "Stale for 23 days.",
                "evidence_rule_ids": ["stale_in_progress"],
            }),
            text_response("done"),
            text_response("I have proposed flagging INS-101 [1]."),
        ])
        found = []
        async for node, payload in stream_agent(
            make_deps(llm, allow_writes=True), "flag INS-101 as at risk"
        ):
            if node == "tools":
                found.extend(payload.get("proposals") or [])
        assert len(found) == 1
        assert found[0].issue_key == "INS-101"


class TestEvidenceLabels:
    """What a citation says it came from has to be true.

    Every tool's output used to be labelled "(computed from Jira)" because the
    label was derived from the tool name. Correct for the risk engine; for the
    write tool it meant a proposal the model had drafted thirty seconds earlier
    appeared beside the answer as though Jira had reported it — the exact
    confusion the approval gate exists to prevent.
    """

    async def test_risk_findings_are_labelled_as_computed(self):
        llm = FakeLLM(responses=[
            tool_response("get_risk_findings", {}),
            text_response("done"),
            text_response("INS-101 has been stale for 23 days [1]."),
        ])
        state = await run_agent(make_deps(llm), "what is at risk?")
        assert "computed from Jira" in state["evidence"][0].label

    async def test_a_proposal_is_not_labelled_as_computed_from_jira(self):
        llm = FakeLLM(responses=[
            tool_response("propose_flag_at_risk", {
                "issue_key": "INS-101",
                "rationale": "Stale for 23 days.",
                "evidence_rule_ids": ["stale_in_progress"],
            }),
            text_response("done"),
            text_response("I have proposed flagging INS-101 [1]."),
        ])
        state = await run_agent(make_deps(llm, allow_writes=True), "flag INS-101")
        label = state["evidence"][0].label
        assert "computed from Jira" not in label
        assert "awaiting approval" in label.lower()

    async def test_an_unregistered_tool_still_produces_a_usable_label(self):
        """A label is not allowed to be empty — the evidence rail would render
        a blank row that a reader cannot interpret."""
        from app.agent.answering import Passage

        assert Passage.from_tool("some_tool", "content").label == "some_tool"
