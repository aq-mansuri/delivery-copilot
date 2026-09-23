"""The agent graph.

A state machine, not a free-running loop. The difference matters here.

A ReAct-style loop ("think, act, observe, repeat until done") lets the model
decide when to stop. Against a client's live tenant that is an unbounded number
of API calls and an unbounded bill, decided by a model that cannot see either.
It also makes failures unreproducible: the same question can take two steps on
Monday and nine on Tuesday.

A graph makes the control flow ours. The model chooses *which* tool, never *how
many times* or *whether to continue* — those are edges, and edges are code. The
consequences are practical rather than theoretical:

- A hard step ceiling, so cost per question has a maximum you can quote.
- Every path terminates, including the ones the model gets wrong.
- The flow is inspectable in a diagram, which matters when a client's security
  review asks what the agent can do.

Nodes here are plain async functions over a TypedDict. LangGraph supplies the
wiring and the conditional edges; nothing in the node bodies depends on it, so
the logic stays testable without the framework.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, AsyncIterator, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.answering import (
    GroundedAnswer,
    Passage,
    answer_question,
    build_context,
)
from app.agent.llm import LLM
from app.agent.tools import ToolRegistry, ToolResult
from app.models.domain import ProposedAction
from app.rag.decomposition import Decomposer, multi_query_search
from app.rag.retrieval import HybridRetriever, ScoredChunk

logger = logging.getLogger(__name__)

# Two tool rounds is enough for "search the docs, then check the risk engine".
# Raising this is a decision with a cost attached, which is exactly why it is a
# named constant rather than a `while True`.
MAX_TOOL_ROUNDS = 3


def _append(left: list, right: list) -> list:
    """Reducer so parallel or repeated nodes accumulate rather than overwrite."""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    question: str
    actor: str

    # Conversation as sent to the model, including tool results.
    messages: Annotated[list[dict[str, Any]], _append]

    hits: list[ScoredChunk]
    subqueries: list[str]
    # Evidence produced by tools. Kept separate from `hits` because it is not
    # retrieval output, and merged only at answer time — see the answer node.
    tool_evidence: Annotated[list[Passage], _append]
    tool_rounds: int

    # The numbered list the answer's citations actually index into, written by
    # the answer node. Callers must report THIS rather than `hits`: once a tool
    # has run, tool output is [1] and the top retrieval chunk is not.
    evidence: list[Passage]

    # Full history, accumulated for tracing.
    tool_calls: Annotated[list[dict[str, Any]], _append]
    # Only what the MOST RECENT reasoning turn asked for. Deliberately not
    # accumulated: the termination check reads this, and an accumulated list
    # makes every turn after the first look like it requested tools — an
    # infinite loop that a reducer quietly creates for you.
    pending_calls: list[dict[str, Any]]

    answer: GroundedAnswer | None
    proposals: Annotated[list[ProposedAction], _append]

    input_tokens: int
    output_tokens: int
    halted_reason: str


class AgentDeps:
    """Everything the nodes need, passed once rather than through global state."""

    def __init__(
        self,
        llm: LLM,
        retriever: HybridRetriever,
        registry: ToolRegistry,
        *,
        allow_writes: bool = False,
        top_k: int = 5,
        decomposer: Decomposer | None = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.registry = registry
        # Optional because the graph must run without it, and measured because
        # it is not free: decomposition is one extra call plus one search per
        # sub-query. `run_eval.py --decompose` is where that trade is decided,
        # not here.
        self.decomposer = decomposer
        # Write tools are withheld unless the caller explicitly enables them.
        # Not offering a tool is more reliable than instructing a model not to
        # use it, and the default is the safe one.
        self.allow_writes = allow_writes
        self.top_k = top_k

    def schemas(self) -> list[dict[str, Any]]:
        return (
            self.registry.schemas()
            if self.allow_writes
            else self.registry.read_only_schemas()
        )


SYSTEM = """You are a delivery assistant for an insurance company.

You have tools for searching Confluence documentation and for reading risk \
findings computed from Jira. Use them before answering — do not answer from \
memory or inference.

Risk findings are computed by deterministic rules and are authoritative. \
Documentation describes intent and may be stale.

When you have gathered what you need, stop calling tools and answer. If the \
tools do not give you what the question needs, say so plainly rather than \
filling the gap."""


def build_graph(deps: AgentDeps):
    """Compile the graph.

    Shape:

        retrieve -> reason -> [tools -> reason]* -> answer -> END
                                 |
                                 +-> (round limit) -> answer

    `reason` is the only node the model controls, and it controls it only by
    choosing tools. Whether to loop again is decided by `_after_reason`, which
    is ours.
    """

    async def retrieve(state: AgentState) -> AgentState:
        """Always retrieve once, before the model sees anything.

        Cheaper and more predictable than making the first search a tool call:
        it guarantees the model has context on its first turn, which removes the
        "let me search for X" round-trip that adds latency and nothing else.

        That guarantee needs the passages to actually be *sent*. An earlier
        version stored them in state and left the first user turn as the bare
        question, so the model — correctly, given what it could see — opened
        every run by calling `search_documentation` to fetch what had already
        been fetched. The node's own docstring claimed otherwise, which is how
        it survived: offline the extra call is free and scripted, so no test
        noticed and no reader doubted the comment.

        The numbering here is the same `build_context` the answer node uses, so
        [1] means the same passage in both turns.
        """
        if deps.decomposer is not None:
            hits, subqueries = await multi_query_search(
                deps.retriever, deps.decomposer, state["question"], top_k=deps.top_k
            )
        else:
            hits = deps.retriever.search(state["question"], top_k=deps.top_k)
            subqueries = []

        context = build_context([Passage.from_hit(hit) for hit in hits])
        opening = (
            f"Context passages:\n\n{context}\n\nQuestion: {state['question']}"
            if hits
            # No passages retrieved. Sending an empty "Context passages:" header
            # reads as "the corpus is empty" and invites the model to answer
            # from memory; saying nothing was found sends it to its tools.
            else f"No documentation was retrieved for this.\n\n"
            f"Question: {state['question']}"
        )

        return {
            "hits": hits,
            "subqueries": list(subqueries),
            "messages": [{"role": "user", "content": opening}],
            "tool_rounds": 0,
            "pending_calls": [],
            "tool_evidence": [],
        }

    async def reason(state: AgentState) -> AgentState:
        response = await deps.llm.complete(
            system=SYSTEM,
            messages=state.get("messages") or [
                {"role": "user", "content": state["question"]}
            ],
            tools=deps.schemas(),
            max_tokens=1500,
            temperature=0.0,
        )

        update: AgentState = {
            "input_tokens": state.get("input_tokens", 0) + response.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + response.output_tokens,
        }

        if response.wants_tools():
            # The assistant turn must be echoed back verbatim alongside the
            # tool results, or the next call has tool_result blocks with no
            # matching tool_use and the API rejects it.
            update["messages"] = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": use.id,
                            "name": use.name,
                            "input": use.input,
                        }
                        for use in response.tool_uses
                    ],
                }
            ]
            calls = [
                {"name": use.name, "input": use.input, "id": use.id}
                for use in response.tool_uses
            ]
            update["tool_calls"] = calls
            update["pending_calls"] = calls
        else:
            # Explicitly cleared. Leaving a stale value here is the same bug in
            # a different place.
            update["pending_calls"] = []
        return update

    async def run_tools(state: AgentState) -> AgentState:
        """Execute the tools the model asked for.

        Errors come back as tool results rather than exceptions, so the model
        can correct itself on the next round instead of the request failing.
        """
        blocks: list[dict[str, Any]] = []
        proposals: list[ProposedAction] = []
        evidence: list[Passage] = []

        for call in state.get("pending_calls", []):
            result: ToolResult = await deps.registry.execute(
                call["name"], call["input"]
            )
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": result.content,
                    "is_error": result.is_error,
                }
            )
            if result.proposal is not None:
                proposals.append(result.proposal)
            if not result.is_error and result.content.strip():
                # Tool output is evidence and must be citable. Without this the
                # answer node numbers only retrieval chunks, so an answer built
                # from risk findings cites an unrelated Confluence page.
                tool = deps.registry.tools.get(call["name"])
                evidence.append(
                    Passage.from_tool(
                        call["name"],
                        result.content,
                        label=tool.evidence_label if tool else "",
                    )
                )

        return {
            "messages": [{"role": "user", "content": blocks}],
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "proposals": proposals,
            "tool_evidence": evidence,
            "pending_calls": [],
        }

    async def answer(state: AgentState) -> AgentState:
        """Final answer, through the grounding checker.

        Deliberately a separate node rather than reusing the model's last text.
        Everything user-facing goes through citation validation — including
        answers produced after a tool round, which is where a model is most
        likely to summarize rather than cite.
        """
        # Tool evidence first: when a tool has run, its output is what the
        # question was actually about, and passage [1] should be that rather
        # than a document retrieved before the model had decided anything.
        evidence: list[Passage] = list(state.get("tool_evidence") or [])
        evidence += [Passage.from_hit(hit) for hit in state.get("hits", [])]

        result = await answer_question(deps.llm, state["question"], evidence)
        return {
            "answer": result,
            # Published because it is the list the citations index into. A
            # caller that numbers `hits` instead shows the wrong source under
            # [1] on every run where a tool fired.
            "evidence": evidence,
            "input_tokens": state.get("input_tokens", 0) + result.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + result.output_tokens,
        }

    def _after_reason(state: AgentState) -> Literal["tools", "answer"]:
        """The termination decision. Ours, not the model's."""
        calls = state.get("pending_calls") or []
        if not calls:
            return "answer"
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            # Reached the ceiling. Answer with what we have rather than
            # truncating silently — a capped run that still answers is far more
            # useful than an error, and the reason is recorded.
            logger.warning("tool round limit reached for %r", state["question"])
            return "answer"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("reason", reason)
    graph.add_node("tools", run_tools)
    graph.add_node("answer", answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges(
        "reason", _after_reason, {"tools": "tools", "answer": "answer"}
    )
    graph.add_edge("tools", "reason")
    graph.add_edge("answer", END)

    return graph.compile()


def _initial_state(question: str, actor: str) -> AgentState:
    """The seed state.

    `messages` starts EMPTY on purpose. The retrieve node writes the opening
    user turn, because that turn has to carry the retrieved passages. Seeding a
    bare question here instead would either lose the context or leave two
    consecutive user turns for the API to reject.
    """
    return {
        "question": question,
        "actor": actor,
        "messages": [],
        "tool_calls": [],
        "pending_calls": [],
        "proposals": [],
        "tool_evidence": [],
        "evidence": [],
        "hits": [],
        "subqueries": [],
        "tool_rounds": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


async def run_agent(
    deps: AgentDeps, question: str, *, actor: str = ""
) -> AgentState:
    graph = build_graph(deps)
    return await graph.ainvoke(_initial_state(question, actor))


async def stream_agent(
    deps: AgentDeps, question: str, *, actor: str = ""
) -> AsyncIterator[tuple[str, Any]]:
    """The same run, reported node by node.

    Yields `(node_name, update)` as each node finishes, then `("final", state)`.

    This exists so the HTTP layer can show a tool call while it is happening
    rather than after everything is over. A question that consults the risk
    engine takes three model calls; a reader watching an undifferentiated
    spinner for that long assumes it has hung, and — worse for this system —
    never sees that the answer came from the rules rather than the model.

    It is a view over `build_graph`, not a second implementation. Both entry
    points compile the same graph from the same state, so there is no streaming
    path that can drift from the awaited one. `tests/test_graph.py` pins that
    by asserting the two produce the same final state.
    """
    graph = build_graph(deps)
    final: AgentState = {}

    async for mode, chunk in graph.astream(
        _initial_state(question, actor), stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            for node, update in chunk.items():
                yield node, update
        else:
            # "values" carries the accumulated state, reducers already applied.
            # Rebuilding it from the updates by hand would mean reimplementing
            # every reducer at the call site and getting one of them wrong.
            final = chunk

    yield "final", final
