"""Agent tools.

The split that matters: **read tools return data, write tools return proposals.**
There is no code path from tool execution to a Jira write. Not a policy the
handler checks — an absence. `ToolRegistry` holds no Jira write client, so a
write tool physically cannot mutate anything, and no prompt injection, jailbreak
or model error can make it.

This is ADR-002 expressed the same way ADR-004 was: encode the rule in the
structure rather than in the discipline of whoever calls it. Executing an
approved action lives in `app/agent/approval.py`, needs a named human, and is
reachable only from the API layer.

Tool descriptions are prompt, not documentation. The model picks tools from them,
so each one says when NOT to use it as well as when to — an omission that shows
up as the model reaching for search when it should be reading the risk engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Sequence

from app.core.sync_result import CompleteSync, SyncResult
from app.models.domain import Citation, ProposedAction, RiskFinding
from app.rag.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    """What comes back from a tool call.

    `proposal` is populated only by write tools. Keeping it a distinct field
    rather than stuffing a proposal into `content` means the graph can route on
    it without parsing text, and the approval gate cannot be bypassed by a tool
    that merely describes a proposal in prose.
    """

    content: str
    proposal: ProposedAction | None = None
    citations: tuple[Citation, ...] = ()
    is_error: bool = False


ToolHandler = Callable[..., Awaitable[ToolResult]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    writes: bool = False
    # How this tool's output is labelled when the answer cites it. Written in
    # the reader's language, not the tool's: someone checking a claim about
    # delivery risk should see where the claim came from, not a function name.
    evidence_label: str = ""

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ---------------------------------------------------------------- read tools


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Natural-language search over Confluence documentation. Use the "
                "user's own words; do not translate into keywords."
            ),
        },
        "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
    },
    "required": ["query"],
}

SEARCH_DESCRIPTION = """Search Confluence documentation (architecture decisions, \
release plans, contracts, compliance requirements, sprint notes).

Use this for questions about WHY something was decided, what a plan says, or what \
a contract commits to.

Do NOT use this to find out which issues are at risk, blocked, or stale — that \
information is computed from Jira by get_risk_findings and is authoritative. \
Documentation may be out of date; the risk engine is not."""


RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_key": {
            "type": "string",
            "description": "Optional. Restrict to one issue, e.g. INS-101.",
        },
        "level": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Optional. Restrict to one severity.",
        },
    },
    "required": [],
}

RISK_DESCRIPTION = """Return current delivery risk findings computed from Jira: \
stale issues, blocked dependencies, sprint carryover, overdue items and missing \
delivery dates.

These are computed by deterministic rules, not inferred. Treat them as fact.

Use this for any question about what is at risk, what is blocked, what is \
slipping, or the state of delivery. Do NOT search documentation for this — \
documentation describes intent, this describes reality."""


def make_search_tool(retriever: HybridRetriever) -> Tool:
    async def handler(query: str, top_k: int = 5) -> ToolResult:
        hits = retriever.search(query, top_k=top_k)
        if not hits:
            return ToolResult(content="No matching documentation found.")

        blocks = [
            f"[{i}] {hit.chunk.citation_label()}\n"
            f"Source: {hit.chunk.page_url}\n{hit.chunk.text}"
            for i, hit in enumerate(hits, start=1)
        ]
        return ToolResult(
            content="\n\n---\n\n".join(blocks),
            citations=tuple(
                Citation(
                    source_type="confluence_page",
                    source_id=hit.chunk.page_id,
                    url=hit.chunk.page_url,
                    excerpt=hit.chunk.citation_label(),
                )
                for hit in hits
            ),
        )

    return Tool(
        name="search_documentation",
        description=SEARCH_DESCRIPTION,
        input_schema=SEARCH_SCHEMA,
        handler=handler,
        evidence_label="Confluence documentation (searched)",
    )


def make_risk_tool(
    findings: Sequence[RiskFinding], as_of: datetime | None = None
) -> Tool:
    async def handler(
        issue_key: str | None = None, level: str | None = None
    ) -> ToolResult:
        selected = list(findings)
        if issue_key:
            selected = [f for f in selected if f.issue_key == issue_key.upper()]
        if level:
            selected = [f for f in selected if f.level.value == level.lower()]

        if not selected:
            # An empty result is an ANSWER, and the wording has to make that
            # unmissable. The previous version said "No risk findings for
            # INS-12. This means the rules found nothing, not that the check
            # failed." — which reads as a caveat, and the answering model
            # classified it as missing context and declined. A lead asking "is
            # INS-12 blocked?" got "the available content does not contain
            # enough information", when the rules knew the answer was no.
            #
            # So: state the negative in the words the question uses, name what
            # was checked so the scope of the "no" is explicit, and say it is
            # definitive. This is the one place where the rules can assert
            # something confidently, and hedging here wastes it.
            # Anchored to when the data was read. Jira keeps changing after a
            # sync, and this service reads findings once — so a confident "not
            # blocked" can be true of the snapshot and false of the tenant. It
            # happened twice on a live demo: a blocker was linked, and the
            # answer stayed "no" with no hint that the two might differ. The
            # assertiveness is right; the timestamp is what makes it safe.
            stamp = (
                f", as of {as_of:%Y-%m-%d %H:%M} UTC" if as_of is not None else ""
            )

            if issue_key:
                scope = f"{issue_key.upper()}"
                opening = f"The risk engine checked {scope} and found no risk findings for it."
            elif level:
                scope = f"{level.lower()} severity"
                opening = f"The risk engine found no risk findings at {scope}."
            else:
                opening = "The risk engine found no risk findings in the current sync."

            return ToolResult(
                content=(
                    f"{opening} None of the rules matched: it is not stale, "
                    f"not blocked (no blocking issue is linked to it and its "
                    f"status is not a blocked one), not carried over between "
                    f"sprints, not overdue, and not missing a delivery date."
                    f"\n\n"
                    f"\n\nThis is a definitive negative result from "
                    f"deterministic rules that ran successfully over a complete "
                    f"sync{stamp} — not missing data. Answer with that rather "
                    f"than declining.\n\n"
                    f"State the negative only for the things listed above. The "
                    f"rules cover delivery risk signals in Jira; they are not a "
                    f"judgement that the work is healthy in every other respect."
                )
            )

        lines = [
            f"[{f.level.value.upper()}] {f.issue_key} ({f.rule_id}): {f.detail}"
            for f in selected
        ]
        return ToolResult(
            content="\n".join(lines),
            citations=tuple(c for f in selected for c in f.citations),
        )

    return Tool(
        name="get_risk_findings",
        description=RISK_DESCRIPTION,
        input_schema=RISK_SCHEMA,
        handler=handler,
        evidence_label="Risk findings — computed from Jira by rule, not inferred",
    )


def make_unavailable_risk_tool(reason: str) -> Tool:
    """The risk tool when the sync it would read from is incomplete.

    Same name, same schema, no findings. ADR-004 keeps a partial sync out of
    reports by making `build_report` accept only `CompleteSync`; the agent is
    the second consumer of the same data and needs the same guarantee. Serving
    `partial_findings` with a caveat attached would not work — "INS-101 is the
    only thing at risk" reads as a complete picture no matter what precedes it,
    and the reader has no way to tell which projects were never opened.

    `is_error=True` so the graph does not file this as evidence the answer may
    cite. The text explains the fix because the model relays it to whoever
    asked, and that person is usually the one who can get the permission
    granted.
    """

    async def handler(
        issue_key: str | None = None, level: str | None = None
    ) -> ToolResult:
        # Arguments are accepted and ignored. The model cannot know the sync
        # failed, so it will filter; refusing on a signature mismatch would
        # surface as a tool crash instead of an explanation.
        return ToolResult(
            content=(
                "Risk findings are unavailable for this sync, so none are "
                f"reported. {reason} Until the sync completes, treat the risk "
                "engine as having no answer rather than as reporting nothing "
                "wrong."
            ),
            is_error=True,
        )

    return Tool(
        name="get_risk_findings",
        description=RISK_DESCRIPTION,
        input_schema=RISK_SCHEMA,
        handler=handler,
        evidence_label="Risk findings — unavailable for this sync",
    )


def risk_tool_for(sync: SyncResult) -> Tool:
    """Pick the risk tool a sync result is entitled to.

    The isinstance check is the gate, and it is here rather than at every call
    site for the same reason `build_report` takes `CompleteSync`: one place
    decides, and no later caller can forget.
    """
    if isinstance(sync, CompleteSync):
        return make_risk_tool(sync.findings, as_of=sync.finished_at)
    return make_unavailable_risk_tool(sync.operator_message())

# --------------------------------------------------------------- write tools


FLAG_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_key": {"type": "string", "description": "e.g. INS-101"},
        "rationale": {
            "type": "string",
            "description": (
                "Why this issue should be flagged, in one sentence a delivery "
                "lead can evaluate without opening the ticket."
            ),
        },
        "evidence_rule_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Rule ids from get_risk_findings supporting this. Required — a "
                "proposal with no evidence will be rejected."
            ),
        },
    },
    "required": ["issue_key", "rationale", "evidence_rule_ids"],
}

FLAG_DESCRIPTION = """Propose flagging a Jira issue as at-risk.

This does NOT modify Jira. It creates a proposal that a named delivery lead must \
approve before anything is written. Say so when you use it — the user should \
never believe a change has been made.

Only propose this when get_risk_findings supports it. Do not propose flags based \
on documentation or inference."""


def make_flag_tool() -> Tool:
    async def handler(
        issue_key: str, rationale: str, evidence_rule_ids: list[str]
    ) -> ToolResult:
        if not evidence_rule_ids:
            return ToolResult(
                content="Rejected: a flag proposal requires at least one "
                "supporting rule id from get_risk_findings.",
                is_error=True,
            )

        proposal = ProposedAction(
            issue_key=issue_key.upper(),
            action_type="flag_at_risk",
            payload={"labels_add": ["at-risk"]},
            rationale=rationale,
            citations=[
                Citation(
                    source_type="risk_rule",
                    source_id=rule_id,
                    url=f"#rule/{rule_id}",
                )
                for rule_id in evidence_rule_ids
            ],
        )
        return ToolResult(
            content=(
                f"Proposed flagging {proposal.issue_key} as at-risk. "
                "AWAITING HUMAN APPROVAL — nothing has been written to Jira."
            ),
            proposal=proposal,
        )

    return Tool(
        name="propose_flag_at_risk",
        description=FLAG_DESCRIPTION,
        input_schema=FLAG_SCHEMA,
        handler=handler,
        writes=True,
        evidence_label="Proposed change — awaiting approval, not written to Jira",
    )


# ------------------------------------------------------------------ registry


class ToolError(RuntimeError):
    pass


@dataclass
class ToolRegistry:
    """Holds tools and executes them by name.

    Note what this class does NOT hold: any client capable of writing to Jira.
    That absence is the security property. A write tool returns a ProposedAction
    because there is nothing here it could write with.
    """

    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, *tools: Tool) -> "ToolRegistry":
        for tool in tools:
            self.tools[tool.name] = tool
        return self

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_anthropic() for tool in self.tools.values()]

    def read_only_schemas(self) -> list[dict[str, Any]]:
        """Offer only read tools.

        Used for plain question answering, where proposing a mutation is never
        appropriate. Withholding a tool is more reliable than instructing the
        model not to use one.
        """
        return [t.to_anthropic() for t in self.tools.values() if not t.writes]

    async def execute(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            # Returned as a result, not raised. A hallucinated tool name should
            # let the model correct itself on the next turn rather than crash
            # a user-facing request.
            return ToolResult(
                content=f"Unknown tool {name!r}. Available: "
                f"{', '.join(sorted(self.tools))}",
                is_error=True,
            )
        try:
            return await tool.handler(**tool_input)
        except TypeError as exc:
            return ToolResult(
                content=f"Invalid arguments for {name}: {exc}", is_error=True
            )
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return ToolResult(content=f"Tool {name} failed: {exc}", is_error=True)
