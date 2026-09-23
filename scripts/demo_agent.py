"""End-to-end agent demo.

    python scripts/demo_agent.py            # scripted LLM, no API key needed
    python scripts/demo_agent.py --live     # real Claude, needs ANTHROPIC_API_KEY

The scripted mode is the point. A demo that requires credentials is a demo that
dies in front of a client when their network blocks you, and it cannot run in
CI. The scripted responses are realistic — including the ones that get rejected.

Five scenarios, chosen to show the system's judgement rather than its happy path:

1. A grounded answer with citations.
2. A tool call into the deterministic risk engine.
3. A refusal on a question the corpus cannot answer.
4. A confabulated answer caught and rejected by the grounding check.
5. A write proposal that stops at the approval gate, then is approved and
   applied with an audit record.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.approval import ApprovalStore, apply_action  # noqa: E402
from app.agent.graph import AgentDeps, run_agent  # noqa: E402
from app.agent.llm import (  # noqa: E402
    FakeLLM,
    text_response,
    tool_response,
)
from app.agent.tools import (  # noqa: E402
    ToolRegistry,
    make_flag_tool,
    make_risk_tool,
    make_search_tool,
)
from app.core.sandbox import sandbox_issues  # noqa: E402
from app.rag.chunking import chunk_page  # noqa: E402
from app.rag.corpus import load_offline_corpus  # noqa: E402
from app.rag.embedders import FakeEmbedder  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from app.risk.rules import RiskConfig, evaluate  # noqa: E402

NOW = datetime.now(timezone.utc)
CFG = RiskConfig(base_url="https://acme.atlassian.net")


def rule(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def make_deps(llm, findings, *, allow_writes=False) -> AgentDeps:
    chunks = [c for page in load_offline_corpus() for c in chunk_page(page)]
    retriever = HybridRetriever(chunks, FakeEmbedder(), candidate_pool=20)
    registry = ToolRegistry().register(
        make_search_tool(retriever), make_risk_tool(findings), make_flag_tool()
    )
    return AgentDeps(llm, retriever, registry, allow_writes=allow_writes)


def show(state, question: str, *, scripted: bool) -> None:
    answer = state["answer"]
    print(f"\n  Q: {question}")

    # Print the numbered evidence the model was given. Without it, a citation
    # like [1] is unverifiable by the reader — and in scripted mode it is
    # frequently wrong, because the script is written before retrieval runs and
    # cannot know which passage will rank first.
    #
    # `evidence`, not `hits`: after a tool round the answer node puts tool
    # output at [1], so numbering the retrieval list prints a Confluence page
    # beside a citation that means the risk engine.
    evidence = state.get("evidence") or []
    if evidence:
        print("\n     passages supplied:")
        for index, passage in enumerate(evidence[:5], start=1):
            print(f"       [{index}] {passage.label}")
    if state["tool_calls"]:
        for call in state["tool_calls"]:
            print(f"     [tool] {call['name']}({call['input'] or ''})")
    print(f"\n  A: {answer.text}")
    if answer.citations:
        print("\n     sources:")
        for source in answer.sources():
            print(f"       - {source['label']}  [{source['found_by']}]")
            print(f"         {source['url']}")
    if scripted and answer.citations:
        print(
            "\n     (scripted mode: the citation index is fixed in the script,"
            "\n      so it may not match the passage that actually ranked first."
            "\n      Run with --live for real citation behaviour.)"
        )
    if answer.refused:
        print(f"\n     REFUSED ({answer.refusal_reason})")
        if answer.rejected_reason:
            print(f"     reason: {answer.rejected_reason}")
    print(f"\n     tokens: {state['input_tokens']} in / {state['output_tokens']} out")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    findings = evaluate(sandbox_issues(NOW), CFG, NOW)

    if args.live:
        import os

        from dotenv import load_dotenv

        # Without this the key in .env is invisible and --live fails with
        # "No ANTHROPIC_API_KEY" even when the key is sitting right there.
        load_dotenv()

        if not os.getenv("ANTHROPIC_API_KEY"):
            # Checked once, up front. Failing four scenarios into a run wastes
            # the reader's time and makes a configuration problem look like a
            # code problem.
            print("--live needs ANTHROPIC_API_KEY.\n")
            print("  Add it to .env:")
            print("    ANTHROPIC_API_KEY=sk-ant-...\n")
            print("  Get one at console.anthropic.com -> API keys.")
            print("  Or drop --live to run the scripted version, which needs no key.")
            return 1

        from app.agent.llm import AnthropicLLM

        make_llm = lambda _script: AnthropicLLM()  # noqa: E731
    else:
        make_llm = lambda script: FakeLLM(responses=list(script))  # noqa: E731

    rule("1. GROUNDED ANSWER — every claim traced to a source")
    # Was "Why do we terminate TLS at the gateway?" — there is no TLS
    # architecture page in the corpus, so the live run correctly refused while
    # the scripted response faked a success. A demo question must be one the
    # corpus can actually answer, or the scripted mode lies.
    q = "Which controls are still outstanding before go-live?"
    state = await run_agent(
        make_deps(
            make_llm([
                text_response("Let me answer from the retrieved documentation."),
                text_response(
                    "Three controls remain unsatisfied before go-live: mutual "
                    "TLS certificates have not been issued for the production "
                    "gateway, penetration testing of the callback endpoint has "
                    "not been scheduled, and the audit trail for approval "
                    "actions is designed but not implemented [1]."
                ),
            ]),
            findings,
        ),
        q,
    )
    show(state, q, scripted=not args.live)

    rule("2. TOOL CALL — risk comes from rules, not from the model")
    q = "What is at risk right now?"
    state = await run_agent(
        make_deps(
            make_llm([
                tool_response("get_risk_findings", {}),
                text_response("Summarising."),
                text_response(
                    "Three issues are flagged: INS-101 has had no activity for 23 "
                    "days, INS-102 is blocked by PLAT-7, and INS-103 has carried "
                    "over three sprints [1]."
                ),
            ]),
            findings,
        ),
        q,
    )
    show(state, q, scripted=not args.live)

    rule("3. REFUSAL — the question presupposes something that does not exist")
    q = "Who approved the Acme security exception?"
    state = await run_agent(
        make_deps(
            make_llm([
                text_response("Checking the documentation."),
                text_response("INSUFFICIENT_EVIDENCE"),
            ]),
            findings,
        ),
        q,
    )
    show(state, q, scripted=not args.live)

    rule("4. CONFABULATION CAUGHT — fluent, plausible, and rejected")
    # The corpus records that the schema question went unanswered. Asking for a
    # DATE invites a fabrication the sources cannot support.
    q = "On exactly which date did Acme confirm the callback schema?"
    state = await run_agent(
        make_deps(
            make_llm([
                text_response("Checking."),
                text_response(
                    "The vendor confirmed the schema last Thursday and "
                    "engineering resumed work immediately afterwards."
                ),
            ]),
            findings,
        ),
        q,
    )
    show(state, q, scripted=not args.live)
    print("\n     ^ the question presupposes a confirmation that never happened.")
    print("       The scripted model answered anyway; the grounding check")
    print("       rejected it before it reached the user.")

    rule("5. WRITE PROPOSAL — stops at the approval gate")
    q = "Flag INS-101 as at risk."
    state = await run_agent(
        make_deps(
            make_llm([
                tool_response("propose_flag_at_risk", {
                    "issue_key": "INS-101",
                    "rationale": "No activity for 23 days while In Progress.",
                    "evidence_rule_ids": ["stale_in_progress"],
                }),
                text_response("Proposed."),
                text_response("I have proposed flagging INS-101 as at-risk [1]."),
            ]),
            findings,
            allow_writes=True,
        ),
        q,
        actor="vp@acme.com",
    )
    show(state, q, scripted=not args.live)

    store = ApprovalStore()
    writes: list[str] = []

    class DemoWriter:
        async def add_labels(self, issue_key: str, labels: list[str]) -> None:
            writes.append(f"labels {labels} -> {issue_key}")

        async def add_comment(self, issue_key: str, body: str) -> None:
            writes.append(f"comment -> {issue_key}: {body[:58]}...")

    for proposal in state["proposals"]:
        pid = store.submit(proposal)
        print(f"\n     proposal {pid}: {proposal.action_type} on {proposal.issue_key}")
        print(f"     rationale: {proposal.rationale}")
        print("     Jira writes so far:", writes or "(none — awaiting approval)")

        approved = store.approve(pid, actor="vp@acme.com", note="Agreed.")
        await apply_action(approved, DemoWriter(), store)

        print("\n     after approval:")
        for write in writes:
            print(f"       {write}")
        entry = store.audit[0]
        print("\n     audit record:")
        for key, value in entry.to_row().items():
            print(f"       {key:<12} {value}")

    rule("The split, restated")
    print("""
  Deterministic  stale detection, blocked dependencies, sprint carryover,
                 overdue, missing delivery dates
  Model          explaining those findings, answering questions with
                 citations, classifying blocker categories, deciding when
                 to decline

  The model never decides what is at risk, and never writes to Jira.
""")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
