"""Narrative generation over a delivery report.

`build_report` (app/core/report.py) is the rules half: it groups findings by
level and refuses anything but a `CompleteSync`. This module is the model half
— it turns those findings into the prose a delivery lead reads before an
executive review, one paragraph per severity level, every claim carrying a
citation back to the finding that produced it.

It does not invent a second way to enforce "no claim without a source". The
risk tool (`app/agent/tools.py`) was already a second consumer of the same
grounding machinery `/ask` uses (ADR-009); this narrative is a third. Each
finding becomes a numbered `Passage`, exactly as retrieval chunks and tool
output already do, and the same `answer_question` / `check_grounding` path
rejects a narrative that makes a claim it cannot cite.

An empty report is a definitive answer, not a gap — no LLM call, and no risk
of the model reading "nothing to summarize" as insufficient context. That
failure already happened once, to a tool answer rather than a narrative: see
the CLAUDE.md section "An empty result is an answer, not a gap".
"""

from __future__ import annotations

from app.agent.answering import GroundedAnswer, Passage, answer_question
from app.agent.llm import LLM
from app.core.report import DeliveryReport
from app.models.domain import RiskFinding

NARRATIVE_QUESTION = (
    "Write this week's delivery risk narrative for a delivery lead ahead of "
    "an executive review."
)

NARRATIVE_SYSTEM_PROMPT = """You write a weekly delivery-risk narrative for an \
insurance company from ONLY the numbered findings provided.

Each finding was produced by a deterministic rule, not inferred — treat every \
one as fact. There is no missing-context case here: the findings ARE the \
complete set of things the rules flagged this week.

Rules, in priority order:

1. Cover every finding provided, once each. Group by severity, HIGH first, and \
name the issue key for every finding you mention.

2. Every factual claim must carry a citation in square brackets referencing the \
finding it restates, like [1] or [2][3]. Cite at least once per paragraph, and \
prefer citing each sentence that states a fact. A paragraph with no citation \
will be rejected.

3. Use the finding's own words. If a finding says an issue has had "no activity \
for 21 days", do not write "abandoned". Do not turn a rule match into a \
diagnosis of root cause the rule did not make.

4. Do not recommend actions and do not add a "next steps" section. A delivery \
lead decides what to do with this; your job is an accurate weekly picture.

5. Do not comment on issues that have no finding in this list. You were given \
only what the rules flagged, not the full state of every project.

6. Be concise. This opens an executive review — a short paragraph per severity \
level actually present is enough."""


def finding_passage(finding: RiskFinding) -> Passage:
    """One finding, one numbered passage.

    `label` carries the issue key and rule id together — the two things a
    reader needs to find this finding again without opening the ticket. `url`
    comes from the finding's own citation when the risk engine recorded one;
    a finding produced by a rule that does not (yet) attach a citation still
    gets a passage, just with nothing to click through to.
    """
    return Passage(
        label=f"{finding.issue_key} — {finding.rule_id}",
        url=finding.citations[0].url if finding.citations else "",
        text=finding.detail,
        found_by="risk_rule",
    )


async def generate_narrative(llm: LLM, report: DeliveryReport) -> GroundedAnswer:
    """The prose half of a `DeliveryReport`.

    Reuses `answer_question` rather than a bespoke narrative path: the
    grounding it enforces is exactly the property this needs, and a second,
    slightly different implementation is how "no claim without a source"
    quietly stops being one rule.
    """
    findings = [f for section in report.sections for f in section.findings]

    if not findings:
        # No model call. The rules ran to completion and found nothing —
        # stated as a positive result, not a caveat, for the same reason
        # `make_risk_tool`'s empty branch is: a hedge here reads as missing
        # data to whoever reads it next, when the truth is the opposite.
        scope = ", ".join(report.projects_covered) or "the synced projects"
        return GroundedAnswer(
            text=(
                f"No risk findings were detected across {scope} "
                f"({report.issues_seen} issues checked against every rule). "
                "This is a definitive result from a complete sync, not an "
                "empty report."
            ),
        )

    passages = [finding_passage(f) for f in findings]
    return await answer_question(
        llm,
        NARRATIVE_QUESTION,
        passages,
        system_prompt=NARRATIVE_SYSTEM_PROMPT,
        max_tokens=1500,
    )
