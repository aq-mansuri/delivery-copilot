"""The approval gate.

The only code in this project that can write to Jira, and it refuses to do so
without a named human and a recorded decision.

Design notes, all of them client-driven rather than aesthetic:

**Approval is a distinct type, not a flag.** Same pattern as ADR-004's
CompleteSync: `ApprovedAction` is what `apply_action` accepts, so an unapproved
proposal cannot reach the writer through any code path, including ones written
later by someone who has not read this file.

**The audit record is written before the API call, not after.** If the write
succeeds and the process dies before logging, the regulator sees a change with
no approval record — the worst possible failure for a regulated insurer.
Recording first can leave an approved-but-unapplied entry, which is merely
untidy and is visible in the log.

**Rejections are recorded too.** "We considered flagging this and a lead said no"
is itself evidence, and it is the record that protects the lead when someone asks
later why nothing was done.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from app.models.domain import ProposedAction

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApprovedAction:
    """A proposal a named human has approved.

    Only constructible through `ApprovalStore.approve`, which is what makes the
    type meaningful. Frozen, so an actor cannot be edited after the fact.
    """

    proposal: ProposedAction
    approved_by: str
    approved_at: datetime
    proposal_id: str


@dataclass(frozen=True)
class AuditEntry:
    proposal_id: str
    issue_key: str
    action_type: str
    decision: Decision
    actor: str
    at: datetime
    rationale: str
    note: str = ""
    applied: bool = False
    apply_error: str = ""
    # Whether a real Jira mutation happened. Distinct from `applied`, which
    # only means the configured writer returned without error — and the default
    # writer is a dry run that holds no Jira client. Conflating the two put
    # "written to Jira" on screen for a tenant that had never been contacted.
    #
    # Stored per entry rather than derived from config when the row is
    # rendered: an audit row is a durable claim about a past event, and a row
    # written under a dry run must still say so after a real writer is
    # injected. Otherwise a deployment setting silently rewrites history.
    written_to_jira: bool = False

    def to_row(self) -> dict[str, str]:
        return {
            "proposal_id": self.proposal_id,
            "issue_key": self.issue_key,
            "action": self.action_type,
            "decision": self.decision.value,
            "actor": self.actor,
            "at": self.at.isoformat(),
            "applied": str(self.applied),
            "written_to_jira": str(self.written_to_jira),
            "rationale": self.rationale,
            "note": self.note,
            "error": self.apply_error,
        }


class IssueWriter(Protocol):
    """Whatever can actually mutate Jira.

    A protocol so the gate can be tested without a tenant, and so the writer can
    be swapped for a dry-run implementation during a client pilot — which is
    usually how the first two weeks of a rollout run.

    `writes_to_jira` is what the audit trail reports. A dry-run writer during a
    pilot is normal and useful; a dry-run writer whose decisions are recorded as
    real changes is a falsified record, and the pilot is exactly when nobody is
    checking closely.
    """

    #: True only if this implementation actually mutates the tenant.
    writes_to_jira: bool

    async def add_labels(self, issue_key: str, labels: list[str]) -> None: ...

    async def add_comment(self, issue_key: str, body: str) -> None: ...


@dataclass
class ApprovalStore:
    """Pending proposals plus the decision log.

    In production this is Postgres. The interface is what matters: proposals go
    in, decisions come out, and every decision is durable before anything is
    applied.
    """

    pending: dict[str, ProposedAction] = field(default_factory=dict)
    audit: list[AuditEntry] = field(default_factory=list)
    _counter: int = 0

    def submit(self, proposal: ProposedAction) -> str:
        if not proposal.citations:
            # Refused at intake. A proposal with no evidence cannot be
            # meaningfully approved — the lead would have nothing to evaluate.
            raise ApprovalError(
                f"Proposal for {proposal.issue_key} has no citations. Every "
                "proposed change must carry the evidence it rests on."
            )
        self._counter += 1
        proposal_id = f"prop_{self._counter:04d}"
        self.pending[proposal_id] = proposal
        return proposal_id

    def approve(self, proposal_id: str, actor: str, note: str = "") -> ApprovedAction:
        if not actor or not actor.strip():
            raise ApprovalError(
                "Approval requires a named actor. An audit trail showing "
                "'approved by (unknown)' is not an audit trail."
            )

        proposal = self.pending.pop(proposal_id, None)
        if proposal is None:
            # Covers both "never existed" and "already decided". Re-approving a
            # decided proposal would double-apply the write.
            raise ApprovalError(
                f"{proposal_id} is not pending. It was never submitted, or has "
                "already been approved or rejected."
            )

        at = datetime.now(timezone.utc)
        self.audit.append(
            AuditEntry(
                proposal_id=proposal_id,
                issue_key=proposal.issue_key,
                action_type=proposal.action_type,
                decision=Decision.APPROVED,
                actor=actor,
                at=at,
                rationale=proposal.rationale,
                note=note,
            )
        )
        return ApprovedAction(
            proposal=proposal,
            approved_by=actor,
            approved_at=at,
            proposal_id=proposal_id,
        )

    def reject(self, proposal_id: str, actor: str, note: str = "") -> None:
        proposal = self.pending.pop(proposal_id, None)
        if proposal is None:
            raise ApprovalError(f"{proposal_id} is not pending.")

        self.audit.append(
            AuditEntry(
                proposal_id=proposal_id,
                issue_key=proposal.issue_key,
                action_type=proposal.action_type,
                decision=Decision.REJECTED,
                actor=actor,
                at=datetime.now(timezone.utc),
                rationale=proposal.rationale,
                note=note,
            )
        )

    def _mark_applied(
        self, proposal_id: str, error: str = "", *, written_to_jira: bool = False
    ) -> None:
        for index, entry in enumerate(self.audit):
            if entry.proposal_id == proposal_id:
                self.audit[index] = AuditEntry(
                    **{
                        **entry.__dict__,
                        "applied": not error,
                        "apply_error": error,
                        # A failed write is not a write, whatever the writer is.
                        "written_to_jira": bool(written_to_jira) and not error,
                    }
                )
                return


async def apply_action(
    approved: ApprovedAction, writer: IssueWriter, store: ApprovalStore
) -> None:
    """Perform an approved change.

    Takes `ApprovedAction` — not `ProposedAction`, not a dict. That signature is
    the gate. A proposal cannot reach this function without passing through
    `ApprovalStore.approve`, which requires a named actor and writes the audit
    record first.
    """
    if not isinstance(approved, ApprovedAction):
        raise ApprovalError(
            f"apply_action requires an ApprovedAction, got "
            f"{type(approved).__name__}. Unapproved changes must never reach "
            "Jira — the audit trail has to show a human made each change."
        )

    proposal = approved.proposal
    try:
        if proposal.action_type == "flag_at_risk":
            await writer.add_labels(
                proposal.issue_key, proposal.payload.get("labels_add", [])
            )
            await writer.add_comment(
                proposal.issue_key,
                f"Flagged as at-risk by {approved.approved_by} via Delivery "
                f"Copilot. Rationale: {proposal.rationale}",
            )
        elif proposal.action_type == "add_comment":
            await writer.add_comment(proposal.issue_key, proposal.payload["body"])
        else:
            raise ApprovalError(
                f"No writer implemented for action_type "
                f"{proposal.action_type!r}."
            )
    except Exception as exc:
        store._mark_applied(approved.proposal_id, error=str(exc))
        raise

    # `getattr` with a False default, not `writer.writes_to_jira`. A writer that
    # forgets to declare itself under-reports; the reverse invents a write into
    # a record a regulator reads. Only one of those errors is safe to make.
    store._mark_applied(
        approved.proposal_id,
        written_to_jira=bool(getattr(writer, "writes_to_jira", False)),
    )
