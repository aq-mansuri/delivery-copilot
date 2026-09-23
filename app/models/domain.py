"""Normalized domain models.

Deliberately decoupled from Atlassian's wire format. The integration layer maps
Jira's deeply-nested JSON into these; everything downstream (risk engine, RAG,
agent) depends only on this module. When Atlassian changes a payload shape, one
mapper changes and nothing else does.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class IssueStatusCategory(str, Enum):
    """Jira status *categories* are stable across projects; status names are not.

    Eight teams named their workflow states differently, so any rule written
    against a status name is a rule that breaks per-project. Rules key off the
    category; the raw name is kept for display only.
    """

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class BlockerCategory(str, Enum):
    """Derived from free-text comments by the LLM, never by keyword matching."""

    VENDOR = "vendor"
    LEGAL_COMPLIANCE = "legal_compliance"
    CROSS_TEAM = "cross_team"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


class ChangelogEntry(BaseModel):
    """A single field transition. This is the backbone of stale detection."""

    field: str
    from_value: str | None = None
    to_value: str | None = None
    at: datetime
    author_id: str | None = None


class Comment(BaseModel):
    id: str
    body: str
    author_id: str | None = None
    created_at: datetime


class IssueLink(BaseModel):
    """A link to another issue, with direction preserved.

    Jira's link model is the single easiest thing to invert. A link type has a
    name ("Blocks") plus an inward description ("is blocked by") and an outward
    one ("blocks"). Which side you're on determines the meaning, and the payload
    tells you by putting the other issue under `inwardIssue` or `outwardIssue`.

    Get this backwards and a genuinely blocked ticket reports as low-severity
    context — the exact false negative the client cannot afford. Hence: direction
    is stored explicitly, never re-derived downstream.
    """

    target_key: str
    target_summary: str | None = None
    target_status_category: IssueStatusCategory | None = None

    link_type: str  # normalized: "blocks" | "depends_on" | "relates_to" | ...
    # True when the *target* is the thing holding this issue up.
    target_blocks_this: bool = False

    def is_resolved(self) -> bool:
        """Unknown status counts as unresolved.

        Deliberate: recall over precision. A link we can't see the status of
        (permissions, deleted project) is more likely to be a real dependency
        than not, and a false positive is cheap.
        """
        return self.target_status_category is IssueStatusCategory.DONE


class Issue(BaseModel):
    key: str
    project_key: str
    summary: str
    status_name: str
    status_category: IssueStatusCategory
    assignee_id: str | None = None
    story_points: float | None = None
    sprint_name: str | None = None

    # The problem child: ~40% blank, and teams disagree on its semantics.
    # Modelled as optional + a confidence flag rather than silently defaulted.
    target_release: str | None = None

    created_at: datetime
    updated_at: datetime
    due_date: datetime | None = None

    comments: list[Comment] = Field(default_factory=list)
    changelog: list[ChangelogEntry] = Field(default_factory=list)
    links: list[IssueLink] = Field(default_factory=list)

    def last_activity_at(self) -> datetime:
        """Most recent *human* signal, not Jira's `updated` field.

        Jira bumps `updated` on automation, bulk edits, and field syncs. Using it
        makes dead tickets look alive, which is the exact failure Priya named.
        """
        candidates = [self.created_at]
        candidates += [c.created_at for c in self.comments]
        candidates += [e.at for e in self.changelog]
        return max(candidates)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Citation(BaseModel):
    """Every downstream claim carries one of these. No exceptions.

    Priya: "'The system says' is not an answer I can give the board."
    """

    source_type: str  # "issue" | "comment" | "confluence_page"
    source_id: str
    url: str
    excerpt: str | None = None


class RiskFinding(BaseModel):
    """Output of a deterministic rule. Never LLM-generated."""

    issue_key: str
    rule_id: str
    level: RiskLevel
    detail: str
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    citations: list[Citation] = Field(default_factory=list)


class ProposedAction(BaseModel):
    """A write that a human must approve before it touches Jira.

    Compliance requirement, not a UX preference: the audit trail has to show a
    human made each change.
    """

    issue_key: str
    action_type: str  # "flag_at_risk" | "add_comment" | "set_field"
    payload: dict
    rationale: str
    citations: list[Citation]
    approved_by: str | None = None
    approved_at: datetime | None = None


class PageSection(BaseModel):
    """A heading and the prose under it.

    Sections, not raw page text, are the retrieval unit. A 4,000-word
    architecture page embedded whole retrieves for everything and answers
    nothing; split at arbitrary character counts and you sever a sentence from
    the heading that gives it meaning. The document's own structure is the
    least-arbitrary boundary available.
    """

    heading: str | None
    level: int  # 0 = content before the first heading
    text: str

    def is_empty(self) -> bool:
        return not self.text.strip()


class Page(BaseModel):
    id: str
    space_key: str
    title: str
    url: str
    version: int
    updated_at: datetime
    labels: list[str] = Field(default_factory=list)
    sections: list[PageSection] = Field(default_factory=list)

    def full_text(self) -> str:
        parts: list[str] = []
        for section in self.sections:
            if section.heading:
                parts.append(section.heading)
            if section.text:
                parts.append(section.text)
        return "\n\n".join(parts)
