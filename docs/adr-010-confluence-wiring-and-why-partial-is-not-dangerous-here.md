# ADR-010: Confluence is wired in with a weaker gate than Jira, on purpose

**Status:** Accepted

## Context

`ConfluenceClient` was written and contract-tested (`tests/test_confluence_contract.py`)
before this ADR, but nothing in `app/` ever imported it. `/health`'s `docs_source`
was a hardcoded string, `"bundled_corpus"`, regardless of what was configured —
so a deployment reading a real Jira tenant was still answering documentation
questions from the fictional "Acme Claims Platform" sample corpus, silently.

The obvious fix is to mirror `load_live_findings` exactly: build a client from
config, load everything, replace the corpus. The part worth writing down is
where it should NOT be a mirror.

## Decision

**Config is the same shape as Jira's, and independently scoped.** `CONFLUENCE_SPACE_KEYS`
plus the same shared `ATLASSIAN_*` credentials gate `has_confluence`, exactly
like `JIRA_PROJECT_KEYS` gates `has_jira` — and the two are deliberately
independent properties. A tenant configured to read Jira must not start
reading Confluence just because the same credentials would work there;
enabling a whole second data source should never be a side effect of enabling
the first one.

**Total failure is never mixed with the bundled corpus.** If every configured
space is unreadable, `load_live_docs` indexes zero chunks and reports
`docs_source: "confluence_unavailable"`. It does not fall back to the sample
pages. This is the same rule `load_live_findings` already follows for Jira,
applied to the other half of the data — and arguably more important here,
because a client reading their own real Jira findings next to invented
Confluence answers has no way to tell the difference, and the Jira half being
real is exactly what makes the whole page look trustworthy.

**Partial failure is NOT treated like Jira's partial failure.** This is the
part that is not a mirror. `run_docs_load` (`app/core/docs_runner.py`) returns
one `DocsLoad` type — not a `CompleteDocsLoad` / `PartialDocsLoad` split — and
a load where one space out of three failed still indexes the other two.

## Rationale

### Why risk findings need the all-or-nothing gate and documentation does not

ADR-004's gate exists because a partial risk sync is dangerous in a specific,
narrow way: it *reads as complete* and *understates risk*. "INS-12 is not
blocked" from a sync that never read the project holding the blocker is a
false negative — confidently wrong, and the client's stated failure mode is
exactly that: telling the board you're on track when you aren't.

A partial documentation corpus does not produce a false confident claim. It
produces more refusals. `/ask` already has a safe, tested, deliberate answer
for "the context doesn't cover this" — it declines (ADR-006) — and that
answer is correct whether the gap is "this was never written down" or "the
space holding it failed to load." Refusing when unsure is the system working
as designed, not a symptom to route around by discarding an otherwise-good
partial corpus.

Put differently: Jira's gate exists to prevent a *false negative* dressed as
completeness. Documentation's failure mode, absent the gate, is a *false
refusal* — safe by construction. Applying the same all-or-nothing rule to
both would trade a real, if incomplete, answer for a guaranteed decline, for
no safety gained.

### Why space keys are not folded into `JIRA_PROJECT_KEYS`

Confluence spaces and Jira projects are different resources on the same
tenant, selected independently, and one client's answer to "should the agent
read our architecture wiki" is unrelated to "should it read our sprint
board." Reusing one setting for both would make disabling one impossible
without disabling the other.

### Why labels are not uppercased like project and space keys

Jira project keys and Confluence space keys are both conventionally
uppercase (`INS`, `ARCH`) and case-insensitive on the API side, so `_keys()`
folds case defensively. Confluence labels are lowercase-with-hyphens by
convention (`architecture`, `compliance-signoff`) and the label filter is
case-sensitive — folding case there would silently turn a working filter into
one that matches nothing. `_labels()` is a separate, smaller parser rather
than a flag on `_keys()`, because "sometimes uppercase, sometimes not" is not
one function's job to decide correctly by guessing which caller it is.

## Consequences

- `Services` gained `docs_source` and `docs_message`, alongside the existing
  `findings_source`. They are independent on purpose: `docs_source` can read
  `"confluence_unavailable"` while `findings_source` reads `"jira"` and the
  risk panel is fully populated, and the UI must show both rather than one
  masking the other.
- `run_docs_load` is a small, separately testable function — mirroring
  `run_sync`'s shape of taking a client as a parameter rather than
  constructing one internally — specifically so the complete/partial/empty
  classification has direct tests, rather than being provable only by reading
  through a live tenant.
- The one thing this does not yet cover: staleness. Jira findings carry
  `synced_at` and a UI banner past a threshold; Confluence pages are read once
  at startup and never refreshed. Content changes far less often than sprint
  status, and `/sync`'s existing UI language ("Re-read Jira now") is specific
  to Jira — folding a documentation refresh into that button would be
  surprising behaviour behind unrelated words. Left for a deliberate follow-up
  rather than bolted on as a side effect of this change.
