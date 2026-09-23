import React, { useState } from "react";

import { ACCENTS } from "./accents.js";
import { Markdown } from "./markdown.jsx";

/**
 * The ledger is the honesty of the thing. Each line is a step that actually
 * happened, tool calls included, so a reader can see that "INS-101 is stale"
 * came out of the risk engine rather than out of the model.
 */
export function Ledger({ steps, running }) {
  if (!steps.length) return null;
  return (
    <ul className="mb-6 space-y-1 text-[13px] text-slate-600">
      {steps.map((step, i) => {
        const active = running && i === steps.length - 1;
        return (
          <li key={i} className={`flex items-start gap-2 ${active || step.kind === "tool" ? "text-ink" : ""}`}>
            <span className={active ? "text-indigo-600" : step.kind === "tool" ? "text-indigo-500" : "text-slate-400"}>
              {active ? "→" : step.kind === "tool" ? "⌁" : "·"}
            </span>
            <span>{step.label}</span>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * The answer, rendered as markdown, with every citation a button that lights
 * its source in the evidence rail.
 *
 * `Markdown` emits React elements and cannot emit HTML — see markdown.jsx. The
 * text comes from a model reading pages anyone at the client can edit, and
 * `dangerouslySetInnerHTML` on that is how a prompt injection becomes a script
 * tag.
 */
export function Answer({ text, onCite }) {
  return (
    <div className="max-w-[68ch] font-serif text-[17px] leading-[1.62] text-ink">
      <Markdown text={text} onCite={onCite} />
    </div>
  );
}

// Only for a refusal `reason` this app actually produces. A plain fetch
// error (empty reason) gets no "why" line rather than a guessed one — a
// network failure is not "the sources do not cover this," and saying so
// would misdiagnose the problem for the person reading it.
const WHY = {
  failed_grounding_check: "A claim could not be traced to any source, so the answer was withheld.",
  model_declined: "The sources do not cover this.",
  model_declined_without_marker: "The sources do not cover this.",
  no_context_retrieved: "Nothing relevant was found to answer from.",
};

/**
 * Refusals are slate, never red, in every section that shows one. A refusal
 * is the system working correctly — colouring it as an error teaches leads
 * that the safe behaviour is a malfunction.
 */
export function Refusal({ reason, text }) {
  const why = WHY[reason];
  return (
    <div className="max-w-[62ch] border-l-2 border-rule pl-3.5 font-sans text-[15px] leading-relaxed text-slate-600">
      {text}
      {why && <span className="mt-1.5 block text-[13px]">{why}</span>}
    </div>
  );
}

export function Evidence({ passages, lit }) {
  if (!passages.length) {
    return (
      <p className="text-[13px] leading-normal text-slate-500">
        Sources appear here before the answer does, so you can see what the
        system read. Click a number in the answer to find its source.
      </p>
    );
  }
  return passages.map((p) => (
    <div
      key={p.index}
      id={`p${p.index}`}
      className={`mb-2 border-l-2 px-3 py-2.5 text-[13.5px] transition-colors motion-reduce:transition-none ${
        lit === p.index ? "border-teal-700 bg-teal-50" : "border-rule"
      }`}
    >
      <span className="mr-1.5 tabular-nums text-teal-700">{p.index}</span>
      <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-teal-800 hover:text-teal-950">
        {p.label}
      </a>
      <span className="mt-1 block text-xs text-slate-500">found by {p.found_by}</span>
    </div>
  ));
}

export function Proposals({ pending, onDecide, busy }) {
  if (!pending.length) {
    return (
      <p className="text-[13px] leading-normal text-slate-500">
        No changes are waiting. Nothing is written to Jira until someone here
        approves it.
      </p>
    );
  }
  return pending.map((p) => (
    <div key={p.id} className="mb-2.5 border border-rule border-l-[3px] border-l-amber-600 bg-white p-3.5">
      <div className="font-semibold text-ink">
        {p.issue_key} · {p.action}
      </div>
      <p className="my-1.5 text-[13px] text-slate-600">{p.rationale}</p>
      <div className="flex gap-2">
        <button
          className={`px-3 py-1.5 text-[13px] ${ACCENTS.amber.solid}`}
          disabled={busy === p.id}
          onClick={() => onDecide(p.id, "approve")}
        >
          Approve change
        </button>
        <button
          className="border border-rule bg-transparent px-3 py-1.5 text-[13px] text-ink hover:bg-slate-50 disabled:opacity-45"
          disabled={busy === p.id}
          onClick={() => onDecide(p.id, "reject")}
        >
          Reject
        </button>
      </div>
    </div>
  ));
}

/**
 * What actually happened to each decision.
 *
 * The outcome is read from `written_to_jira`, not from `applied`. `applied`
 * only means the configured writer returned without error, and the shipped
 * default is a dry run that holds no Jira client — so this line used to say
 * "written to Jira" about a tenant the process had never contacted. It is the
 * one claim in the UI that nothing was checking.
 *
 * The flag is per row rather than global, so rows recorded during a dry-run
 * pilot keep saying so after a real writer is deployed.
 */
function outcome(a) {
  if (a.decision === "rejected") return "";
  if (a.error) return `, not applied — ${a.error}`;
  if (a.written_to_jira === "True") return ", written to Jira";
  if (a.applied === "True") return ", recorded — dry run, nothing written to Jira";
  return ", approved but not yet applied";
}

export function Audit({ entries }) {
  if (!entries.length) {
    return <p className="text-[13px] leading-normal text-slate-500">Approvals and rejections are recorded here.</p>;
  }
  return (
    <div className="text-[12.5px] text-slate-600">
      {entries
        .slice()
        .reverse()
        .map((a, i) => (
          <div key={i} className="border-b border-rule py-1.5 last:border-0">
            {a.issue_key} {a.decision} by {a.actor}
            {outcome(a)}
          </div>
        ))}
    </div>
  );
}

const LEVEL_LABEL = { high: "High", medium: "Medium", low: "Low" };

/**
 * The weekly narrative: rules pick this week's findings (`/report`'s
 * `sections`), the model writes them up, every claim cites the finding
 * behind it.
 *
 * Its own citation rail, separate from the ask flow's Evidence panel — the
 * numbers here index into this report's findings, not whatever `/ask` last
 * retrieved, and reusing one `lit` state for both would light the wrong
 * passage in one panel every time the other one was clicked.
 */
export function Summary({ report, onGenerate, generating, error }) {
  const [lit, setLit] = useState(null);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-4">
        <span className="text-[13px] text-slate-600">
          {report ? (
            <>
              {report.issues_seen} issues checked across{" "}
              {report.projects_covered.join(", ") || "no projects"}
              {report.sections.map((s) => (
                <span key={s.level}>
                  {" "}
                  · {s.findings.length} {LEVEL_LABEL[s.level] || s.level}
                </span>
              ))}
            </>
          ) : (
            "Rules pick this week's findings; the model writes them up, one citation per claim."
          )}
        </span>
        <button className={`px-4 py-2 text-[13px] ${ACCENTS.violet.solid}`} disabled={generating} onClick={onGenerate}>
          {generating ? "Generating…" : report ? "Regenerate" : "Generate weekly summary"}
        </button>
      </div>

      {error && <Refusal reason="" text={error} />}

      {report && report.narrative_answered && (
        <div className="mb-4">
          <Answer text={report.narrative} onCite={setLit} />
        </div>
      )}

      {report && !report.narrative_answered && (
        <div className="mb-4">
          <Refusal reason={report.refusal_reason} text={report.narrative} />
        </div>
      )}

      {report && report.sources.length > 0 && (
        <div className="mb-4">
          {report.sources.map((s) => (
            <div
              key={s.index}
              id={`s${s.index}`}
              className={`mb-2 border-l-2 px-3 py-2.5 text-[13.5px] transition-colors motion-reduce:transition-none ${
                lit === s.index ? "border-violet-700 bg-violet-50" : "border-rule"
              }`}
            >
              <span className="mr-1.5 tabular-nums text-violet-700">{s.index}</span>
              {s.url ? (
                <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-violet-800 hover:text-violet-950">
                  {s.label}
                </a>
              ) : (
                <span className="text-slate-600">{s.label}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {report && !report.narrative_answered && (
        <ul className="list-disc space-y-1.5 pl-5 text-[13.5px] text-slate-600">
          {report.sections.flatMap((s) =>
            s.findings.map((f) => (
              <li key={`${f.issue_key}-${f.rule_id}`}>
                <strong className="text-ink">{f.issue_key}</strong> ({LEVEL_LABEL[s.level] || s.level}, {f.rule_id}):{" "}
                {f.detail}
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
