import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  decide,
  generateReport,
  getHealth,
  getProposals,
  resync,
  streamAsk,
} from "./api.js";
import { cardClass, headingClass } from "./accents.js";
import {
  Answer,
  Audit,
  Evidence,
  Ledger,
  Proposals,
  Refusal,
  Summary,
} from "./components.jsx";

// The write example names a real at-risk issue, supplied by /health, because
// the issue keys differ between the sandbox tenant and a client's. Hardcoded to
// "INS-101" it was correct offline and wrong the moment JIRA_PROJECT_KEYS was
// set — the demo's most important button asking the agent to flag a ticket that
// does not exist. Omitted entirely when nothing is at risk, rather than shown
// with a key nobody can look up.
const examplesFor = (health) => [
  "Which controls are still outstanding before go-live?",
  "What is at risk right now?",
  ...(health?.top_risk_issue
    ? [`Flag ${health.top_risk_issue} as at risk`]
    : []),
  "Who approved the Acme security exception?",
];

/**
 * The ways this deployment is not what it appears to be.
 *
 * Returns nothing on the sandbox: there the issue keys are obviously invented,
 * so the caveats are noise. It is the half-live configuration that misleads —
 * real tickets in the risk panel alongside a sample Confluence corpus and
 * approvals that never leave the process.
 */
function caveats(health) {
  if (!health || health.findings_source !== "jira") return [];
  const out = [];
  if (health.docs_source === "bundled_corpus")
    out.push(
      "Documentation answers come from the bundled sample corpus, not your Confluence.",
    );
  if (health.writes_to_jira === false)
    out.push(
      "Approvals are a dry run — nothing is written back, and each decision says so.",
    );
  return out;
}

/**
 * How old the findings are, in words a person reads without doing arithmetic.
 *
 * Shown because this service reads Jira once and the answers are only as fresh
 * as that read. A confident "INS-12 is not blocked" from a snapshot taken
 * before the blocker was linked is the failure this surfaces — it happened
 * twice before the age was on screen.
 */
function freshness(seconds) {
  if (seconds < 90) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  return hours < 24 ? `${hours} hr ago` : `${Math.round(hours / 24)} days ago`;
}

// Past this, the picture is old enough that a reader should be told plainly
// rather than left to read a timestamp. Two minutes of demo editing is the
// case that matters.
const STALE_AFTER_SECONDS = 120;

/**
 * The clock notice stands alone and is not part of `caveats`.
 *
 * The others qualify how much of the system is live. This one says the findings
 * are dated in the future, which changes what the numbers on screen mean —
 * folding it into a list of caveats invites it to be skimmed with them.
 */
function clockNotice(health) {
  const days = health?.clock_offset_days || 0;
  if (!days) return null;
  return (
    `Risk rules are being evaluated as of ${days} days from now, because Jira ` +
    `will not backdate an issue and stale detection needs history a fresh ` +
    `sandbox does not have. Thresholds are unchanged; the clock moved.`
  );
}

const STAGES = {
  retrieving: "Searching Jira and Confluence",
  reasoning: "Deciding what to check",
  generating: "Drafting an answer from the sources",
  validating: "Checking every claim against its source",
};

// Named rather than shown raw, because the name is the point: a reader should
// be able to see that the risk claim came from the rules engine and the
// background came from Confluence.
const TOOLS = {
  search_documentation: "Searching Confluence documentation",
  get_risk_findings:
    "Reading the risk engine — computed from Jira, not inferred",
  propose_flag_at_risk: "Drafting a change for someone to approve",
};

// A plain banner, neutral wash — used for state that qualifies how much of
// the system is live, not state that changes what the numbers mean (those
// two get the amber treatment below, on purpose: see the CSS header note).
function Banner({ children }) {
  return (
    <div className="border-b border-rule bg-wash px-7 py-2.5 text-[13.5px] text-slate-600">
      {children}
    </div>
  );
}

// Amber, and only for the two banners that change what the numbers on screen
// mean rather than merely how much of the system is live: the demo clock and
// a stale sync. Amber is otherwise reserved for "a human must decide"
// (Waiting on you); these earn the same colour because a wrong read here is
// just as costly as an unapproved change.
function AmberBanner({ children }) {
  return (
    <div className="border-b border-rule border-l-4 border-l-amber-600 bg-amber-50 px-6 py-2.5 text-[13.5px] text-ink">
      {children}
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [actor, setActor] = useState("vp@acme.com");
  const [running, setRunning] = useState(false);

  const [steps, setSteps] = useState([]);
  const [passages, setPassages] = useState([]);
  const [writing, setWriting] = useState(false);
  const [answer, setAnswer] = useState(null);
  const [refusal, setRefusal] = useState(null);
  const [error, setError] = useState(null);
  const [meta, setMeta] = useState(null);
  const [lit, setLit] = useState(null);

  const [syncing, setSyncing] = useState(false);
  const [pending, setPending] = useState([]);
  const [audit, setAudit] = useState([]);
  const [busy, setBusy] = useState(null);
  const [health, setHealth] = useState(null);

  const [report, setReport] = useState(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [reportError, setReportError] = useState(null);

  // Aborts an in-flight request when a new question is asked. Without it the
  // old stream keeps writing into state and the two answers interleave.
  const abortRef = useRef(null);

  const refresh = useCallback(async () => {
    const data = await getProposals();
    setPending(data.pending);
    setAudit(data.audit);
  }, []);

  const refreshHealth = useCallback(
    () =>
      getHealth()
        .then(setHealth)
        .catch(() => {}),
    [],
  );

  useEffect(() => {
    refresh();
    refreshHealth();
    // Re-read the clock, not the data. Without this the age freezes at
    // whatever it was when the page loaded, which is worse than no age at all.
    const timer = setInterval(refreshHealth, 30000);
    return () => clearInterval(timer);
  }, [refresh, refreshHealth]);

  async function onResync() {
    setSyncing(true);
    try {
      await resync();
      setError(null);
    } catch (e) {
      setError({ message: e.message });
    } finally {
      setSyncing(false);
      refreshHealth();
    }
  }

  async function ask(text) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setSteps([]);
    setPassages([]);
    setWriting(false);
    setAnswer(null);
    setRefusal(null);
    setError(null);
    setMeta(null);
    setLit(null);

    try {
      await streamAsk(
        text,
        (name, data) => {
          if (name === "stage") {
            if (data.writes_offered) setWriting(true);
            // Consecutive duplicates are collapsed: the graph returns to
            // "reasoning" after every tool round, and four identical lines
            // read as a stutter rather than as progress.
            setSteps((s) =>
              s[s.length - 1]?.label === (STAGES[data.stage] || data.stage)
                ? s
                : [
                    ...s,
                    { kind: "stage", label: STAGES[data.stage] || data.stage },
                  ],
            );
          } else if (name === "tool") {
            setSteps((s) => [
              ...s,
              {
                kind: "tool",
                label: TOOLS[data.name] || `Calling ${data.name}`,
              },
            ]);
          } else if (name === "passages") {
            // Sent twice: retrieval first, then what the answer's citations
            // actually index into. Replacing is the whole point — after a tool
            // round they are different lists, and holding the first would light
            // the wrong source when a reader clicks [1].
            setPassages(data.passages);
          } else if (name === "proposal") {
            // Straight to the pending list, while the agent is still writing
            // its answer. Waiting for "done" would have the change appear after
            // the sentence describing it.
            refresh();
          } else if (name === "answer") setAnswer(data);
          else if (name === "refusal") setRefusal(data);
          else if (name === "error") setError(data);
          else if (name === "done") setMeta(data);
        },
        controller.signal,
      );
    } catch (e) {
      if (e.name !== "AbortError") setError({ message: e.message });
    } finally {
      setRunning(false);
      refresh();
    }
  }

  async function onGenerateReport() {
    setGeneratingReport(true);
    try {
      setReport(await generateReport());
      setReportError(null);
    } catch (e) {
      setReportError(e.message);
    } finally {
      setGeneratingReport(false);
    }
  }

  async function onDecide(id, verb) {
    if (!actor.trim()) {
      setError({
        message:
          "Enter your email. Every decision is recorded against a person.",
      });
      return;
    }
    setBusy(id);
    try {
      await decide(id, verb, actor.trim());
      setError(null);
    } catch (e) {
      setError({ message: e.message });
    } finally {
      setBusy(null);
      refresh();
    }
  }

  return (
    <>
      <header className="flex items-baseline gap-3.5 border-b border-rule px-7 py-4">
        <h1 className="m-0 text-base font-semibold tracking-tight">
          Delivery Copilot
        </h1>
        <span className="ml-auto text-[13px] text-slate-600">
          Approving as{" "}
          <input
            value={actor}
            onChange={(e) => setActor(e.target.value)}
            aria-label="Your email"
            className="w-[200px] rounded border border-rule bg-white px-2 py-1 font-sans text-[13px] text-ink"
          />
        </span>
      </header>

      {health && health.can_answer === false && (
        <Banner>
          Questions are unavailable: {health.missing_config.join(", ")} is not
          configured. Approvals still work.
        </Banner>
      )}

      {/* A partial sync is not a smaller report, it is no report — the risk
          engine returns nothing rather than a fraction that reads as the whole
          picture. Saying so here is what stops silence being read as calm. */}
      {health && health.sync_message && (
        <Banner>
          {health.sync_message} Until it completes, no risk findings are served.
        </Banner>
      )}

      {/* What is real and what is not, in one place.
          Only shown once a real tenant is configured — on the sandbox the issue
          keys give it away, but pointed at a client's Jira the mixed state is
          invisible and reads as a fully live system. */}
      {caveats(health).length > 0 && (
        <Banner>Reading your live Jira. {caveats(health).join(" ")}</Banner>
      )}

      {clockNotice(health) && <AmberBanner>{clockNotice(health)}</AmberBanner>}

      {health && health.sync_age_seconds >= STALE_AFTER_SECONDS && (
        <AmberBanner>
          These findings were read from Jira{" "}
          {freshness(health.sync_age_seconds)}. Anything changed since is not
          reflected — including whether an issue is blocked.{" "}
          <button
            className="border-0 bg-transparent p-0 text-amber-900 underline underline-offset-2 disabled:text-slate-500 disabled:no-underline"
            disabled={syncing}
            onClick={onResync}
          >
            {syncing ? "Re-reading…" : "Re-read Jira now"}
          </button>
        </AmberBanner>
      )}

      <div className="mx-auto max-w-[1240px] px-7 pt-6">
        <section className={cardClass("violet")}>
          <h2 className={headingClass("violet")}>Weekly Summary</h2>
          <Summary
            report={report}
            onGenerate={onGenerateReport}
            generating={generatingReport}
            error={reportError}
          />
        </section>
      </div>

      <main className="mx-auto grid max-w-[1240px] grid-cols-1 gap-7 p-7 md:grid-cols-[minmax(0,1fr)_360px]">
        <section className={cardClass("indigo")}>
          <h2 className={headingClass("indigo")}>Ask a question</h2>

          <form
            className="mb-6 flex gap-2.5"
            onSubmit={(e) => {
              e.preventDefault();
              if (question.trim().length >= 3) ask(question.trim());
            }}
          >
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about delivery risk, blockers or plans"
              minLength={3}
              required
              className="flex-1 rounded border border-rule bg-white px-3.5 py-2.5 font-sans text-[15px] text-ink"
            />
            <button
              disabled={running}
              className="rounded border border-indigo-700 bg-indigo-700 px-4 py-2.5 font-sans text-[15px] text-white hover:bg-indigo-800 disabled:border-slate-300 disabled:bg-slate-300 disabled:text-slate-600"
            >
              {running ? "Asking" : "Ask"}
            </button>
          </form>

          {health && (
            <div className="-mt-4 mb-5 text-[12.5px] text-slate-600">
              Jira read {freshness(health.sync_age_seconds)}
              {health.risk_findings !== null &&
                ` · ${health.risk_findings} findings`}
              {" · "}
              <button
                className="border-0 bg-transparent p-0 text-indigo-700 underline underline-offset-2 disabled:text-slate-500 disabled:no-underline"
                disabled={syncing}
                onClick={onResync}
              >
                {syncing ? "re-reading…" : "re-read"}
              </button>
            </div>
          )}

          <div className="mb-6 flex flex-wrap gap-2">
            {examplesFor(health).map((text) => (
              <button
                key={text}
                type="button"
                onClick={() => {
                  setQuestion(text);
                  ask(text);
                }}
                className="rounded border border-indigo-200 bg-transparent px-2.5 py-1.5 text-[13px] text-indigo-800 hover:bg-indigo-50"
              >
                {text}
              </button>
            ))}
          </div>

          <Ledger steps={steps} running={running} />

          {writing && !answer && !refusal && (
            <p className="mb-5 max-w-[62ch] border-l-2 border-amber-600 pl-3.5 text-[13.5px] leading-relaxed text-slate-600">
              This reads as a request to change something. Anything the agent
              proposes appears under “Waiting on you” and is not written to Jira
              until you approve it.
            </p>
          )}

          {answer && <Answer text={answer.text} onCite={setLit} />}
          {refusal && <Refusal {...refusal} />}
          {error && <Refusal reason="" text={error.message} />}

          {meta && (
            <div className="mt-5 border-t border-rule pt-3 text-[12.5px] text-slate-600">
              {meta.ms} ms · ${meta.cost_usd.toFixed(4)} · {meta.input_tokens}{" "}
              tokens in, {meta.output_tokens} out
            </div>
          )}
        </section>

        <aside className="space-y-7">
          <section className={cardClass("teal")}>
            <h2 className={headingClass("teal")}>Evidence</h2>
            <Evidence passages={passages} lit={lit} />
          </section>

          <section className={cardClass("amber")}>
            <h2 className={headingClass("amber")}>Waiting on you</h2>
            <Proposals pending={pending} onDecide={onDecide} busy={busy} />
          </section>

          <section className={cardClass("slate")}>
            <h2 className={headingClass("slate")}>Decisions</h2>
            <Audit entries={audit} />
          </section>
        </aside>
      </main>
    </>
  );
}
