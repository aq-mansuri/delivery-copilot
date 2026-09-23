# Three-minute demo

## What to cut, and why

The temptation is to show everything: seven scripts, nine ADRs, a 495-test
suite. Three minutes fits roughly four beats, so almost all of it goes.

**Cut:** the architecture diagram, the Jira integration, the LangGraph state
machine, the tracing, the container. All good work; none of it is a reason to
hire you. An interviewer assumes a competent engineer can wire up an API.

**Keep:** the discovery insight, the refusal, the approval gate, one number.
Those are the four things most candidates cannot show.

The whole demo is `--live`. Scripted mode makes a question the corpus cannot
answer look answered, and if anyone spots that, the credibility of everything
else goes with it.

---

## 0:00 — the problem, not the product (25s)

> "An insurer's engineering leads spend half of every Monday building a status
> deck, and the numbers are always slightly wrong. The ask was 'put AI on top of
> Jira'. Discovery found something more specific: the signal they actually want
> is a ticket that's been In Progress for three weeks with no activity. It looks
> healthy on the board. It's dead.
>
> That's a changelog query, not an AI problem. So the most valuable half of this
> system has no model in it."

Say nothing about the stack yet. The architectural decision *is* the pitch.

## 0:25 — a grounded answer (45s)

Ask: **"Which controls are still outstanding before go-live?"**

Let the evidence rail populate before the answer arrives. Point at it once:

> "Sources appear before the answer does. Everything it says is traceable —"

Click a citation. The passage lights.

> "— and you can check any claim in one click. The client's words were: 'the
> system says' is not an answer I can give the board."

## 1:10 — the refusal (40s)

Ask: **"Who approved the Acme security exception?"**

> "There is no such exception. The question presupposes one."

Read the response aloud — the model names the false premise and declines.

> "Retrieval can't catch this. It returns highly relevant pages, because the
> question is about real topics. It just asks for a fact that isn't there.
> Refusal has to happen at generation time, and it's enforced in code: any claim
> that can't be traced gets the whole answer rejected, not flagged."

This is the beat that lands. Everyone has seen a RAG demo answer a question.
Almost nobody has seen one decline for a stated reason.

## 1:50 — the approval gate (40s)

Type: **"Flag INS-101 as at risk"**

Do not pre-seed this. Typing it is the beat — the same box, the same endpoint,
and the ledger shows the agent reading the risk engine before it drafts
anything. A proposal appears in the right rail while the answer is still being
written.

> "The agent can propose flagging an issue at-risk. It cannot write to Jira. Not
> policy — the tool registry holds no write client, so there's nothing it could
> write with."

Approve it. Show the audit row, then the ticket in Jira with the label and the
attributed comment on it.

Decide beforehand which mode you are demoing in and say which:
`JIRA_ALLOW_WRITES` on writes for real and the row reads "written to Jira"; off
is a dry run and the row says that instead. Both are honest; the one that is not
is a dry run whose row claims a write.

> "Recorded against a named person, written before the API call. If the write
> succeeds and logging dies, a regulator sees a change with no approval record.
> That failure is worse than the reverse.
>
> And it says 'dry run' because it is one. `applied` means the writer returned;
> it never meant Jira changed. Those were the same flag until I checked."

If anyone asks why the row does not say "written to Jira": no real writer is
implemented. Say that plainly rather than reaching for the config.

## 2:30 — one number (30s)

> "Every claim the system makes is graded by a second model against its cited
> source: 85% supported, 9% overreach, 2% unsupported, 100% refusal on
> unanswerable questions.
>
> And the grader is itself calibrated against held-out labelled cases — 88%
> detection, no dangerous misses. Without that, a groundedness score is just a
> number a model produced about itself."

Stop there.

---

## If they ask for more

Have these ready, one sentence each, and let them pull:

- **"How do you know retrieval is any good?"** → `run_eval.py`, recall by
  question category, cross-source was the weak one at 50%.
- **"What broke?"** → The seeder created Jira links backwards while the reader
  parsed them correctly. Two bugs that cancelled out invisibly, found by diffing
  one real payload against my fixtures.
- **"What doesn't work?"** → Approvals don't survive a restart, `X-Actor` is
  self-asserted so the audit trail records a claim not an identity, findings
  refresh only when asked (`POST /sync`; the UI shows the age), and temporal
  questions are structurally unsupported.
- **"What did you get wrong?"** → The strongest one to have ready. The approval
  gate was tested, correct, and reachable only from a script — the UI was the
  retrieval half of the system, so "flag INS-101" came back as an answer *about*
  INS-101. Wiring it up surfaced two live-only bugs: the retrieve node never
  actually sent its passages to the model, and citation numbers shown to the
  user disagreed with the answer's after any tool call. ADR-009.

That last answer matters more than it looks. A candidate who can name the gaps
in their own system is one a client will believe when they say something works.

## Recording it

Screen only, no webcam. Cursor still except when clicking. No music. Cut every
loading pause over two seconds — a six-second model call is honest, six seconds
of silence on a recording is not.

Upload unlisted, link it in the README, and keep it under three minutes even if
it means dropping a beat.
