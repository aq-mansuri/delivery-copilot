"""Calibration set for the groundedness judge.

Claims with known verdicts, drawn from real live output and from the seed
corpus. These measure the judge itself. Without them a groundedness score is a
number produced by an unvalidated instrument.

The `overreach` cases matter most — they are the failure actually observed, and
a judge asked "is this supported?" about a plausible inference tends to say yes.

Three cases were relabelled after the first calibration run. They had been
marked `overreach` on the reasoning that they were "on the same subject". The
judge called them `unsupported`, and it was right: a passage recording that a
test was not scheduled is silent on WHY, so a causal claim introduces a subject
the passage never raises. The rubric was underspecified, not the judge.

That is the point of calibrating. It tested the taxonomy, not only the model.

Second round: two recommendation cases were still labelled `overreach` while the
sharpened rubric already placed recommendations under `unsupported`. The judge
followed the rubric; the fixtures had not been updated to match it. Exact
agreement rose 79% -> 89%, and both remaining disagreements were this same
fixture lag.

Keep rubric and labels in sync deliberately. A calibration set that contradicts
its own rubric measures the contradiction.

## Caveat on the 100% score

This set has been relabelled twice, both times to agree with the judge. Each
change was correct on the merits — the rubric was underspecified, then the
fixtures lagged it — but the consequence is that the set is now fitted to the
judge's reading.

So 100% proves the rubric is self-consistent and that a cheaper model applies it
the same way. It does NOT prove the judge handles cases nobody has inspected.

`HELD_OUT` below exists for that: cases written against the rubric without
checking what the judge says about them first. Score them separately and do not
relabel them to agree. When they disagree, decide on the merits and record the
decision — but a held-out set that gets relabelled is no longer held out.
"""

from app.agent.judge import CalibrationCase, Verdict

SLA_PASSAGE = (
    "Acme commit to a 5 business day turnaround on integration support "
    "requests. This commitment has not been met during the current "
    "integration; two schema confirmation requests remain unanswered beyond "
    "the agreed window. The agreement contains no service credit mechanism "
    "for integration support delays."
)

STALE_PASSAGE = (
    "INS-101 has sat untouched for three weeks. The team is awaiting schema "
    "confirmation from Acme and has had no reply to two follow-ups. The "
    "assignee has moved onto other work in the meantime."
)

CONTROLS_PASSAGE = (
    "Mutual TLS is specified but certificates have not been issued for the "
    "production gateway. Penetration testing of the callback endpoint has not "
    "been scheduled. No exceptions have been raised for this integration."
)

CASES = [
    # --- supported: restatement, paraphrase, arithmetic already in the text
    CalibrationCase(
        claim="Acme committed to a 5 business day turnaround on integration support requests.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.SUPPORTED,
        note="direct restatement",
    ),
    CalibrationCase(
        claim="Two schema confirmation requests are still unanswered past the agreed window.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.SUPPORTED,
        note="paraphrase",
    ),
    CalibrationCase(
        claim="INS-101 has not been worked on for three weeks.",
        passage_text=STALE_PASSAGE,
        expected=Verdict.SUPPORTED,
        note="paraphrase",
    ),
    CalibrationCase(
        claim="No exceptions have been raised for this integration.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.SUPPORTED,
    ),

    # --- overreach: on-topic, goes beyond the source
    CalibrationCase(
        claim="Acme is in breach of contract.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.OVERREACH,
        note="legal conclusion; source says a commitment was not met",
    ),
    CalibrationCase(
        claim="We should formally log the SLA breach to apply commercial pressure.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="recommendation; observed verbatim in live output. The passage "
        "states terms and advises nothing, so the claim adds a subject.",
    ),
    # Relabelled after calibration. The judge called these unsupported, and on
    # review it had the better reading: the passage is SILENT on motive,
    # outcome and cause, and silence about a subject is not weak support for it.
    # Overreach now means "same subject, stronger claim"; introducing a subject
    # the passage never raises is unsupported.
    CalibrationCase(
        claim="The vendor is deliberately stalling the integration.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="motive; passage says nothing about why",
    ),
    CalibrationCase(
        claim="INS-101 will miss the Q4 target.",
        passage_text=STALE_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="prediction; passage states no target and no forecast",
    ),
    CalibrationCase(
        claim="The team should escalate INS-101 to vendor management immediately.",
        passage_text=STALE_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="recommendation; passage records status, never advises action",
    ),
    CalibrationCase(
        claim="Penetration testing was skipped because the team was under schedule pressure.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="cause; passage records that it was not scheduled, never why",
    ),

    # Added after relabelling left only three true overreach cases. Each of
    # these is the sharpened definition: the passage DOES say this, the claim
    # says it harder. No new subject is introduced.
    CalibrationCase(
        claim="Acme has repeatedly failed to meet its support obligations.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.OVERREACH,
        note="'repeatedly failed' from two unanswered requests — same subject, stronger",
    ),
    CalibrationCase(
        claim="INS-101 has been abandoned.",
        passage_text=STALE_PASSAGE,
        expected=Verdict.OVERREACH,
        note="'abandoned' from 'untouched three weeks, assignee moved on'",
    ),
    CalibrationCase(
        claim="The integration is completely blocked on Acme.",
        passage_text=STALE_PASSAGE,
        expected=Verdict.OVERREACH,
        note="'completely' overstates a source describing one awaited confirmation",
    ),
    CalibrationCase(
        claim="Security readiness for this integration is seriously deficient.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.OVERREACH,
        note="judgement built on listed unsatisfied controls",
    ),

    # --- unsupported: the passage does not address it
    CalibrationCase(
        claim="The contract renews annually each July.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="true of a different contract in the corpus; not in this passage",
    ),
    CalibrationCase(
        claim="INS-102 is blocked by the platform team.",
        passage_text=STALE_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="about a different issue",
    ),
    CalibrationCase(
        claim="Compliance sign-off has already been granted.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="contradicts the passage",
    ),
    CalibrationCase(
        claim="The parental leave policy allows six months at full pay.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="entirely off topic",
    ),
]


# Written against the rubric, deliberately NOT checked against the judge before
# being committed. If these disagree, decide on the merits and record why —
# relabelling them to agree turns a held-out set back into a fitted one.
HELD_OUT = [
    CalibrationCase(
        claim="The claims integration cannot process live policyholder data yet.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.SUPPORTED,
        note="restatement of the controls gate",
    ),
    CalibrationCase(
        claim="Certificate issuance is the only thing standing between us and go-live.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.OVERREACH,
        note="'only thing' overstates a passage listing three unsatisfied controls",
    ),
    CalibrationCase(
        claim="The production gateway has no certificates issued.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.SUPPORTED,
    ),
    CalibrationCase(
        claim="Penetration testing will be scheduled once certificates arrive.",
        passage_text=CONTROLS_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="sequencing the passage never states",
    ),
    CalibrationCase(
        claim="Acme has been unresponsive for longer than the contract allows.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.SUPPORTED,
        note="paraphrase of 'unanswered beyond the agreed window'",
    ),
    CalibrationCase(
        claim="The absence of service credits leaves us without commercial recourse.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.OVERREACH,
        note="'without recourse' is a conclusion; source states only the absence",
    ),
    CalibrationCase(
        claim="Acme employs fewer support staff than its contract volume requires.",
        passage_text=SLA_PASSAGE,
        expected=Verdict.UNSUPPORTED,
        note="explanation the passage never offers",
    ),
    CalibrationCase(
        claim="The assignee is no longer actively working INS-101.",
        passage_text=STALE_PASSAGE,
        expected=Verdict.SUPPORTED,
    ),
]
