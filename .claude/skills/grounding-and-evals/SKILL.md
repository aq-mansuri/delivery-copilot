---
name: grounding-and-evals
description: Use when building, tuning, or debugging retrieval quality, RAG citations, refusal behaviour, or evaluation harnesses in this project — including changing chunking, retrieval scoring, the grounding checker, prompts that affect citations, or any eval metric. Covers why score floors cannot produce refusal, why grounding is enforced per paragraph, how to read run_eval.py and diagnose_grounding.py output, and the measure-before-changing discipline. Trigger on recall, MRR, eval, groundedness, citation, refusal, hallucination, chunking, BM25, RRF, or retrieval tuning.
---

# Grounding and evaluation

## Measure before changing. Always.

Every tuning change in this project is made against a number. Two design
decisions were reversed by measuring first, and both would have shipped wrong:

**A retrieval score floor cannot produce refusal** (ADR-006). Measured: answerable
questions scored BM25 1.87–7.25, unanswerable ones 2.54–5.21. The ranges overlap,
because negatives ask for absent *facts* about present *topics*. Relevance and
answerability are different properties. Refusal is a generation-time obligation.

**Per-sentence grounding produced a 100% refusal rate** (ADR-008). The cause was
not strictness but an off-by-one: models write the citation after the full stop
(`...in parallel. [3]`), so a naive sentence split credited it to the next
sentence. Every rejection traced to that.

A 100% refusal rate is not safety. It is a false negative wearing a safety
jacket, and its failure mode is that the client stops opening the tool.

## The tools

    python scripts/run_eval.py                  # retrieval: recall@k, MRR, by category
    python scripts/run_eval.py --decompose      # with query decomposition
    python scripts/diagnose_grounding.py --live # per-paragraph verdicts on real output

`docs/baseline.json` predates the fixes. Never regenerate it.

Report the *category breakdown*, not just the aggregate. 80% overall can be 100%
on prose and 20% on identifiers — a specific fixable problem the single number
hides.

## Known metric weaknesses — do not quote these numbers without the caveat

- **Cross-source recall is any-of, not all-of.** A question needing two pages
  scores a hit when one is retrieved. The reported 100% is actually "found one of
  two."
- **Citation correctness is unmeasured.** Recall says the right passage was
  retrieved, not that the answer cited the right one.
- **Claim-level grounding is unmeasured.** This is the live gap ADR-008 opened
  knowingly — see below.
- **A passing case may pass for the wrong reason.** Temporal questions score 100%
  because only one page is about release plans, not because retrieval can order
  by time.

## The grounding checker

Enforced per **paragraph**: every paragraph making claims must carry at least one
resolvable citation. `per_sentence=True` exists for evals.

Handled deliberately: trailing citations pulled back onto their own sentence;
markdown headings, table rows and rules are structure not claims; a short
uncited lead-in followed by cited content is framing.

Two rejections, distinct because they mean different things:
- **out-of-range citation** — the model invented a source. More alarming.
- **uncited claim** — an assertion with no traceable origin. More common.

Refusals are recorded separately as `model_declined` (marker emitted) and
`model_declined_without_marker` (declined in prose). Keeping them distinct is
deliberate: a fallback that hides its own use never gets fixed, and a correct
decline miscounted as a grounding failure corrupts the refusal metric.

## The open gap

Paragraph-level citation means **traceable, not supported**. Observed live: the
model produced recommendations ("formally log the SLA breach") citing a passage
that states SLA terms but recommends nothing, and asserted "a clear contractual
breach" where the source says a commitment was not met. A legal conclusion the
sources do not draw.

Claim-level groundedness — does each assertion follow from its cited passage,
judged against a rubric — is the next thing to build.

## When adding eval cases

- Ground truth is **page ids, not chunk ids**. Chunk boundaries move whenever the
  chunker changes, and an eval set that breaks on every tweak stops being run.
- Semantic cases must not share vocabulary with their target page, or they are
  keyword cases wearing a semantic label.
- Negative cases need **topically adjacent bait**. A negative with no nearby
  content passes trivially and proves nothing.
- Include cases that are **expected to fail**. A named gap beats one discovered
  in front of a client.
- If you cannot write a specific page id, the case cannot be scored and does not
  belong in the file yet.

## Not an eval question

Some questions are answered by the risk engine or a JQL query, not by search:
"which issues are blocked", "what is at risk", "which issue was updated most
recently". Scoring these as retrieval measures the wrong component and inflates
recall with questions retrieval was never asked. See `_engine_cases` in
`docs/eval_set.json`.


## Refusal has two failure modes, not one

Most of this file is about the system answering when it should decline. The
inverse is harder to spot, because a wrong refusal and a right one are the same
sentence on screen.

**A reported absence is evidence.** When a deterministic tool says "the rules
checked INS-12 and found nothing", that is the answer to "is INS-12 blocked?",
and the answer is no. The answering model will classify it as missing context
and decline unless the tool's wording makes the assertion unmistakable — state
the negative in the question's own words, name what was checked so the scope of
the "no" is explicit, and say the result is definitive.

`refusal_rate` on its own cannot catch this: a system that declines everything
scores 100% on the unanswerable set. Read it alongside the answer rate on
questions that *do* have an answer, and keep at least one eval case whose
correct answer is a confident negative.

**Fix it in the tool, not in `answering.SYSTEM_PROMPT`.** The prompt governs
refusal everywhere; a tool only speaks when its rules actually ran. Changing the
prompt to rescue one case puts the baselined refusal behaviour at risk across
every question.
