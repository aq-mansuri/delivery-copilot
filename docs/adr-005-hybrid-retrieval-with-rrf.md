# ADR-005: Hybrid retrieval fused with reciprocal rank

**Status:** Accepted

## Context

Roughly half the expected questions carry literal identifiers — ticket keys, vendor
names, exact technical terms ("mutual TLS"). Dense embeddings handle paraphrase well
and exact tokens badly: `INS-101` and `INS-107` are near-neighbours in vector space,
and a rare vendor name may barely move the embedding.

The complementary weakness also holds: BM25 fails on "why did we choose X" against a
page that never uses the word "choose".

## Decision

Run BM25 and dense retrieval independently over a candidate pool larger than the
final result size, then fuse by reciprocal rank:

    score(d) = Σ 1 / (k + rank_r(d))   over retrievers r,  k = 60

## Rejected alternative

Weighted sum of normalized scores. BM25 scores are unbounded and corpus-dependent;
cosine similarity is bounded and is not. Combining them requires normalizing two
distributions that shift whenever the corpus changes, and the weights need retuning
per tenant. RRF discards magnitudes, keeps ranks, and is scale-free — it survives a
corpus change without retuning, which matters when the same system has to work
across several clients without per-client babysitting.

## Consequences

- Two indexes to maintain and keep in sync on re-ingest.
- `k = 60` is a convention, not a derived value. It should be validated against the
  eval set rather than accepted on faith.
- Candidate pool must exceed final `top_k`; fusing two top-5 lists gives the fusion
  almost nothing to work with.
- Retrieval reason (keyword / semantic / both) is exposed per hit, so "why was this
  page cited" has an answer.
