# ADR 0002 — hybrid retrieval via reciprocal rank fusion

date: 2026-10-16
status: accepted

## context

Dense retrieval (hashing TF-IDF, ADR 0001) misses keyword-exact queries;
BM25 misses paraphrases. Neither alone wins on the 20-question eval set,
and a naive score blend can't combine them — cosine similarities and BM25
scores live on different scales and a bad blend inherits the worst of both.

## decision

Keep three retrievers behind one interface in `retriever.py`
(`index` / `embed` / `search`) — dense, BM25, and hybrid — and fuse with
reciprocal rank fusion, not score averaging. RRF only needs ranks, so the
scale mismatch never enters the picture.

- BM25 scores get squashed to [0, 1] with s/(1+s), not max-normalised.
  A nonsense query matching only stopwords gives tiny raw scores
  everywhere; max-normalising would inflate the best tiny score to 1.0 and
  sail past the `min_score` gate, killing the "I don't know" refusal the
  evals assert on. The saturating squash keeps weak matches weak.
- `hybrid_alpha` (default 0.5) weighs the dense list's ranks vs the BM25
  list's; `hybrid_depth` (default 20) sets how deep each backend reads
  before fusion — RRF needs depth to see where the backends agree.
- Fusion stays in `retriever.py`; `pipeline.py` picks a retriever by the
  `retriever:` config key and never sees the fusion mechanics.

## consequences

- hybrid hits 20/20 recall@3 on the eval set, matching BM25's top end while
  catching the paraphrases dense-only wins on.
- alpha is a real knob, not decoration: 1.0 = dense only, 0.0 = BM25 only,
  so either backend can be reverted without code.
- Cost: every query pays for both retrievals plus the fusion — roughly 2x
  the retrieve-stage latency. The latency probe (evals/latency_probe.py)
  shows retrieval is still the cheapest stage, so this is affordable.
