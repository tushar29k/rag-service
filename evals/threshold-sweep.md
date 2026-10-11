# Threshold sweep — min_score precision/recall trade-off

Date: 2026-10-11. Corpus: data/sample_docs.json (4 fixture policy docs), golden set: evals/golden.jsonl (18 answerable + 2 unanswerable), top_k=3, hashing embeddings + mock generator, rerank/cache/latency-logging off. Grid: 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00.

A correct answer = expected_doc in the top-3 citations. False positive = answered an unanswerable question or answered off the wrong doc. False negative = refused an answerable question.

## dense

| min_score | precision | recall | f1 | refused | evidence |
|---|---|---|---|---|---|
| 0.00 | 0.900 | 1.000 | 0.947 | 0/20 | 18/18 |
| 0.05 | 0.900 | 1.000 | 0.947 | 0/20 | 18/18 |
| 0.10 | 0.900 | 1.000 | 0.947 | 0/20 | 18/18 |
| 0.15 | 1.000 | 0.944 | 0.971 | 3/20 | 17/18 |
| 0.20 | 1.000 | 0.722 | 0.839 | 7/20 | 13/18 |
| 0.25 | 1.000 | 0.333 | 0.500 | 14/20 | 6/18 |
| 0.30 | 1.000 | 0.167 | 0.286 | 17/20 | 3/18 |
| 0.35 | 1.000 | 0.056 | 0.105 | 19/20 | 1/18 |
| 0.40 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 0.50 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 0.60 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 0.70 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 0.80 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 0.90 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 1.00 | – | 0.000 | 0.000 | 20/20 | 0/18 |

**Best F1: 0.971 at min_score=0.15** (precision 1.000, recall 0.944, refused 3/20).

## bm25

| min_score | precision | recall | f1 | refused | evidence |
|---|---|---|---|---|---|
| 0.00 | 0.900 | 1.000 | 0.947 | 0/20 | 18/18 |
| 0.05 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.10 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.15 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.20 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.25 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.30 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.35 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.40 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.50 | 1.000 | 1.000 | 1.000 | 2/20 | 18/18 |
| 0.60 | 1.000 | 1.000 | 1.000 | 2/20 | 18/18 |
| 0.70 | 1.000 | 0.778 | 0.875 | 6/20 | 14/18 |
| 0.80 | 1.000 | 0.444 | 0.615 | 12/20 | 8/18 |
| 0.90 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 1.00 | – | 0.000 | 0.000 | 20/20 | 0/18 |

**Best F1: 1.000 at min_score=0.60** (precision 1.000, recall 1.000, refused 2/20).

## hybrid

| min_score | precision | recall | f1 | refused | evidence |
|---|---|---|---|---|---|
| 0.00 | 0.900 | 1.000 | 0.947 | 0/20 | 18/18 |
| 0.05 | 0.900 | 1.000 | 0.947 | 0/20 | 18/18 |
| 0.10 | 0.900 | 1.000 | 0.947 | 0/20 | 18/18 |
| 0.15 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.20 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.25 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.30 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.35 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.40 | 0.947 | 1.000 | 0.973 | 1/20 | 18/18 |
| 0.50 | 1.000 | 1.000 | 1.000 | 2/20 | 18/18 |
| 0.60 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 0.70 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 0.80 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 0.90 | – | 0.000 | 0.000 | 20/20 | 0/18 |
| 1.00 | – | 0.000 | 0.000 | 20/20 | 0/18 |

**Best F1: 1.000 at min_score=0.50** (precision 1.000, recall 1.000, refused 2/20).

## Findings

- dense: best F1 0.971 at 0.15; at 0.00 precision 0.900 / recall 1.000 (gate off: answers everything, including the 2 unanswerable questions).
- bm25: best F1 1.000 at 0.60; at 0.00 precision 0.900 / recall 1.000 (gate off: answers everything, including the 2 unanswerable questions).
- hybrid: best F1 1.000 at 0.50; at 0.00 precision 0.900 / recall 1.000 (gate off: answers everything, including the 2 unanswerable questions).

## Recommendation

min_score is backend-specific — one number does not fit all. Tune per retriever backend on your own corpus; this sweep is the procedure. Config default stays 0.15 until the neural-embedding swap lands, then re-run this script and take the new best-F1 threshold (ties go to the higher, more conservative threshold).

## Caveats

- 20-question fixture set: directionally right, not statistically powerful. Only 2 unanswerable questions drive all precision differences — add more adversarial/near-miss questions before trusting small precision gaps.
- Hybrid (RRF) scores saturate near 1.0, so the final min_score gate barely bites it — its refusals come from the per-sub-list pre-gate inside HybridRetriever, not the pipeline filter.
- Hashing TF-IDF scores are bag-of-words; the neural embedding backend will shift every curve — re-tune after the swap.
