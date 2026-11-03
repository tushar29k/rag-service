# Hybrid diagnosis results — 2026-10-18

Experiment: `evals/hybrid_diagnosis.py` (deterministic; re-ran twice, identical).
Corpus: 2067 SQuAD v1.1 dev paragraphs (committed slice `evals/squad_slice.json`).
302 questions. Metric: recall@3 of the gold paragraph (the exact paragraph
each question was written from). No rerank, no rewrite, no semantic cache.

## Baseline (reproduces 2026-10-17 exactly)

| backend | recall@3 |
|---|---|
| dense (hashing TF-IDF fallback) | 223/302 (73.8%) |
| bm25 | 262/302 (86.8%) |
| hybrid (RRF, alpha=0.5, k=60, depth=20) | 248/302 (82.1%) |

## Where fusion loses — bm25 vs hybrid contingency

both hit: 247 | bm25-only: 15 | hybrid-only: 1 | neither: 39
Net fusion effect: **−14** (fusion helped exactly once).

Dense rank of the gold paragraph on the 15 bm25-only questions:
top-3: 0 | #4–10: 5 | #11–20: 7 | absent from top-20: 3.

On every question fusion lost, dense ranked gold #4 or worse.

Examples (question | article | bm25 rank | dense rank):
- "When were the finalists announced?" | Super_Bowl_50 | #1 | absent
- "What was the score for the 2015 NFC championship game?" | Super_Bowl_50 | #3 | #10
- "Who were the ESPN Deportes commentators for Super Bowl 50?" | Super_Bowl_50 | #1 | #18

## hybrid_alpha sweep (recall@3)

| alpha (dense weight) | recall@3 |
|---|---|
| 0.00 | 262/302 (86.8%) |
| 0.15 | 260/302 (86.1%) |
| 0.25 | 256/302 (84.8%) |
| 0.35 | 255/302 (84.4%) |
| 0.50 (default) | 248/302 (82.1%) |
| 0.65 | 244/302 (80.8%) |
| 0.80 | 240/302 (79.5%) |
| 1.00 | 224/302 (74.2%) |

Monotonic decline as dense weight rises. alpha=0.0's hit set is
**exactly** BM25's hit set (verified question-for-question, symmetric
difference 0) — tuning alpha 0.5 → 0.0 recovers all 14 lost questions.

## Read

Equal-weight RRF lets the weak ranker's noise outvote the strong ranker's
correct top ranks. Weight by measured strength, not by default. Scope note:
dense here is the hashing TF-IDF fallback, not real embeddings — with real
embeddings the right alpha would not be 0. No repo defaults changed.
