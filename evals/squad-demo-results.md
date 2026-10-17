# SQuAD slice demo results

Date: 2026-10-17
Dataset: SQuAD v1.1 dev (https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
Slice: every 35th question of 10570 (deterministic, spread across all 48 articles)
Questions evaluated: 302 (overrides --limit when regenerating)
Corpus: 2067 SQuAD paragraphs, one doc per paragraph, chunk_size 120
Embedder: hashing TF-IDF (the zero-dependency default)
Generator: extractive mock (not scored here)

## recall@3 per backend

Gold = the exact paragraph the question was written from, retrieved
in the top-3 citations (para_id carried in chunk metadata).

| backend | recall@3 | p50 ms/query | p99 ms/query | wall s |
|---------|----------|--------------|--------------|--------|
| dense | 223/302 (73.8%) | 9.7 | 53.1 | 4.0 |
| bm25 | 262/302 (86.8%) | 8.2 | 31.2 | 3.0 |
| hybrid | 248/302 (82.1%) | 20.1 | 43.8 | 6.5 |

## Scope notes (read before quoting these numbers)

- The embedder is the hashing TF-IDF fallback, not sentence-
  transformers — recall here measures retrieval mechanics and the
  corpus, not embedding quality. Expect real embeddings to shift
  dense/hybrid up.
- Questions are asked cold with no context about which article
  they came from; BM25's keyword matching has the natural edge on
  SQuAD's entity-heavy questions.
- Per-question latency includes the mock generation step, so it
  reads high next to the retrieve-only timings in
  evals/latency_probe.py.
- Slice is deterministic (every 35th question) and committed at
  evals/squad_slice.json — re-running gives the same 302 questions.
