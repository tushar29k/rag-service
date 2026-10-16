# ADR 0001 — embedder choice: hashing TF-IDF by default, sentence-transformers on request

date: 2026-10-16
status: accepted

## context

The service needed an embedder for dense retrieval that works on a fresh
checkout, in CI, and on the free Render tier — no downloads, no GPU, no
model server. But the evals had to measure something honest, and a real
neural embedder had to be swappable in without rewriting the pipeline.

## decision

Ship with a hashing TF-IDF embedder (`TfidfEmbedder` in `embedder.py`,
dim 2048) as the default. Sentence-transformers is wired behind the same
`build_embedder()` shape — set `embedder_backend: st` in `config.yaml` and
it lazy-loads (e.g. bge-base-en-v1.5). Any failure at import or model-load
time falls back to hashing with a loud log line, so an offline box never
crashes.

The hashing scheme uses md5 bucketing, not Python's `hash()` — hash() is
salted per process, md5 isn't, so the same token lands in the same bucket
on every run. IDF weights are fit in one pass over the indexed corpus;
`embed_dim` in the config must match the model or the store asserts.

## consequences

- Zero-dependency boot everywhere: evals, CI, demo, cold Render starts.
- Honest limitation: bag-of-words hashing misses semantics (word order,
  synonyms). The 20-question eval set shows it — dense 19/20 recall@3 vs
  BM25's 20/20; the hybrid retriever (ADR 0002) closes the gap.
- Swapping to neural embeddings changes vector geometry — re-tune the
  backend-specific `min_score` after the swap (see the threshold sweep,
  evals/threshold-sweep.md), and re-fit the store (vectors aren't portable).
- The `# SWAP:` marker in `embedder.py` stays the production hint.
