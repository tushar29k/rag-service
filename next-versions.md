# rag-service — next versions plan

Current state (v1): chunk → embed → retrieve → generate with a hashing TF-IDF
embedder, numpy brute-force store, extractive mock generator, CLI + FastAPI,
evals at 8/8. Every item below is one day's commit: implement it, keep evals
green, commit with a human-style message.

## v2.1 — real embeddings & hybrid retrieval

- [ ] Swap hashing TF-IDF for sentence-transformers (`backend: st` in config.yaml; lazy import; fall back to hashing when offline) — done when `python embedder.py` self-test passes with both backends
- [ ] BM25 retriever (rank-bm25) alongside dense, same interface as the dense path — done when both retrievers answer the demo questions
- [ ] Hybrid fusion: reciprocal rank fusion of BM25 + dense with an `alpha` weight in config — done when `pipeline.py` supports `retriever: hybrid`
- [ ] Evals: extend golden.jsonl to 20 questions; report recall@3 for dense vs BM25 vs hybrid — done when run_eval.py prints the three-way table
- [ ] Reranker interface + cross-encoder backend (bge-reranker via sentence-transformers), rerank top-20 → top-5 — done when `rerank: true` in config changes the top-5
- [ ] Evals: nDCG@5 lift from reranking recorded in eval output — done when the lift number prints
- [ ] Query rewriting: multi-query expansion with a mock LLM function (marked `# SWAP:`), dedupe merged results — done when one ambiguous query fans out to 3 variants

## v2.2 — production store & generation

- [ ] pgvector store backend implementing the same 4-method store interface; docker-compose for local pgvector — done when `store_backend: pgvector` passes store.py self-tests
- [ ] Batch `/index` endpoint accepting a list of documents; index versioning recorded in /health — done when version bumps on re-index
- [ ] Real LLM generation backend: OpenAI-compatible client with a citation-required prompt; extractive mock stays as fallback — done when `generator: openai` works with a key and falls back cleanly without one
- [ ] Streaming answers endpoint (SSE) from the generate stage — done when curl shows tokens arriving incrementally
- [ ] Per-stage latency logging to JSONL + a p50/p99 summary script — done when 100 queries produce a latency table
- [ ] Semantic cache: exact-match + embedding-similarity cache with a hit-rate metric in /health — done when repeated queries report hits
- [ ] Config profiles: dev/prod YAML overlays (`config.prod.yaml`) — done when the service loads the overlay via env var

## v2.3 — chunking & quality science

- [ ] Sentence-aware recursive chunker option next to the word-window chunker — done when `chunker: recursive` is selectable in config
- [ ] Chunking experiment script: 4 strategies × same queries, winner recorded in docs — done when results table is committed to evals/
- [ ] Metadata filtering extended: multi-filter + range filters in the store interface — done when store.py self-test covers them
- [ ] RAGAS-style evals: faithfulness + answer-relevance scoring with a mock judge (marked `# SWAP:`) — done when run_eval.py prints both scores
- [ ] Failure-mode audit doc: the 7 known RAG failure points mapped to this codebase — done when docs/failure-modes.md exists
- [ ] Threshold tuning: sweep min_score, record precision/recall trade-off — done when the sweep table is in evals/
- [ ] README: production swap guide — what changes for 100k docs, Qdrant, real embeddings — done when README has a "Going to production" section

## v2.4 — ship it

- [ ] Dockerfile + docker-compose (API + pgvector) — done when `docker compose up` serves queries
- [ ] /health expansion: index stats, active backend names, last eval score snapshot — done when all fields present
- [ ] CI: GitHub Action running evals on push, failing if recall@3 drops — done when .github/workflows/evals.yml exists and passes
- [ ] Load test script: concurrent queries, p50/p99 report — done when load_test.py prints the table
- [ ] ADRs: 3 architecture decision records (embedder choice, hybrid fusion, store backend) — done when docs/adr/ has 3 files
- [ ] Demo on a real public dataset (e.g. SQuAD slice): index + eval numbers committed — done when demo script + results exist
- [ ] Release notes v2.0 in README with real numbers (latency, recall@3, cost) — done when README leads with the numbers
