# rag-service — a production-shaped RAG system you can actually run

No GPU, no API keys, no heavy downloads: the embedder is hashing TF-IDF and
the "LLM" is extractive, so the **entire pipeline runs on this machine**.
Every file has `# SWAP:` comments marking exactly what changes in production.

## Run it

```bash
cd rag-service
pip install -r requirements.txt

python3 chunker.py            # unit-check chunking + overlap
python3 embedder.py           # unit-check the embedder
python3 store.py              # unit-check the vector store
python3 cli.py --demo         # index sample docs, answer 3 questions
python3 evals/run_eval.py     # recall@3 + faithfulness on the golden set
uvicorn service:app --reload  # the HTTP API (then POST /query)
```

## The shape (this is what production RAG looks like)

```
cli.py / service.py      entry points (CLI demo, FastAPI)
pipeline.py              RAGPipeline: index + answer, every stage TIMED
chunker.py               overlapping word-window chunking
embedder.py              TfidfEmbedder (fit/transform) — SWAP for sentence-transformers
store.py                 VectorStore: brute-force + metadata PRE-filter — SWAP for Qdrant
config.yaml              chunk size, top_k, index version — not hardcoded
evals/golden.jsonl       the golden set — versioned with the code
evals/run_eval.py        recall@k + faithfulness + p50 latency
```

## Exercises (do these — this is the production learning)

1. **Break the chunking.** Set `chunk_size: 30` in config.yaml, re-run evals.
   Which questions lose recall? Why do small chunks hurt some queries?
2. **Add hybrid search.** Implement keyword (BM25-ish) scoring in `store.py`,
   combine 50/50 with vector scores, re-run evals. Did recall@3 move?
3. **Measure the bottleneck.** Look at the `breakdown` timings in `cli.py`
   output. Now imagine `generate` takes 3000ms (a real LLM) — what % of total
   is retrieval? This is why seniors optimise the LLM call first.
4. **Add a reranker.** Write `rerank.py` that re-scores the top-10 with a
   stricter similarity and keeps top-3. Time it. When is it worth it?
5. **Streaming endpoint.** `service.py` imports StreamingResponse but never
   uses it — add `POST /query/stream` yielding tokens as they're generated.
6. **Access control.** Add `department` to a doc's metadata and prove a query
   *without* the right filter can't retrieve it (pre-filter, never post-filter).
