# rag-service

A retrieval-augmented generation (RAG) system built like the real thing — chunking, embeddings, a vector store, retrieval, grounded answers with citations — but with zero API keys and zero GPU, so you can actually run the whole loop on your laptop.

## Live demo

**[https://tushar29k-rag-service.onrender.com](https://tushar29k-rag-service.onrender.com)** — Index documents and ask questions — answers come back with citations, retrieval scores, and per-stage latency.
> Hosted on Render's free tier — the first visit after a while can take ~30s while the instance wakes up.

## The idea

Most RAG tutorials stop at "embed the docs, search, done". The interesting part of production RAG is everything around that: how you chunk documents, how you *measure* whether retrieval is any good, how latency breaks down per stage, and how you stop the model from confidently answering out of irrelevant chunks. This project is a complete, runnable loop — index documents, ask questions, get grounded answers with citations and per-stage timings — where every heavyweight piece has a clearly marked swap point for its production equivalent.

## How it works

The pipeline lives in `pipeline.py` and runs five stages for every question:

1. **Chunking** (`chunker.py`) — documents are split into overlapping windows (120 words, 20-word overlap by default), or, with `chunker: recursive` in config.yaml, into sentence-aware chunks whose boundaries only land between sentences. The overlap matters: a sentence straddling a boundary would otherwise get cut in half and the retriever could miss it entirely.
2. **Embeddings** (`embedder.py`) — chunks are turned into vectors with a hashing TF-IDF embedder (see "Honest notes"). The interface is `fit`/`transform`, deliberately shaped like a real embedder so swapping one in is a one-class change.
3. **Store** (`store.py`) — vectors go into a numpy brute-force store with metadata **pre**-filtering (filter first, score second — doing it the other way round silently kills recall). Same four-method interface as Qdrant/pgvector.
4. **Retrieval** — two backends behind one interface (`retriever.py`), picked with `retriever:` in config.yaml: `dense` embeds the question and pulls top-k chunks by cosine similarity, `bm25` scores keyword overlap with rank-bm25. Anything scoring below a relevance threshold (`min_score`) is dropped so the generator refuses instead of hallucinating off a junk chunk.
5. **Generation** — an extractive mock picks the highest-overlap sentences from the retrieved chunks and cites them like `[1]`, following the grounded-generation prompt pattern a real LLM call would use.

Every stage is timed, and the per-stage `breakdown` comes back with every answer — that's the bit you'd ship to a latency dashboard in production.

## How to run

Prerequisites: Python 3.10+ and pip. No API keys, no downloads beyond pip packages.

```bash
cd rag-service
pip install -r requirements.txt
```

**Sanity-check each piece** (each module self-tests when run directly):

```bash
python3 chunker.py     # chunks 300 fake words, asserts the overlap is intact
python3 embedder.py    # checks related docs score higher than unrelated ones
python3 store.py       # checks filtered search + save/load round-trip
```

**Run the demo** — indexes the sample docs and answers 3 questions:

```bash
python3 cli.py --demo
```

You should see something like:

```
indexed 5 chunks (version v1)

Q: What is the refund window?
A: Customers in India are eligible for a full refund within 30 days of purchase...
  (0.2ms {'embed': 0.0, 'retrieve': 0.2, 'prompt': 0.0, 'generate': 0.0})
  [score 0.192] Refund policy. Customers in India are eligible for a full refund...
```

Everything runs in well under a millisecond here — the timings get interesting once a real (slow) LLM sits in the `generate` stage.

**Interactive mode** — ask your own questions (empty line quits; prefix `nofilter:` to skip the country filter):

```bash
python3 cli.py
```

**Run the evals:**

```bash
python3 evals/run_eval.py
```

**Run the HTTP API:**

```bash
uvicorn service:app --reload
```

Then `POST /query` with `{"text": "What is the refund window?", "filters": {"country": "IN"}}` — you get the answer, citations with scores, latency, and the per-stage breakdown. `POST /index` adds documents, `GET /health` shows index stats (version, doc and chunk counts), the active backend names, cache stats, and the last eval score snapshot.

### Use your own key

The live demo above runs on the author's key. To point your own copy at a
real model for the generation stage:

1. **Get a free key.** Go to `aistudio.google.com/api-keys` and click
   **Create API key** — pick "Create API key in new project" (no Cloud
   project and no credit card needed). Alternative: an OpenRouter key
   (`openrouter.ai`) used with a `:free` model slug.
2. **Local run:** `export LLM_API_KEY=your-key-here` before starting the
   server — or put it in a `.env` file you never commit.
3. **Render deploy:** dashboard → your service → Environment → add
   `LLM_API_KEY` → Save. Render redeploys automatically and the fresh
   build reads the key at startup (the client is created once at import,
   so a restart is required — there is no hot-swap).
4. **Confirm it's live:** the stamp in the demo header turns green
   (`● live LLM · gemini-3.8-flash`), or `GET /info` returns
   `"real_llm": true`.
5. **Keep the key safe:** keys live in environment variables or a secret
   manager only — never in code, never in a commit.

## Project layout

```
llm_client.py     free-tier LLM client (gemini | openrouter), stdlib only
cli.py              demo / interactive entry point
service.py          FastAPI wrapper: /index, /query, /health
pipeline.py         RAGPipeline — index + answer, every stage timed
chunker.py          word-window + sentence-aware recursive chunking
embedder.py         hashing TF-IDF embedder (fit/transform)  ← swap for sentence-transformers
retriever.py        dense (embedder + store) and bm25 backends, same interface  ← `retriever:` in config.yaml
store.py            numpy brute-force store + metadata pre-filter  ← swap for Qdrant
config.yaml         chunk size, overlap, top_k, min_score, index version
data/sample_docs.json   the docs the demo indexes
evals/golden.jsonl      8 golden Q&A pairs the evals assert against
evals/run_eval.py       recall@3 + evidence coverage + p50 latency
```

## Evals

`python3 evals/run_eval.py` runs 8 golden questions from `evals/golden.jsonl` against the pipeline and checks two things per question:

- **recall@3** — is the expected document among the top-3 retrieved chunks?
- **evidence coverage** — do the retrieved chunks actually contain the answer keywords?

It also reports p50 latency. Current numbers: **recall@3: 8/8, evidence: 8/8**. One of the 8 questions ("Who is the CEO?") is deliberately unanswerable — for that one, the check is that the system refuses instead of making something up. If you change chunking or retrieval settings, re-run this; it's the fastest way to see whether you helped or hurt.

Run the same golden set against the BM25 backend with `python3 evals/run_eval.py --retriever bm25` (or set `retriever: bm25` in `config.yaml` to make it the default for the CLI and the API).

## Honest notes

- **The embedder is hashing TF-IDF, not a neural model.** It's a bag-of-words trick with md5 bucketing. Great for zero-dependency local runs, useless for semantic similarity ("refund" won't match "money back"). Swap point: `embedder.py`, marked with `# SWAP` — a sentence-transformers class with the same `fit`/`transform` shape drops straight in.
- **The "LLM" is an extractive mock** (`_mock_llm` in `pipeline.py`) unless you set `LLM_API_KEY`. It quotes sentences with keyword overlap; it can't do negation or synthesis, and it says so in every answer. With a key set, `RAGPipeline` calls a real model (Gemini via Google AI Studio's free tier, or OpenRouter with `LLM_PROVIDER=openrouter` and a `:free` model; `LLM_MODEL` overrides the default) over the retrieved context instead — retrieval, reranking, citations, and the response contract are untouched. If the API call fails, the mock answers and the response carries a `[model unavailable — showing offline mock result]` note. On the live Render demo, set the env vars in the Render dashboard.
- **The store is brute-force numpy** — O(n·d) per query, fine to ~100K chunks. Swap point: `store.py`, same four methods (`upsert`/`search`/`save`/`load`) on Qdrant or pgvector.
- **`min_score` (0.15) was tuned by hand** against the golden set, not derived from anything principled. Re-tune it when the corpus changes.
- **Config profiles:** `config.yaml` is the dev default. Setting `RAG_PROFILE=prod` layers `config.prod.yaml` on top of it at startup (nested dicts merge key by key, scalars and lists are replaced — see `load_config` in `pipeline.py`). A missing overlay file fails loudly instead of silently running dev defaults. The active profile is reported by `GET /info`. Make your own with `config.<name>.yaml` — it must live next to `config.yaml`.

## Going to production

The runbook for taking the dev setup above to a real corpus. Every swap point is already wired — this section is where each one points.

### 1. The embedder swap (do this first)

`embedder.py` is hashing TF-IDF by default. Set `embedder_backend: st` in `config.yaml` and `build_embedder()` lazy-loads sentence-transformers (`all-MiniLM-L6-v2`, 384 dims); if the package or the model can't load, it falls back to hashing with a log line — never a crash. Bigger model, no code change: set `st_model` to e.g. `bge-base-en-v1.5` (that's the `# SWAP` marker in `embedder.py`).

Two things bite here:

- **`embed_dim` must match the model.** `config.yaml` ships `embed_dim: 2048` — that's the hashing width. Set it to 384 for all-MiniLM-L6-v2 (768 for bge-base) or every vector lands in a store built for the wrong dimension.
- **Re-tune `min_score`.** The 2026-10-11 threshold sweep (`evals/threshold-sweep.md`, produced by `evals/threshold_sweep.py`) found best F1 at dense 0.15 / BM25 0.60 / hybrid 0.50 on hashing embeddings. Neural embeddings shift every curve — re-run the sweep and take the new best-F1 threshold (ties go to the higher, more conservative value). One number does not fit all backends.

The semantic cache has its own embedder with a `# SWAP` note (`cache.py::_embed`) — hashing bag-of-words again. Point it at the pipeline's embedder for production so cache hits use real similarity.

### 2. Indexing 100k docs

The numpy store (`store.py`) is O(n·d) per query — fine to ~100K chunks. At 100k docs with default chunking (`chunk_size: 120`, `overlap: 20`) you're looking at roughly 5–10 chunks per doc, well past where brute force hurts — so this is also the store-swap moment (step 3). The mechanics of getting docs in:

- **Batch the writes.** `POST /index` accepts one doc or a list, and every call bumps `index_version`. Push a few hundred to a thousand docs per request from your indexing client — `/health` reports the version, so you can watch it climb and resume where a failed batch left off.
- **Indexing is idempotent.** `pipeline.index_documents()` dedupes chunks on content hash before upsert, so re-sending a batch is a no-op — resume safety is free.
- **Pick the chunker deliberately.** `chunker: word` (default) won the 4-way experiment in `evals/chunking-experiment.md` on the fixture corpus; `recursive` is the better production pick for long-form docs since chunk boundaries only ever land between sentences; `sentence_window` maximizes precision at the cost of more chunks. More chunks = more storage and deeper retrieval lists to fuse, so don't default to sentence_window.
- **Metadata first, filters later.** `/query` `filters` run against chunk `meta` via `match_metadata()` in `store.py` — exact match plus mongo-style `$eq/$ne/$gt/$gte/$lt/$lte`. Put whatever you'll filter on (source, year, department) in the doc metadata at index time. The pgvector/Qdrant backends only support exact-match filters today, so keep production filters to exact matches until that's extended.

### 3. Swapping the store (Qdrant or pgvector)

The store contract is four methods on one interface (`store.py`): `upsert(texts, embs, metas)`, `search(query_vec, top_k, filters)` returning `(text, score, meta)` tuples best-first, `save(path)`, `load(path)`. The pipeline talks to the interface, never the implementation:

- **Qdrant:** implement the four methods on `QdrantClient` in a new `store_qdrant.py` (the file-level `# SWAP` comment in `store.py` says exactly this), then flip `store_backend:` in `config.yaml`.
- **pgvector:** already shipped as `store_pgvector.py` — `docker-compose.yml` stands up Postgres + pgvector locally, `PGVECTOR_DSN` points at it, `store_backend: pgvector` selects it. Its `search()` returns `1 - cosine_distance` so the score scale lines up with the numpy store's cosine similarity and the pipeline's `min_score` gate applies unchanged.
- **The pre-filter invariant:** whichever backend you write, filters run *before* scoring (the recall guarantee `store.py::_filtered_ids` encodes — retrieve-then-throw-away silently tanks recall). `BM25Retriever.search` reuses `match_metadata()` so dense and sparse agree on what a filter means.

### 4. Config: what changes for prod

`RAG_PROFILE=prod` layers `config.prod.yaml` over `config.yaml` at startup (nested dicts deep-merge, scalars and lists are replaced; a missing overlay raises instead of silently running dev defaults — see `load_config` in `pipeline.py`). The shipped overlay moves `top_k` 3→5, `min_score` 0.15→0.2, `hybrid_depth` 20→40, the semantic cache to 1024 entries, and prod timings into `logs/prod-latency.jsonl`. The active profile shows up in `GET /info`.

Features worth switching on for a real corpus — all off by default so the demo's behavior doesn't change under you:

- `reranker: true` — pulls `rerank_depth: 20` candidates and re-scores jointly down to `rerank_top_n: 5`. Use `rerank_backend: cross_encoder` when the model (`rerank_model: BAAI/bge-reranker-base`) can be cached; it degrades to the lexical stand-in with a log line when it can't, and the lexical path is honest but weak (measured nDCG@5 lift of −0.050 in `evals/run_eval.py`).
- `rewrite: true` — fans one ambiguous query into `rewrite_variants: 3` variants, lists merge and dedupe (mock LLM behind the `# SWAP` in `rewriter.py` — point it at the real model).
- `retriever: hybrid` — reciprocal rank fusion of dense + BM25 at `hybrid_alpha: 0.5`.

### 5. Latency and cost: what to expect

Every stage is timed (`pipeline._record_latency` appends one JSON line per query to `logs/latency.jsonl`, gitignored), and every answer carries its per-stage `breakdown`. `python3 evals/latency_summary.py` prints the p50/p99 table. Where the time goes:

- Hashing embed + numpy store: sub-millisecond. The `st` swap moves the embed stage to tens of milliseconds per query on CPU; reranking adds a cross-encoder call per candidate (20 → 5).
- Generation dominates once it's real: the extractive mock is instant; a real LLM call (Gemini free tier via `LLM_API_KEY`, or `generator: openai` with `OPENAI_API_KEY`/`LLM_API_KEY`, `LLM_MODEL` and `OPENAI_BASE_URL` overrides in `llm_client.py`) takes seconds — streamable via `POST /query-stream` (SSE: meta → token* → done).
- The semantic cache (`semantic_cache.enabled`, threshold 0.85) skips the whole pipeline on repeated questions. `/health` reports hits/misses/hit_rate and cached answers carry `"cached": true`.

### 6. Re-running the evals after every swap

The eval suite is the detector — run it after each change, not just at the end:

```bash
python3 evals/run_eval.py                 # recall@3, evidence, faithfulness + answer-relevance, all three retrievers
python3 evals/run_eval.py --retriever bm25
python3 evals/threshold_sweep.py          # re-tune min_score on the new backend; writes evals/threshold-sweep.md
```

The 2026-10-11 baseline: dense 19/20, BM25 20/20, hybrid 20/20 recall@3; faithfulness 0.944–1.000 via `evals/judge.py`; rerank lift pending the neural cross-encoder. If a swap drops recall@3, it regressed.

## Things worth trying

- Set `chunk_size: 30` in `config.yaml` and re-run evals. Which questions lose recall? Why do small chunks hurt some queries?
- Add a reranker: re-score the top-10 with a stricter similarity, keep top-3. Time it. When is it worth it?
- `service.py` imports `StreamingResponse` but never uses it — add `POST /query/stream`.
- Add a `department` field to a doc's metadata and prove a query *without* the right filter can't retrieve it (pre-filter, never post-filter).
