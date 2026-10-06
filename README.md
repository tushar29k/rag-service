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

Then `POST /query` with `{"text": "What is the refund window?", "filters": {"country": "IN"}}` — you get the answer, citations with scores, latency, and the per-stage breakdown. `POST /index` adds documents, `GET /health` shows the index version and chunk count.

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

## Things worth trying

- Set `chunk_size: 30` in `config.yaml` and re-run evals. Which questions lose recall? Why do small chunks hurt some queries?
- Add a reranker: re-score the top-10 with a stricter similarity, keep top-3. Time it. When is it worth it?
- `service.py` imports `StreamingResponse` but never uses it — add `POST /query/stream`.
- Add a `department` field to a doc's metadata and prove a query *without* the right filter can't retrieve it (pre-filter, never post-filter).
