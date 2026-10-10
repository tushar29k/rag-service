# RAG failure-mode audit

The seven ways a retrieval-augmented system fails, mapped to the actual
code in this repo: which file owns each failure point, what it does
about it today, and what's still a known gap. The eval suite is the
shared detector — every failure below has an assertion or a metric
in `evals/run_eval.py` (or an eval file that feeds it) watching for it.

## 1. Retrieval misses — the right chunks never surface

**Owner:** `retriever.py` (`DenseRetriever`, `BM25Retriever`,
`HybridRetriever`), `store.py` (`VectorStore`), `embedder.py`.

Three backends answer the same interface behind `build_retriever()`, and
the pipeline doesn't know which one is wired in — dense for fuzzy
semantics, BM25 for keyword-exact matches (model names, error codes),
hybrid fusing both with reciprocal rank fusion (`HybridRetriever`,
`hybrid_alpha: 0.5`, `hybrid_depth: 20`). `reranker.py` adds a second
stage: retrieve wide (`rerank_depth: 20`), re-score jointly down to
`rerank_top_n: 5`. `rewriter.py` fans an ambiguous query out into
variants before retrieval (each variant retrieves independently, lists
merge and dedupe).

**How it's caught:** `evals/run_eval.py` scores recall@3 and evidence
coverage on `evals/golden.jsonl` for all three backends side by side,
plus nDCG@5 lift for reranking. A miss shows up as a recall drop, not a
silent wrong answer.

**Still open:** rerank and rewrite are off by default (`config.yaml`) so
the live demo's behavior doesn't shift under you — a real corpus with
ambiguous queries should turn them on.

## 2. Bad chunking — the answer is cut in half across a boundary

**Owner:** `chunker.py` (`chunk_text`, `recursive_chunk_text`,
`fixed_char_chunk_text`, `sentence_window_chunk_text`), picked with
`chunker:` in `config.yaml` (`chunk_size: 120`, `overlap: 20`).

The default word-window chunker re-covers the tail of each window via
overlap so a straddling sentence survives; `recursive` only breaks at
sentence boundaries; `sentence_window` trades more chunks for precision.
`fixed_char` exists purely as the dumb baseline. Re-indexing the same
docs is a no-op: `pipeline.py`'s `index_documents()` dedupes chunks on
content hash before upsert.

**How it's caught:** `evals/chunking-experiment.md` + `evals/chunking_experiment.py`
scored all four strategies on the same queries — the choice is measured,
not stylistic. The `hash` field on every chunk meta (`pipeline.py`) also
makes chunk-boundary regressions visible as index-count changes.

**Still open:** one monster sentence with no boundaries degrades
`recursive` back to word windows; abbreviations bleed through the cheap
sentence splitter (`_split_sentences`).

## 3. Lost in the middle / overlong context — the answer is buried

**Owner:** `pipeline.py` (`build_prompt`, `answer`), `config.yaml`
(`top_k: 3`).

Retrieval returns at most `top_k` chunks (3 by default) and the prompt
numbers them as citations `[1]..[n]`, so the model reads a short,
referenced context block — not the whole store. When rerank is on, the
pipeline pulls 20 candidates deep but only the re-scored top 5 reach
the prompt. The grounded prompt tells the model to answer using ONLY
the context.

**How it's caught:** evidence coverage in `run_eval.py` asserts the
must-contain keywords sit in the retrieved chunks; `latency_probe.py`
keeps the retrieve stage's share of total time visible, since deep
top-k reads cost latency before they cost accuracy.

**Still open:** no summarization or map-reduce for long multi-chunk
answers — the ceiling is one prompt's worth of chunks.

## 4. Stale index — answers cite a corpus that has moved on

**Owner:** `pipeline.py` (`index_documents`, `_bump_index_version`),
`service.py` (`/health`, `/index`), `cache.py` (`SemanticCache.clear`).

Every `/index` call bumps `index_version` (`v1` -> `v2` -> ...) and the
version ships in every answer, stream meta event, and `/health` — you
can always tell which build of the index produced an answer. Content
hashes make re-indexing idempotent (returns 0 new chunks instead of
duplicating). And the semantic cache is cleared on every index call,
so a cached answer can never outlive the corpus it was grounded on.

**How it's caught:** answers and `/health` both report `index_version`;
the evals re-index from `data/sample_docs.json` on every run, so
stale-index bugs would show as golden-set drift between runs.

**Still open:** `store.py`'s numpy backend is in-memory — a service
restart rebuilds from nothing until `/index` is called again; the demo
re-seeds via the UI. `store_pgvector.py` exists for the persistent path
(docker-compose + `PGVECTOR_DSN`).

## 5. Threshold mis-tuning — the refusal gate fires at the wrong point

**Owner:** `pipeline.py` (`answer`, `answer_stream`), `config.yaml`
(`min_score: 0.15`), `retriever.py` (score normalization), `reranker.py`.

Brute-force search always returns *something*, even for nonsense — so
after retrieval, anything scoring below `min_score` is dropped and the
generator refuses ("I don't know") instead of answering off a junk
chunk. The backends keep this honest by design: BM25 scores are squashed
as `s/(1+s)` (not max-normalized, so stopword matches stay weak),
hybrid pre-gates each sub-list at `min_score` before fusion (RRF would
otherwise fuse junk ranks into passing scores), and reranker scores land
in `[0, 1]` so the same gate applies unchanged.

**How it's caught:** `run_eval.py` includes deliberately unanswerable
golden questions asserting the "don't know" refusal, plus BM25 weak-query
asserts in `retriever.py`'s `__main__` (`"xylophone quantum"` scores
< 0.15, stopword-only queries score < 0.15).

**Still open:** `min_score: 0.15` was tuned by hand against the golden
set, not derived — it must be re-tuned when the corpus changes, and the
doc says so in `config.yaml`.

## 6. Hallucinated answers — fluent claims with no grounding

**Owner:** `pipeline.py` (`build_prompt`, `_generate`, `_mock_llm`).

Every factual claim must cite its chunk (`[1]`, `[2]` — "No citation =
no claim — never guess or fill in gaps from memory"). The extractive
mock refuses when retrieval came back empty rather than inventing, and
every answer returns `citations` with text, score, and meta so a reader
(or the UI) can check the sources. Real-model temperature is pinned low
(0.2) in `_generate`.

**How it's caught:** `evals/judge.py` scores faithfulness (claims
supported by retrieved context) and answer-relevance on the answerable
golden questions; unanswerable ones are judged on refusal instead — a
clean "I don't know" is trivially faithful.

**Still open:** the judges are mock heuristics (`judge.py` carries a
`# SWAP` note for a real LLM judge); the mock generator is extractive
by design and can't do negation or synthesis — it says so in its output.

## 7. Silent mock fallback — the model fails and nobody notices

**Owner:** `llm_client.py` (`FreeLLMClient`, `FreeLLMError`),
`pipeline.py` (`_generate`, `_generate_stream`), `service.py` (`/info`),
`ui/index.html` (status badge).

When the real-model call fails (timeout, 429/5xx after one backoff
retry, unreachable provider), `_generate` falls back to the extractive
mock — but never silently: the answer gets an appended
`[model unavailable — showing offline mock result]` note, the error is
printed to stderr, and `rag.last_llm_error` is exposed on `/info`
alongside `real_llm` (false when no key configured a live client at
startup). The demo page shows a green "live LLM" badge with the model
name or an amber "offline mock" badge from that same endpoint.

**How it's caught:** `/info` distinguishes "no key configured" from
"key configured but the last call failed" (`last_error` non-null);
`test_llm_client.py` covers the client; evals run against the mock, so
fallback doesn't change their results.

**Still open:** a mid-stream failure in `answer_stream` surfaces the
note only after the already-yielded tokens — the meta event can't be
retracted.
