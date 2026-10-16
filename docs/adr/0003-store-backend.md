# ADR 0003 — store backend: a 4-method interface, numpy first, pgvector later

date: 2026-10-16
status: accepted

## context

The index needed to run offline and in-memory for the demo, CI, and tests —
but "works on 200 chunks" is not a storage story. Whatever replaced the
numpy store had to drop in without touching retrieval code, and filtering
had to behave identically on both backends (filter-then-score, never
retrieve-then-discard, so filters can't silently tank recall).

## decision

Define the store as a 4-method interface: `upsert(texts, embs, metas)`,
`search(query_vec, top_k, filters)`, `save(path)`, `load(path)`. Two
invariants every backend obeys:

1. Row alignment — row i of the embedding matrix is always the same chunk
   as texts[i] / metas[i]; never break that.
2. Filter-first — metadata filtering happens before scoring, using the
   shared `match_metadata()` in `store.py` (exact match, multi-filter AND,
   mongo-style $gt/$lt/$eq operators). pgvector only gets exact-match
   filtering honestly documented, because server-side operators differ.

The default is the numpy `VectorStore` (brute-force cosine, persisted as
npz + json). `store_pgvector.py` implements the same interface against a
pgvector server and is selected by `store_backend: pgvector`; the swap lives
in `build_store()` in `retriever.py` so `DenseRetriever` stays untouched.

## consequences

- Adding a store backend (Qdrant, etc.) means one new file + one branch in
  `build_store()` — retrieval, fusion, and the pipeline don't move.
- numpy is O(n) per query: fine for the demo's thousands of chunks, a real
  wall past ~100k (see the README's "Going to production" section).
- The pgvector self-test needs a live server, so it isn't runnable in this
  sandbox or CI — the backend ships marked for a live environment, with
  docker-compose.yml ready for it.
