"""Retrieval backends: dense (embeddings + cosine), BM25 (sparse keywords),
and hybrid (reciprocal rank fusion of the two).

All three answer the same interface, so pipeline.py never knows which
one is behind it:
    index(texts, metas)         # add chunks
    embed(question)             # question -> backend-native representation
    search(q, top_k, filters)    # [(text, score, meta)] best first, scores in [0, 1]

Scores are cosine similarity for dense and s/(1+s)-squashed BM25 for
sparse — both land in [0, 1] so pipeline's min_score gate treats them the
same: 0 means "no real match", 1 means "as good as it gets".

Why the squash and not max-normalisation? A nonsense question matching
only stopwords gives tiny raw BM25 scores everywhere; max-normalising
would inflate the best tiny score to 1.0 and sail straight past min_score,
killing the "I don't know" refusal the evals assert on. The saturating
squash keeps weak matches weak.
"""
from embedder import _tokenize, build_embedder
from store import VectorStore

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # caught properly in BM25Retriever.__init__


class DenseRetriever:
    """The original embedder + vector-store path — same behaviour, new clothes."""

    def __init__(self, cfg):
        self.embedder = build_embedder(cfg)
        # store sized from the embedder, not from config — st's 384 dims and
        # hashing's embed_dim are different animals
        self.store = VectorStore(dim=self.embedder.dim)
        self._fitted = False

    @property
    def metas(self):
        return self.store.metas

    def index(self, texts, metas):
        if not self._fitted:
            self.embedder.fit(texts)   # learn IDF from the first batch only
            self._fitted = True
        self.store.upsert(texts, self.embedder.transform(texts), metas)

    def embed(self, question):
        return self.embedder.transform([question])[0]

    def search(self, q_emb, top_k=5, filters=None):
        return self.store.search(q_emb, top_k=top_k, filters=filters)

    def __len__(self):
        return len(self.store)


class BM25Retriever:
    """rank-bm25 over the chunk texts. Keyword-exact where dense is fuzzy —
    catches the queries dense misses (model names, error codes, SKUs) and
    vice versa. Filters are applied by masking, so filtered docs never
    compete on scores they didn't earn.

    Query terms showing up in more than half the corpus are dropped before
    scoring — a df-based stopword detector, no stopword list needed.
    Without it, "who is the CEO" matches every chunk on "is"/"the" and
    defeats the min_score refusal the evals assert on."""

    def __init__(self, cfg):
        if BM25Okapi is None:
            raise ImportError("pip install rank-bm25  (needed for "
                              "retriever: bm25)")
        self.texts, self._metas, self._tok = [], [], []
        self._bm25 = None
        self._df = {}

    @property
    def metas(self):
        return self._metas

    def index(self, texts, metas):
        # same tokeniser as the dense path — both backends must agree on
        # what a "word" is, or the comparison is apples to oranges
        for t, m in zip(texts, metas):
            self.texts.append(t)
            self._metas.append(m)
            self._tok.append(_tokenize(t))
        # BM25Okapi fits at construction, so refit on every batch — a few
        # thousand chunks refit in milliseconds, no incremental index needed
        self._bm25 = BM25Okapi(self._tok) if self._tok else None
        self._df = {}
        for toks in self._tok:
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1

    def embed(self, question):
        return _tokenize(question)   # the "embed" stage for sparse: just tokens

    def search(self, qtok, top_k=5, filters=None):
        if self._bm25 is None:
            return []
        # drop corpus-ubiquitous query terms ("the", "is", ...) — they
        # contribute noise, not signal, and inflate junk matches
        kept = [t for t in qtok if self._df.get(t, 0) <= len(self._tok) / 2]
        # filter first, score second — same reason as the vector store:
        # retrieve-then-throw-away silently tanks recall
        ids = [i for i, m in enumerate(self._metas)
               if not filters or
               all(m.get(k) == v for k, v in filters.items())]
        if not ids:
            return []
        raw = self._bm25.get_scores(kept)
        ranked = sorted(((raw[i] / (1 + raw[i]), i) for i in ids),
                        key=lambda x: -x[0])[:top_k]
        return [(self.texts[i], float(s), self._metas[i]) for s, i in ranked]

    def __len__(self):
        return len(self.texts)


def build_retriever(cfg):
    """Pick the retriever backend from config.yaml.

    "dense" is the default (embeddings + cosine). "bm25" is the sparse
    keyword path. "hybrid" fuses both with reciprocal rank fusion —
    keyword-exact and fuzzy matches reinforce instead of competing.
    Anything unrecognised falls back to dense — a typo in config
    shouldn't take the service down.
    """
    name = cfg.get("retriever")
    if name == "bm25":
        return BM25Retriever(cfg)
    if name == "hybrid":
        return HybridRetriever(cfg)
    return DenseRetriever(cfg)


class HybridRetriever:
    """Reciprocal rank fusion of dense + BM25 over one shared corpus.

    Each backend ranks its own deep list, then ranks are fused as
    alpha/(k+rank) for dense + (1-alpha)/(k+rank) for BM25. Scores are
    normalised by the max possible fused score (rank 1 in both lists =
    1/(k+1)), so they land in [0, 1] and pipeline's min_score gate
    behaves the same as for the single backends.

    Each sub-list is pre-gated at min_score before fusion. Without that,
    junk queries still produce ranks (1..N everywhere) and RRF would fuse
    them into passing scores — defeating the "I don't know" refusal the
    evals assert on. The gate keeps each backend honest first, fusion
    only re-orders the survivors.
    """

    def __init__(self, cfg):
        self._dense = DenseRetriever(cfg)
        # raises ImportError if rank-bm25 isn't installed — hybrid needs it
        self._sparse = BM25Retriever(cfg)
        self.alpha = float(cfg.get("hybrid_alpha", 0.5))
        self.k = 60       # RRF damping: bigger k flattens rank differences
        self.depth = int(cfg.get("hybrid_depth", 20))  # per-list fuse depth
        self.min_score = float(cfg.get("min_score", 0.15))

    @property
    def metas(self):
        return self._dense.metas   # one corpus, indexed in lockstep

    def index(self, texts, metas):
        self._dense.index(texts, metas)
        self._sparse.index(texts, metas)

    def embed(self, question):
        return (self._dense.embed(question),   # dense vector
                self._sparse.embed(question))  # sparse tokens

    def search(self, q, top_k=5, filters=None):
        q_emb, q_tok = q
        # fuse deep lists, return the shallow top_k — RRF needs the depth
        # to see where the backends agree and disagree
        dense = [(t, s, m) for t, s, m in
                 self._dense.search(q_emb, top_k=self.depth, filters=filters)
                 if s >= self.min_score]
        sparse = [(t, s, m) for t, s, m in
                  self._sparse.search(q_tok, top_k=self.depth, filters=filters)
                  if s >= self.min_score]
        if not dense and not sparse:
            return []
        fused = {}   # chunk key -> [fused score, text, meta]
        for weight, ranked in ((self.alpha, dense), (1 - self.alpha, sparse)):
            for rank, (text, _, meta) in enumerate(ranked, start=1):
                key = meta.get("hash", text)   # chunk identity for dedupe
                if key not in fused:
                    fused[key] = [0.0, text, meta]
                fused[key][0] += weight / (self.k + rank)
        # best case: rank 1 in both lists -> 1/(k+1); normalise to [0, 1]
        norm = self.k + 1
        out = sorted(((s * norm, t, m) for s, t, m in fused.values()),
                     key=lambda x: -x[0])[:top_k]
        return [(t, float(s), m) for s, t, m in out]

    def __len__(self):
        return len(self._dense)


if __name__ == "__main__":
    docs = ["the refund window is 30 days for customers in india",
            "maternity leave is 26 weeks in india",
            "standard shipping takes 3 to 5 business days"]

    for name in ("dense", "bm25", "hybrid"):
        r = build_retriever({"retriever": name, "embed_dim": 256})
        r.index(docs, [{"i": i} for i in range(3)])
        assert len(r) == 3 and len(r.metas) == 3
        top = r.search(r.embed("how long is the refund window"), top_k=1)
        assert top[0][0] == docs[0], f"{name} retriever ranked wrong"
        print(f"{name} retriever OK "
              f"(top score {top[0][1]:.3f}, filters:",
              [m for _, _, m in r.search(r.embed("shipping"),
                                        top_k=5, filters={"i": 2})], ")")

    # hybrid on a nonsense query: both sub-lists gate out before fusion,
    # so nothing survives to be fused — the refusal path stays intact
    r = build_retriever({"retriever": "hybrid"})
    r.index(["refund policy thirty days", "the policy is the policy"],
            [{"i": 0}, {"i": 1}])
    assert r.search(r.embed("xylophone quantum"), top_k=3) == []
    # and alpha really is a weight: alpha=1 must equal the dense ranking,
    # alpha=0 must equal the bm25 ranking
    docs2 = ["refund window thirty days india", "shipping takes five days",
             "maternity leave twenty six weeks"]
    for alpha, twin in ((1.0, "dense"), (0.0, "bm25")):
        h = build_retriever({"retriever": "hybrid", "hybrid_alpha": alpha})
        h.index(docs2, [{"i": i} for i in range(3)])
        s = build_retriever({"retriever": twin, "embed_dim": 256})
        s.index(docs2, [{"i": i} for i in range(3)])
        q = "refund window"
        hr = [t for t, _, _ in h.search(h.embed(q), top_k=1)]
        sr = [t for t, _, _ in s.search(s.embed(q), top_k=1)]
        assert hr == sr, f"alpha={alpha} hybrid rank-1 != {twin}: {hr} vs {sr}"
    print("hybrid fusion OK (refusal empty, alpha endpoints match backends)")

    # a nonsense question must score near 0, so pipeline's min_score gate
    # still refuses it instead of answering off stopword matches
    r = build_retriever({"retriever": "bm25"})
    r.index(["refund policy thirty days", "the policy is the policy"],
            [{"i": 0}, {"i": 1}])
    s = r.search(r.embed("xylophone quantum"), top_k=1)[0][1]
    assert s < 0.15, f"bm25 weak-match score too high: {s}"
    s2 = r.search(r.embed("what is the"), top_k=1)[0][1]
    assert s2 < 0.15, f"bm25 stopword-only query scored too high: {s2}"
    print(f"bm25 weak-query guard OK (scores {s:.3f}, {s2:.3f})")
