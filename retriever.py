"""Retrieval backends: dense (embeddings + cosine) and BM25 (sparse keywords).

Both backends answer the same interface, so pipeline.py never knows which
one is behind it — and hybrid fusion later is just a third backend:
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
    keyword path. Anything unrecognised falls back to dense — a typo in
    config shouldn't take the service down.
    """
    if cfg.get("retriever") == "bm25":
        return BM25Retriever(cfg)
    return DenseRetriever(cfg)


if __name__ == "__main__":
    docs = ["the refund window is 30 days for customers in india",
            "maternity leave is 26 weeks in india",
            "standard shipping takes 3 to 5 business days"]

    for name in ("dense", "bm25"):
        r = build_retriever({"retriever": name, "embed_dim": 256})
        r.index(docs, [{"i": i} for i in range(3)])
        assert len(r) == 3 and len(r.metas) == 3
        top = r.search(r.embed("how long is the refund window"), top_k=1)
        assert top[0][0] == docs[0], f"{name} retriever ranked wrong"
        print(f"{name} retriever OK "
              f"(top score {top[0][1]:.3f}, filters:",
              [m for _, _, m in r.search(r.embed("shipping"),
                                        top_k=5, filters={"i": 2})], ")")

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
