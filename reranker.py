"""Reranking: second-stage re-scoring of the retriever's candidates.

Pipeline stage order when rerank is on: retrieve (top-20 deep) -> rerank
-> top-5. The first stage is fast and recall-oriented (bi-encoder, BM25,
or the RRF hybrid); the reranker is slow and precision-oriented — a
cross-encoder reads the query and the chunk *together*, so it sees
word-order and negation that the first stage can't.

Both rerankers answer the same interface, and their scores land in
[0, 1] so pipeline.py's min_score gate treats them exactly like
retriever scores — 0 means "no real match", 1 means "as good as it
gets":
    rerank(question, candidates, top_n)  # [(text, score, meta)] best first

candidates are the retriever's (text, score, meta) triples; the
reranker re-scores and re-orders them, it never adds new ones.
"""
import math

from embedder import _tokenize  # same word-splitting as both retrievers

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None  # caught properly in CrossEncoderReranker.__init__


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def rerank(self, question, candidates, top_n=5):
        """Re-score candidates, return top_n as [(text, score, meta)]."""
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """The real one: (query, chunk) pairs scored jointly by a
    cross-encoder (bge-reranker by default, via sentence-transformers).

    Scores are raw logits out of the model — the sigmoid maps them into
    [0, 1] so pipeline's min_score gate stays meaningful. A strong
    cross-encoder opinion overrides the first-stage score: a junk query
    gets low logits (refused by the gate), a confident match gets a
    high one. The gate is tuned on first-stage scores, though, so the
    refusal threshold behaves a little differently here — the nDCG/lift
    eval in the next roadmap item will measure it properly.
    """

    def __init__(self, model_name):
        if CrossEncoder is None:
            raise ImportError("pip install sentence-transformers  (needed "
                              "for rerank_backend: cross_encoder)")
        self.model_name = model_name
        self._model = None

    def load(self):
        # downloads the model on first call if it's not cached — that's
        # why build_reranker() calls this at startup: fail fast to the
        # lexical fallback there, not mid-request on the demo
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, question, candidates, top_n=5):
        if not candidates:
            return []
        model = self.load()
        logits = model.predict([(question, text)
                                for text, _, _ in candidates])
        # sigmoid -> [0, 1]; keep the candidate text/meta attached so the
        # pipeline's citation shape doesn't change
        scored = [(_sigmoid(float(s)), text, meta)
                  for (text, _, meta), s in zip(candidates, logits)]
        scored.sort(key=lambda x: -x[0])
        return [(t, s, m) for s, t, m in scored[:top_n]]


class LexicalReranker(Reranker):
    """Zero-dep fallback: rank candidates by query-token overlap.

    # SWAP: use CrossEncoderReranker (rerank_backend: cross_encoder) where
    the sentence-transformers package and the model are available — this
    one can't tell "X was Y" from "Y was X", it's just counting shared
    words. Good enough as a wiring check, not as a quality improvement.
    """

    def rerank(self, question, candidates, top_n=5):
        if not candidates:
            return []
        q_toks = set(_tokenize(question))
        scored = []
        for text, _, meta in candidates:
            # fraction of the query's tokens the chunk covers
            s = len(q_toks & set(_tokenize(text))) / len(q_toks) \
                if q_toks else 0.0
            scored.append((s, text, meta))
        scored.sort(key=lambda x: -x[0])
        return [(t, float(s), m) for s, t, m in scored[:top_n]]


def build_reranker(cfg):
    """Pick the reranker backend from config.yaml.

    "cross_encoder" is the real neural one (bge-reranker via
    sentence-transformers). It degrades to the lexical fallback with a
    loud log line when the package or the model can't load, so an
    offline box still reranks instead of crashing. "lexical" picks the
    fallback directly. Anything unrecognised -> lexical — a typo in
    config shouldn't take the service down.
    """
    model = cfg.get("rerank_model", "BAAI/bge-reranker-base")
    if cfg.get("rerank_backend", "cross_encoder") == "cross_encoder":
        try:
            r = CrossEncoderReranker(model)
            r.load()   # fail fast here: falls back below, not mid-request
            return r
        except Exception as e:  # noqa: BLE001 - any load failure means fall back
            print(f"reranker: cross-encoder unavailable ({e}) — "
                  "using lexical fallback")
    return LexicalReranker()


if __name__ == "__main__":
    docs = ["the refund window is 30 days for customers in india",
            "maternity leave is 26 weeks in india",
            "standard shipping takes 3 to 5 business days"]
    metas = [{"i": i} for i in range(3)]
    cands = [(d, 0.5, m) for d, m in zip(docs, metas)]

    # the interface both backends answer
    r = LexicalReranker()
    top = r.rerank("how long is the refund window", cands, top_n=2)
    assert top[0][0] == docs[0], "lexical reranker ranked wrong"
    assert top[0][1] > top[1][1], "lexical scores not descending"
    assert all(0.0 <= s <= 1.0 for _, s, _ in top), "scores out of [0, 1]"
    assert r.rerank("q", [], top_n=5) == [], "empty candidates must stay empty"
    print("lexical reranker OK (top-2 ordered, scores in [0, 1])")

    # the real one — needs the package + a downloaded model; offline it
    # must refuse cleanly and the factory must hand back the fallback
    if CrossEncoder is None:
        got = build_reranker({"rerank_model": "BAAI/bge-reranker-base"})
        assert isinstance(got, LexicalReranker)
        print("cross-encoder unavailable — factory fell back to lexical OK")
    else:
        print("CrossEncoder importable — rerank() smoke test skipped here "
              "(model download happens in the pipeline test instead)")
