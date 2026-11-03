"""Embeddings: hashing TF-IDF by default, sentence-transformers on request.

Hashing is a bag-of-words trick, not a neural model — that's deliberate.
It runs anywhere with no downloads, and the fit/transform interface is the
same shape a real embedder gives you.

The real embedder is wired in too: set `embedder_backend: st` in
config.yaml and build_embedder() lazy-loads sentence-transformers.
Any failure at import or model-load time falls back to hashing with a
loud log line, so an offline box never crashes.

# SWAP (production): use a bigger model via st_model in config.yaml,
# e.g. bge-base-en-v1.5 — no code change needed.
"""
import hashlib
import math
import numpy as np


def _tokenize(text):
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


class TfidfEmbedder:
    def __init__(self, dim=2048):
        self.dim = dim
        self.df = np.zeros(dim)   # document frequency per bucket
        self.n = 0                # documents seen
        self._idf = None

    @staticmethod
    def _bucket(token, dim):
        # md5, not Python's hash() — hash() is salted per process, md5 isn't,
        # so the same token lands in the same bucket on every run.
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % dim

    def fit(self, docs):
        """One pass over the corpus to learn IDF weights.

        Call this once before transform() — the assert will remind you
        if you forget.
        """
        for doc in docs:
            for tok in set(_tokenize(doc)):
                self.df[self._bucket(tok, self.dim)] += 1
            self.n += 1
        # smoothed IDF: words that show up everywhere get squashed,
        # rare words get boosted
        self._idf = np.log((1 + self.n) / (1 + self.df)) + 1

    def transform(self, texts):
        """Turn texts into L2-normalised float32 vectors, shape (n, dim).

        Normalised so that cosine similarity later is just a dot product.
        """
        assert self._idf is not None, "call fit() first"
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _tokenize(text)
            if not toks:
                continue
            tf = {}
            for tok in toks:
                b = self._bucket(tok, self.dim)
                tf[b] = tf.get(b, 0) + 1
            for b, c in tf.items():
                X[i, b] = (1 + math.log(c)) * self._idf[b]  # sublinear TF × IDF
            norm = np.linalg.norm(X[i])
            if norm:
                X[i] /= norm   # empty docs stay zero vectors — fine, they score nothing
        return X


class STEmbedder:
    """sentence-transformers behind the same fit/transform shape as
    TfidfEmbedder, so the pipeline never knows the difference."""

    def __init__(self, model="all-MiniLM-L6-v2"):
        # lazy import: the package is heavy and optional — embedder.py must
        # still import cleanly on a box that never heard of it
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_embedding_dimension()

    def fit(self, docs):
        pass  # no fitting to do — the model's already trained

    def transform(self, texts):
        # normalised the same way, so the store's cosine math doesn't change
        v = self._model.encode(texts, normalize_embeddings=True,
                               convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32)


def build_embedder(cfg):
    """Pick the embedder backend from config.yaml.

    "hashing" is the default: zero deps, works everywhere. "st" gives real
    neural embeddings — but if the import or the model load fails (offline
    box, no cached model), we fall back to hashing instead of crashing the
    service. The fallback is loud so nobody deploys st and silently gets
    hashing.
    """
    if cfg.get("embedder_backend") == "st":
        try:
            return STEmbedder(model=cfg.get("st_model", "all-MiniLM-L6-v2"))
        except Exception as e:  # noqa: BLE001 — any failure here means offline
            # fall back to hashing — a degraded embedder beats a dead service
            print(f"[embedder] st backend failed ({type(e).__name__}: {e}); "
                  "falling back to hashing")
    return TfidfEmbedder(dim=cfg.get("embed_dim", 2048))


if __name__ == "__main__":
    docs = ["the cat sat on the mat", "dogs are great pets", "the mat was red"]

    # backend 1: hashing — the default, always available
    e = TfidfEmbedder(dim=512)
    e.fit(docs)
    v = e.transform(["cat on mat", "quantum physics"])
    sim_cat = float(v[0] @ e.transform(["the cat sat"])[0])
    sim_unrelated = float(v[0] @ e.transform(["quantum physics"])[0])
    print(f"hashing: similar: {sim_cat:.3f}, unrelated: {sim_unrelated:.3f}")
    assert sim_cat > sim_unrelated, "hashing embedder broken"
    print("hashing backend OK")

    # backend 2: sentence-transformers — real if the package + model load,
    # graceful hashing fallback otherwise (that IS the tested behaviour offline)
    import os as _os
    for k in list(_os.environ):  # sandbox proxy vars confuse the hf client;
        if "proxy" in k.lower():  # harmless in prod, which won't have them
            _os.environ.pop(k)
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use the cached model if any

    e2 = build_embedder({"embedder_backend": "st",
                         "st_model": "all-MiniLM-L6-v2"})
    if isinstance(e2, TfidfEmbedder):
        # st didn't load here — assert the fallback is a working embedder
        e2.fit(docs)
        v2 = e2.transform(["cat on mat"])
        assert v2.shape == (1, 512)
        print("st unavailable offline — fell back to hashing, fallback OK")
    else:
        e2.fit(docs)  # no-op by design
        v2 = e2.transform(["cat on mat", "quantum physics"])
        assert v2.shape == (2, e2.dim), f"unexpected shape {v2.shape}"
        s_cat = float(v2[0] @ e2.transform(["the cat sat"])[0])
        s_unrelated = float(v2[0] @ v2[1])
        print(f"st: similar: {s_cat:.3f}, unrelated: {s_unrelated:.3f}, "
              f"dim={e2.dim}")
        assert s_cat > s_unrelated, "st embedder broken"
        print("st backend OK")
