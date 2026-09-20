"""Embeddings with zero heavy dependencies: hashing TF-IDF.

Yeah, it's a bag-of-words trick, not a neural model — that's deliberate.
It runs anywhere with no downloads, and the fit/transform interface is the
same shape a real embedder gives you, so the swap is a one-class change.

# SWAP (production): replace TfidfEmbedder with:
#   from sentence_transformers import SentenceTransformer
#   class STEmbedder:
#       def __init__(self, model="all-MiniLM-L6-v2"): self.m = SentenceTransformer(model)
#       def fit(self, docs): pass
#       def transform(self, texts): return self.m.encode(texts, normalize_embeddings=True)
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


if __name__ == "__main__":
    docs = ["the cat sat on the mat", "dogs are great pets", "the mat was red"]
    e = TfidfEmbedder(dim=512)
    e.fit(docs)
    v = e.transform(["cat on mat", "quantum physics"])
    sim_cat = float(v[0] @ e.transform(["the cat sat"])[0])
    sim_unrelated = float(v[0] @ e.transform(["quantum physics"])[0])
    print(f"similar: {sim_cat:.3f}, unrelated: {sim_unrelated:.3f}")
    assert sim_cat > sim_unrelated, "embedder broken"
    print("embedder OK")
