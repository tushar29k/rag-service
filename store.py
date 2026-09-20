"""Vector store: numpy brute-force + metadata pre-filtering + persistence.

Brute force is O(n·d) per query — fine to ~100K chunks, and perfect for
learning. The interface (upsert/search/save/load) is what Qdrant/pgvector
provide, so the pipeline never knows the difference.

# SWAP (production): implement the same 4 methods on QdrantClient /
# pinecone / pgvector. Only this file changes.
"""
import json
import numpy as np


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.emb = np.zeros((0, dim), dtype=np.float32)
        self.texts = []    # chunk texts, aligned with emb rows
        self.metas = []    # chunk metadata dicts, aligned with emb rows

    def upsert(self, texts, embs, metas):
        """Add chunks. Idempotent callers should dedupe BEFORE calling."""
        assert embs.shape[1] == self.dim
        self.emb = np.vstack([self.emb, embs]) if len(self.emb) else embs
        self.texts.extend(texts)
        self.metas.extend(metas)

    def _filtered_ids(self, filters):
        if not filters:
            return list(range(len(self.texts)))
        # PRE-filter: candidates outside the filter are never scored.
        # (Post-filtering — retrieve then discard — silently kills recall.)
        return [i for i, m in enumerate(self.metas)
                if all(m.get(k) == v for k, v in filters.items())]

    def search(self, query_vec, top_k=5, filters=None):
        """Returns [(text, score, meta)] sorted by cosine similarity."""
        ids = self._filtered_ids(filters)
        if not ids:
            return []
        sims = self.emb[ids] @ query_vec          # cosine (vectors normalised)
        order = np.argsort(-sims)[:top_k]
        return [(self.texts[ids[i]], float(sims[i]), self.metas[ids[i]])
                for i in order]

    def save(self, path):
        np.savez(path + ".npz", emb=self.emb)
        json.dump({"texts": self.texts, "metas": self.metas, "dim": self.dim},
                  open(path + ".json", "w"))

    @classmethod
    def load(cls, path):
        meta = json.load(open(path + ".json"))
        store = cls(meta["dim"])
        store.emb = np.load(path + ".npz")["emb"]
        store.texts, store.metas = meta["texts"], meta["metas"]
        return store

    def __len__(self):
        return len(self.texts)


if __name__ == "__main__":
    s = VectorStore(dim=4)
    s.upsert(["refund policy text", "leave policy text"],
             np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
             [{"dept": "sales"}, {"dept": "hr"}])
    res = s.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=5,
                   filters={"dept": "sales"})
    assert res[0][0] == "refund policy text" and len(res) == 1
    s.save("/tmp/test_store")
    s2 = VectorStore.load("/tmp/test_store")
    assert len(s2) == 2
    print("store OK")
