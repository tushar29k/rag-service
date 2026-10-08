"""Vector store: numpy brute-force + metadata pre-filtering + persistence.

Brute force is O(n·d) per query, which sounds bad but is honestly fine up
to ~100K chunks — and perfect for learning, because there's nowhere for a
bug to hide. The interface (upsert/search/save/load) mirrors what
Qdrant/pgvector give you, so the pipeline never knows the difference.

# SWAP (production): implement the same 4 methods on QdrantClient /
# pinecone / pgvector. Only this file changes.
"""
import json
import numpy as np


# mongo-style range operators the filter language understands
_RANGE_OPS = {
    "$eq": lambda v, t: v == t,
    "$ne": lambda v, t: v != t,
    "$gt": lambda v, t: v is not None and v > t,
    "$gte": lambda v, t: v is not None and v >= t,
    "$lt": lambda v, t: v is not None and v < t,
    "$lte": lambda v, t: v is not None and v <= t,
}


def match_metadata(meta, filters):
    """True when a chunk's metadata satisfies every filter (ANDed).

    Filter values are exact matches, or mongo-style operator dicts:
        {"dept": "sales"}                          exact match
        {"dept": "sales", "year": 2024}             multi-filter (AND)
        {"year": {"$gte": 2020, "$lte": 2023}}      range filter
    A dict with $-keys is operators; anything else is exact. Missing
    keys never satisfy a range filter ($ne matches them, mongo-style).
    """
    for key, cond in filters.items():
        val = meta.get(key)
        if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
            for op, target in cond.items():
                if op not in _RANGE_OPS:
                    raise ValueError(f"unknown filter operator: {op}")
                if not _RANGE_OPS[op](val, target):
                    return False
        elif val != cond:
            return False
    return True


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.emb = np.zeros((0, dim), dtype=np.float32)
        # texts and metas stay row-aligned with emb — row i is always the
        # same chunk in all three, don't break that invariant
        self.texts = []
        self.metas = []

    def upsert(self, texts, embs, metas):
        """Add chunks to the index.

        Not deduped here — if you index the same doc twice you get it
        twice. Callers dedupe first (pipeline.py does it with content
        hashes).
        """
        assert embs.shape[1] == self.dim
        self.emb = np.vstack([self.emb, embs]) if len(self.emb) else embs
        self.texts.extend(texts)
        self.metas.extend(metas)

    def _filtered_ids(self, filters):
        if not filters:
            return list(range(len(self.texts)))
        # Filter FIRST, score second. The tempting alternative — retrieve
        # then throw away — silently tanks recall when the filter would
        # have removed most of the top-k.
        return [i for i, m in enumerate(self.metas)
                if match_metadata(m, filters)]

    def search(self, query_vec, top_k=5, filters=None):
        """Top-k chunks as (text, score, meta) tuples, best first."""
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
    s.upsert(["refund policy text", "leave policy text", "shipping policy text"],
             np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                      dtype=np.float32),
             [{"dept": "sales", "year": 2022},
              {"dept": "hr", "year": 2021},
              {"dept": "sales", "year": 2024}])
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    texts = lambda res: [t for t, _, _ in res]
    # exact match still works
    assert texts(s.search(q, top_k=5, filters={"dept": "sales"})) == \
        ["refund policy text", "shipping policy text"]
    # multi-filter: every condition ANDed
    assert texts(s.search(q, top_k=5,
                          filters={"dept": "sales", "year": 2022})) == \
        ["refund policy text"]
    # range filters: mongo-style $gte/$lte
    assert texts(s.search(q, top_k=5, filters={"year": {"$gte": 2022}})) == \
        ["refund policy text", "shipping policy text"]
    assert texts(s.search(q, top_k=5,
                          filters={"year": {"$gte": 2022, "$lte": 2023}})) == \
        ["refund policy text"]
    # range + exact combined
    assert texts(s.search(q, top_k=5, filters={"dept": "sales",
                                              "year": {"$lt": 2023}})) == \
        ["refund policy text"]
    # missing keys never satisfy a range filter...
    assert s.search(q, top_k=5, filters={"region": {"$gte": 1}}) == []
    # ...but $ne keeps docs missing the key, mongo-style
    assert len(s.search(q, top_k=5, filters={"dept": {"$ne": "hr"}})) == 2
    try:
        s.search(q, top_k=5, filters={"year": {"$between": [2020, 2023]}})
        raise AssertionError("unknown operator should fail loud")
    except ValueError:
        pass
    s.save("/tmp/test_store")
    s2 = VectorStore.load("/tmp/test_store")
    assert len(s2) == 3
    print("store OK")
