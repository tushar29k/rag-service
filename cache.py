"""SemanticCache: exact-match + embedding-similarity cache for /query.

Two tiers. Exact: sha of the normalized question — repeats are instant.
Semantic: cosine over hashing bag-of-words vectors — paraphrases that
clear `threshold` answer from cache too. Misses just run the pipeline.

# SWAP (production): swap _embed for a real embedder (the pipeline's
# TfidfEmbedder / sentence-transformers) when one is configured — the
# interface is one function text -> sparse vector, cosine over those.
"""
import hashlib
import math
import re

# buckets for the hashing vectors — small on purpose: this is a
# near-duplicate check, not a retrieval model
_DIM = 256


def _normalize(text):
    # case, punctuation and spacing all fold away, so "What is X?!"
    # and "what is x" hit the same cache entry
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _embed(text):
    # hashing bag-of-words: zero deps, microseconds per query. good
    # enough to catch paraphrases; the real retriever still owns
    # retrieval quality, this just spots "asked this before"
    vec = {}
    for tok in _normalize(text).split():
        # md5, not hash() — hash() is salted per process, so vectors
        # wouldn't line up across workers
        b = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _DIM
        vec[b] = vec.get(b, 0) + 1
    return vec


def _cosine(a, b):
    # sparse dot product — most buckets are empty, skip them
    num = sum(w * b.get(k, 0) for k, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return num / (na * nb) if na and nb else 0.0


class SemanticCache:
    def __init__(self, threshold=0.85, max_entries=256):
        self.threshold = threshold
        self.max_entries = max_entries
        # key -> (query vector, cached answer dict). insertion-ordered,
        # so eviction is plain FIFO
        self._entries = {}
        self.hits = self.misses = self.exact_hits = self.semantic_hits = 0

    def lookup(self, query):
        key = hashlib.sha256(_normalize(query).encode()).hexdigest()[:16]
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
            self.exact_hits += 1
            return entry[1]
        # no exact hit — best cosine over stored queries, threshold gate
        qvec = _embed(query)
        best_resp, best_sim = None, 0.0
        for vec, resp in self._entries.values():
            sim = _cosine(qvec, vec)
            if sim > best_sim:
                best_resp, best_sim = resp, sim
        if best_resp is not None and best_sim >= self.threshold:
            self.hits += 1
            self.semantic_hits += 1
            return best_resp
        self.misses += 1
        return None

    def store(self, query, response):
        # cap memory — oldest entry evicted first, the cache shouldn't
        # grow without bound on a busy service
        if self.max_entries and len(self._entries) >= self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        key = hashlib.sha256(_normalize(query).encode()).hexdigest()[:16]
        self._entries[key] = (_embed(query), response)

    def clear(self):
        # re-indexing can change what a question grounds to — a stale
        # cached answer is worse than a miss, so rebuilds empty the cache
        self._entries.clear()

    def stats(self):
        # the /health shape — hit_rate is the number the roadmap item
        # wants visible, everything else is the debugging trail
        total = self.hits + self.misses
        return {"enabled": True, "threshold": self.threshold,
                "hits": self.hits, "misses": self.misses,
                "exact_hits": self.exact_hits,
                "semantic_hits": self.semantic_hits,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "size": len(self._entries)}
