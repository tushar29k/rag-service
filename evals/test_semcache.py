"""Proves the semantic cache tiers: exact repeat = hit, paraphrase =
semantic hit, fresh query = miss, /health hit_rate > 0 after repeats."""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
from pipeline import RAGPipeline  # noqa: E402

pipe = RAGPipeline(overrides={"generator": "mock", "latency_log": None})
assert pipe.cache is not None, "cache should be enabled from config.yaml"

docs = json.load(open("data/sample_docs.json"))
pipe.index_documents([{"text": d["text"], "metadata": d.get("metadata", {})}
                      for d in docs])
print("chunks indexed:", len(pipe.retriever))

q1 = "what is the refund policy"
r1 = pipe.answer(q1)
assert r1["cached"] is False, "first ask must be a miss"
print("q1 (fresh)      -> cached =", r1["cached"])

r2 = pipe.answer(q1)
assert r2["cached"] is True, "exact repeat must hit"
print("q1 (repeat)     -> cached =", r2["cached"])

r3 = pipe.answer("WHAT is the Refund Policy?!")  # normalization test
assert r3["cached"] is True, "case/punct variant must exact-hit"
print("q1 (recased)    -> cached =", r3["cached"])

# paraphrase: one extra word — cosine 4/sqrt(4*5)=0.894 >= 0.85 threshold
para = "what is the refund policy really"
r4 = pipe.answer(para)
assert r4["cached"] is True, "near-duplicate must be a semantic hit"
print("paraphrase      -> cached =", r4["cached"])

fresh = "how long is the warranty on electronics"
r5 = pipe.answer(fresh)
assert r5["cached"] is False, "unrelated question must miss"
print("fresh question  -> cached =", r5["cached"])

st = pipe.cache.stats()
print("stats:", st)
assert st["exact_hits"] >= 2, "want >= 2 exact hits"
assert st["semantic_hits"] >= 1, "want >= 1 semantic hit"
assert st["hit_rate"] > 0, "hit_rate must be > 0"
assert st["misses"] >= 2, "want the fresh asks counted as misses"

# re-indexing clears the cache — grounded answers may have changed
n = pipe.index_documents([{"text": "A new document about hybrid retrieval systems.",
                           "metadata": {}}])
r6 = pipe.answer(q1)
assert r6["cached"] is False, "post-index repeat must be a miss (cache cleared)"
print("after re-index  -> cached =", r6["cached"],
      "(new chunks:", n, ")")

print("ALL CACHE CHECKS PASSED")
