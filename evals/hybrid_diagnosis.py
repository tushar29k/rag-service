"""Why is hybrid worse than BM25 alone? Diagnose + fix, real numbers.

Yesterday's SQuAD slice (302 questions, 2067 paragraphs):
  dense  73.8% | bm25 86.8% | hybrid 82.1%   <- fusion HURTS by 4.7pts

This script:
  1. verifies the baseline reproduces,
  2. finds where fusion loses (per-question gold ranks per backend),
  3. sweeps hybrid_alpha to see if re-weighting recovers the loss.

No mocks, no invented data: the repo's own pipeline, retrievers and the
committed SQuAD slice. Deterministic — re-running gives identical numbers.

    python3 evals/hybrid_diagnosis.py
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "evals")

HERE = os.path.dirname(os.path.abspath(__file__))
SLICE_PATH = os.path.join(HERE, "squad_slice.json")

from pipeline import RAGPipeline  # noqa: E402

BASE_OVERRIDES = {"rerank": False, "rewrite": False,
                  "semantic_cache": {"enabled": False}, "latency_log": ""}

ALPHAS = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]
DEPTH = 20


def build(name, extra=None):
    ov = dict(BASE_OVERRIDES, retriever=name)
    if extra:
        ov.update(extra)
    return RAGPipeline(overrides=ov)


def gold_rank(retriever, question, gold_para_id, depth=DEPTH):
    """Best 1-based rank of any chunk from the gold paragraph, or None."""
    res = retriever.search(retriever.embed(question), top_k=depth)
    best = None
    for i, (_, _, meta) in enumerate(res, start=1):
        if meta.get("para_id") == gold_para_id:
            best = i
            break
    return best


def main():
    slice_meta = json.load(open(SLICE_PATH))
    questions = slice_meta["questions"]
    docs = [{"metadata": {"doc_id": p["para_id"], "para_id": p["para_id"],
                          "title": p["title"], "source": "squad-v1.1-dev",
                          "doc_type": "encyclopedia"},
             "text": p["context"]}
            for p in slice_meta["paragraphs"]]
    print("corpus: %d paragraphs, questions: %d"
          % (len(docs), len(questions)), flush=True)

    pipes = {}
    for name in ("dense", "bm25", "hybrid"):
        print("indexing %s ..." % name, flush=True)
        pipes[name] = build(name)
        pipes[name].index_documents(docs)

    # per-question gold ranks, one pass per backend
    ranks = {}
    for name in ("dense", "bm25"):
        r = pipes[name].retriever
        ranks[name] = [gold_rank(r, q["question"], q["para_id"])
                       for q in questions]
    # hybrid at alpha=0.5, then the sweep reuses the same index
    hr = pipes["hybrid"].retriever
    ranks["hybrid_0.5"] = [gold_rank(hr, q["question"], q["para_id"])
                           for q in questions]
    alpha_hits = {}
    for a in ALPHAS:
        hr.alpha = a
        hits = sum(1 for q in questions
                   if (rk := gold_rank(hr, q["question"], q["para_id"]))
                   is not None and rk <= 3)
        alpha_hits[a] = hits

    def recall(key):
        hits = sum(1 for rk in ranks[key] if rk is not None and rk <= 3)
        return hits

    n = len(questions)
    rd, rb, rh = recall("dense"), recall("bm25"), recall("hybrid_0.5")
    print("\n== baseline (verify vs yesterday) ==")
    print("dense  %d/%d (%.1f%%)   [yesterday 223/302 73.8%%]"
          % (rd, n, 100.0 * rd / n))
    print("bm25   %d/%d (%.1f%%)   [yesterday 262/302 86.8%%]"
          % (rb, n, 100.0 * rb / n))
    print("hybrid %d/%d (%.1f%%)   [yesterday 248/302 82.1%%]"
          % (rh, n, 100.0 * rh / n))

    hit = lambda rk: rk is not None and rk <= 3  # noqa: E731
    bm25_hit = [hit(rk) for rk in ranks["bm25"]]
    hyb_hit = [hit(rk) for rk in ranks["hybrid_0.5"]]
    both = sum(1 for b, h in zip(bm25_hit, hyb_hit) if b and h)
    bm25_only = sum(1 for b, h in zip(bm25_hit, hyb_hit) if b and not h)
    hyb_only = sum(1 for b, h in zip(bm25_hit, hyb_hit) if h and not b)
    neither = sum(1 for b, h in zip(bm25_hit, hyb_hit) if not b and not h)
    print("\n== bm25 vs hybrid(alpha=0.5) contingency ==")
    print("both hit: %d | bm25-only: %d | hybrid-only: %d | neither: %d"
          % (both, bm25_only, hyb_only, neither))
    print("net fusion effect: %+d  (hybrid-only minus bm25-only)" %
          (hyb_only - bm25_only))

    # where does the gold sit in the dense list on the bm25-only questions?
    print("\n== dense rank of gold on the %d bm25-only questions ==" %
          bm25_only)
    buckets = {"1-3": 0, "4-10": 0, "11-20": 0, "absent(top20)": 0}
    examples = []
    for q, b, h, rd_, rb_ in zip(questions, bm25_hit, hyb_hit,
                                 ranks["dense"], ranks["bm25"]):
        if b and not h:
            if rd_ is None:
                buckets["absent(top20)"] += 1
            elif rd_ <= 3:
                buckets["1-3"] += 1
            elif rd_ <= 10:
                buckets["4-10"] += 1
            else:
                buckets["11-20"] += 1
            if len(examples) < 3:
                examples.append((q["question"][:80], q["title"], rb_, rd_))
    for k, v in buckets.items():
        print("  dense rank %-12s : %d" % (k, v))
    print("\n  examples (question | article | bm25 rank | dense rank):")
    for qtext, title, rbm, rdn in examples:
        print("  - %s | %s | bm25 #%s | dense #%s"
              % (qtext, title, rbm, rdn))

    print("\n== hybrid_alpha sweep (recall@3) ==")
    for a in ALPHAS:
        h = alpha_hits[a]
        tag = ""
        if a == 0.5:
            tag = "  <- default"
        print("  alpha=%.2f  %d/%d (%.1f%%)%s"
              % (a, h, n, 100.0 * h / n, tag))
    best = max(ALPHAS, key=lambda a: alpha_hits[a])
    print("\nbest alpha=%.2f -> %d/%d (%.1f%%), delta vs default 0.5: %+d"
          % (best, alpha_hits[best], n, 100.0 * alpha_hits[best] / n,
             alpha_hits[best] - alpha_hits[0.5]))
    print("delta vs pure bm25: %+d" % (alpha_hits[best] - rb))


if __name__ == "__main__":
    main()
