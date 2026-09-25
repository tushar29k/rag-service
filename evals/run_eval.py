"""Eval runner: recall@k + evidence coverage on the golden set.

    python3 evals/run_eval.py                    # three-way comparison table
    python3 evals/run_eval.py --retriever hybrid # detail for one backend

What it checks, per question:
  recall@3  — did we retrieve the right document in the top 3?
  evidence  — do the retrieved chunks actually contain the answer keywords?

Why "evidence" and not answer faithfulness? The generator here is an
extractive mock (see pipeline.py # SWAP), so answer-level faithfulness
wouldn't tell us anything real. What this pipeline actually owns is
retrieval quality — that's what we assert on. Once a real LLM goes in,
add an LLM-as-judge check on top: "is the answer supported by the
citations?"

The default run scores dense, bm25 and hybrid on the same 20-question
golden set and prints the three-way table — one command, no config
edits, same corpus for all three.
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
from pipeline import RAGPipeline

BACKENDS = ("dense", "bm25", "hybrid")


def _score_one(rag, g):
    res = rag.answer(g["question"], filters=g["filters"], top_k=3)
    docs = [c["meta"].get("doc_id") for c in res["citations"]]
    rec = g["expected_doc"] in docs if g["expected_doc"] else True
    if g["expected_doc"] is None:
        # deliberately unanswerable question: the system must say it
        # doesn't know, not go rummaging for an answer anyway
        ev = "don't know" in res["answer"].lower()
    else:
        context = " ".join(c["text"] for c in res["citations"]).lower()
        ev = all(kw.lower() in context for kw in g["must_contain"])
    return rec, ev, res["latency_ms"]


def evaluate(name, golden):
    # one backend at a time, same corpus — the comparison is the point
    rag = RAGPipeline(overrides={"retriever": name})
    rag.index_documents(json.load(open("data/sample_docs.json")))
    scores = [_score_one(rag, g) for g in golden]
    recall = sum(1 for rec, _, _ in scores if rec)
    evidence = sum(1 for _, ev, _ in scores if ev)
    latencies = [ms for _, _, ms in scores]
    return recall, evidence, latencies


def detail(name, golden):
    rag = RAGPipeline(overrides={"retriever": name})
    print(f"retriever backend: {rag.cfg.get('retriever', 'dense')}")
    rag.index_documents(json.load(open("data/sample_docs.json")))
    print(f"{'question':45s} {'recall':6s} {'evidence':8s} {'ms':>6s}")
    for g in golden:
        rec, ev, ms = _score_one(rag, g)
        print(f"{g['question'][:45]:45s} {str(rec):6s} {str(ev):8s} "
              f"{ms:6.1f}")


def main():
    ap = argparse.ArgumentParser()
    # run the same golden set against one backend without editing config
    ap.add_argument("--retriever", choices=BACKENDS, default=None,
                    help="detail view for one backend; default compares "
                         "all three")
    args = ap.parse_args()
    golden = [json.loads(l) for l in open("evals/golden.jsonl")]
    n = len(golden)

    if args.retriever:
        detail(args.retriever, golden)
        return

    rows = [(name,) + evaluate(name, golden) for name in BACKENDS]
    print(f"golden set: {n} questions, recall@3 + evidence per backend")
    print(f"{'retriever':10s} {'recall@3':>8s} {'evidence':>8s} {'p50_ms':>7s}")
    for name, recall, evidence, latencies in rows:
        print(f"{name:10s} {recall:>3d}/{n:<4d} {evidence:>3d}/{n:<4d} "
              f"{sorted(latencies)[n // 2]:>7.1f}")


if __name__ == "__main__":
    main()
