"""Eval runner: recall@k + evidence coverage on the golden set.

    python3 evals/run_eval.py

Metrics:
  recall@3  — is the right document among the top-3 retrieved?
  evidence  — do the retrieved chunks contain the answer keywords?

Why "evidence" and not answer faithfulness? This project's generator is an
extractive mock (see pipeline.py # SWAP). Answer-level faithfulness needs a
real LLM (or an LLM judge) — what THIS pipeline owns is retrieval quality,
so that's what we assert on. When you swap in a real LLM, add an
LLM-as-judge check: "is the answer supported by the citations?"
"""
import json
import sys

sys.path.insert(0, ".")
from pipeline import RAGPipeline


def main():
    rag = RAGPipeline()
    rag.index_documents(json.load(open("data/sample_docs.json")))
    golden = [json.loads(l) for l in open("evals/golden.jsonl")]

    recall_hits, evidence_hits, latencies = 0, 0, []
    print(f"{'question':45s} {'recall':6s} {'evidence':8s} {'ms':>6s}")
    for g in golden:
        res = rag.answer(g["question"], filters=g["filters"], top_k=3)
        docs = [c["meta"].get("doc_id") for c in res["citations"]]
        rec = g["expected_doc"] in docs if g["expected_doc"] else True
        if g["expected_doc"] is None:
            # unanswerable question: the system must refuse, not retrieve
            ev = "don't know" in res["answer"].lower()
        else:
            context = " ".join(c["text"] for c in res["citations"]).lower()
            ev = all(kw.lower() in context for kw in g["must_contain"])
        recall_hits += rec
        evidence_hits += ev
        latencies.append(res["latency_ms"])
        print(f"{g['question'][:45]:45s} {str(rec):6s} {str(ev):8s} "
              f"{res['latency_ms']:6.1f}")
    n = len(golden)
    print(f"\nrecall@3: {recall_hits}/{n}   evidence: {evidence_hits}/{n}   "
          f"p50: {sorted(latencies)[n//2]:.1f}ms")


if __name__ == "__main__":
    main()
