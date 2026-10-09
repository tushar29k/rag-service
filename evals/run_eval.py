"""Eval runner: recall@k + evidence coverage on the golden set,
plus the nDCG@5 lift reranking buys you, plus RAGAS-style judge scores
(faithfulness + answer relevance) on the answerable questions.

    python3 evals/run_eval.py                    # three-way comparison table
    python3 evals/run_eval.py --retriever hybrid # detail for one backend

What it checks, per question:
  recall@3        — did we retrieve the right document in the top 3?
  evidence        — do the retrieved chunks actually contain the answer keywords?
  ndcg@5          — how well is the top-5 ordered, rerank on vs off?
  faithfulness    — are the answer's claims supported by the retrieved
                    context? (mock judge — see evals/judge.py # SWAP)
  answer_relevance — does the answer address the question, judged via
                    reverse-generated questions? (mock judge)

Recall/evidence/nDCG assert on retrieval quality; the two judge scores
assert on the generation stage. The unanswerable golden questions are
skipped for the judge scores — they test refusal, not groundedness, and
a clean "I don't know" is trivially faithful.

Graded relevance (0/1/2) for nDCG comes straight from the golden
annotations: expected_doc + must_contain keywords, which already existed
before this eval. 2 = a chunk of the right doc containing the answer
keywords, 1 = the right doc without them, 0 = anything else. Fixed
annotations, so the lift number can't be gamed — re-running gives the
same grades.

The default run scores dense, bm25 and hybrid on the same 20-question
golden set, prints the three-way table, then the rerank lift per
backend, then the judge scores per backend — one command, no config
edits, same corpus for everything.
"""
import argparse
import json
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "evals")
from pipeline import RAGPipeline
from judge import faithfulness, answer_relevance

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


def _corpus(rag):
    # texts + metas of the full indexed corpus — every retriever hides
    # the texts somewhere different, metas is uniform across backends
    r = rag.retriever
    if hasattr(r, "texts"):
        texts = r.texts
    elif hasattr(r, "store"):
        texts = r.store.texts
    else:  # hybrid wraps the dense retriever's store
        texts = r._dense.store.texts
    return texts, r.metas


def _dcg(rels, k=5):
    return sum((2.0**r - 1.0) / math.log2(i + 2)
               for i, r in enumerate(rels[:k]))


def _grade_corpus(g, texts, metas):
    # one grade per corpus chunk, keyed by chunk hash — the answer()
    # citations carry the same hash, so ranked chunks map back cleanly
    if g["expected_doc"] is None:
        # deliberately unanswerable: every chunk is grade 0; the ideal
        # ranking is empty (the min_score gate should refuse everything)
        return {m["hash"]: 0 for m in metas}
    doc = g["expected_doc"]
    kws = [k.lower() for k in g["must_contain"]]
    return {m["hash"]: (2 if all(k in t.lower() for k in kws) else 1)
            if m.get("doc_id") == doc else 0
            for t, m in zip(texts, metas)}


def _ndcg_one(grades, ranked_hashes):
    ideal = sorted(grades.values(), reverse=True)
    idcg = _dcg(ideal)
    if idcg == 0:
        # unanswerable question: perfect means nothing survived the
        # gate, anything retrieved is a false positive
        return 1.0 if not ranked_hashes else 0.0
    ranked = [grades[h] for h in ranked_hashes]
    return _dcg(ranked) / idcg


def evaluate_rerank(name, golden):
    # same backend twice, rerank off vs on — the config's rerank
    # true/false semantics do the work, no pipeline surgery here
    docs = json.load(open("data/sample_docs.json"))
    off = RAGPipeline(overrides={"retriever": name, "rerank": False})
    on = RAGPipeline(overrides={"retriever": name, "rerank": True})
    off.index_documents(docs)
    on.index_documents(docs)
    reranker_name = (type(on.reranker).__name__
                     if on.reranker else "none")
    off_ndcg, on_ndcg = [], []
    for g in golden:
        grades = _grade_corpus(g, *_corpus(off))
        for rag, acc in ((off, off_ndcg), (on, on_ndcg)):
            res = rag.answer(g["question"], filters=g["filters"], top_k=5)
            ranked = [c["meta"]["hash"] for c in res["citations"]]
            acc.append(_ndcg_one(grades, ranked))
    return (sum(off_ndcg) / len(off_ndcg), sum(on_ndcg) / len(on_ndcg),
            reranker_name)


def evaluate_judge(name, golden):
    # answerable questions only — refusal is tested elsewhere, and a
    # clean "I don't know" is trivially faithful. same corpus, one
    # backend at a time, mock judge from evals/judge.py
    qa = [g for g in golden if g["expected_doc"] is not None]
    rag = RAGPipeline(overrides={"retriever": name})
    rag.index_documents(json.load(open("data/sample_docs.json")))
    f_scores, r_scores = [], []
    for g in qa:
        res = rag.answer(g["question"], filters=g["filters"], top_k=3)
        context = "\n\n".join(c["text"] for c in res["citations"])
        f_scores.append(faithfulness(res["answer"], context))
        r_scores.append(answer_relevance(g["question"], res["answer"]))
    n = len(qa)
    return (sum(f_scores) / n, sum(r_scores) / n, n)


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

    # rerank lift: same questions, same backends, top-5 ordered with and
    # without the reranker — the lift number is the whole point of the
    # rerank stage, so it gets its own section
    lifts = [(name,) + evaluate_rerank(name, golden) for name in BACKENDS]
    default = next(l for l in lifts if l[0] == "dense")
    print(f"\nrerank lift (reranker: {default[3]}, ndcg@5, {n} questions)")
    print(f"{'retriever':10s} {'no rerank':>9s} {'rerank':>6s} {'lift':>7s}")
    for name, off, on, _ in lifts:
        print(f"{name:10s} {off:>9.3f} {on:>6.3f} {on - off:>+7.3f}")
    _, off, on, _ = default
    print(f"rerank ndcg@5: {on:.3f} (lift {on - off:+.3f})")

    # ragas-style answer scores: mock LLM judge, answerable questions
    # only. faithfulness checks the answer against the context it was
    # generated from; answer relevance checks it against the question.
    # the extractive mock quotes its context, so expect faithfulness to
    # sit high — the metric's real job starts when a generative LLM swaps
    # in behind judge.py's # SWAP
    judges = [(name,) + evaluate_judge(name, golden) for name in BACKENDS]
    jn = judges[0][3]
    print(f"\njudge scores (mock judge, n={jn} answerable questions)")
    print(f"{'retriever':10s} {'faithfulness':>12s} {'answer_rel':>10s}")
    for name, f, r, _ in judges:
        print(f"{name:10s} {f:>12.3f} {r:>10.3f}")


if __name__ == "__main__":
    main()
