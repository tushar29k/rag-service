"""Sweep min_score and record the precision/recall trade-off.

    python3 evals/threshold_sweep.py                 # full sweep, all backends
    python3 evals/threshold_sweep.py --out evals/threshold-sweep.md

What it measures, per (backend, threshold):
  precision — of the questions the system answered (non-empty retrieval),
              how many were answered *correctly*
  recall    — of the 18 answerable questions, how many were answered correctly
  f1        — harmonic mean, the thing to maximize when picking a threshold
  refused   — questions where nothing survived the gate (out of 20)
  evidence  — must_contain keywords in the retrieved chunks (out of 18)

A correct answer = the expected_doc is in the top-3 citations. Answering an
unanswerable question, or answering off the wrong doc, is a false positive;
refusing an answerable question is a false negative. The 2 unanswerable
golden questions are the only false-positive sources this corpus has, so
precision only bites when the gate is loose enough to answer them.

min_score is NOT portable across backends — 0.15 means different things for
hashing-cosine (dense), normalized BM25, and RRF-fused (hybrid) scores. The
sweep makes that visible instead of pretending one number fits all.

One pipeline per (backend, threshold): the hybrid retriever snapshots
min_score at construction for its sub-list pre-gate, so cfg edits after
construction wouldn't move it. Cache and latency logging are off for the
sweep — cached answers would freeze the first threshold's verdicts, and 900
timed queries don't belong in logs/latency.jsonl.
"""
import argparse
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "evals")

from pipeline import RAGPipeline

BACKENDS = ("dense", "bm25", "hybrid")
# wide enough to cover every backend's score scale: dense lives under
# ~0.25, bm25 under ~0.7, hybrid clusters at 0.96-1.0 (RRF saturation)
GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def _verdict(rag, g):
    # citations empty = the gate refused -> the generator says "I don't
    # know". everything downstream keys off that, not the answer text
    res = rag.answer(g["question"], filters=g["filters"], top_k=3)
    docs = [c["meta"].get("doc_id") for c in res["citations"]]
    answered = bool(docs)
    if g["expected_doc"] is None:
        return ("fp", None) if answered else ("tn", None)
    correct = g["expected_doc"] in docs
    ev = None
    if answered:
        context = " ".join(c["text"] for c in res["citations"]).lower()
        ev = all(kw.lower() in context for kw in g["must_contain"])
    if not answered:
        return ("fn", ev)
    return (("tp", ev) if correct else ("fp", ev))


def sweep_backend(name, golden, docs):
    rows = []
    for t in GRID:
        rag = RAGPipeline(overrides={
            "retriever": name,
            "min_score": t,
            # fresh verdicts every threshold: no cache, no latency spam
            "semantic_cache": {"enabled": False},
            "latency_log": None,
        })
        rag.index_documents(docs)
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        evidence = 0
        n_ans = sum(1 for g in golden if g["expected_doc"] is not None)
        for g in golden:
            v, ev = _verdict(rag, g)
            counts[v] += 1
            if ev:
                evidence += 1
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision and (precision + recall) else 0.0)
        rows.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "refused": fn + counts["tn"],
            "evidence": evidence,
            "n_ans": n_ans,
            "tp": tp, "fp": fp, "fn": fn, "tn": counts["tn"],
        })
    return rows


def _fmt(v):
    return f"{v:.3f}" if v is not None else "   –  "


def print_table(name, rows):
    print(f"\nbackend: {name}")
    print(f"{'min_score':>9s} {'precision':>9s} {'recall':>6s} "
          f"{'f1':>6s} {'refused':>7s} {'evidence':>8s}")
    for r in rows:
        print(f"{r['threshold']:>9.2f} {_fmt(r['precision']):>9s} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['refused']:>3d}/20 {r['evidence']:>3d}/{r['n_ans']:<4d}")


def md_report(results, golden_n):
    # results: {backend: [rows]}. markdown mirror of the stdout tables,
    # plus the recommendation the sweep exists to produce
    lines = [
        "# Threshold sweep — min_score precision/recall trade-off",
        "",
        "Date: 2026-10-11. Corpus: data/sample_docs.json (4 fixture policy "
        "docs), golden set: evals/golden.jsonl (18 answerable + 2 "
        "unanswerable), top_k=3, hashing embeddings + mock generator, "
        "rerank/cache/latency-logging off. Grid: " +
        ", ".join(f"{t:.2f}" for t in GRID) + ".",
        "",
        "A correct answer = expected_doc in the top-3 citations. False "
        "positive = answered an unanswerable question or answered off the "
        "wrong doc. False negative = refused an answerable question.",
        "",
    ]
    for name, rows in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| min_score | precision | recall | f1 | refused | "
                     "evidence |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            p = f"{r['precision']:.3f}" if r["precision"] is not None else "–"
            lines.append(f"| {r['threshold']:.2f} | {p} | {r['recall']:.3f} | "
                         f"{r['f1']:.3f} | {r['refused']}/20 | "
                         f"{r['evidence']}/{r['n_ans']} |")
        best = max(rows, key=lambda r: (r["f1"], r["threshold"]))
        lines.append("")
        lines.append(f"**Best F1: {best['f1']:.3f} at min_score={best['threshold']:.2f}** "
                     f"(precision { _fmt(best['precision']).strip() }, recall "
                     f"{best['recall']:.3f}, refused {best['refused']}/20).")
        lines.append("")
    lines.append("## Findings")
    lines.append("")
    for name, rows in results.items():
        best = max(rows, key=lambda r: (r["f1"], r["threshold"]))
        worst_lo = rows[0]
        lines.append(
            f"- {name}: best F1 {best['f1']:.3f} at {best['threshold']:.2f}; "
            f"at 0.00 precision {_fmt(worst_lo['precision']).strip()} / "
            f"recall {worst_lo['recall']:.3f} (gate off: answers everything, "
            "including the 2 unanswerable questions).")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "min_score is backend-specific — one number does not fit all. "
        "Tune per retriever backend on your own corpus; this sweep is the "
        "procedure. Config default stays 0.15 until the neural-embedding "
        "swap lands, then re-run this script and take the new best-F1 "
        "threshold (ties go to the higher, more conservative threshold).")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- 20-question fixture set: directionally right, not statistically "
        "powerful. Only 2 unanswerable questions drive all precision "
        "differences — add more adversarial/near-miss questions before "
        "trusting small precision gaps.")
    lines.append(
        "- Hybrid (RRF) scores saturate near 1.0, so the final min_score "
        "gate barely bites it — its refusals come from the per-sub-list "
        "pre-gate inside HybridRetriever, not the pipeline filter.")
    lines.append(
        "- Hashing TF-IDF scores are bag-of-words; the neural embedding "
        "backend will shift every curve — re-tune after the swap.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="write the markdown sweep report (default: print "
                         "tables only)")
    ap.add_argument("--backend", choices=BACKENDS, default=None,
                    help="sweep one backend only")
    args = ap.parse_args()
    golden = [json.loads(l) for l in open("evals/golden.jsonl")]
    docs = json.load(open("data/sample_docs.json"))
    backends = (args.backend,) if args.backend else BACKENDS
    results = {}
    for name in backends:
        print(f"sweeping {name} ...", file=sys.stderr, flush=True)
        rows = sweep_backend(name, golden, docs)
        results[name] = rows
        print_table(name, rows)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md_report(results, len(golden)) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
