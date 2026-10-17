"""SQuAD slice demo: index 2067 real SQuAD paragraphs through the repo's
own pipeline, answer 302 real questions per retriever backend, and commit
the numbers. This is the honest external-data check — the golden set is
hand-built, this one is not.

    python3 evals/squad_demo.py                # full run, writes results md
    python3 evals/squad_demo.py --limit 20      # quick smoke of the flow
    python3 evals/squad_demo.py --regenerate    # re-download the dev set and
                                               # rebuild evals/squad_slice.json

What it checks per backend (dense / bm25 / hybrid):
  recall@3 — is the question's own paragraph in the top-3 citations?
             "gold" is the exact SQuAD paragraph the question was written
             from, keyed by para_id carried in chunk metadata.

Timing is retrieval + mock generation per answer. The embedder here is the
zero-dependency hashing TF-IDF default (sentence-transformers isn't
installed in this sandbox — see the scope note in squad-demo-results.md).
No rerank, no rewrite, no semantic cache, no latency logging: the numbers
are the retrievers, unassisted.

The slice is committed at evals/squad_slice.json so the demo runs offline;
--regenerate re-downloads SQuAD v1.1 dev and rebuilds it deterministically
(every 35th question, spread across all articles).
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, ".")
sys.path.insert(0, "evals")
from pipeline import RAGPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
SLICE_PATH = os.path.join(HERE, "squad_slice.json")
RESULTS_PATH = os.path.join(HERE, "squad-demo-results.md")
SQUAD_URL = ("https://rajpurkar.github.io/SQuAD-explorer/"
             "dataset/dev-v1.1.json")
BACKENDS = ("dense", "bm25", "hybrid")


def regenerate_slice():
    # full dev set (~5MB), deterministic subset kept — a few hundred QA
    # pairs so the demo stays fast on the offline backends
    print("downloading SQuAD v1.1 dev set...")
    with urllib.request.urlopen(SQUAD_URL, timeout=120) as r:
        d = json.load(r)
    paras, qas = [], []
    for a in d["data"]:
        for pi, p in enumerate(a["paragraphs"]):
            para_id = "%s::%d" % (a["title"], pi)
            paras.append({"para_id": para_id, "title": a["title"],
                          "context": p["context"]})
            for qa in p["qas"]:
                if qa.get("is_impossible"):
                    continue
                qas.append({"id": qa["id"], "question": qa["question"],
                            "answers": [x["text"] for x in qa["answers"]],
                            "para_id": para_id, "title": a["title"]})
    sel = [q for i, q in enumerate(qas) if i % 35 == 0]
    with open(SLICE_PATH, "w") as f:
        json.dump({"dataset": "SQuAD v1.1 dev", "url": SQUAD_URL,
                   "slice_note": "every 35th question of %d (deterministic, "
                                 "spread across all articles)" % len(qas),
                   "paragraphs": paras, "questions": sel}, f)
    print("slice: %d questions, %d paragraphs -> %s"
          % (len(sel), len(paras), SLICE_PATH))


def run_backend(name, docs, questions):
    # each paragraph is one indexed doc; para_id rides in the metadata so
    # citations map back to the exact gold paragraph afterwards
    rag = RAGPipeline(overrides={"retriever": name, "rerank": False,
                                 "rewrite": False,
                                 "semantic_cache": {"enabled": False},
                                 # "" not None: the pipeline drops None
                                 # overrides, "" disables the latency log
                                 "latency_log": ""})
    n_chunks = rag.index_documents(docs)
    hits, latencies = 0, []
    t0 = time.perf_counter()
    for q in questions:
        t = time.perf_counter()
        res = rag.answer(q["question"], top_k=3)
        latencies.append((time.perf_counter() - t) * 1000)
        got = {c["meta"].get("para_id") for c in res["citations"]}
        if q["para_id"] in got:
            hits += 1
    wall = time.perf_counter() - t0
    latencies.sort()
    return {"hits": hits, "total": len(questions), "chunks": n_chunks,
            "p50_ms": round(latencies[len(latencies) // 2], 1),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
            "wall_s": round(wall, 1)}


def write_results(slice_meta, rows):
    n, paras = len(slice_meta["questions"]), len(slice_meta["paragraphs"])
    lines = [
        "# SQuAD slice demo results",
        "",
        "Date: %s" % time.strftime("%Y-%m-%d"),
        "Dataset: %s (%s)" % (slice_meta["dataset"], slice_meta["url"]),
        "Slice: %s" % slice_meta["slice_note"],
        "Questions evaluated: %d (overrides --limit when regenerating)" % n,
        "Corpus: %d SQuAD paragraphs, one doc per paragraph, chunk_size 120" % paras,
        "Embedder: hashing TF-IDF (the zero-dependency default)",
        "Generator: extractive mock (not scored here)",
        "",
        "## recall@3 per backend",
        "",
        "Gold = the exact paragraph the question was written from, retrieved",
        "in the top-3 citations (para_id carried in chunk metadata).",
        "",
        "| backend | recall@3 | p50 ms/query | p99 ms/query | wall s |",
        "|---------|----------|--------------|--------------|--------|",
    ]
    for name in BACKENDS:
        r = rows[name]
        lines.append("| %s | %d/%d (%.1f%%) | %.1f | %.1f | %.1f |"
                     % (name, r["hits"], r["total"],
                        100.0 * r["hits"] / r["total"],
                        r["p50_ms"], r["p99_ms"], r["wall_s"]))
    lines += [
        "",
        "## Scope notes (read before quoting these numbers)",
        "",
        "- The embedder is the hashing TF-IDF fallback, not sentence-",
        "  transformers — recall here measures retrieval mechanics and the",
        "  corpus, not embedding quality. Expect real embeddings to shift",
        "  dense/hybrid up.",
        "- Questions are asked cold with no context about which article",
        "  they came from; BM25's keyword matching has the natural edge on",
        "  SQuAD's entity-heavy questions.",
        "- Per-question latency includes the mock generation step, so it",
        "  reads high next to the retrieve-only timings in",
        "  evals/latency_probe.py.",
        "- Slice is deterministic (every 35th question) and committed at",
        "  evals/squad_slice.json — re-running gives the same 302 questions.",
    ]
    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("results written to %s" % RESULTS_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N questions (quick smoke)")
    ap.add_argument("--regenerate", action="store_true",
                    help="re-download SQuAD and rebuild the slice")
    args = ap.parse_args()

    if args.regenerate or not os.path.exists(SLICE_PATH):
        regenerate_slice()
    slice_meta = json.load(open(SLICE_PATH))
    questions = slice_meta["questions"][:args.limit or None]

    docs = [{"metadata": {"doc_id": p["para_id"], "para_id": p["para_id"],
                          "title": p["title"], "source": "squad-v1.1-dev",
                          "doc_type": "encyclopedia"},
             "text": p["context"]}
            for p in slice_meta["paragraphs"]]

    print("corpus: %d paragraphs, questions: %d" % (len(docs), len(questions)))
    rows = {}
    for name in BACKENDS:
        print("backend: %s ..." % name, flush=True)
        rows[name] = run_backend(name, docs, questions)
        r = rows[name]
        print("  recall@3 %d/%d (%.1f%%), p50 %.1fms, wall %.1fs"
              % (r["hits"], r["total"], 100.0 * r["hits"] / r["total"],
                 r["p50_ms"], r["wall_s"]))
    write_results(slice_meta, rows)


if __name__ == "__main__":
    main()
