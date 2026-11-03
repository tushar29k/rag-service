"""SQuAD-format QA slice: the standard exact-match / F1 machinery,
run against this service's full answer path.

    python3 evals/run_squad_qa.py

The slice (evals/squad_qa_slice.json, built by
evals/make_squad_qa_slice.py) is 20 hand-written questions in the
SQuAD v1.1 shape — small on purpose, and labelled a slice wherever
it is reported. It complements evals/squad_demo.py: that one runs
302 real SQuAD questions and grades RETRIEVAL (is the source
paragraph in the top 3?); this one grades the ANSWER PATH end to end
against gold answer spans, the way SQuAD itself is scored.

Per question, per backend (all 10 passages indexed together, so
retrieval has to find the right passage among the nine wrong ones):

  retrieval hit  the gold passage is among the citations
  exact match    the gold span appears verbatim in the answer
                 (SQuAD normalisation: lowercase, punctuation and
                 articles stripped) — for an extractive generator
                 this is the honest headline
  token F1       SQuAD-style F1 between the gold span and the
                 answer's best-matching sentence. Sentence-level,
                 because the pipeline's answer is quoted sentences
                 plus a boilerplate note, not a bare span; scoring
                 the whole answer blob would grade the boilerplate,
                 not the extraction.
  refused        the pipeline answered don't-know on a question its
                 corpus plainly supports — a slice failure, counted

Deterministic (mock generator forced, hashing/BM25 backends), and
the results file (evals/squad-qa-results.md) regenerates
byte-identically.
"""
import json
import re
import string
import sys
from collections import Counter

sys.path.insert(0, ".")
sys.path.insert(0, "evals")
from pipeline import RAGPipeline

BACKENDS = ("dense", "bm25", "hybrid")
OVERRIDES = {"generator": "mock", "semantic_cache": False}


def _norm(text):
    # the official SQuAD normalisation: lowercase, drop punctuation,
    # drop articles, collapse whitespace
    text = text.lower()
    text = "".join(c for c in text if c not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(gold, pred):
    gold_toks, pred_toks = _norm(gold).split(), _norm(pred).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    hits = sum(common.values())
    if not hits or not gold_toks or not pred_toks:
        return 0.0
    p, r = hits / len(pred_toks), hits / len(gold_toks)
    return 2 * p * r / (p + r)


def _best_sentence_f1(gold, answer):
    sentences = [s for s in answer.replace("!", ".").split(".")
                 if s.strip()]
    return max((_f1(gold, s) for s in sentences), default=0.0)


def _load_items():
    raw = json.load(open("evals/squad_qa_slice.json"))
    items = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                ans = qa["answers"][0]
                start = ans["answer_start"]
                assert para["context"][start:start + len(ans["text"])] \
                    == ans["text"], f"{qa['id']}: span offset drift"
                items.append({"id": qa["id"], "title": article["title"],
                              "context": para["context"],
                              "question": qa["question"],
                              "gold": ans["text"]})
    return raw["version"], items


def _evaluate(backend, items):
    rag = RAGPipeline(overrides={"retriever": backend, **OVERRIDES})
    docs, seen = [], set()
    for it in items:
        if it["title"] not in seen:
            seen.add(it["title"])
            docs.append({"text": it["context"],
                         "metadata": {"doc_id": it["title"]}})
    rag.index_documents(docs)
    rows = []
    for it in items:
        res = rag.answer(it["question"], top_k=3)
        cited = {c["meta"].get("doc_id") for c in res["citations"]}
        rows.append({
            "id": it["id"], "title": it["title"],
            "hit": it["title"] in cited,
            "em": _norm(it["gold"]) in _norm(res["answer"]),
            "f1": _best_sentence_f1(it["gold"], res["answer"]),
            "refused": "don't know" in res["answer"].lower(),
        })
    return rows


def main():
    version, items = _load_items()
    n = len(items)
    lines = [
        "# SQuAD-format QA slice — results",
        "",
        f"Slice version `{version}`: {n} hand-written questions over "
        "10 passages in SQuAD v1.1 format (evals/squad_qa_slice.json). "
        "A slice, not the SQuAD dataset — the full-scale retrieval "
        "slice over real SQuAD data is evals/squad_demo.py. Scored by "
        "evals/run_squad_qa.py against the full answer path with the "
        "extractive mock generator.",
        "",
        "| backend | retrieval hit | exact match | mean token F1 | "
        "refused |",
        "|---|---|---|---|---|",
    ]
    detail = ["", "Per-question detail (hit / EM / F1):", ""]
    all_rows = {}
    for name in BACKENDS:
        rows = _evaluate(name, items)
        all_rows[name] = {r["id"]: r for r in rows}
        hits = sum(r["hit"] for r in rows)
        ems = sum(r["em"] for r in rows)
        f1 = sum(r["f1"] for r in rows) / n
        refused = sum(r["refused"] for r in rows)
        print(f"{name:7s} hit {hits}/{n} · EM {ems}/{n} · "
              f"F1 {f1:.3f} · refused {refused}")
        lines.append(f"| {name} | {hits}/{n} | {ems}/{n} | {f1:.3f} | "
                     f"{refused} |")
    detail.append("| id | passage | " +
                  " | ".join(f"{b} hit · EM · F1" for b in BACKENDS) +
                  " |")
    detail.append("|---|---|" + "---|" * len(BACKENDS))
    for it in items:
        cells = []
        for name in BACKENDS:
            r = all_rows[name][it["id"]]
            cells.append(f"{'✓' if r['hit'] else '✗'} · "
                         f"{'✓' if r['em'] else '✗'} · {r['f1']:.2f}")
        detail.append(f"| {it['id']} | {it['title']} | " +
                      " | ".join(cells) + " |")
    with open("evals/squad-qa-results.md", "w") as f:
        f.write("\n".join(lines + detail) + "\n")
    print("\nwrote evals/squad-qa-results.md")


if __name__ == "__main__":
    main()
