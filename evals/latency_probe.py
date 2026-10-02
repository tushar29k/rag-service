"""Latency probe: 100 queries -> logs/latency.jsonl -> p50/p99 table.

    python3 evals/latency_probe.py

Runs the 20 golden questions x 5 passes against the offline mock backend,
then prints the summary. The log file is truncated first so the table
reflects exactly these 100 queries — a fresh, comparable snapshot every
time. logs/ is gitignored; the JSONL stays local.
"""
import json
import os
import sys

sys.path.insert(0, ".")
from pipeline import RAGPipeline
from latency_summary import main as print_summary

N_PASSES = 5


def main():
    rag = RAGPipeline(overrides={"retriever": "dense", "generator": "mock"})
    rag.index_documents(json.load(open("data/sample_docs.json")))
    # fresh snapshot: wipe whatever previous runs logged
    log = rag.latency_log
    if log:
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        open(log, "w").close()
    golden = [json.loads(l) for l in open("evals/golden.jsonl")
              if l.strip()]
    n = 0
    for _ in range(N_PASSES):
        for g in golden:
            rag.answer(g["question"], filters=g["filters"], top_k=3)
            n += 1
    print(f"ran {n} queries against the mock backend")
    sys.argv = [sys.argv[0]]
    print_summary()


if __name__ == "__main__":
    main()
