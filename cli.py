"""Demo CLI: builds the index from data/sample_docs.json and answers questions.

    python3 cli.py            # interactive Q&A
    python3 cli.py --demo     # 3 canned questions, with per-stage timings
    python3 cli.py --demo --retriever bm25   # same, through the BM25 backend
"""
import argparse
import json

from pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    # run the demo questions through either backend without editing config
    ap.add_argument("--retriever", choices=("dense", "bm25", "hybrid"),
                    default=None)
    args = ap.parse_args()

    rag = RAGPipeline(overrides={"retriever": args.retriever})
    print(f"retriever backend: {rag.cfg.get('retriever', 'dense')}")
    docs = json.load(open("data/sample_docs.json"))
    n = rag.index_documents(docs)
    print(f"indexed {n} chunks (version {rag.index_version})\n")

    questions = (["What is the refund window?",
                  "How long is maternity leave in India?",
                  "What does the warranty cover?"]
                 if args.demo else None)

    def ask(q):
        # demo nicety: everything is filtered to Indian policies so the
        # metadata pre-filter actually gets exercised
        res = rag.answer(q, filters={"country": "IN"})
        print(f"Q: {q}\nA: {res['answer']}")
        print(f"  ({res['latency_ms']}ms {res['breakdown']})")
        for c in res["citations"]:
            print(f"  [score {c['score']}] {c['text'][:90]}...")

    if questions:
        for q in questions:
            ask(q)
            print()
    else:
        print("Ask questions (empty line quits). Prefix 'nofilter:' to skip the country filter.")
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            filt = None if q.startswith("nofilter:") else {"country": "IN"}
            q = q.replace("nofilter:", "")
            res = rag.answer(q, filters=filt)
            print(f"A: {res['answer']}  ({res['latency_ms']}ms)")


if __name__ == "__main__":
    main()
