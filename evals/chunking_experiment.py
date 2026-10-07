"""Chunking experiment: 4 strategies x the same queries.

    cd <repo root> && python3 evals/chunking_experiment.py

Builds a small fixture corpus (4 policy docs, ~400 words each, facts
sprinkled through multi-paragraph text), indexes it four times — once per
chunking strategy — and asks the SAME 8 questions through the offline
pipeline (hashing embedder, mock generator, dense retriever). Measures
recall@3 and evidence recall per strategy, declares a winner, and writes
the results table to evals/chunking-experiment.md.

Why a dedicated fixture instead of evals/golden.jsonl? The sample docs
average ~55 words — one chunk per doc under every strategy, so chunking
can't move the needle. The metric definitions (recall@3, evidence) are
the same ones evals/run_eval.py uses.
"""
import os
import sys

sys.path.insert(0, ".")
from pipeline import RAGPipeline

STRATEGIES = ["word", "recursive", "fixed_char", "sentence_window"]
CHUNK_SIZE, OVERLAP, TOP_K = 120, 20, 3
OUT_PATH = "evals/chunking-experiment.md"

CORPUS = [
    {"doc_id": "refund-policy", "text": """Our return policy is designed to be simple and fair. You have 30 days from the date of purchase to return any unused item in its original packaging for a full refund to your original payment method. To start a return, log in to your account and print a prepaid return label from the orders page. You must keep the original receipt for all returns, as we cannot process refunds without proof of purchase. Once the warehouse receives your item, inspection takes one to two business days, and the refund posts within five business days after that. Shipping fees are non-refundable in all cases, including when the item itself is returned for a full refund. If your order arrived damaged, do not start a standard return: contact support with photos of the damage and you will get a free replacement within seven days, with no return shipping needed. Damaged-item replacements ship by the fastest available method at our expense, and a tracking number is emailed as soon as the parcel leaves the warehouse. Items bought during the holiday season get an extended forty-five day return window instead of the standard thirty days, which covers purchases made between the first of November and the twenty-fifth of December. Clearance items and gift cards are excluded from returns entirely and cannot be exchanged. If you paid with a gift card, the refund goes back onto a new gift card rather than to a bank account. For exchanges, we recommend placing a new order once your refund is issued, since exchange inventory cannot be reserved while a return is in transit. Support can override the return window by up to seven days for extenuating circumstances, but this is at their discretion and requires a supervisor's approval. Returns without the original packaging are accepted but incur a fifteen percent restocking fee deducted from the refund amount. International returns follow the same rules, except the customer covers return shipping and the refund posts after customs clearance."""},
    {"doc_id": "shipping-policy", "text": """We ship to most countries with tracked delivery on every order. Standard shipping takes 3 to 5 business days for domestic orders placed before the daily cutoff at 2pm local time. Orders placed after the cutoff ship the next business day. We offer free shipping on orders over $50 after discounts are applied, which means the merchandise subtotal must clear fifty dollars before the free option appears at checkout. Express delivery arrives in one to two business days and is available for most postal codes; a few remote regions fall back to standard times and the checkout page says so before you pay. International shipping takes 10 to 15 business days depending on the destination country, and customs duties are calculated at checkout so there are no surprise charges on delivery. Every parcel gets a tracking number emailed within an hour of dispatch, and you can follow it from the warehouse to your door on the tracking page. If a parcel is marked delivered but you cannot find it, wait twenty-four hours and check with neighbours or the building reception before filing a lost-parcel claim. Lost parcels are reshipped free of charge once the carrier confirms the loss, which usually takes three business days of investigation. Split shipments happen when items come from different warehouses; each parcel is tracked separately and shipping is still charged once per order. We do not ship to PO boxes for express orders because the couriers require a signature on delivery. Hazardous materials such as lithium batteries ship by ground only, which adds two to four days on top of the quoted times. During peak season in December, add up to three extra business days to every estimate above."""},
    {"doc_id": "warranty-policy", "text": """Every product ships with a standard warranty that lasts two years from the date of delivery, covering manufacturing defects in materials and workmanship. The battery is covered for one year only, since batteries are consumable parts with a naturally limited lifespan. To file a claim, open the support portal, enter your order number, and describe the fault; a prepaid shipping label is issued within one business day. Please note that water damage is excluded from coverage under all warranty tiers, even the extended plan, because moisture indicators cannot distinguish accidents from misuse. Accidental damage such as cracked screens from drops is not covered by the standard warranty but can be added through the accidental-damage plan at checkout. The warranty is void if the device has been opened or repaired by anyone other than our authorised service centres, which you can verify by the intact tamper seal. Refurbished units carry a one-year warranty instead of two, with the same exclusions otherwise. If a valid claim cannot be repaired, we replace the unit with the same model or the nearest current equivalent at no cost to you. Warranty service does not extend the original warranty period; the clock keeps running from the original delivery date. Transfer of ownership transfers the remaining warranty automatically, with no registration needed by the new owner. Commercial use halves all warranty periods, so a two-year consumer warranty becomes one year for business buyers. Keep your proof of purchase: claims without an order number take longer while support looks up the sale. Extended warranties must be bought within thirty days of delivery and cannot be added retroactively after a fault appears."""},
    {"doc_id": "pricing-policy", "text": """Our pricing is transparent: the price you see is the price you pay, with no hidden fees at checkout. New customers get 10 percent off their first order when they create an account and verify their email address; the discount applies automatically to the first cart. Bulk buyers save more: any order containing five or more units of the same item gets 15 percent off those units, and the discount stacks with seasonal promotions. Students with a valid student ID receive 20 percent off full-price items through the education store, verified once per academic year. Clearance items are final sale and cannot be returned or exchanged, which is why they are marked down so aggressively. Price matching applies to identical items sold by authorised retailers within fourteen days of purchase; marketplace sellers and auction listings do not qualify. We run two major sales a year, in March and September, with discounts of up to 30 percent across most categories. Loyalty members earn one point per dollar spent, and every hundred points converts to five dollars of store credit. Referral credit gives both you and the referred friend ten dollars once their first order ships. Gift cards never expire and can be combined with other discounts, except on clearance items. Tax is calculated based on the shipping address and shown before payment. If a price drops within seven days of your purchase, contact support for a one-time price adjustment to the lower amount. Corporate accounts get custom quotes for orders above ten thousand dollars through the sales team."""},
]

QUESTIONS = [
    {"question": "How long do I have to return something I bought?",
     "expected_doc": "refund-policy", "must_contain": ["30 days"]},
    {"question": "My order arrived damaged. What do I get?",
     "expected_doc": "refund-policy",
     "must_contain": ["replacement", "seven days"]},
    {"question": "How long does standard shipping take?",
     "expected_doc": "shipping-policy",
     "must_contain": ["3 to 5 business days"]},
    {"question": "Is there a free shipping option?",
     "expected_doc": "shipping-policy",
     "must_contain": ["orders over $50"]},
    {"question": "How long does the standard warranty last?",
     "expected_doc": "warranty-policy", "must_contain": ["two years"]},
    {"question": "Is water damage covered by the warranty?",
     "expected_doc": "warranty-policy",
     "must_contain": ["water damage", "excluded"]},
    {"question": "What discount do new customers receive?",
     "expected_doc": "pricing-policy", "must_contain": ["10 percent"]},
    {"question": "Can I return clearance items?",
     "expected_doc": "pricing-policy", "must_contain": ["final sale"]},
]


def _texts(rag):
    # same corpus-peek trick as run_eval._corpus — every retriever hides
    # the texts somewhere different, metas is uniform across backends
    r = rag.retriever
    if hasattr(r, "texts"):
        return r.texts
    if hasattr(r, "store"):
        return r.store.texts
    return r._dense.store.texts  # hybrid wraps the dense retriever's store


def _score_one(rag, q):
    # recall@3 + evidence, same definitions as evals/run_eval.py
    res = rag.answer(q["question"], top_k=TOP_K)
    docs = [c["meta"].get("doc_id") for c in res["citations"]]
    recall = q["expected_doc"] in docs
    context = " ".join(c["text"] for c in res["citations"]).lower()
    evidence = all(kw.lower() in context for kw in q["must_contain"])
    return recall, evidence


def run_strategy(name):
    rag = RAGPipeline(overrides={"chunker": name, "retriever": "dense",
                                 "generator": "mock",
                                 "semantic_cache": {"enabled": False}})
    rag.index_documents([{"text": d["text"],
                          "metadata": {"doc_id": d["doc_id"]}}
                         for d in CORPUS])
    texts = _texts(rag)
    cells = [_score_one(rag, q) for q in QUESTIONS]
    return {"chunks": len(texts),
            "mean_words": sum(len(t.split()) for t in texts) / len(texts),
            "recall": sum(1 for r, _ in cells if r),
            "evidence": sum(1 for _, e in cells if e),
            "cells": cells}


def _cell(rec, ev):
    # one symbol per question per strategy for the detail matrix
    if rec and ev:
        return "R+E"
    if rec:
        return "R"
    if ev:
        return "E"
    return "–"


def main():
    results = {s: run_strategy(s) for s in STRATEGIES}
    n = len(QUESTIONS)
    # winner: most evidence, then most recall, then fewest chunks (a
    # smaller index is cheaper to store and search, all else equal)
    ranked = sorted(STRATEGIES,
                    key=lambda s: (results[s]["evidence"],
                                   results[s]["recall"],
                                   -results[s]["chunks"]),
                    reverse=True)
    winner, runner_up = ranked[0], ranked[1]
    w, r = results[winner], results[runner_up]

    missed_all = [q["question"] for q, cs in
                  zip(QUESTIONS, zip(*[results[s]["cells"]
                                       for s in STRATEGIES]))
                  if not any(rec or ev for rec, ev in cs)]

    lines = []
    lines.append("# Chunking experiment — 4 strategies, same queries")
    lines.append("")
    lines.append("Date: 2026-10-07. Corpus: 4 fixture policy docs (~400 "
                 "words each, facts buried mid-document), 8 questions, "
                 "dense retriever + hashing embeddings + mock generator, "
                 "chunk_size=120 words, overlap=20, top_k=3. Every strategy "
                 "saw the same corpus and the same questions.")
    lines.append("")
    lines.append("| strategy | chunks | mean words/chunk | recall@3 | "
                 "evidence |")
    lines.append("|---|---|---|---|---|")
    for s in STRATEGIES:
        d = results[s]
        lines.append(f"| {s} | {d['chunks']} | {d['mean_words']:.0f} | "
                     f"{d['recall']}/{n} | {d['evidence']}/{n} |")
    lines.append("")
    lines.append("Per-question detail (R = right doc retrieved, "
                 "E = answer keywords in the retrieved chunks):")
    lines.append("")
    header = "| question | " + " | ".join(STRATEGIES) + " |"
    lines.append(header)
    lines.append("|---|---|---|---|---|")
    for i, q in enumerate(QUESTIONS):
        row = " | ".join(_cell(*results[s]["cells"][i])
                         for s in STRATEGIES)
        lines.append(f"| {q['question']} | {row} |")
    lines.append("")
    lines.append(f"**Winner: {winner}** — evidence {w['evidence']}/{n}, "
                 f"recall@3 {w['recall']}/{n}, {w['chunks']} chunks "
                 f"(vs {runner_up}: evidence {r['evidence']}/{n}, "
                 f"recall@3 {r['recall']}/{n}, {r['chunks']} chunks).")
    if w["evidence"] == r["evidence"] and w["recall"] == r["recall"]:
        lines.append(f"The top two tied on retrieval quality; {winner} wins "
                     f"on index size ({w['chunks']} vs {r['chunks']} chunks).")
    if missed_all:
        lines.append("Questions missed by every strategy: " +
                     "; ".join(f'"{q}"' for q in missed_all) +
                     " — these need better retrieval (query rewriting, "
                     "hybrid), not better chunking.")
    lines.append("")
    lines.append("Caveats: small fixture corpus; hashing embeddings are "
                 "bag-of-words, not neural — boundary placement is what "
                 "this measures, and the same script reruns unchanged "
                 "against st embeddings. The live default stays "
                 "`chunker: word`; promoting the winner is a separate "
                 "decision once confirmed on the neural path.")
    lines.append("")

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"fixture: {len(CORPUS)} docs, {n} questions, top_k={TOP_K}")
    print(f"{'strategy':15s} {'chunks':>6s} {'recall@3':>8s} "
          f"{'evidence':>8s}")
    for s in STRATEGIES:
        d = results[s]
        print(f"{s:15s} {d['chunks']:>6d} {d['recall']:>3d}/{n:<4d} "
              f"{d['evidence']:>3d}/{n:<4d}")
    print(f"winner: {winner} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
