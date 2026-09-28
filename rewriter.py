"""Query rewriting: multi-query expansion before retrieval.

A vague query misses because the chunks use different words ("refund
window" vs "return period"). Fanning one query out into a few variants —
original + synonym paraphrase + keyword expansion — gives retrieval extra
chances to hit the right chunks, and dedupe collapses the overlap back
into one ranked list.

# SWAP (production): point rewrite_query at a real LLM — e.g.
#   variants = llm.generate(f"Write 3 search queries paraphrasing: {question}")
# The list-of-strings contract stays the same, so the swap is one line.
"""
import re

from embedder import _tokenize  # same tokens the retriever sees

# paraphrase stand-ins: what the mock knows how to reword. Real LLMs
# do this open-ended; the table just has to cover enough to exercise
# the fan-out, merge and dedupe wiring offline.
_SYNONYMS = {
    "refund": "reimbursement",
    "window": "period",
    "long": "duration",
    "cost": "price",
    "shipping": "delivery",
    "leave": "time off",
    "warranty": "guarantee",
    "claim": "compensation",
    "policy": "terms",
    "return": "refund",
    "period": "window",
    "price": "cost",
    "delivery": "shipping",
    "fee": "charge",
    "cover": "include",
    "country": "region",
}

# no retrieval signal — dropped before expansion so variant 3 stays
# topical instead of echoing "what is the"
_STOPWORDS = {"what", "is", "the", "how", "long", "does", "do", "a", "an",
              "of", "in", "to", "for", "are", "it", "and", "or", "on"}


def _mock_variants(question, n=3):
    """Deterministic offline stand-in for the LLM rewriter.

    Variant 1 is always the original — the user's own words stay in play.
    Variant 2 swaps known synonyms (the vocabulary-gap bridge). Variant 3
    appends related terms (keyword expansion, catches looser matches). If
    the table knows no synonyms, both fall back to generic expansion, so
    the fan-out is always exactly n distinct variants.
    """
    toks = [t for t in _tokenize(question.lower()) if t not in _STOPWORDS]
    v1 = question
    # paraphrase: reword the user's own sentence in place
    v2 = re.sub(r"[a-z0-9]+",
                lambda m: _SYNONYMS.get(m.group(0), m.group(0)), question)
    # expansion: original plus the synonyms the paraphrase used
    extra = [s for t in toks
             for s in (_SYNONYMS.get(t),) if s and s != t]
    v3 = (v1 + " " + " ".join(extra)) if extra else v1 + " details meaning"
    if v2 == v1:
        # no synonyms known — expand instead of echoing the original
        v2 = v1 + " explanation"
    # dedupe + pad: the caller always gets exactly n variants
    out = []
    for v in (v1, v2, v3):
        if v not in out:
            out.append(v)
    while len(out) < n:
        out.append(f"{v1} details")
    return out[:n]


def rewrite_query(question, num_variants=3):
    """The rewriter stage — mock today, real LLM behind the # SWAP above."""
    return _mock_variants(question, n=num_variants)


def dedupe_results(lists):
    """Merge per-variant retrieval lists into one ranked, deduped list.

    A chunk found by two variants keeps its best score — scores stay in
    [0, 1], so pipeline's min_score gate behaves the same as before.
    """
    merged = {}  # chunk key -> (text, best score, meta)
    for hits in lists:
        for text, score, meta in hits:
            key = meta.get("hash", text)
            if key not in merged or score > merged[key][1]:
                merged[key] = (text, score, meta)
    return sorted(merged.values(), key=lambda x: -x[1])


def build_rewriter(cfg):
    """None unless rewrite: true — the default path must not change."""
    if not cfg.get("rewrite"):
        return None
    return QueryRewriter(cfg)


class QueryRewriter:
    def __init__(self, cfg):
        self.num_variants = int(cfg.get("rewrite_variants", 3))

    def rewrite(self, question):
        return rewrite_query(question, num_variants=self.num_variants)


if __name__ == "__main__":
    # done-when: one ambiguous query fans out to exactly 3 variants,
    # merged results dedupe with the best score kept
    v = rewrite_query("what is the refund window")
    assert len(v) == 3 and len(set(v)) == 3, v
    assert v[0] == "what is the refund window"  # original always first
    print("variants:", v)
    hits1 = [("refund window is 30 days", 0.9, {"hash": "a"}),
             ("other chunk", 0.5, {"hash": "b"})]
    hits2 = [("refund window is 30 days", 0.7, {"hash": "a"}),
             ("return period", 0.6, {"hash": "c"})]
    merged = dedupe_results([hits1, hits2])
    assert [t for t, _, _ in merged] == [
        "refund window is 30 days", "return period", "other chunk"], merged
    assert merged[0][1] == 0.9  # duplicate keeps its best score
    print("merge + dedupe OK")
