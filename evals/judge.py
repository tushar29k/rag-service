"""RAGAS-style answer scoring: faithfulness + answer relevance.

RAGAS scores a RAG answer on two things that recall@k never sees:

  faithfulness     — are the claims in the answer actually supported by
                     the retrieved context? (claims supported / claims total)
  answer relevance — does the answer address the question? RAGAS
                     reverse-engineers this: it has the judge generate
                     questions from the answer, then measures how similar
                     those questions are to the original one.

Both need an LLM judge in production. Here the judge is a deterministic
mock behind `_mock_judge` — marked # SWAP: — so the whole thing runs
offline and stays green in CI. The public functions below
(judge_claims, faithfulness, answer_relevance) keep their contracts when
the real judge swaps in.

# SWAP (production): point _mock_judge at a real LLM, e.g.
#   verdict = json.loads(llm.generate(prompt,
#                    response_format={"type": "json_object"}))
# Claims come out cleaner and verdicts stop being token-overlap, but
# the function signatures and the faithfulness ratio stay identical.
"""
import json
import math
import os
import re
import sys

# self-test runs as `python3 evals/judge.py` — pull in the repo root so
# the same token view as retrieval is importable either way
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from embedder import _tokenize  # same token view retrieval uses

# stopwords carry no claim content — excluded from overlap math
_STOP = {"what", "is", "the", "how", "long", "does", "do", "a", "an",
        "of", "in", "to", "for", "are", "it", "and", "or", "on", "this",
        "that", "with", "from", "by", "as", "an", "not", "no", "can",
        "will", "be", "have", "has", "was", "were", "their", "they"}

# _mock_llm appends this disclaimer to every answer; it's commentary
# about the mock itself, not a grounded claim about the knowledge base,
# so the judge scores the answer content only
_DISCLAIMER = "Note: this is an extractive mock"


def _content_toks(text):
    return [t for t in _tokenize(text.lower()) if t not in _STOP]


def _split_claims(answer):
    # one claim per sentence, like RAGAS's claim extraction step; the
    # disclaimer sentence is meta-commentary, not a grounded claim
    body = answer.split(_DISCLAIMER)[0]
    claims = [s.strip().rstrip(".") for s in re.split(r"[.!?]", body)
              if s.strip() and len(_content_toks(s)) >= 2]
    return claims


def _cosine(a, b):
    # token-multiset cosine — crude but deterministic; a real embedding
    # behind the same interface is the obvious upgrade with real judges
    va = {t: a.count(t) for t in set(a)}
    vb = {t: b.count(t) for t in set(b)}
    dot = sum(n * vb.get(t, 0) for t, n in va.items())
    na, nb = math.sqrt(sum(n * n for n in va.values())), \
        math.sqrt(sum(n * n for n in vb.values()))
    return dot / (na * nb) if na and nb else 0.0


def _mock_judge(prompt):
    """Deterministic stand-in for the LLM judge.

    The prompt carries REQUEST / CONTEXT / ANSWER sections (see
    build_faithfulness_prompt below). This mock parses them, then:
      verdict_claims -> per-claim supported/unsupported by token overlap
      verdict_relevance -> pseudo-questions from the answer's top tokens,
                           scored against the original question by cosine

    A real judge returns these as JSON; here the verdicts are computed
    straight from the parsed sections, so the prompt format is the only
    thing the real swap has to honor.
    """
    sections = {}
    current = None
    for line in prompt.splitlines():
        # the request code lives entirely on its own header line — the
        # free-text instructions are decoration, not part of it
        if line.startswith("REQUEST:"):
            current, sections["REQUEST"] = None, line[len("REQUEST:"):].strip()
        elif line.startswith("CONTEXT:"):
            current, sections["CONTEXT"] = "CONTEXT", ""
        elif line.startswith("ANSWER:"):
            current, sections["ANSWER"] = "ANSWER", ""
        elif current:
            sections[current] += line + "\n"
    request = sections.get("REQUEST", "")
    context = sections.get("CONTEXT", "").strip()
    answer = sections.get("ANSWER", "").strip()

    if request == "verdict_claims":
        ctx_toks = set(_tokenize(context.lower()))
        verdicts = []
        for claim in _split_claims(answer):
            toks = _content_toks(claim)
            hit = sum(1 for t in toks if t in ctx_toks) / len(toks)
            # verbatim extractive quotes land near 1.0; a fabricated
            # claim about nothing in the context lands near 0
            verdicts.append({"claim": claim, "supported": hit >= 0.5,
                             "overlap": round(hit, 2)})
        return json.dumps({"verdicts": verdicts})

    if request == "verdict_relevance":
        # reverse-engineer: the answer's top content tokens, grouped into
        # three pseudo-questions; relevance is their mean cosine to the
        # original question — same shape as RAGAS, content-free words
        freq, order = {}, {}
        for i, t in enumerate(_content_toks(answer.split(_DISCLAIMER)[0])):
            freq[t] = freq.get(t, 0) + 1
            order.setdefault(t, i)
        top = sorted(freq, key=lambda t: (-freq[t], order[t]))[:9]
        questions = [" ".join(top[i:i + 3]) + "?"
                     for i in range(0, min(len(top), 9), 3)]
        return json.dumps({"questions": questions})
    raise ValueError(f"unknown judge request: {request!r}")


def build_faithfulness_prompt(answer, context):
    return ("Split the answer into claims and judge each one: is it "
            "supported by the context below? Reply with JSON "
            '{"verdicts": [{"claim": ..., "supported": bool}]}.\n'
            "REQUEST: verdict_claims\n"
            f"CONTEXT:\n{context}\nANSWER:\n{answer}")


def build_relevance_prompt(question, answer):
    return ("Generate 3 questions the answer below could be answering, "
            "given the original question for context. Reply with JSON "
            '{"questions": [...]}.\n'
            "REQUEST: verdict_relevance\n"
            f"ORIGINAL QUESTION: {question}\nANSWER:\n{answer}")


def judge_claims(answer, context):
    """Per-claim verdicts — [{claim, supported, overlap}]."""
    raw = _mock_judge(build_faithfulness_prompt(answer, context))
    return json.loads(raw)["verdicts"]


def faithfulness(answer, context):
    """Fraction of the answer's claims supported by the context.

    No claims (a clean "I don't know") counts as faithful — refusing to
    invent is exactly what you want, and the unanswerable golden questions
    exercise that path.
    """
    verdicts = judge_claims(answer, context)
    if not verdicts:
        return 1.0
    return sum(1 for v in verdicts if v["supported"]) / len(verdicts)


def answer_relevance(question, answer):
    """Mean cosine(question, generated questions from answer).

    The mock judge builds the questions from the answer's own top tokens,
    so an answer that ignores the question shares no vocabulary and the
    score collapses — which is the point of the metric.
    """
    raw = _mock_judge(build_relevance_prompt(question, answer))
    questions = json.loads(raw)["questions"]
    if not questions:
        return 0.0
    q_toks = _content_toks(question)
    return sum(_cosine(q_toks, _content_toks(gq)) for gq in questions) \
        / len(questions)


if __name__ == "__main__":
    # done-when for the module: the mock judge scores a faithful answer
    # near 1.0, a fabricated one near 0, and relevance tracks topicality
    ctx = "The refund window is 30 days for orders placed in India."
    good = "The refund window is 30 days."
    bad = "Refunds are issued in 90 days with a 50 percent fee."
    assert faithfulness(good, ctx) == 1.0, judge_claims(good, ctx)
    assert faithfulness(bad, ctx) < 0.5, judge_claims(bad, ctx)
    rel = answer_relevance("What is the refund window?", good)
    irrel = answer_relevance("What is the refund window?",
                             "Employees get 26 weeks of maternity leave.")
    assert rel > irrel, (rel, irrel)
    print(f"judge self-test OK — faithful 1.0, fabricated "
          f"{faithfulness(bad, ctx):.2f}, relevance {rel:.2f} > {irrel:.2f}")
