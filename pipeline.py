"""RAGPipeline: chunk -> embed -> retrieve -> prompt -> generate.

Every stage is timed, and the `breakdown` in each result is the bit I'd
actually ship to a latency dashboard — when answers get slow, this tells
you which stage to blame first.
"""
import hashlib
import time
import yaml

from chunker import chunk_text
from retriever import build_retriever


class Timer:
    def __init__(self):
        self.marks, self._t = {}, time.perf_counter()

    def mark(self, name):
        now = time.perf_counter()
        self.marks[name] = round((now - self._t) * 1000, 1)
        self._t = now

    def total(self):
        return round(sum(self.marks.values()), 1)


def _mock_llm(question, context_chunks):
    """Stand-in for the LLM: quotes the 2-3 sentences with the most word
    overlap with the question. Refuses when retrieval came back empty.

    It's dumb on purpose — negation and synthesis are exactly what a real
    model buys you over this.

    # SWAP (production): call your LLM here —
    #   answer = openai.chat.completions.create(model=..., messages=[...])
    # The prompt format below already follows the grounded-generation pattern.
    """
    if not context_chunks:
        return "I don't know — nothing in the knowledge base covers this."
    q_toks = set(question.lower().split())
    scored = []
    for text, _, _ in context_chunks:
        for sent in [s.strip() for s in text.replace("!", ".").split(".")
                     if s.strip()]:
            overlap = len(q_toks & set(sent.lower().split()))
            scored.append((overlap, sent))
    scored.sort(key=lambda x: -x[0])
    best = [s for _, s in scored[:3]]
    note = ("\n\nNote: this is an extractive mock, not a real LLM — it quotes "
            "sentences with keyword overlap and can't do negation or synthesis. "
            "In production this is where the LLM call goes (# SWAP above).")
    return (". ".join(best) + "." + note) if best else "I don't know."


class RAGPipeline:
    def __init__(self, config_path="config.yaml", overrides=None):
        self.cfg = yaml.safe_load(open(config_path))
        if overrides:
            # lets evals/cli run one backend without editing config.yaml
            self.cfg.update({k: v for k, v in overrides.items()
                             if v is not None})
        # dense, bm25 or hybrid behind the same interface — the pipeline
        # stages below don't know or care which one is wired in
        self.retriever = build_retriever(self.cfg)
        self.index_version = self.cfg.get("index_version", "v1")

    @staticmethod
    def _hash(text):
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def index_documents(self, docs):
        """Index a batch of {text, metadata} docs.

        Returns how many NEW chunks were added — re-indexing the same docs
        is a no-op thanks to content hashing.
        """
        all_chunks, all_metas = [], []
        for doc in docs:
            for i, ch in enumerate(chunk_text(
                    doc["text"], self.cfg["chunk_size"], self.cfg["overlap"])):
                all_metas.append({**doc.get("metadata", {}),
                                  "chunk": i, "hash": self._hash(ch)})
                all_chunks.append(ch)
        # cheap trick: dedupe on content hash, so re-running the indexer
        # over the same docs doesn't pile up duplicates
        seen = {m["hash"] for m in self.retriever.metas}
        new = [(t, m) for t, m in zip(all_chunks, all_metas)
               if m["hash"] not in seen]
        if not new:
            return 0
        texts = [t for t, _ in new]
        self.retriever.index(texts, [m for _, m in new])
        return len(new)

    def build_prompt(self, question, retrieved):
        # numbered citations like [1] so the answer can point at its sources
        context = "\n\n".join(
            f"[{i+1}] {text}" for i, (text, _, _) in enumerate(retrieved))
        return f"""Answer the question using ONLY the context below. If the
context doesn't contain the answer, say you don't know. Cite sources like [1].

Context:
{context}

Question: {question}
Answer:"""

    def answer(self, question, filters=None, top_k=None):
        top_k = top_k or self.cfg.get("top_k", 3)
        t = Timer()
        # embed = vectorise for dense, tokenise for bm25 — the "embed" stage
        # means "turn the question into what the retriever eats"
        q = self.retriever.embed(question)
        t.mark("embed")
        retrieved = self.retriever.search(q, top_k=top_k, filters=filters)
        # Brute-force search always returns *something*, even for nonsense
        # questions. This threshold turns low-score retrievals into
        # "nothing relevant" so the generator refuses instead of answering
        # off a junk chunk. Tuned by hand against the golden set — re-tune
        # it if the corpus changes.
        min_score = self.cfg.get("min_score", 0.15)
        retrieved = [(tx, s, m) for tx, s, m in retrieved if s >= min_score]
        t.mark("retrieve")
        prompt = self.build_prompt(question, retrieved)
        t.mark("prompt")
        answer = _mock_llm(question, retrieved)
        t.mark("generate")
        return {"answer": answer,
                "citations": [{"text": tx, "score": round(s, 3),
                               "meta": m} for tx, s, m in retrieved],
                "latency_ms": t.total(), "breakdown": t.marks,
                "index_version": self.index_version}
