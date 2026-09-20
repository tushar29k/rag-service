"""RAGPipeline: chunk -> embed -> retrieve -> prompt -> generate.

Every stage is timed. The `breakdown` in the result is what you'd ship to
your latency dashboard — it's how you find which stage to optimise.
"""
import hashlib
import time
import yaml

from chunker import chunk_text
from embedder import TfidfEmbedder
from store import VectorStore


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
    """Extractive stand-in: returns the 2 sentences with highest keyword
    overlap with the question; refuses when nothing retrieved.

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
    def __init__(self, config_path="config.yaml"):
        self.cfg = yaml.safe_load(open(config_path))
        self.embedder = TfidfEmbedder(dim=self.cfg.get("embed_dim", 2048))
        self.store = VectorStore(dim=self.cfg.get("embed_dim", 2048))
        self.index_version = self.cfg.get("index_version", "v1")
        self._fitted = False

    @staticmethod
    def _hash(text):
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def index_documents(self, docs):
        """docs: [{text, metadata}]. Returns chunk count. Idempotent via hash."""
        all_chunks, all_metas = [], []
        for doc in docs:
            for i, ch in enumerate(chunk_text(
                    doc["text"], self.cfg["chunk_size"], self.cfg["overlap"])):
                all_metas.append({**doc.get("metadata", {}),
                                  "chunk": i, "hash": self._hash(ch)})
                all_chunks.append(ch)
        # dedupe: same content hash already indexed -> skip
        seen = {m["hash"] for m in self.store.metas}
        new = [(t, m) for t, m in zip(all_chunks, all_metas)
               if m["hash"] not in seen]
        if not new:
            return 0
        texts = [t for t, _ in new]
        if not self._fitted:
            self.embedder.fit(texts)     # learn IDF on first batch
            self._fitted = True
        embs = self.embedder.transform(texts)
        self.store.upsert(texts, embs, [m for _, m in new])
        return len(new)

    def build_prompt(self, question, retrieved):
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
        q_emb = self.embedder.transform([question])[0]
        t.mark("embed")
        retrieved = self.store.search(q_emb, top_k=top_k, filters=filters)
        # RELEVANCE THRESHOLD: brute force always returns *something*.
        # In production, scores below this mean "nothing relevant" — without
        # it, the generator happily answers from irrelevant chunks.
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
