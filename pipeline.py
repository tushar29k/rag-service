"""RAGPipeline: chunk -> embed -> retrieve -> prompt -> generate.

Every stage is timed, and the `breakdown` in each result is the bit I'd
actually ship to a latency dashboard — when answers get slow, this tells
you which stage to blame first.
"""
import hashlib
import sys
import time
import yaml

from chunker import chunk_text
from llm_client import FreeLLMClient, FreeLLMError
from reranker import build_reranker
from retriever import build_retriever
from rewriter import build_rewriter, dedupe_results


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
    #
    # UPDATE: the swap is now real — RAGPipeline wires FreeLLMClient
    # (llm_client.py, Gemini / OpenRouter free tiers) when LLM_API_KEY is
    # set; _mock_llm stays the offline fallback, unchanged.
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
        # rerank is off by default — only pay for the cross-encoder (or
        # the fallback) when the config explicitly asks for it
        self.reranker = (build_reranker(self.cfg)
                         if self.cfg.get("rerank") else None)
        # off unless config opts in — the default answer() path stays
        # byte-for-byte identical with rewriting disabled
        self.rewriter = build_rewriter(self.cfg)
        self.index_version = self.cfg.get("index_version", "v1")
        # real model when a key is configured, None otherwise — the mock
        # path in answer() stays byte-for-byte identical with no key
        self.llm = FreeLLMClient.from_env()
        self.last_llm_error = None  # last api failure, if any — on /info

    def _generate(self, prompt, question, retrieved):
        # the real-LLM path: grounded answer over the retrieved context.
        # any failure falls back to the extractive mock with a visible note
        if self.llm is None:
            return _mock_llm(question, retrieved)
        try:
            out = self.llm.generate(prompt, max_tokens=512, temperature=0.2)
            self.last_llm_error = None  # recovered
            return out
        except FreeLLMError as e:
            self.last_llm_error = str(e)  # key-free — safe for /info
            print(f"rag: model call failed ({e}) — mock instead",
                  file=sys.stderr)
            return (_mock_llm(question, retrieved)
                    + "\n\n[model unavailable — showing offline mock result]")

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
        # query rewriting: one query fans out to variants (mock LLM, off
        # entirely unless config says rewrite: true). each variant retrieves
        # independently, then the lists merge and dedupe — overlap across
        # variants collapses into one ranked list, best score kept
        variants = (self.rewriter.rewrite(question)
                    if self.rewriter else [question])
        t.mark("rewrite")
        # embed = vectorise for dense, tokenise for bm25 — the "embed" stage
        # means "turn each variant into what the retriever eats"
        per_variant = (int(self.cfg.get("rerank_depth", 20))
                       if self.reranker else top_k)
        hits = [self.retriever.search(self.retriever.embed(v),
                                      top_k=per_variant, filters=filters)
                for v in variants]
        t.mark("retrieve")
        if self.reranker:
            # two-stage retrieval: a wide deduped net, then the reranker
            # re-scores the candidates down to the final list. min_score is
            # applied to the reranked scores below — the reranker's opinion
            # is the better judge.
            top_n = int(self.cfg.get("rerank_top_n", 5))
            candidates = dedupe_results(hits)[:per_variant]
            retrieved = self.reranker.rerank(question, candidates,
                                             top_n=top_n)
            t.mark("rerank")
        else:
            retrieved = dedupe_results(hits)[:top_k]
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
        answer = self._generate(prompt, question, retrieved)
        t.mark("generate")
        return {"answer": answer,
                "citations": [{"text": tx, "score": round(s, 3),
                               "meta": m} for tx, s, m in retrieved],
                "latency_ms": t.total(), "breakdown": t.marks,
                "index_version": self.index_version}
