"""HTTP service over RAGPipeline. Run: uvicorn service:app --reload

Endpoints:
  POST /index  {text, metadata}         -> chunk + embed + upsert
  POST /query  {text, filters?, top_k?} -> grounded answer + citations + timings
  GET  /health                          -> index version + chunk count
"""
try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse  # noqa (streaming exercise)
except ImportError as e:
    raise SystemExit("pip install fastapi uvicorn  (then re-run)") from e

from pipeline import RAGPipeline

rag = RAGPipeline()   # built once at import — rebuilding per request
                      # would re-index everything and be painfully slow
app = FastAPI(title="rag-service")


@app.post("/index")
def index(doc: dict):
    n = rag.index_documents([{"text": doc["text"],
                              "metadata": doc.get("metadata", {})}])
    return {"chunks_indexed": n, "total_chunks": len(rag.store),
            "index_version": rag.index_version}


@app.post("/query")
def query(q: dict):
    return rag.answer(q["text"], filters=q.get("filters"),
                      top_k=q.get("top_k"))


@app.get("/health")
def health():
    return {"status": "ok", "index_version": rag.index_version,
            "chunks": len(rag.store)}
