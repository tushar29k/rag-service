"""HTTP service over RAGPipeline. Run: uvicorn service:app --reload

Endpoints:
  POST /index        {text, metadata} or [{text, metadata}, ...]  -> chunk + embed + upsert
  POST /query        {text, filters?, top_k?} -> grounded answer + citations + timings
  POST /query-stream {text, filters?, top_k?} -> SSE: meta, then tokens, then done
  GET  /health                          -> index version + chunk count
"""
import json
from typing import Union

try:
    from fastapi import Body, FastAPI
    from fastapi.responses import StreamingResponse
except ImportError as e:
    raise SystemExit("pip install fastapi uvicorn  (then re-run)") from e

from pipeline import RAGPipeline

rag = RAGPipeline()   # built once at import — rebuilding per request
                      # would re-index everything and be painfully slow
app = FastAPI(title="rag-service")


@app.post("/index")
def index(body: Union[dict, list] = Body(...)):
    # accepts one doc or a batch — batches go through the same
    # chunk/embed/upsert path, just more of them in one request
    docs = body if isinstance(body, list) else [body]
    n = rag.index_documents([{"text": d["text"],
                              "metadata": d.get("metadata", {})}
                             for d in docs])
    return {"docs_received": len(docs), "chunks_indexed": n,
            "total_chunks": len(rag.retriever),
            "index_version": rag.index_version}


@app.post("/query")
def query(q: dict):
    return rag.answer(q["text"], filters=q.get("filters"),
                      top_k=q.get("top_k"))


@app.post("/query-stream")
def query_stream(q: dict):
    # server-sent events: meta first (citations + version), then one
    # event per answer token as the generate stage emits them, then done
    def sse():
        for ev in rag.answer_stream(q["text"], filters=q.get("filters"),
                                    top_k=q.get("top_k")):
            yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/health")
def health():
    cache = (rag.cache.stats() if rag.cache else
             {"enabled": False, "hits": 0, "misses": 0, "hit_rate": 0.0,
              "size": 0})
    return {"status": "ok", "index_version": rag.index_version,
            "chunks": len(rag.retriever), "cache": cache}


@app.get("/info")
def info():
    # real_llm is True only when a key configured a live client at startup
    client = rag.llm
    return {"real_llm": client is not None,
            "generator": rag.cfg.get("generator", "auto"),
            "profile": rag.profile,  # None unless RAG_PROFILE set
            "provider": client.provider if client else None,
            "model": client.model if client else None,
            "last_error": rag.last_llm_error}

# -- demo ui -----------------------------------------------------------------
# open / in a browser to click through the api instead of curling it.
import os as _os
from fastapi.responses import FileResponse as _FileResponse

_UI_INDEX = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ui", "index.html")


@app.get("/", include_in_schema=False)
def _demo_ui():
    return _FileResponse(_UI_INDEX)

