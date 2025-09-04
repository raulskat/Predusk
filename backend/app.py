from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.types import (
    ProcessTextRequest,
    ProcessTextResponse,
    QueryRequest,
    QueryResponse,
    Citation,
)
from backend.rag_engine import chunk_text, upsert_chunks_to_vector_db, retrieve_and_rerank, generate_answer


settings = get_settings()

app = FastAPI(title="Mini RAG API", version="0.1.0")

origins = ["*"] if settings.ALLOWED_ORIGINS == "*" else [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
allow_credentials = False if origins == ["*"] else True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-text", response_model=ProcessTextResponse)
def process_text(req: ProcessTextRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    chunks = chunk_text(req.text, req.chunk_size, req.chunk_overlap)
    try:
        count = upsert_chunks_to_vector_db(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    return ProcessTextResponse(chunks_indexed=count, message="Indexed")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        results = retrieve_and_rerank(req.query, req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    # While Step 2 isn't implemented, return a placeholder
    context_chunks = [r.get("text", "") if isinstance(r, dict) else str(r) for r in results]
    answer = generate_answer(req.query, context_chunks)

    citations = []
    for i, r in enumerate(results):
        if isinstance(r, dict):
            citations.append(Citation(chunk_id=str(r.get("id", i)), text=r.get("text", ""), score=r.get("score")))
        else:
            citations.append(Citation(chunk_id=str(i), text=str(r)))

    return QueryResponse(answer=answer, citations=citations)
