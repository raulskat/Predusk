from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class ProcessTextRequest(BaseModel):
    text: str = Field(..., description="Raw input text to index")
    chunk_size: int = Field(800, ge=100, le=4000)
    chunk_overlap: int = Field(200, ge=0, le=2000)


class ProcessTextResponse(BaseModel):
    chunks_indexed: int
    message: str = "OK"


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(6, ge=1, le=50)


class Citation(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation] = []

