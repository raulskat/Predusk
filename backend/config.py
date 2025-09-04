from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv


# Load variables from a local .env if present
load_dotenv()


class Settings:
    # Vector DB
    def _get(name: str, default: str | None = None) -> str | None:
        val = os.getenv(name, default)
        if val is None:
            return None
        return val.strip()

    PINECONE_API_KEY: str | None = _get.__func__("PINECONE_API_KEY")
    PINECONE_INDEX: str | None = _get.__func__("PINECONE_INDEX")
    # New SDK prefers host or cloud+region
    PINECONE_INDEX_HOST: str | None = _get.__func__("PINECONE_INDEX_HOST")
    PINECONE_CLOUD: str | None = _get.__func__("PINECONE_CLOUD", "AWS")
    PINECONE_REGION: str | None = _get.__func__("PINECONE_REGION", "US_EAST_1")
    PINECONE_ENVIRONMENT: str | None = _get.__func__("PINECONE_ENVIRONMENT")  # legacy, ignored by new SDK
    PINECONE_NAMESPACE: str = _get.__func__("PINECONE_NAMESPACE", "default") or "default"

    # Models
    GOOGLE_API_KEY: str | None = _get.__func__("GOOGLE_API_KEY")  # Gemini
    COHERE_API_KEY: str | None = _get.__func__("COHERE_API_KEY")
    EMBEDDING_MODEL: str = _get.__func__("EMBEDDING_MODEL", "text-embedding-004") or "text-embedding-004"
    RERANK_MODEL: str = _get.__func__("RERANK_MODEL", "rerank-english-v3.0") or "rerank-english-v3.0"
    LLM_MODEL: str = _get.__func__("LLM_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    # Server
    ALLOWED_ORIGINS: str = _get.__func__("ALLOWED_ORIGINS", "*") or "*"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
