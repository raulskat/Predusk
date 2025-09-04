from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv


# Load variables from a local .env if present
load_dotenv()


def _env(name: str, default: str | None = None, strip: bool = True) -> str | None:
    val = os.getenv(name, default)
    if val is None:
        return None
    if strip and isinstance(val, str):
        return val.strip()
    return val


def _env_host(name: str) -> str | None:
    val = _env(name)
    if not val:
        return None
    # sanitize: remove scheme, whitespace, and trailing slashes
    return val.replace("https://", "").replace("http://", "").strip().strip("/")


class Settings:
    # Vector DB
    PINECONE_API_KEY: str | None = _env("PINECONE_API_KEY")
    PINECONE_INDEX: str | None = _env("PINECONE_INDEX")
    # New SDK prefers host or cloud+region
    PINECONE_INDEX_HOST: str | None = _env_host("PINECONE_INDEX_HOST")
    PINECONE_CLOUD: str | None = _env("PINECONE_CLOUD", "AWS")
    PINECONE_REGION: str | None = _env("PINECONE_REGION", "US_EAST_1")
    PINECONE_ENVIRONMENT: str | None = _env("PINECONE_ENVIRONMENT")  # legacy, ignored by new SDK
    PINECONE_NAMESPACE: str = _env("PINECONE_NAMESPACE", "default") or "default"

    # Models
    GOOGLE_API_KEY: str | None = _env("GOOGLE_API_KEY")  # Gemini
    COHERE_API_KEY: str | None = _env("COHERE_API_KEY")
    EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "text-embedding-004") or "text-embedding-004"
    RERANK_MODEL: str = _env("RERANK_MODEL", "rerank-english-v3.0") or "rerank-english-v3.0"
    LLM_MODEL: str = _env("LLM_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    # Server
    ALLOWED_ORIGINS: str = _env("ALLOWED_ORIGINS", "*") or "*"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
