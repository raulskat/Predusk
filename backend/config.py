from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv


# Load variables from a local .env if present
load_dotenv()


class Settings:
    # Vector DB
    PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX: str | None = os.getenv("PINECONE_INDEX")
    # New SDK prefers host or cloud+region
    PINECONE_INDEX_HOST: str | None = os.getenv("PINECONE_INDEX_HOST")
    PINECONE_CLOUD: str | None = os.getenv("PINECONE_CLOUD", "AWS")
    PINECONE_REGION: str | None = os.getenv("PINECONE_REGION", "US_EAST_1")
    PINECONE_ENVIRONMENT: str | None = os.getenv("PINECONE_ENVIRONMENT")  # legacy, ignored by new SDK
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "default")

    # Models
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")  # Gemini
    COHERE_API_KEY: str | None = os.getenv("COHERE_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-english-v3.0")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    # Server
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
