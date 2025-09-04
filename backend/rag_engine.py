from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any

import time
import uuid

from backend.config import get_settings

# External SDKs are imported lazily in functions to avoid import errors


def _window_slices(tokens: List[str], size: int, overlap: int) -> Iterable[Tuple[int, int]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap >= size:
        raise ValueError("overlap must be < size")
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + size)
        yield start, end
        if end == n:
            break
        start = end - overlap


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """
    Simple whitespace tokenizer with sliding window chunking.
    Returns a list of chunk strings.
    """
    tokens = text.split()
    chunks: List[str] = []
    for s, e in _window_slices(tokens, size=chunk_size, overlap=chunk_overlap):
        chunks.append(" ".join(tokens[s:e]))
    return chunks


def _ensure_pinecone_index_host(dimension: int | None = None) -> str:
    """Return an index host for Pinecone, creating the index if needed.
    Prefers PINECONE_INDEX_HOST if provided. Otherwise, attempts to create
    `PINECONE_INDEX` with provided or inferred dimension.
    """
    settings = get_settings()
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set")

    from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, GcpRegion, AzureRegion, VectorType, Metric

    if settings.PINECONE_INDEX_HOST:
        return settings.PINECONE_INDEX_HOST

    if not settings.PINECONE_INDEX:
        raise RuntimeError("PINECONE_INDEX or PINECONE_INDEX_HOST must be set")

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    # Try to describe existing index to get host
    try:
        desc = pc.describe_index(settings.PINECONE_INDEX)
        if hasattr(desc, "host") and desc.host:
            return desc.host
    except Exception:
        pass

    if dimension is None:
        raise RuntimeError("Index does not exist and dimension is unknown to create it")

    # Map cloud/region
    cloud = (settings.PINECONE_CLOUD or "AWS").upper()
    region = (settings.PINECONE_REGION or "US_EAST_1").upper()
    if cloud == "AWS":
        cloud_enum = CloudProvider.AWS
        # Fallback if invalid region is provided
        region_enum = getattr(AwsRegion, region, AwsRegion.US_EAST_1)
    elif cloud in ("GCP", "GOOGLE", "GOOGLE_CLOUD"):
        cloud_enum = CloudProvider.GCP
        region_enum = getattr(GcpRegion, region, GcpRegion.US_CENTRAL1)
    elif cloud in ("AZURE",):
        cloud_enum = CloudProvider.AZURE
        region_enum = getattr(AzureRegion, region, AzureRegion.EASTUS)
    else:
        cloud_enum = CloudProvider.AWS
        region_enum = AwsRegion.US_EAST_1

    config = pc.create_index(
        name=settings.PINECONE_INDEX,
        dimension=dimension,
        metric=Metric.COSINE,
        vector_type=VectorType.DENSE,
        spec=ServerlessSpec(cloud=cloud_enum, region=region_enum),
    )
    return config.host


def _get_pinecone_index(dimension: int | None = None):
    settings = get_settings()
    host = _ensure_pinecone_index_host(dimension)
    from pinecone import Pinecone

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return pc.Index(host=host)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    settings = get_settings()
    if not settings.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    import google.generativeai as genai

    genai.configure(api_key=settings.GOOGLE_API_KEY)
    model_name = settings.EMBEDDING_MODEL

    # Batch embed by iterating; the SDK does not guarantee batch API for all models
    embeddings: List[List[float]] = []
    for t in texts:
        resp = genai.embed_content(model=model_name, content=t)
        vec = resp["embedding"] if isinstance(resp, dict) else getattr(resp, "embedding", None)
        if vec is None:
            raise RuntimeError("Embedding API did not return 'embedding'")
        embeddings.append(list(map(float, vec)))
    return embeddings


def upsert_chunks_to_vector_db(chunks: List[str]) -> int:
    """Embed chunks and upsert to Pinecone with metadata."""
    if not chunks:
        return 0

    vectors = _embed_texts(chunks)
    dim = len(vectors[0]) if vectors and isinstance(vectors[0], list) else None
    index = _get_pinecone_index(dimension=dim)

    settings = get_settings()
    ts = int(time.time())
    # Build tuples: (id, vector, metadata)
    payload = []
    for i, (text, vec) in enumerate(zip(chunks, vectors)):
        vid = f"chunk-{ts}-{i}-{uuid.uuid4().hex[:8]}"
        meta = {"text": text, "chunk_index": i, "created_at": ts}
        payload.append((vid, vec, meta))

    index.upsert(vectors=payload, namespace=settings.PINECONE_NAMESPACE)
    return len(payload)


def retrieve_and_rerank(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    settings = get_settings()
    # 1) Embed query
    qvec = _embed_texts([query])[0]
    dim = len(qvec)
    index = _get_pinecone_index(dimension=dim)

    # 2) Retrieve from Pinecone
    res = index.query(
        vector=qvec,
        top_k=max(top_k * 3, top_k),  # overfetch for reranker
        include_metadata=True,
        namespace=settings.PINECONE_NAMESPACE,
    )
    matches = getattr(res, "matches", []) or res.get("matches", [])

    # If no reranker API key, return matches as-is
    if not settings.COHERE_API_KEY:
        results = []
        for m in matches[:top_k]:
            mid = getattr(m, "id", None) or m.get("id")
            meta = getattr(m, "metadata", None) or m.get("metadata", {})
            score = getattr(m, "score", None) or m.get("score")
            results.append({"id": mid, "text": meta.get("text", ""), "score": score})
        return results

    # 3) Rerank with Cohere
    import cohere

    client = cohere.Client(settings.COHERE_API_KEY)
    docs = [ (getattr(m, "metadata", None) or m.get("metadata", {})).get("text", "") for m in matches ]
    rerank_model = settings.RERANK_MODEL
    rr = client.rerank(model=rerank_model, query=query, documents=docs, top_n=min(top_k, len(docs)))

    # rr.results is a list with 'index' mapping to docs index and 'relevance_score'
    results: List[Dict[str, Any]] = []
    for item in rr.results:
        idx = getattr(item, "index", None) if hasattr(item, "index") else item.get("index")
        score = getattr(item, "relevance_score", None) if hasattr(item, "relevance_score") else item.get("relevance_score")
        if idx is None or idx >= len(matches):
            continue
        m = matches[idx]
        mid = getattr(m, "id", None) or m.get("id")
        meta = getattr(m, "metadata", None) or m.get("metadata", {})
        results.append({"id": mid, "text": meta.get("text", ""), "score": float(score) if score is not None else None})
    return results


def generate_answer(query: str, context_chunks: List[str]) -> str:
    settings = get_settings()
    if not settings.GOOGLE_API_KEY:
        return "LLM API key not configured."

    import google.generativeai as genai

    genai.configure(api_key=settings.GOOGLE_API_KEY)
    model = genai.GenerativeModel(settings.LLM_MODEL)

    context = "\n\n".join(f"- {c}" for c in context_chunks if c)
    system = (
        "You are a helpful assistant. Answer the user's question using only the provided context. "
        "Be concise. If the answer cannot be found in the context, say you don't know."
    )
    prompt = f"""
{system}

Context:
{context}

Question: {query}

Answer with citations by referring to chunk numbers like [#1], [#2] where appropriate.
""".strip()

    try:
        resp = model.generate_content(prompt)
        # The response object may contain .text; else access candidates
        if hasattr(resp, "text") and resp.text:
            return resp.text
        return str(resp)
    except Exception as e:
        return f"LLM error: {e}"
