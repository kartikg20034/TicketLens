from __future__ import annotations
import os
from functools import lru_cache

MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DIMENSIONS = 1536

@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for real embeddings")
    return OpenAI(api_key=key)

def embed(text: str) -> list[float]:
    response = _client().embeddings.create(model=MODEL, input=text)
    return response.data[0].embedding

def embed_many(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = _client().embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
