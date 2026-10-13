"""Shared local embedding helpers for the BGE-backed RAG pipeline."""

from __future__ import annotations

import math
import re
from typing import Any


MODEL_ID = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSIONS = 768


def load_embedding_model():
    """Load BGE from the Hugging Face cache, downloading it on first use."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is not installed. Install requirements with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    print(f"Loading local embedding model: {MODEL_ID}")
    try:
        model = SentenceTransformer(MODEL_ID)
    except Exception as error:
        raise RuntimeError(
            f"Could not load {MODEL_ID} ({type(error).__name__}). Check the model cache "
            "and install compatibility; the first load downloads the model weights."
        ) from error
    device = getattr(model, "device", "automatic device selection")
    print(f"Embedding model loaded on {device}")
    return model


def encode_texts(model: Any, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Encode text locally as normalized, 768-dimensional vectors."""
    if not texts:
        return []
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if len(encoded) != len(texts):
        raise ValueError(f"Expected {len(texts)} embeddings; received {len(encoded)}")

    vectors: list[list[float]] = []
    for index, values in enumerate(encoded):
        vector = [float(value) for value in values]
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding {index + 1} has {len(vector)} dimensions; "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Embedding {index + 1} contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"Embedding {index + 1} is not normalized (norm={norm:.6f})")
        vectors.append(vector)
    return vectors


def is_reusable_embedding(embedding_model: str, embedding_dimensions: int) -> bool:
    """Accept only vectors produced by this model at the expected dimensions."""
    return (
        embedding_model == MODEL_ID
        and int(embedding_dimensions) == EMBEDDING_DIMENSIONS
    )


def retrieval_function_uses_model(definition: str | None) -> bool:
    """Return whether the function explicitly filters to reference BGE rows."""
    if not definition:
        return False
    normalized = definition.casefold()
    model_filter = re.search(
        rf"\b(?:[a-z_][a-z0-9_]*\.)?embedding_model\s*=\s*['\"]{re.escape(MODEL_ID.casefold())}['\"]",
        normalized,
    )
    reference_filter = re.search(
        r"\b(?:[a-z_][a-z0-9_]*\.)?split\s*=\s*['\"]reference['\"]",
        normalized,
    )
    return bool(model_filter and reference_filter)
