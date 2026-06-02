"""
Sub-phase 2.6 — Embedding model registry.

Supported models (from phase-2-plan.md):
  all-MiniLM-L6-v2       — current Phase 1 baseline (384-dim)
  BAAI/bge-small-en-v1.5  — strongest drop-in local upgrade (384-dim, same storage)
  nomic-embed-text-v1.5   — dimension-flexible (Matryoshka: 768→64)
  text-embedding-3-small   — cloud path (OpenAI API)

Dual-index migration rule (architectural rule):
  Never replace embeddings in place. Always build a new index, shadow-run
  with ≥ 0.95 ANN Recall@k vs exact on sample queries, then switch.

Usage:
  from app.services.vector_backends.embedding_models import get_model_info, embed_with_model
  info = get_model_info("BAAI/bge-small-en-v1.5")  # EmbeddingModelInfo(dim=384, ...)
  vec = embed_with_model("some text", "BAAI/bge-small-en-v1.5")
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelInfo:
    name: str
    dim: int
    description: str
    is_local: bool              # False = requires API key
    is_drop_in: bool            # True = same dim as baseline, can share storage schema
    requires_package: str       # install hint


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, EmbeddingModelInfo] = {
    "all-MiniLM-L6-v2": EmbeddingModelInfo(
        name="all-MiniLM-L6-v2",
        dim=384,
        description="Phase 1 baseline — sentence-transformers, 384-dim",
        is_local=True,
        is_drop_in=True,
        requires_package="sentence-transformers",
    ),
    "BAAI/bge-small-en-v1.5": EmbeddingModelInfo(
        name="BAAI/bge-small-en-v1.5",
        dim=384,
        description="Strongest drop-in local upgrade — 384-dim, same storage",
        is_local=True,
        is_drop_in=True,
        requires_package="sentence-transformers",
    ),
    "nomic-embed-text-v1.5": EmbeddingModelInfo(
        name="nomic-embed-text-v1.5",
        dim=768,
        description="Best open model for dimension flexibility (Matryoshka: 768→64)",
        is_local=True,
        is_drop_in=False,
        requires_package="sentence-transformers",
    ),
    "text-embedding-3-small": EmbeddingModelInfo(
        name="text-embedding-3-small",
        dim=1536,
        description="OpenAI cloud embedder — best simple cloud option",
        is_local=False,
        is_drop_in=False,
        requires_package="openai",
    ),
}

# Default model (Phase 1 baseline)
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_model_info(model_name: str) -> EmbeddingModelInfo:
    """Return EmbeddingModelInfo for a registered model name."""
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown embedding model: {model_name!r}. "
            f"Registered models: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[model_name]


def list_models() -> list[EmbeddingModelInfo]:
    """Return all registered embedding models."""
    return list(_REGISTRY.values())


def model_checksum(model_name: str, backend: str, dim: int) -> str:
    """
    Deterministic checksum for (model, backend, dim) triplet.

    Used by VectorIndexState to detect configuration drift.
    """
    payload = f"{model_name}:{backend}:{dim}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Embedding function dispatcher
# ---------------------------------------------------------------------------

def embed_with_model(text: str, model_name: str = DEFAULT_MODEL) -> bytes:
    """
    Embed text using the specified model. Returns float32 bytes.

    Falls back to the Phase 1 `embed_text` for the baseline model.
    For other models, uses sentence-transformers directly.
    Raises ImportError if the required package is not installed.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    info = get_model_info(model_name)

    if model_name == DEFAULT_MODEL:
        # Use existing Phase 1 pipeline
        from app.services.semantic_classifier import embed_text
        return embed_text(text)

    if info.is_local:
        return _embed_sentence_transformers(text, model_name)

    if model_name == "text-embedding-3-small":
        return _embed_openai(text)

    raise NotImplementedError(f"No embed implementation for {model_name!r}")


def _embed_sentence_transformers(text: str, model_name: str) -> bytes:
    """Embed using sentence-transformers (lazy-loaded per model)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers"
        )
    import numpy as np

    # Lazy model cache (process-scoped, keyed by model_name)
    cache = _embed_sentence_transformers.__dict__.setdefault("_cache", {})
    if model_name not in cache:
        cache[model_name] = SentenceTransformer(model_name)
    model = cache[model_name]

    vec = model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32).tobytes()


def _embed_openai(text: str) -> bytes:
    """Embed using OpenAI API (text-embedding-3-small)."""
    import os
    import numpy as np
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is not installed. Install with: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    vec = response.data[0].embedding
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()
