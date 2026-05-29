"""
Sub-phase 2.7 — HyDE (Hypothetical Document Embeddings).

Key insight (from phase-2-plan.md):
  Strongest gains for semantic-drift queries — where the phrasing of the query
  differs significantly from how the information is stored in memory.

  Example: query "what database are we using in prod?" may not semantically
  match the stored memory "PostgreSQL chosen for production due to concurrency
  requirements" because the phrasing styles are very different. HyDE bridges
  this gap by generating a hypothetical memory in the style of stored memories.

Architecture decision:
  Use HyDE ONLY for vague or low-confidence semantic queries.
  Not for every query — it adds 200-500ms LLM latency plus an embedding call.
  The retrieval pipeline gates HyDE on:
    - enable_hyde=True in RetrievalConfig
    - Query confidence below hyde_confidence_threshold (default 0.7)
    - OR explicit intent override "semantic" or "exploratory"

HyDE vector combination strategies:
  - "max":    element-wise max of query_vec and hyde_vec (favors overlap)
  - "avg":    arithmetic average (balanced)
  - "query":  use only query_vec (HyDE generated but not used — for A/B testing)
  - "hyde":   use only hyde_vec (for evaluation)

Default: "avg" — balances original query semantics with hypothetical document context.

Usage:
  from app.services.retrieval.hyde import generate_hyde_vector, HyDEConfig
  cfg = HyDEConfig(llm_fn=my_llm)
  vec = generate_hyde_vector("what database do we use?", cfg)
  # vec is list[float], same dim as base model (384 for all-MiniLM-L6-v2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HyDEConfig:
    """Configuration for HyDE generation."""
    llm_fn: Callable[[str], str] | None = None   # required for LLM path
    combination: str = "avg"                      # max | avg | query | hyde
    confidence_threshold: float = 0.70            # below this → use HyDE
    embedding_model: str = "all-MiniLM-L6-v2"
    max_hyde_tokens: int = 120                    # rough cap on generated text


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_HYDE_PROMPT = """\
Write a short, concise memory note that would likely answer the following user query.
Write it as if it were a stored project memory — factual, terse, third-person.
Do not say "hypothetical" or "example". Just write the memory note directly.

Query: {query}

Constraints:
- 40 to 100 words
- Include likely entities, decision rationale, or core fact
- Style: written as a stored project note, not a question or answer
- Do not start with "This memory" or "Memory:"
Write only the memory note:"""


# ---------------------------------------------------------------------------
# Core HyDE functions
# ---------------------------------------------------------------------------

def generate_hyde_text(query: str, llm_fn: Callable[[str], str]) -> str:
    """
    Generate a hypothetical memory document for a query via LLM.

    Args:
        query:  User query string.
        llm_fn: Callable (prompt: str) -> str.

    Returns:
        Hypothetical document text (stripped).

    Raises:
        ValueError if LLM returns empty text.
        RuntimeError on LLM call failure (propagated to caller for graceful skip).
    """
    if not query or not query.strip():
        raise ValueError("Cannot generate HyDE for empty query")
    prompt = _HYDE_PROMPT.format(query=query.strip())
    result = llm_fn(prompt)
    if not result or not result.strip():
        raise ValueError("LLM returned empty HyDE document")
    # Trim to approximate token cap (4 chars/token × max_hyde_tokens)
    return result.strip()[:480]


def embed_text_to_vec(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed text and return a normalized float32 numpy array.
    """
    if model_name == "all-MiniLM-L6-v2":
        from app.services.semantic_classifier import embed_text
        raw = embed_text(text)
    else:
        from app.services.vector_backends.embedding_models import embed_with_model
        raw = embed_with_model(text, model_name)

    vec = np.frombuffer(raw, dtype=np.float32).copy()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def combine_vectors(
    query_vec: np.ndarray,
    hyde_vec: np.ndarray,
    strategy: str = "avg",
) -> list[float]:
    """
    Combine query and HyDE vectors according to strategy.

    Strategies:
      "avg":   arithmetic mean of both vectors, then re-normalise
      "max":   element-wise maximum, then re-normalise
      "query": return query_vec unchanged
      "hyde":  return hyde_vec unchanged
    """
    if strategy == "query":
        return query_vec.tolist()
    if strategy == "hyde":
        return hyde_vec.tolist()
    if strategy == "avg":
        combined = (query_vec + hyde_vec) / 2.0
    elif strategy == "max":
        combined = np.maximum(query_vec, hyde_vec)
    else:
        raise ValueError(f"Unknown HyDE combination strategy: {strategy!r}")

    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined.tolist()


def generate_hyde_vector(
    query: str,
    config: HyDEConfig,
    query_vector: list[float] | None = None,
) -> list[float] | None:
    """
    Full HyDE pipeline: generate hypothetical doc, embed, combine with query vector.

    Args:
        query:        User query string.
        config:       HyDEConfig (llm_fn must be set).
        query_vector: Pre-computed query embedding as list[float] (optional;
                      generated if not provided).

    Returns:
        Combined vector as list[float], or None if HyDE is unavailable/fails.
        The caller falls back to plain query_vector if None is returned.
    """
    if config.llm_fn is None:
        return None

    try:
        # 1. Generate hypothetical document
        hyde_text = generate_hyde_text(query, config.llm_fn)

        # 2. Embed hypothetical document
        hyde_vec = embed_text_to_vec(hyde_text, config.embedding_model)

        # 3. Get or compute query vector
        if query_vector is not None:
            q_vec = np.array(query_vector, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
        else:
            q_vec = embed_text_to_vec(query, config.embedding_model)

        # 4. Combine
        return combine_vectors(q_vec, hyde_vec, config.combination)

    except Exception:
        return None  # Graceful degradation — caller uses plain query vector


def should_use_hyde(intent: str, confidence: float, threshold: float = 0.70) -> bool:
    """
    Decide whether to invoke HyDE based on intent and confidence.

    HyDE is most valuable when:
      - Intent is exploratory (vague, open-ended)
      - Confidence is low (heuristic didn't fire cleanly)
      - NOT for factual / temporal lookups (those are precise — HyDE muddies them)
    """
    if intent in ("exploratory", "general"):
        return True
    if confidence < threshold:
        return True
    return False
