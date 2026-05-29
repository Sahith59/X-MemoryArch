"""
Sub-phase 2.5 — Cross-encoder reranker.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Hugging Face via sentence-transformers)

Behaviour:
  - If sentence_transformers is not installed, or model loading fails,
    rerank() returns the original order unchanged (graceful passthrough).
  - Top-N candidates (default 20) are scored by the cross-encoder;
    remaining candidates are appended at the end with a low sentinel score.
  - is_available() → bool indicates whether the model was loaded successfully.

Usage:
  from app.services.retrieval.reranker import CrossEncoderReranker
  reranker = CrossEncoderReranker()
  ranked = reranker.rerank(query, memories, top_n=20)
  # → list of (memory_id, score) sorted desc
"""
from __future__ import annotations

import threading

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_SENTINEL_SCORE = -9999.0


class CrossEncoderReranker:
    """
    Lazy-loading cross-encoder reranker.

    The model is loaded on first call to rerank() or is_available.
    Thread-safe via a lock; multiple concurrent calls block until the first
    load completes.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        self._available: bool | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        if self._available is None:
            self._load()
        return self._available  # type: ignore[return-value]

    def _load(self) -> None:
        with self._lock:
            if self._available is not None:
                return
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                self._available = True
            except Exception:
                self._model = None
                self._available = False

    def rerank(
        self,
        query: str,
        memories: list,
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Rerank memories using the cross-encoder.

        Args:
            query:    The original retrieval query.
            memories: Memory ORM objects in current rank order.
            top_n:    Number of top candidates to pass to the cross-encoder.
                      Memories beyond top_n are appended at the end.

        Returns:
            list of (memory_id, score) sorted by score descending.
            When not available, returns original order with ordinal scores.
        """
        if not memories:
            return []

        if not self.available:
            return _passthrough(memories)

        candidates = memories[:top_n]
        tail = memories[top_n:]

        # Build (query, document) pairs — title + content as document
        pairs = [
            (query, f"{m.title}. {m.content[:512]}")
            for m in candidates
        ]

        try:
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception:
            return _passthrough(memories)

        scored = sorted(
            zip([m.id for m in candidates], raw_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Append tail with sentinel scores (ordered by existing rank)
        for i, m in enumerate(tail):
            scored.append((m.id, _SENTINEL_SCORE - i))

        return scored


def _passthrough(memories: list) -> list[tuple[str, float]]:
    """Return memories in their original order with ordinal scores (N, N-1, …)."""
    n = len(memories)
    return [(m.id, float(n - i)) for i, m in enumerate(memories)]


# ---------------------------------------------------------------------------
# Module-level singleton (avoids re-loading model on every retrieve() call)
# ---------------------------------------------------------------------------

_default_reranker: CrossEncoderReranker | None = None
_reranker_lock = threading.Lock()


def get_default_reranker(model_name: str = DEFAULT_MODEL) -> CrossEncoderReranker:
    """Return a shared singleton reranker instance."""
    global _default_reranker
    if _default_reranker is None:
        with _reranker_lock:
            if _default_reranker is None:
                _default_reranker = CrossEncoderReranker(model_name)
    return _default_reranker
