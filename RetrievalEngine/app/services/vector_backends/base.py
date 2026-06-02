"""
VectorBackend Protocol — pluggable interface for all vector search backends.

Sub-phase 2.1: SQLiteExactBackend (exact cosine, correctness-first)
Sub-phase 2.6: ChromaBackend, FAISSBackend, sqlite-vec, QdrantBackend

Key design constraint (from phase-2-plan.md):
  SQLite is ALWAYS the source of truth. The orchestration layer pre-filters in
  SQLite first (privacy, status, etc.), then passes `allowed_ids` to the backend.
  The privacy gate is a SQLite authority — never delegated to vector backend filters.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """
    Pluggable vector similarity backend.

    All implementations must be safe to swap at runtime. The retrieval service
    pre-filters allowed_ids in SQLite before calling search(), so backends need
    not re-implement the privacy/status gate.
    """

    @property
    def name(self) -> str:
        """Short backend identifier, e.g. 'sqlite_exact', 'faiss', 'qdrant'."""
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        project_id: str,
        allowed_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return (memory_id, cosine_similarity) pairs, sorted descending.

        Args:
            query_vector:  Float32 unit-normed query embedding.
            top_k:         Max results to return.
            project_id:    Scope search to this project.
            allowed_ids:   Pre-filtered whitelist from SQLite hard filters.
                           If None, search all embedded memories in the project.
                           If empty list, return [] immediately.
        """
        ...

    def upsert(
        self,
        memory_id: str,
        vector: list[float],
        project_id: str,
        embedding_model: str,
        embedding_dim: int,
        payload: dict | None = None,
    ) -> None:
        """Insert or update a single memory's embedding."""
        ...

    def delete(self, memory_id: str) -> None:
        """Remove a memory's embedding from the index."""
        ...

    def batch_backfill(
        self,
        rows: list[tuple[str, list[float], dict]],
    ) -> None:
        """
        Bulk insert (memory_id, vector, payload) rows.
        Used for initial backfill and index migrations.
        """
        ...

    def stats(self) -> dict:
        """Return backend stats: total indexed, model, backend name, etc."""
        ...
