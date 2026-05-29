"""
SQLiteExactBackend — Phase 2.1 correctness-first vector backend.

Reads float32 embedding blobs from the Phase 1 `memories` table and computes
exact cosine similarity in numpy. No ANN approximation — every eligible
embedding is scored. This is the baseline for ANN Recall@k validation
(Sub-phase 2.6 requires ≥ 0.95 overlap vs exact before enabling any ANN backend).

Performance characteristics:
  - Scales linearly with number of embedded memories
  - Adequate up to ~10K memories at sub-200ms p95 latency on CPU
  - For > 10K memories, migrate to ChromaBackend or FAISSBackend in Sub-phase 2.6

Usage in retrieval pipeline:
  backend = SQLiteExactBackend(db)
  results = backend.search(query_vec, top_k=50, project_id=pid, allowed_ids=ids)
"""
from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class SQLiteExactBackend:
    """Exact cosine similarity over Phase 1 embedding blobs."""

    def __init__(self, db: "Session") -> None:
        self._db = db

    @property
    def name(self) -> str:
        return "sqlite_exact"

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        project_id: str,
        allowed_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return top-k (memory_id, cosine_similarity) pairs, sorted descending.

        Embeddings are stored as float32 bytes (np.array.tobytes()).
        Cosine similarity is valid when embeddings are unit-normed at write time
        (Phase 1's embed_text normalizes before storing).
        """
        if allowed_ids is not None and len(allowed_ids) == 0:
            return []

        from app import models  # Phase 1 models via extended __path__

        query_vec = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Load embeddings — filter by project and allowed_ids in one ORM query
        orm_q = (
            self._db.query(models.Memory.id, models.Memory.embedding)
            .filter(
                models.Memory.project_id == project_id,
                models.Memory.embedding.isnot(None),
            )
        )
        if allowed_ids is not None:
            orm_q = orm_q.filter(models.Memory.id.in_(allowed_ids))

        rows = orm_q.all()
        if not rows:
            return []

        ids: list[str] = []
        matrix_rows: list[np.ndarray] = []

        for mid, emb_bytes in rows:
            try:
                vec = np.frombuffer(emb_bytes, dtype=np.float32)
                if vec.shape[0] != query_vec.shape[0]:
                    continue
                matrix_rows.append(vec)
                ids.append(mid)
            except Exception:
                continue

        if not matrix_rows:
            return []

        matrix = np.vstack(matrix_rows)  # (N, dim)
        sims = matrix @ query_vec        # cosine similarity (unit-normed vectors)

        # Partial sort — only need top_k
        top_k_actual = min(top_k, len(ids))
        top_indices = np.argpartition(sims, -top_k_actual)[-top_k_actual:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        return [(ids[i], float(sims[i])) for i in top_indices]

    def upsert(
        self,
        memory_id: str,
        vector: list[float],
        project_id: str,
        embedding_model: str,
        embedding_dim: int,
        payload: dict | None = None,
    ) -> None:
        """Update the embedding on the Memory row directly in Phase 1's table."""
        from app import models

        mem = self._db.get(models.Memory, memory_id)
        if mem is None:
            return
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        mem.embedding = vec.tobytes()
        mem.embedding_model = embedding_model
        mem.embedding_dim = embedding_dim
        self._db.commit()

    def delete(self, memory_id: str) -> None:
        from app import models

        mem = self._db.get(models.Memory, memory_id)
        if mem:
            mem.embedding = None
            self._db.commit()

    def batch_backfill(
        self,
        rows: list[tuple[str, list[float], dict]],
    ) -> None:
        from app import models

        for memory_id, vector, _payload in rows:
            mem = self._db.get(models.Memory, memory_id)
            if mem is None:
                continue
            vec = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            mem.embedding = vec.tobytes()
        self._db.commit()

    def stats(self) -> dict:
        from app import models

        total = (
            self._db.query(models.Memory.id)
            .filter(models.Memory.embedding.isnot(None))
            .count()
        )
        return {
            "backend": self.name,
            "total_indexed": total,
            "ann": False,
            "description": "Exact cosine similarity — correctness baseline",
        }
