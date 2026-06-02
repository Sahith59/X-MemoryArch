"""
Sub-phase 2.6 — Dual-index migration + backfill service.

Architectural rule 11 (from phase-2-plan.md):
  Never enable ANN without verifying ≥ 0.95 Recall@k vs exact on sample queries.
  Shadow-run the new ANN index alongside exact, compare overlap, then switch.

Architectural rule (from phase-2-plan.md):
  Never replace embeddings in place. Always build a new index with a shadow run,
  verify Recall@k, then switch.

Workflow:
  1. BackfillService.backfill_project()  — load memories, upsert to target backend
  2. BackfillService.validate_ann_recall() — shadow-run, compute Recall@k vs exact
  3. If ≥ 0.95: BackfillService.activate_backend() — update VectorIndexState
  4. If < 0.95: reject, keep exact baseline

Usage:
  from app.services.vector_backends.migration import BackfillService
  svc = BackfillService()
  state = svc.backfill_project(db, project_id, ann_backend, "BAAI/bge-small-en-v1.5")
  validation = svc.validate_ann_recall(db, project_id, ann_backend)
  if validation["passes_threshold"]:
      svc.activate_backend(db, project_id, ann_backend, state)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.services.vector_backends.base import VectorBackend

_MIN_RECALL = 0.95
_DEFAULT_BATCH = 100
_DEFAULT_SAMPLE = 50
_DEFAULT_K = 10


# ---------------------------------------------------------------------------
# Backfill + validation service
# ---------------------------------------------------------------------------

class BackfillService:
    """
    Orchestrates dual-index migration and ANN recall validation.

    Does NOT call embedding models during backfill — it uses the float32
    embedding bytes already stored in the Phase 1 `memories` table.
    Call embed_with_model() separately if re-embedding is needed.
    """

    def backfill_project(
        self,
        db: "Session",
        project_id: str,
        target_backend: "VectorBackend",
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = _DEFAULT_BATCH,
        only_missing: bool = False,
    ) -> "VectorIndexState":
        """
        Load embedded memories from SQLite and upsert them to target_backend.

        Args:
            db:               SQLAlchemy session.
            project_id:       Project to backfill.
            target_backend:   Destination ANN backend.
            embedding_model:  Model name to record in VectorIndexState.
            batch_size:       Memories per batch (controls memory usage).
            only_missing:     If True, skip memories already in the backend
                              (not supported by all backends — conservative default=False
                               means full re-index).

        Returns:
            The upserted or created VectorIndexState record.
        """
        from app import models as phase1_models
        from app.p2_models import VectorIndexState
        from app.services.vector_backends.embedding_models import (
            get_model_info, model_checksum,
        )

        _now = datetime.now(timezone.utc)

        # Load all active memories with embeddings
        rows = (
            db.query(phase1_models.Memory)
            .filter(
                phase1_models.Memory.project_id == project_id,
                phase1_models.Memory.embedding.isnot(None),
                phase1_models.Memory.status == "active",
            )
            .all()
        )

        dim = 0
        backfill_rows = []
        for mem in rows:
            try:
                vec = np.frombuffer(mem.embedding, dtype=np.float32)
                dim = int(vec.shape[0])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                backfill_rows.append((
                    mem.id,
                    vec.tolist(),
                    {
                        "project_id": project_id,
                        "embedding_model": embedding_model,
                        "memory_type": mem.type,
                    },
                ))
            except Exception:
                continue

        # Batch upsert to target backend
        for i in range(0, len(backfill_rows), batch_size):
            batch = backfill_rows[i:i + batch_size]
            target_backend.batch_backfill(batch)

        total = len(backfill_rows)
        checksum = model_checksum(embedding_model, target_backend.name, dim)

        # Upsert VectorIndexState — deactivate old, create new
        existing = (
            db.query(VectorIndexState)
            .filter(
                VectorIndexState.project_id == project_id,
                VectorIndexState.is_active == True,  # noqa: E712
            )
            .first()
        )
        if existing:
            existing.is_active = False
            existing.updated_at = _now

        state = VectorIndexState(
            id=str(uuid.uuid4()),
            project_id=project_id,
            backend=target_backend.name,
            embedding_model=embedding_model,
            embedding_dim=max(dim, 0),
            index_checksum=checksum,
            total_indexed=total,
            last_backfill_at=_now,
            is_active=False,  # Only activated after validate_ann_recall passes
            created_at=_now,
            updated_at=_now,
        )
        db.add(state)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        return state

    def validate_ann_recall(
        self,
        db: "Session",
        project_id: str,
        ann_backend: "VectorBackend",
        k: int = _DEFAULT_K,
        sample_size: int = _DEFAULT_SAMPLE,
        min_recall: float = _MIN_RECALL,
    ) -> dict:
        """
        Shadow-run ann_backend vs SQLiteExactBackend; compute Recall@k.

        Architectural rule 11: ANN must achieve ≥ 0.95 Recall@k vs exact
        before it can be activated.

        Returns:
          {
            "overall_ann_recall": float,
            "passes_threshold": bool,
            "per_query_recalls": list[float],
            "threshold": float,
            "sample_size": int,
          }
        """
        from app import models as phase1_models
        from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
        from app.services.benchmark.eval_metrics import ann_recall_at_k

        exact_backend = SQLiteExactBackend(db)

        # Sample queries: use embedding vectors from stored memories as proxies
        sample_mems = (
            db.query(phase1_models.Memory)
            .filter(
                phase1_models.Memory.project_id == project_id,
                phase1_models.Memory.embedding.isnot(None),
                phase1_models.Memory.status == "active",
            )
            .limit(sample_size)
            .all()
        )

        if not sample_mems:
            return {
                "overall_ann_recall": 1.0,  # vacuous: no data to compare
                "passes_threshold": True,
                "per_query_recalls": [],
                "threshold": min_recall,
                "sample_size": 0,
            }

        per_query_recalls: list[float] = []

        for mem in sample_mems:
            try:
                qvec = np.frombuffer(mem.embedding, dtype=np.float32).tolist()
            except Exception:
                continue

            exact_results = [
                mid for mid, _ in exact_backend.search(qvec, top_k=k, project_id=project_id)
            ]
            ann_results = [
                mid for mid, _ in ann_backend.search(qvec, top_k=k, project_id=project_id)
            ]
            r = ann_recall_at_k(exact_results, ann_results, k=k)
            per_query_recalls.append(r)

        overall = (
            sum(per_query_recalls) / len(per_query_recalls) if per_query_recalls else 1.0
        )

        return {
            "overall_ann_recall": overall,
            "passes_threshold": overall >= min_recall,
            "per_query_recalls": per_query_recalls,
            "threshold": min_recall,
            "sample_size": len(per_query_recalls),
        }

    def activate_backend(
        self,
        db: "Session",
        project_id: str,
        backend: "VectorBackend",
        state: "VectorIndexState",
    ) -> None:
        """
        Mark a VectorIndexState as active (after validation passes).

        Deactivates any previously active state for this project.
        """
        from app.p2_models import VectorIndexState

        # Deactivate all other states for this project
        others = (
            db.query(VectorIndexState)
            .filter(
                VectorIndexState.project_id == project_id,
                VectorIndexState.id != state.id,
                VectorIndexState.is_active == True,  # noqa: E712
            )
            .all()
        )
        for other in others:
            other.is_active = False

        state.is_active = True
        state.updated_at = datetime.now(timezone.utc)

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

    def get_active_state(
        self, db: "Session", project_id: str
    ) -> "VectorIndexState | None":
        """Return the currently active VectorIndexState for a project."""
        from app.p2_models import VectorIndexState
        return (
            db.query(VectorIndexState)
            .filter(
                VectorIndexState.project_id == project_id,
                VectorIndexState.is_active == True,  # noqa: E712
            )
            .first()
        )
