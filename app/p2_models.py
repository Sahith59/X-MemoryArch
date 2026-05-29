"""
Phase 2 SQLAlchemy models.

Uses Phase 1's Base (resolved via namespace extension in conftest/startup) so
that Base.metadata.create_all() creates both Phase 1 and Phase 2 tables in
the same database.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base  # Phase 1's Base via extended __path__


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RetrievalRun(Base):
    """
    Telemetry row written for every retrieval call.

    Drives: latency tracking, privacy leakage audit (forbidden_candidate_count),
    Recall@k evaluation, and eventually weighted RRF learning.
    """
    __tablename__ = "retrieval_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String, nullable=True)
    filters_applied: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON

    # Per-leg candidate counts (before fusion)
    candidate_count_bm25: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_count_dense: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_count_entity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fused_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Privacy leakage audit — candidates blocked by hard filter (must always be 0 in production)
    forbidden_candidate_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Result
    selected_memory_ids: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Context assembly metrics (populated in Sub-phase 2.2)
    packet_token_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    packet_compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    judge_relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Backend identity
    backend_used: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    def set_selected_ids(self, ids: list[str]) -> None:
        self.selected_memory_ids = json.dumps(ids)

    def get_selected_ids(self) -> list[str]:
        return json.loads(self.selected_memory_ids) if self.selected_memory_ids else []

    def set_filters(self, d: dict) -> None:
        self.filters_applied = json.dumps(d)


class VectorIndexState(Base):
    """
    Sub-phase 2.6 — Per-project vector index metadata.

    Tracks which ANN backend + embedding model is active, backfill progress,
    and a checksum for drift detection. One active row per project at any time.
    Multiple historical rows are allowed (is_active=False) for audit.
    """
    __tablename__ = "vector_index_state"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String, nullable=False, index=True)

    # Which backend and model are in use
    backend: Mapped[str] = mapped_column(String, nullable=False)          # sqlite_exact/chroma/faiss/qdrant
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)

    # Backfill tracking
    index_checksum: Mapped[str | None] = mapped_column(String, nullable=True)   # sha256 of (backend+model+dim)
    total_indexed: Mapped[int] = mapped_column(Integer, default=0)
    last_backfill_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Only one row per project should be is_active=True
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
