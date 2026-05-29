"""
Phase 2.1 — Retrieval API router.

POST /projects/{project_id}/retrieve
  Run the full hybrid retrieval pipeline for a project.

Mounts onto the Phase 1 FastAPI app at merge time (Sub-phase merge).
For development: can be tested independently via the service layer.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.retrieval.retrieval_service import RetrievalConfig, RetrievalResult, retrieve
from app.services.vector_backends.sqlite_exact import SQLiteExactBackend

router = APIRouter(prefix="/projects/{project_id}", tags=["retrieval"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=100)
    max_clearance: str = Field("internal", pattern="^(public|internal|sensitive|secret)$")
    include_superseded: bool = False
    embed_query: bool = True
    intent: str | None = None


class MemoryOut(BaseModel):
    id: str
    title: str
    content: str
    type: str
    importance: int
    confidence: float | None
    canonical_type: str | None
    privacy_level: str
    source_quote: str | None
    created_at: Any  # datetime — serialised as ISO string
    updated_at: Any

    class Config:
        from_attributes = True


class RetrieveResponse(BaseModel):
    run_id: str
    query: str
    memories: list[MemoryOut]
    total_returned: int
    latency_ms: int
    candidate_count_bm25: int
    candidate_count_dense: int
    candidate_count_entity: int
    fused_count: int
    forbidden_candidate_count: int


# ---------------------------------------------------------------------------
# Dependency — get DB session (wired to Phase 1's get_db at merge time)
# ---------------------------------------------------------------------------

def _get_db():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_memories(
    project_id: str,
    body: RetrieveRequest,
    db: Session = Depends(_get_db),
) -> RetrieveResponse:
    """
    Run hybrid retrieval (BM25 + dense + entity) with RRF fusion.

    Hard filters are always applied:
      - privacy_level ≤ max_clearance
      - status ≠ superseded (unless include_superseded=True)
      - review_status ≠ rejected
      - valid_until IS NULL OR valid_until > now
    """
    # Verify project exists
    from app import crud
    project = crud.get_project(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    cfg = RetrievalConfig(
        top_k=body.top_k,
        max_clearance=body.max_clearance,
        include_superseded=body.include_superseded,
        embed_query=body.embed_query,
        intent=body.intent,
    )

    vector_backend = SQLiteExactBackend(db)
    result: RetrievalResult = retrieve(
        db=db,
        project_id=project_id,
        query=body.query,
        vector_backend=vector_backend,
        config=cfg,
    )

    memories_out = []
    for m in result.memories:
        memories_out.append(MemoryOut(
            id=m.id,
            title=m.title,
            content=m.content,
            type=m.type,
            importance=m.importance,
            confidence=getattr(m, "confidence", None),
            canonical_type=getattr(m, "canonical_type", None),
            privacy_level=m.privacy_level,
            source_quote=getattr(m, "source_quote", None),
            created_at=m.created_at,
            updated_at=m.updated_at,
        ))

    return RetrieveResponse(
        run_id=result.run_id,
        query=result.query,
        memories=memories_out,
        total_returned=len(memories_out),
        latency_ms=result.latency_ms,
        candidate_count_bm25=result.candidate_count_bm25,
        candidate_count_dense=result.candidate_count_dense,
        candidate_count_entity=result.candidate_count_entity,
        fused_count=result.fused_count,
        forbidden_candidate_count=result.forbidden_candidate_count,
    )
