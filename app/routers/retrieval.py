"""
Phase 2.1/2.2/2.4 — Retrieval API router.

Endpoints:
  POST /projects/{project_id}/retrieve
    Run the full hybrid retrieval pipeline. Returns ranked memories.

  POST /projects/{project_id}/retrieve/context
    Retrieve + assemble a token-budgeted context packet.
    Wires into the existing ContextPacket model.
    Returns the packet content with Compression Ratio + RCD proxy metrics.

  POST /projects/{project_id}/memories/generate-cluster-summaries
    Generate RAPTOR-style cluster summaries for clusters ≥ min_cluster_size.
    Stores each summary as a Memory row (type="cluster_summary").

Mounts onto the Phase 1 FastAPI app at merge time.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.retrieval.retrieval_service import RetrievalConfig, RetrievalResult, retrieve
from app.services.vector_backends.sqlite_exact import SQLiteExactBackend

router = APIRouter(prefix="/projects/{project_id}", tags=["retrieval"])


# ---------------------------------------------------------------------------
# Request / Response schemas — /retrieve
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
    retrieval_hint: str | None
    created_at: Any
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
# Request / Response schemas — /retrieve/context
# ---------------------------------------------------------------------------

class ContextRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    token_budget: int = Field(4000, ge=100, le=32000, description="Max tokens for the assembled packet")
    top_k: int = Field(20, ge=1, le=100, description="Candidates to retrieve before assembly")
    max_clearance: str = Field("internal", pattern="^(public|internal|sensitive|secret)$")
    include_superseded: bool = False
    embed_query: bool = True
    target_tool: str = Field("Claude", max_length=64)
    include_source_quote: bool = True
    include_retrieval_hint: bool = True


class ContextPacketResponse(BaseModel):
    packet_id: str
    run_id: str
    query: str
    content: str
    token_count: int
    token_budget: int
    compression_ratio: float
    rcd_proxy: float
    included_memory_count: int
    excluded_memory_count: int
    used_abstractive_fallback: bool
    latency_ms: int


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def _get_db():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# POST /retrieve
# ---------------------------------------------------------------------------

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_memories(
    project_id: str,
    body: RetrieveRequest,
    db: Session = Depends(_get_db),
) -> RetrieveResponse:
    """
    Run hybrid retrieval (BM25 + dense + entity) with RRF fusion.

    Hard filters always applied:
      - privacy_level ≤ max_clearance
      - status ≠ superseded (unless include_superseded=True)
      - review_status ≠ rejected
      - valid_until IS NULL OR valid_until > now
    """
    from app import crud
    if crud.get_project(db, project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    cfg = RetrievalConfig(
        top_k=body.top_k,
        max_clearance=body.max_clearance,
        include_superseded=body.include_superseded,
        embed_query=body.embed_query,
        intent=body.intent,
    )
    result: RetrievalResult = retrieve(
        db=db,
        project_id=project_id,
        query=body.query,
        vector_backend=SQLiteExactBackend(db),
        config=cfg,
    )

    return RetrieveResponse(
        run_id=result.run_id,
        query=result.query,
        memories=[
            MemoryOut(
                id=m.id,
                title=m.title,
                content=m.content,
                type=m.type,
                importance=m.importance,
                confidence=getattr(m, "confidence", None),
                canonical_type=getattr(m, "canonical_type", None),
                privacy_level=m.privacy_level,
                source_quote=getattr(m, "source_quote", None),
                retrieval_hint=getattr(m, "retrieval_hint", None),
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in result.memories
        ],
        total_returned=len(result.memories),
        latency_ms=result.latency_ms,
        candidate_count_bm25=result.candidate_count_bm25,
        candidate_count_dense=result.candidate_count_dense,
        candidate_count_entity=result.candidate_count_entity,
        fused_count=result.fused_count,
        forbidden_candidate_count=result.forbidden_candidate_count,
    )


# ---------------------------------------------------------------------------
# POST /retrieve/context
# ---------------------------------------------------------------------------

@router.post("/retrieve/context", response_model=ContextPacketResponse)
def retrieve_context(
    project_id: str,
    body: ContextRetrieveRequest,
    db: Session = Depends(_get_db),
) -> ContextPacketResponse:
    """
    Retrieve memories and assemble a token-budgeted context packet.

    Pipeline:
      1. Hybrid retrieval (BM25 + dense + entity + RRF fusion)
      2. Token-budgeted extractive assembly (highest-ranked first, drop if over budget)
      3. Abstractive fallback: cluster_label summary for excluded clustered memories
      4. Persist to ContextPacket table (wires into existing Phase 1 model)
      5. Update RetrievalRun with packet metrics (compression_ratio, token_budget)

    Returns the assembled packet with Compression Ratio + RCD proxy metrics.
    """
    from app import crud, models as phase1_models
    from app.p2_models import RetrievalRun
    from app.services.retrieval.context_assembly import assemble_context_packet

    project = crud.get_project(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    # Step 1: Retrieve
    cfg = RetrievalConfig(
        top_k=body.top_k,
        max_clearance=body.max_clearance,
        include_superseded=body.include_superseded,
        embed_query=body.embed_query,
    )
    result: RetrievalResult = retrieve(
        db=db,
        project_id=project_id,
        query=body.query,
        vector_backend=SQLiteExactBackend(db),
        config=cfg,
    )

    # Step 2: Assemble
    assembly = assemble_context_packet(
        memories=result.memories,
        scores=result.rrf_scores,
        token_budget=body.token_budget,
        include_source_quote=body.include_source_quote,
        include_retrieval_hint=body.include_retrieval_hint,
        query=body.query,
        project_name=project.name,
    )

    # Step 3: Persist ContextPacket
    packet = phase1_models.ContextPacket(
        project_id=project_id,
        target_tool=body.target_tool,
        intent=body.query,
        included_memory_ids=json.dumps(assembly.included_memory_ids),
        content=assembly.content,
        token_estimate=assembly.token_count,
    )
    db.add(packet)

    # Step 4: Update RetrievalRun with packet metrics
    run = db.query(RetrievalRun).filter(RetrievalRun.id == result.run_id).first()
    if run:
        run.packet_token_budget = body.token_budget
        run.packet_compression_ratio = assembly.compression_ratio
        run.token_count = assembly.token_count

    try:
        db.commit()
        db.refresh(packet)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to persist context packet")

    return ContextPacketResponse(
        packet_id=packet.id,
        run_id=result.run_id,
        query=body.query,
        content=assembly.content,
        token_count=assembly.token_count,
        token_budget=body.token_budget,
        compression_ratio=assembly.compression_ratio,
        rcd_proxy=assembly.rcd_proxy,
        included_memory_count=len(assembly.included_memory_ids),
        excluded_memory_count=len(assembly.excluded_memory_ids),
        used_abstractive_fallback=assembly.used_abstractive_fallback,
        latency_ms=result.latency_ms,
    )


# ---------------------------------------------------------------------------
# POST /memories/generate-cluster-summaries  (Sub-phase 2.4)
# ---------------------------------------------------------------------------

class ClusterSummaryItemOut(BaseModel):
    cluster_id: int
    cluster_label: str
    memory_count: int
    summary_memory_id: str
    summary_text: str
    used_llm: bool


class GenerateClusterSummariesResponse(BaseModel):
    project_id: str
    summaries_generated: int
    min_cluster_size: int
    results: list[ClusterSummaryItemOut]


@router.post(
    "/memories/generate-cluster-summaries",
    response_model=GenerateClusterSummariesResponse,
)
def generate_cluster_summaries(
    project_id: str,
    min_cluster_size: int = 10,
    db: Session = Depends(_get_db),
) -> GenerateClusterSummariesResponse:
    """
    Generate RAPTOR-style cluster summaries for all clusters ≥ min_cluster_size.

    Each summary is stored as a Memory row (type="cluster_summary") so it
    participates in BM25 and dense retrieval. Re-running updates existing rows.
    """
    from app import crud
    from app.services.retrieval.cluster_summaries import generate_cluster_summaries as _generate

    if crud.get_project(db, project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    results = _generate(db=db, project_id=project_id, min_cluster_size=min_cluster_size)

    return GenerateClusterSummariesResponse(
        project_id=project_id,
        summaries_generated=len(results),
        min_cluster_size=min_cluster_size,
        results=[
            ClusterSummaryItemOut(
                cluster_id=r.cluster_id,
                cluster_label=r.cluster_label,
                memory_count=r.memory_count,
                summary_memory_id=r.summary_memory_id,
                summary_text=r.summary_text,
                used_llm=r.used_llm,
            )
            for r in results
        ],
    )
