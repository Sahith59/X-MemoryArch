"""
Phase 2.1/2.2/2.4/2.6 — Retrieval API router.

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

  POST /projects/{project_id}/vector-index/backfill
    Backfill memories from SQLite into an ANN backend.

  GET /projects/{project_id}/vector-index/status
    Return active VectorIndexState and available backends.

  POST /projects/{project_id}/vector-index/validate
    Shadow-run ANN vs exact; compute Recall@k. If ≥ 0.95, activate.

  GET /vector-backends
    List all backend types and their availability.

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


# ---------------------------------------------------------------------------
# Sub-phase 2.6 — Vector index management endpoints
# ---------------------------------------------------------------------------

class BackfillRequest(BaseModel):
    backend: str = Field("chroma", description="Backend type: chroma, faiss, qdrant")
    embedding_model: str = Field("all-MiniLM-L6-v2")
    batch_size: int = Field(100, ge=1, le=1000)
    backend_config: dict = Field(default_factory=dict, description="Backend-specific config")


class BackfillResponse(BaseModel):
    project_id: str
    state_id: str
    backend: str
    embedding_model: str
    embedding_dim: int
    total_indexed: int
    index_checksum: str | None
    is_active: bool
    message: str


class VectorIndexStatusResponse(BaseModel):
    project_id: str
    active_state: dict | None
    available_backends: list[dict]


class ValidateResponse(BaseModel):
    project_id: str
    backend: str
    overall_ann_recall: float
    passes_threshold: bool
    threshold: float
    sample_size: int
    per_query_recalls: list[float]
    activated: bool
    message: str


@router.post("/vector-index/backfill", response_model=BackfillResponse)
def backfill_vector_index(
    project_id: str,
    body: BackfillRequest,
    db: Session = Depends(_get_db),
) -> BackfillResponse:
    """
    Backfill embedded memories from SQLite into an ANN backend.

    Builds a new index without activating it (activation requires
    validate to pass Recall@k ≥ 0.95).

    Architectural rule: Never replace embeddings in place. Always build
    a new index, shadow-run, verify recall, then switch.
    """
    from app import crud
    from app.services.vector_backends.backend_factory import BackendType, create_backend
    from app.services.vector_backends.migration import BackfillService

    if crud.get_project(db, project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    try:
        BackendType(body.backend)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown backend: {body.backend!r}. Valid: chroma, faiss, qdrant, sqlite_exact",
        )

    if body.backend == "sqlite_exact":
        raise HTTPException(
            status_code=400,
            detail="sqlite_exact is the baseline — backfill only applies to ANN backends",
        )

    try:
        target_backend = create_backend(body.backend, body.backend_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create backend: {exc}")

    svc = BackfillService()
    try:
        state = svc.backfill_project(
            db=db,
            project_id=project_id,
            target_backend=target_backend,
            embedding_model=body.embedding_model,
            batch_size=body.batch_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backfill failed: {exc}")

    return BackfillResponse(
        project_id=project_id,
        state_id=state.id,
        backend=state.backend,
        embedding_model=state.embedding_model,
        embedding_dim=state.embedding_dim,
        total_indexed=state.total_indexed,
        index_checksum=state.index_checksum,
        is_active=state.is_active,
        message=(
            f"Backfilled {state.total_indexed} memories to {state.backend}. "
            "Run /validate to activate (requires Recall@k ≥ 0.95)."
        ),
    )


@router.get("/vector-index/status", response_model=VectorIndexStatusResponse)
def get_vector_index_status(
    project_id: str,
    db: Session = Depends(_get_db),
) -> VectorIndexStatusResponse:
    """
    Return the active VectorIndexState for the project and all available backends.
    """
    from app import crud
    from app.p2_models import VectorIndexState
    from app.services.vector_backends.backend_factory import list_available_backends

    if crud.get_project(db, project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    state = (
        db.query(VectorIndexState)
        .filter(
            VectorIndexState.project_id == project_id,
            VectorIndexState.is_active == True,  # noqa: E712
        )
        .first()
    )

    active = None
    if state:
        active = {
            "id": state.id,
            "backend": state.backend,
            "embedding_model": state.embedding_model,
            "embedding_dim": state.embedding_dim,
            "total_indexed": state.total_indexed,
            "index_checksum": state.index_checksum,
            "last_backfill_at": state.last_backfill_at.isoformat() if state.last_backfill_at else None,
            "is_active": state.is_active,
        }

    return VectorIndexStatusResponse(
        project_id=project_id,
        active_state=active,
        available_backends=list_available_backends(),
    )


class ValidateRequest(BaseModel):
    backend: str = Field("chroma", description="Backend type to validate")
    backend_config: dict = Field(default_factory=dict)
    k: int = Field(10, ge=1, le=100)
    sample_size: int = Field(50, ge=1, le=500)
    min_recall: float = Field(0.95, ge=0.0, le=1.0)
    auto_activate: bool = Field(True, description="Activate backend if recall passes threshold")


@router.post("/vector-index/validate", response_model=ValidateResponse)
def validate_vector_index(
    project_id: str,
    body: ValidateRequest,
    db: Session = Depends(_get_db),
) -> ValidateResponse:
    """
    Shadow-run ANN backend vs SQLiteExact; compute Recall@k.

    Architectural rule 11: ANN must achieve ≥ 0.95 Recall@k vs exact
    before activation. If auto_activate=True and recall passes, the
    most recent inactive state for this backend is activated.
    """
    from app import crud
    from app.p2_models import VectorIndexState
    from app.services.vector_backends.backend_factory import BackendType, create_backend
    from app.services.vector_backends.migration import BackfillService

    if crud.get_project(db, project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    try:
        BackendType(body.backend)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {body.backend!r}")

    try:
        ann_backend = create_backend(body.backend, body.backend_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create backend: {exc}")

    svc = BackfillService()
    validation = svc.validate_ann_recall(
        db=db,
        project_id=project_id,
        ann_backend=ann_backend,
        k=body.k,
        sample_size=body.sample_size,
        min_recall=body.min_recall,
    )

    activated = False
    if validation["passes_threshold"] and body.auto_activate:
        # Find the most recent inactive state for this backend
        state = (
            db.query(VectorIndexState)
            .filter(
                VectorIndexState.project_id == project_id,
                VectorIndexState.backend == body.backend,
                VectorIndexState.is_active == False,  # noqa: E712
            )
            .order_by(VectorIndexState.created_at.desc())
            .first()
        )
        if state:
            svc.activate_backend(db=db, project_id=project_id, backend=ann_backend, state=state)
            activated = True

    passes = validation["passes_threshold"]
    return ValidateResponse(
        project_id=project_id,
        backend=body.backend,
        overall_ann_recall=validation["overall_ann_recall"],
        passes_threshold=passes,
        threshold=validation["threshold"],
        sample_size=validation["sample_size"],
        per_query_recalls=validation["per_query_recalls"],
        activated=activated,
        message=(
            f"Recall@{body.k} = {validation['overall_ann_recall']:.3f} "
            f"({'PASS' if passes else 'FAIL'} threshold={body.min_recall}). "
            + ("Backend activated." if activated else "")
        ),
    )


# ---------------------------------------------------------------------------
# Sub-phase 2.7 — Contextual embeddings endpoint
# ---------------------------------------------------------------------------

class ContextualEmbeddingsRequest(BaseModel):
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Model for re-embedding")
    batch_size: int = Field(50, ge=1, le=500)
    force_regenerate: bool = Field(False, description="Re-generate even if prefix exists")
    use_llm: bool = Field(False, description="Use LLM for richer prefix (requires llm config)")


class ContextualEmbeddingsResponse(BaseModel):
    project_id: str
    total_processed: int
    already_had_prefix: int
    newly_prefixed: int
    failed: int
    used_llm: bool
    memory_ids_updated: list[str]
    message: str


@router.post(
    "/memories/generate-contextual-embeddings",
    response_model=ContextualEmbeddingsResponse,
)
def generate_contextual_embeddings(
    project_id: str,
    body: ContextualEmbeddingsRequest,
    db: Session = Depends(_get_db),
) -> ContextualEmbeddingsResponse:
    """
    Generate context prefixes and re-embed all active memories for a project.

    Architectural basis (Anthropic research):
      Prepending a 50-100 token context prefix before embedding reduces
      retrieval failure by 67%.

    After calling this endpoint:
      1. All active memories have contextual_prefix set and embeddings updated.
      2. Run POST /vector-index/backfill to rebuild the ANN index.
      3. Run POST /vector-index/validate to activate the new index.

    The LLM path (use_llm=True) requires ANTHROPIC_API_KEY to be set.
    Falls back to template prefix if LLM is unavailable.
    """
    from app import crud
    from app.services.retrieval.contextual_embeddings import generate_contextual_embeddings as _gen

    if crud.get_project(db, project_id) is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    llm_fn = None
    if body.use_llm:
        try:
            import os
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

            def _claude_prefix(prompt: str) -> str:
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text if msg.content else ""

            llm_fn = _claude_prefix
        except Exception:
            llm_fn = None  # graceful degradation to template

    try:
        result = _gen(
            db=db,
            project_id=project_id,
            llm_fn=llm_fn,
            batch_size=body.batch_size,
            force_regenerate=body.force_regenerate,
            embedding_model=body.embedding_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Contextual embedding generation failed: {exc}")

    return ContextualEmbeddingsResponse(
        project_id=project_id,
        total_processed=result.total_processed,
        already_had_prefix=result.already_had_prefix,
        newly_prefixed=result.newly_prefixed,
        failed=result.failed,
        used_llm=result.used_llm,
        memory_ids_updated=result.memory_ids_updated,
        message=(
            f"Generated contextual prefixes for {result.newly_prefixed}/{result.total_processed} memories. "
            f"Skipped {result.already_had_prefix} already prefixed. "
            f"Failed: {result.failed}. "
            "Now run /vector-index/backfill to rebuild the ANN index."
        ),
    )


# ---------------------------------------------------------------------------
# GET /vector-backends  (global, not project-scoped)
# ---------------------------------------------------------------------------

_global_router = APIRouter(tags=["retrieval"])


@_global_router.get("/vector-backends")
def list_vector_backends() -> list[dict]:
    """List all vector backend types and their availability on this host."""
    from app.services.vector_backends.backend_factory import list_available_backends
    return list_available_backends()
