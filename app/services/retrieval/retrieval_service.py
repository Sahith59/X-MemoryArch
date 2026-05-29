"""
Phase 2.1 — Retrieval pipeline orchestrator.

Pipeline (Sub-phase 2.1 — correctness-first, no reranking/MMR yet):

  Query
    → Hard Filters (privacy gate, status, superseded, valid_until)
    → Three Candidate Generators [BM25 | Dense | Entity]  (parallel in design,
                                                            sequential in this impl)
    → RRF Fusion (k=60, unweighted)
    → Top-K selection
    → Telemetry (retrieval_runs)
    → RetrievalResult

Sub-phase 2.5 adds: query-intent routing, weighted RRF, cross-encoder, MMR.
Sub-phase 2.7 adds: HyDE, contextual embeddings.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.services.vector_backends.base import VectorBackend


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Per-call retrieval configuration."""
    top_k: int = 10
    max_clearance: str = "internal"    # public | internal | sensitive | secret
    include_superseded: bool = False   # expose superseded memories (history mode)
    bm25_candidates: int = 50          # max BM25 leg candidates
    dense_candidates: int = 50         # max dense leg candidates
    entity_candidates: int = 20        # max entity leg candidates
    rrf_k: int = 60                    # RRF constant (Cormack et al.)
    embed_query: bool = True           # False → skip dense leg (e.g., FTS-only mode)
    intent: str | None = None          # caller-supplied intent hint (Sub-phase 2.5 uses this)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Structured result returned to the caller and logged to retrieval_runs."""
    memories: list                     # list[models.Memory] — typed at runtime
    run_id: str
    latency_ms: int
    candidate_count_bm25: int
    candidate_count_dense: int
    candidate_count_entity: int
    fused_count: int
    forbidden_candidate_count: int
    selected_memory_ids: list[str]
    rrf_scores: dict                   # {memory_id: rrf_score} — used by context assembly
    query: str
    config: RetrievalConfig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def retrieve(
    db: "Session",
    project_id: str,
    query: str,
    vector_backend: "VectorBackend",
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    """
    Run the full Phase 2.1 retrieval pipeline and return a RetrievalResult.

    Args:
        db:             SQLAlchemy session (Phase 1 database).
        project_id:     Scope all retrieval to this project.
        query:          Natural-language query string.
        vector_backend: Pluggable vector backend (SQLiteExactBackend for Phase 2.1).
        config:         RetrievalConfig (defaults used if None).

    Returns:
        RetrievalResult with selected memories and full telemetry.
    """
    from app.services.retrieval.candidate_generators import (
        apply_hard_filters,
        generate_bm25_candidates,
        generate_dense_candidates,
        generate_entity_candidates,
    )
    from app.services.retrieval.fusion import rrf_fuse
    from app import models as p2_models
    from app.p2_models import RetrievalRun  # Phase 2 model

    if config is None:
        config = RetrievalConfig()

    start = time.monotonic()

    # ------------------------------------------------------------------
    # Step 1: Hard filters — get allowed_ids (privacy gate + status)
    # ------------------------------------------------------------------
    allowed_ids, forbidden_count = apply_hard_filters(
        db=db,
        project_id=project_id,
        max_clearance=config.max_clearance,
        include_superseded=config.include_superseded,
    )

    # ------------------------------------------------------------------
    # Step 2: Embed query for dense leg
    # ------------------------------------------------------------------
    query_vector: list[float] | None = None
    if config.embed_query and allowed_ids:
        try:
            from app.services.semantic_classifier import embed_text
            emb_bytes = embed_text(query)
            query_vector = np.frombuffer(emb_bytes, dtype=np.float32).tolist()
        except Exception:
            query_vector = None  # dense leg disabled when embedding unavailable

    # ------------------------------------------------------------------
    # Step 3: Three candidate generators
    # ------------------------------------------------------------------
    bm25_candidates = generate_bm25_candidates(
        db=db,
        project_id=project_id,
        query=query,
        allowed_ids=allowed_ids,
        top_k=config.bm25_candidates,
    )

    dense_candidates = generate_dense_candidates(
        vector_backend=vector_backend,
        query_vector=query_vector,
        allowed_ids=allowed_ids,
        top_k=config.dense_candidates,
        project_id=project_id,
    )

    entity_candidates = generate_entity_candidates(
        db=db,
        project_id=project_id,
        query=query,
        allowed_ids=allowed_ids,
        top_k=config.entity_candidates,
    )

    # ------------------------------------------------------------------
    # Step 4: RRF fusion
    # ------------------------------------------------------------------
    fused = rrf_fuse(
        ranked_lists=[
            [mid for mid, _ in bm25_candidates],
            [mid for mid, _ in dense_candidates],
            [mid for mid, _ in entity_candidates],
        ],
        k=config.rrf_k,
    )

    # ------------------------------------------------------------------
    # Step 5: Load top-K memory objects (preserve RRF order)
    # ------------------------------------------------------------------
    top_k_fused = fused[: config.top_k]
    top_k_ids = [mid for mid, _ in top_k_fused]
    rrf_scores = {mid: score for mid, score in top_k_fused}

    memories_by_id: dict = {}
    if top_k_ids:
        from app import models as phase1_models
        rows = (
            db.query(phase1_models.Memory)
            .filter(phase1_models.Memory.id.in_(top_k_ids))
            .all()
        )
        memories_by_id = {m.id: m for m in rows}

    # Preserve RRF rank order
    memories = [memories_by_id[mid] for mid in top_k_ids if mid in memories_by_id]

    # ------------------------------------------------------------------
    # Step 6: Telemetry — write to retrieval_runs
    # ------------------------------------------------------------------
    latency_ms = int((time.monotonic() - start) * 1000)

    run = RetrievalRun(
        id=str(uuid.uuid4()),
        project_id=project_id,
        query=query,
        intent=config.intent,
        candidate_count_bm25=len(bm25_candidates),
        candidate_count_dense=len(dense_candidates),
        candidate_count_entity=len(entity_candidates),
        fused_count=len(fused),
        forbidden_candidate_count=forbidden_count,
        backend_used=vector_backend.name,
        latency_ms=latency_ms,
    )
    run.set_selected_ids(top_k_ids)
    run.set_filters({
        "max_clearance": config.max_clearance,
        "include_superseded": config.include_superseded,
        "top_k": config.top_k,
    })

    try:
        db.add(run)
        db.commit()
    except Exception:
        db.rollback()  # telemetry failure must never break retrieval

    return RetrievalResult(
        memories=memories,
        run_id=run.id,
        latency_ms=latency_ms,
        candidate_count_bm25=len(bm25_candidates),
        candidate_count_dense=len(dense_candidates),
        candidate_count_entity=len(entity_candidates),
        fused_count=len(fused),
        forbidden_candidate_count=forbidden_count,
        selected_memory_ids=top_k_ids,
        rrf_scores=rrf_scores,
        query=query,
        config=config,
    )
