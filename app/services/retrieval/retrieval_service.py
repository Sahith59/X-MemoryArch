"""
Phase 2 — Retrieval pipeline orchestrator.

Pipeline:
  Query
    → Hard Filters (privacy gate, status, superseded, valid_until)         [2.1]
    → Three Candidate Generators [BM25 | Dense | Entity]                   [2.1]
    → RRF Fusion (k=60)                                                     [2.1]
    → Graph Expansion (entity boost, code anchor, 1-hop, 2-hop)            [2.4]
    → Intent Classification (heuristic-first, optional LLM fallback)       [2.5]
    → Weighted Ranking (5 signals, intent-dependent weights)                [2.5]
    → Cross-Encoder Reranker (top-N, opt-in, graceful degradation)         [2.5]
    → MMR Diversification (λ=0.70, structural similarity)                  [2.5]
    → Top-K selection
    → Telemetry (retrieval_runs)
    → RetrievalResult

Sub-phase 2.7 adds: HyDE, contextual embeddings.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

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
    intent: str | None = None          # caller-supplied intent override (skips classifier)
    # Sub-phase 2.4 — graph expansion knobs
    enable_entity_boost: bool = True
    enable_graph_expansion: bool = True
    enable_2hop: bool = True
    max_expansion_total: int = 15
    entity_boost_weight: float = 0.15
    # Sub-phase 2.5 — ranking / reranker / MMR knobs
    enable_weighted_ranking: bool = True
    enable_reranker: bool = False      # opt-in — adds ~50-200ms latency
    enable_mmr: bool = True
    mmr_lambda: float = 0.70           # 0.65–0.80 range
    reranker_top_n: int = 20           # cross-encoder sees this many candidates
    intent_llm_fn: Callable[[str], str] | None = None  # LLM for low-confidence intents
    # Sub-phase 2.7 — HyDE knobs
    enable_hyde: bool = False          # opt-in — adds LLM latency + extra embedding
    hyde_llm_fn: Callable[[str], str] | None = None   # LLM for hypothetical doc generation
    hyde_combination: str = "avg"      # avg | max | query | hyde
    hyde_confidence_threshold: float = 0.70  # below this intent confidence → use HyDE


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
    rrf_scores: dict                   # {memory_id: final_score} — used by context assembly
    query: str
    config: RetrievalConfig
    expanded_via_links: int = 0        # Sub-phase 2.4: memories added by graph expansion
    # Sub-phase 2.5 telemetry
    intent_detected: str = "general"
    intent_confidence: float = 0.5
    reranked: bool = False
    mmr_applied: bool = False
    # Sub-phase 2.7 telemetry
    hyde_used: bool = False


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
    Run the full Phase 2 retrieval pipeline and return a RetrievalResult.

    Args:
        db:             SQLAlchemy session (Phase 1 database).
        project_id:     Scope all retrieval to this project.
        query:          Natural-language query string.
        vector_backend: Pluggable vector backend (SQLiteExactBackend for Phase 2).
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
    from app.p2_models import RetrievalRun

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
    # Step 2: Embed query for dense leg (+ optional HyDE augmentation)
    # ------------------------------------------------------------------
    query_vector: list[float] | None = None
    hyde_used = False
    if config.embed_query and allowed_ids:
        try:
            from app.services.semantic_classifier import embed_text
            emb_bytes = embed_text(query)
            query_vector = np.frombuffer(emb_bytes, dtype=np.float32).tolist()
        except Exception:
            query_vector = None

    # Sub-phase 2.7: HyDE augmentation (applied after base embedding)
    if (
        config.enable_hyde
        and config.hyde_llm_fn is not None
        and query_vector is not None
    ):
        from app.services.retrieval.hyde import (
            HyDEConfig, generate_hyde_vector, should_use_hyde,
        )
        from app.services.retrieval.intent_classifier import classify_intent

        # Quick intent check to gate HyDE (avoids HyDE on precise factual queries)
        _intent, _conf = classify_intent(query)
        if should_use_hyde(_intent.value, _conf, config.hyde_confidence_threshold):
            hyde_cfg = HyDEConfig(
                llm_fn=config.hyde_llm_fn,
                combination=config.hyde_combination,
            )
            augmented = generate_hyde_vector(query, hyde_cfg, query_vector)
            if augmented is not None:
                query_vector = augmented
                hyde_used = True

    # ------------------------------------------------------------------
    # Step 3: Three candidate generators
    # ------------------------------------------------------------------
    bm25_candidates = generate_bm25_candidates(
        db=db, project_id=project_id, query=query,
        allowed_ids=allowed_ids, top_k=config.bm25_candidates,
    )
    dense_candidates = generate_dense_candidates(
        vector_backend=vector_backend, query_vector=query_vector,
        allowed_ids=allowed_ids, top_k=config.dense_candidates,
        project_id=project_id,
    )
    entity_candidates = generate_entity_candidates(
        db=db, project_id=project_id, query=query,
        allowed_ids=allowed_ids, top_k=config.entity_candidates,
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
    # Step 5: Graph expansion (2.4)
    # ------------------------------------------------------------------
    from app.services.retrieval.graph_expansion import graph_expand

    pre_expand_ids = [mid for mid, _ in fused[: config.top_k]]
    pre_expand_scores = {mid: score for mid, score in fused[: config.top_k]}

    augmented_scores, expanded_count = graph_expand(
        db=db, project_id=project_id,
        top_k_ids=pre_expand_ids, rrf_scores=pre_expand_scores,
        allowed_ids=set(allowed_ids), query=query,
        enable_entity_boost=config.enable_entity_boost,
        enable_graph_expansion=config.enable_graph_expansion,
        enable_2hop=config.enable_2hop,
        max_expansion_total=config.max_expansion_total,
        entity_boost_weight=config.entity_boost_weight,
    )

    # Candidate pool = top (top_k + reranker buffer) by augmented score
    pool_size = max(config.top_k, config.reranker_top_n) if config.enable_reranker else config.top_k
    all_scored = sorted(augmented_scores.items(), key=lambda x: x[1], reverse=True)
    pool_ids = [mid for mid, _ in all_scored[:pool_size]]
    pool_scores = {mid: score for mid, score in all_scored[:pool_size]}

    # ------------------------------------------------------------------
    # Step 6: Load candidate memory objects
    # ------------------------------------------------------------------
    memories_by_id: dict = {}
    if pool_ids:
        from app import models as phase1_models
        rows = (
            db.query(phase1_models.Memory)
            .filter(phase1_models.Memory.id.in_(pool_ids))
            .all()
        )
        memories_by_id = {m.id: m for m in rows}

    candidate_memories = [memories_by_id[mid] for mid in pool_ids if mid in memories_by_id]

    # ------------------------------------------------------------------
    # Step 7: Intent classification (2.5)
    # ------------------------------------------------------------------
    from app.services.retrieval.intent_classifier import classify_intent, IntentLabel

    if config.intent and config.intent in [i.value for i in IntentLabel]:
        detected_intent = IntentLabel(config.intent)
        intent_confidence = 1.0
    else:
        detected_intent, intent_confidence = classify_intent(
            query, llm_fn=config.intent_llm_fn
        )

    # ------------------------------------------------------------------
    # Step 8: Weighted ranking (2.5)
    # ------------------------------------------------------------------
    final_scores = pool_scores  # default to augmented RRF scores
    reranked = False
    mmr_applied = False

    if config.enable_weighted_ranking and candidate_memories:
        from app.services.retrieval.ranking import weighted_rank, load_entity_map

        entity_map = load_entity_map(db, pool_ids)
        ranked_list = weighted_rank(
            memories=candidate_memories,
            rrf_scores=pool_scores,
            intent=detected_intent,
            query=query,
            entity_map=entity_map,
        )
        final_scores = {mid: score for mid, score in ranked_list}
        # Re-order candidate_memories by new ranking
        rank_order = {mid: i for i, (mid, _) in enumerate(ranked_list)}
        candidate_memories.sort(key=lambda m: rank_order.get(m.id, 9999))

    # ------------------------------------------------------------------
    # Step 9: Cross-encoder reranker (2.5, opt-in)
    # ------------------------------------------------------------------
    if config.enable_reranker and candidate_memories:
        from app.services.retrieval.reranker import get_default_reranker

        reranker = get_default_reranker()
        if reranker.available:
            reranked_list = reranker.rerank(
                query=query,
                memories=candidate_memories,
                top_n=config.reranker_top_n,
            )
            final_scores = {mid: score for mid, score in reranked_list}
            reranked_order = {mid: i for i, (mid, _) in enumerate(reranked_list)}
            candidate_memories.sort(key=lambda m: reranked_order.get(m.id, 9999))
            reranked = True

    # ------------------------------------------------------------------
    # Step 10: MMR diversification (2.5)
    # ------------------------------------------------------------------
    if config.enable_mmr and candidate_memories and len(candidate_memories) > 1:
        from app.services.retrieval.mmr import mmr_select
        from app.services.retrieval.ranking import load_entity_map

        entity_map = load_entity_map(db, [m.id for m in candidate_memories])
        selected_ids = mmr_select(
            memories=candidate_memories,
            scores=final_scores,
            top_k=config.top_k,
            lambda_=config.mmr_lambda,
            entity_map=entity_map,
        )
        mmr_applied = True
    else:
        selected_ids = [m.id for m in candidate_memories[: config.top_k]]

    # ------------------------------------------------------------------
    # Step 11: Build final ordered memory list
    # ------------------------------------------------------------------
    final_by_id = {m.id: m for m in candidate_memories}
    memories = [final_by_id[mid] for mid in selected_ids if mid in final_by_id]
    final_rrf_scores = {mid: final_scores.get(mid, 0.0) for mid in selected_ids}

    # ------------------------------------------------------------------
    # Step 12: Telemetry — write to retrieval_runs
    # ------------------------------------------------------------------
    latency_ms = int((time.monotonic() - start) * 1000)

    run = RetrievalRun(
        id=str(uuid.uuid4()),
        project_id=project_id,
        query=query,
        intent=detected_intent.value,
        candidate_count_bm25=len(bm25_candidates),
        candidate_count_dense=len(dense_candidates),
        candidate_count_entity=len(entity_candidates),
        fused_count=len(fused),
        forbidden_candidate_count=forbidden_count,
        backend_used=vector_backend.name,
        latency_ms=latency_ms,
    )
    run.set_selected_ids(selected_ids)
    run.set_filters({
        "max_clearance": config.max_clearance,
        "include_superseded": config.include_superseded,
        "top_k": config.top_k,
        "intent": detected_intent.value,
        "mmr_lambda": config.mmr_lambda,
        "hyde_used": hyde_used,
    })

    try:
        db.add(run)
        db.commit()
    except Exception:
        db.rollback()

    return RetrievalResult(
        memories=memories,
        run_id=run.id,
        latency_ms=latency_ms,
        candidate_count_bm25=len(bm25_candidates),
        candidate_count_dense=len(dense_candidates),
        candidate_count_entity=len(entity_candidates),
        fused_count=len(fused),
        forbidden_candidate_count=forbidden_count,
        selected_memory_ids=selected_ids,
        rrf_scores=final_rrf_scores,
        query=query,
        config=config,
        expanded_via_links=expanded_count,
        intent_detected=detected_intent.value,
        intent_confidence=intent_confidence,
        reranked=reranked,
        mmr_applied=mmr_applied,
        hyde_used=hyde_used,
    )
