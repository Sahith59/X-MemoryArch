"""
Sub-phase 2.3 — Benchmark runner.

Orchestrates:
  1. Synthetic corpus evaluation (all 9 scenario types)
  2. Privacy leakage suite (systematic cross-clearance isolation)
  3. ANN Recall@k validation (exact backend vs itself as baseline;
     real ANN backends validated in Sub-phase 2.6)
  4. Score reporting vs phase-2-plan.md targets

Usage:
  from app.services.benchmark.runner import run_benchmark
  suite = run_benchmark(db, project_id, corpus_fixture)
  print(suite.full_report())
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.services.benchmark.eval_metrics import (
    BenchmarkSuite,
    EvalReport,
    ann_recall_at_k,
    evaluate,
)
from app.services.benchmark.corpus import CorpusFixture, TestCase


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(
    db: "Session",
    corpus: CorpusFixture,
    top_k: int = 10,
) -> BenchmarkSuite:
    """
    Run the full synthetic benchmark against the given corpus.

    Returns a BenchmarkSuite with per-scenario EvalReports and overall metrics.
    All queries run with embed_query=False (deterministic, no sentence_transformers).
    """
    from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend

    vector_backend = SQLiteExactBackend(db)

    # Group test cases by scenario
    scenario_cases: dict[str, list[TestCase]] = {}
    for tc in corpus.test_cases:
        scenario_cases.setdefault(tc.scenario, []).append(tc)

    reports: list[EvalReport] = []

    for scenario_name, cases in sorted(scenario_cases.items()):
        report = EvalReport(scenario_name=scenario_name)

        for tc in cases:
            cfg = RetrievalConfig(
                top_k=top_k,
                max_clearance=tc.max_clearance,
                embed_query=tc.embed_query,
                include_superseded=False,
            )

            t0 = time.monotonic()
            result = retrieve(
                db=db,
                project_id=corpus.project_id,
                query=tc.query,
                vector_backend=vector_backend,
                config=cfg,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            retrieved_ids = result.selected_memory_ids
            expected_ids = corpus.resolve_expected_ids(tc)
            excluded_ids = corpus.resolve_excluded_ids(tc)

            # Count privacy violations: excluded IDs that appeared in results
            violations = sum(1 for eid in excluded_ids if eid in retrieved_ids)

            qr = evaluate(
                query=tc.query,
                retrieved_ids=retrieved_ids,
                relevant_ids=expected_ids,
                latency_ms=latency_ms,
                privacy_violations=violations,
            )
            report.query_results.append(qr)

        reports.append(report)

    return BenchmarkSuite(reports=reports)


# ---------------------------------------------------------------------------
# Privacy leakage suite
# ---------------------------------------------------------------------------

def run_privacy_suite(
    db: "Session",
    corpus: CorpusFixture,
    top_k: int = 20,
) -> dict[str, int]:
    """
    Systematic cross-clearance privacy leakage test.

    For every clearance level, run all queries at that level and verify that
    memories above the caller's clearance never appear in the final results.

    NOTE: forbidden_candidate_count in the retrieval result counts memories
    BLOCKED by the hard filter — non-zero is CORRECT when a project has
    higher-clearance memories. It is NOT a leakage metric. Real leakage =
    above-clearance memories appearing in selected_memory_ids, which this
    function checks directly.

    Returns:
      {clearance_level: violation_count}  — all must be 0 to pass.
    """
    from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    from app import models

    vector_backend = SQLiteExactBackend(db)
    clearance_order = ["public", "internal", "sensitive", "secret"]
    violations: dict[str, int] = {level: 0 for level in clearance_order}

    # Get all memories in the project with their privacy levels
    all_memories = (
        db.query(models.Memory)
        .filter(models.Memory.project_id == corpus.project_id)
        .all()
    )
    level_index = {lv: i for i, lv in enumerate(clearance_order)}
    secret_by_level: dict[str, set[str]] = {lv: set() for lv in clearance_order}
    for m in all_memories:
        secret_by_level[m.privacy_level].add(m.id)

    # Run all queries at each clearance level, check for leakage
    queries = list({tc.query for tc in corpus.test_cases})

    for clearance in clearance_order:
        cfg = RetrievalConfig(
            top_k=top_k,
            max_clearance=clearance,
            embed_query=False,
        )
        allowed_levels = clearance_order[: level_index[clearance] + 1]
        forbidden_levels = clearance_order[level_index[clearance] + 1:]
        forbidden_ids = set()
        for fl in forbidden_levels:
            forbidden_ids |= secret_by_level[fl]

        for query in queries:
            result = retrieve(
                db=db,
                project_id=corpus.project_id,
                query=query,
                vector_backend=vector_backend,
                config=cfg,
            )
            # Real leakage: above-clearance memories that escaped into final results.
            leaked = [mid for mid in result.selected_memory_ids if mid in forbidden_ids]
            violations[clearance] += len(leaked)

    return violations


# ---------------------------------------------------------------------------
# ANN Recall@k validation
# ---------------------------------------------------------------------------

def run_ann_validation(
    db: "Session",
    corpus: CorpusFixture,
    ann_backend,
    k: int = 10,
    min_recall: float = 0.95,
) -> dict[str, object]:
    """
    Validate an ANN backend against the SQLiteExactBackend baseline.

    Rule 11 from phase-2-plan.md: never enable ANN without verifying
    ≥ 0.95 Recall@k vs exact on sample queries.

    Returns:
      {
        "overall_ann_recall": float,
        "passes_threshold": bool,
        "per_query_recalls": list[float],
        "threshold": float,
      }
    """
    from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend

    exact_backend = SQLiteExactBackend(db)
    cfg = RetrievalConfig(top_k=k, embed_query=False, max_clearance="secret")

    queries = list({tc.query for tc in corpus.test_cases})
    per_query_recalls: list[float] = []

    for query in queries:
        exact_result = retrieve(
            db=db, project_id=corpus.project_id,
            query=query, vector_backend=exact_backend, config=cfg,
        )
        ann_result = retrieve(
            db=db, project_id=corpus.project_id,
            query=query, vector_backend=ann_backend, config=cfg,
        )
        r = ann_recall_at_k(
            exact_results=exact_result.selected_memory_ids,
            ann_results=ann_result.selected_memory_ids,
            k=k,
        )
        per_query_recalls.append(r)

    overall = sum(per_query_recalls) / len(per_query_recalls) if per_query_recalls else 0.0
    return {
        "overall_ann_recall": overall,
        "passes_threshold": overall >= min_recall,
        "per_query_recalls": per_query_recalls,
        "threshold": min_recall,
    }
