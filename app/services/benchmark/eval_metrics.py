"""
Sub-phase 2.3 — Retrieval evaluation metrics.

Implements:
  Recall@k     = |relevant ∩ retrieved[:k]| / |relevant|
  Precision@k  = |relevant ∩ retrieved[:k]| / k
  MRR@k        = 1 / rank_of_first_relevant  (within top-k, else 0)
  NDCG@k       = DCG@k / IDCG@k  (binary relevance: 1 if relevant, 0 otherwise)

All functions accept:
  retrieved:  ordered list of memory IDs (highest rank first)
  relevant:   set or list of gold-label memory IDs
  k:          cutoff (default matches function name)

Usage:
  from app.services.benchmark.eval_metrics import recall_at_k, mrr_at_k, ndcg_at_k, evaluate
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: list[str] | set[str], k: int) -> float:
    """
    Recall@k: fraction of relevant memories found in the top-k retrieved.

    Returns 0.0 when relevant is empty (no gold labels → not evaluable).
    Returns 1.0 when all relevant items appear within top-k.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for mid in retrieved[:k] if mid in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: list[str], relevant: list[str] | set[str], k: int) -> float:
    """Precision@k: fraction of top-k retrieved that are relevant."""
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for mid in retrieved[:k] if mid in relevant_set)
    return hits / k


def mrr_at_k(retrieved: list[str], relevant: list[str] | set[str], k: int) -> float:
    """
    MRR@k: reciprocal rank of the first relevant result within top-k.

    Returns 0.0 when no relevant item appears in top-k.
    """
    relevant_set = set(relevant)
    for rank, mid in enumerate(retrieved[:k], start=1):
        if mid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str] | set[str], k: int) -> float:
    """
    NDCG@k with binary relevance (1 = relevant, 0 = not).

    DCG@k  = Σ rel_i / log2(i + 1)  for i in 1..k
    IDCG@k = Σ 1 / log2(i + 1)      for i in 1..min(|relevant|, k)
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, mid in enumerate(retrieved[:k], start=1)
        if mid in relevant_set
    )

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregated evaluation result
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Metrics for a single (query, retrieved_ids, relevant_ids) triple."""
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    latency_ms: int = 0
    privacy_violations: int = 0    # non-zero = leakage bug


@dataclass
class EvalReport:
    """Aggregated evaluation results across a set of queries."""
    scenario_name: str
    query_results: list[QueryResult] = field(default_factory=list)

    # Targets (from phase-2-plan.md)
    TARGET_RECALL_5: float = 0.80
    TARGET_MRR_10: float = 0.78
    TARGET_NDCG_10: float = 0.55
    TARGET_LATENCY_P50_MS: int = 150
    TARGET_LATENCY_P95_MS: int = 400

    @property
    def n(self) -> int:
        return len(self.query_results)

    @property
    def recall_at_1(self) -> float:
        return _mean(r.recall_at_1 for r in self.query_results)

    @property
    def recall_at_3(self) -> float:
        return _mean(r.recall_at_3 for r in self.query_results)

    @property
    def recall_at_5(self) -> float:
        return _mean(r.recall_at_5 for r in self.query_results)

    @property
    def precision_at_5(self) -> float:
        return _mean(r.precision_at_5 for r in self.query_results)

    @property
    def mrr_at_10(self) -> float:
        return _mean(r.mrr_at_10 for r in self.query_results)

    @property
    def ndcg_at_10(self) -> float:
        return _mean(r.ndcg_at_10 for r in self.query_results)

    @property
    def latency_p50_ms(self) -> float:
        if not self.query_results:
            return 0.0
        lats = sorted(r.latency_ms for r in self.query_results)
        return lats[len(lats) // 2]

    @property
    def latency_p95_ms(self) -> float:
        if not self.query_results:
            return 0.0
        lats = sorted(r.latency_ms for r in self.query_results)
        idx = min(int(len(lats) * 0.95), len(lats) - 1)
        return lats[idx]

    @property
    def total_privacy_violations(self) -> int:
        return sum(r.privacy_violations for r in self.query_results)

    @property
    def meets_recall_target(self) -> bool:
        return self.recall_at_5 >= self.TARGET_RECALL_5

    @property
    def meets_mrr_target(self) -> bool:
        return self.mrr_at_10 >= self.TARGET_MRR_10

    @property
    def meets_ndcg_target(self) -> bool:
        return self.ndcg_at_10 >= self.TARGET_NDCG_10

    @property
    def meets_latency_p50(self) -> bool:
        return self.latency_p50_ms <= self.TARGET_LATENCY_P50_MS

    @property
    def meets_latency_p95(self) -> bool:
        return self.latency_p95_ms <= self.TARGET_LATENCY_P95_MS

    @property
    def privacy_clean(self) -> bool:
        return self.total_privacy_violations == 0

    def summary(self) -> str:
        """Human-readable one-paragraph score report."""
        lines = [
            f"=== Benchmark: {self.scenario_name} ({self.n} queries) ===",
            f"  Recall@1={self.recall_at_1:.3f}  Recall@3={self.recall_at_3:.3f}  "
            f"Recall@5={self.recall_at_5:.3f} (target ≥{self.TARGET_RECALL_5})"
            f" {'✓' if self.meets_recall_target else '✗'}",
            f"  MRR@10={self.mrr_at_10:.3f} (target ≥{self.TARGET_MRR_10})"
            f" {'✓' if self.meets_mrr_target else '✗'}",
            f"  NDCG@10={self.ndcg_at_10:.3f} (target ≥{self.TARGET_NDCG_10})"
            f" {'✓' if self.meets_ndcg_target else '✗'}",
            f"  Latency p50={self.latency_p50_ms:.0f}ms (target ≤{self.TARGET_LATENCY_P50_MS}ms)"
            f" {'✓' if self.meets_latency_p50 else '✗'}",
            f"  Latency p95={self.latency_p95_ms:.0f}ms (target ≤{self.TARGET_LATENCY_P95_MS}ms)"
            f" {'✓' if self.meets_latency_p95 else '✗'}",
            f"  Privacy violations={self.total_privacy_violations} (must be 0)"
            f" {'✓' if self.privacy_clean else '✗ LEAKAGE DETECTED'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregate across multiple reports
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSuite:
    """Aggregated results across all scenarios in a benchmark run."""
    reports: list[EvalReport] = field(default_factory=list)

    @property
    def overall_recall_at_5(self) -> float:
        all_results = [r for rep in self.reports for r in rep.query_results]
        return _mean(r.recall_at_5 for r in all_results)

    @property
    def overall_mrr_at_10(self) -> float:
        all_results = [r for rep in self.reports for r in rep.query_results]
        return _mean(r.mrr_at_10 for r in all_results)

    @property
    def overall_ndcg_at_10(self) -> float:
        all_results = [r for rep in self.reports for r in rep.query_results]
        return _mean(r.ndcg_at_10 for r in all_results)

    @property
    def total_queries(self) -> int:
        return sum(rep.n for rep in self.reports)

    @property
    def privacy_clean(self) -> bool:
        return all(rep.privacy_clean for rep in self.reports)

    @property
    def total_privacy_violations(self) -> int:
        return sum(rep.total_privacy_violations for rep in self.reports)

    def full_report(self) -> str:
        lines = ["=" * 60, "BENCHMARK SUITE REPORT", "=" * 60]
        for rep in self.reports:
            lines.append(rep.summary())
        lines.append("=" * 60)
        lines.append(f"OVERALL ({self.total_queries} queries across {len(self.reports)} scenarios)")
        lines.append(f"  Recall@5 = {self.overall_recall_at_5:.3f}")
        lines.append(f"  MRR@10   = {self.overall_mrr_at_10:.3f}")
        lines.append(f"  NDCG@10  = {self.overall_ndcg_at_10:.3f}")
        lines.append(f"  Privacy  = {'CLEAN' if self.privacy_clean else 'LEAKAGE DETECTED'}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-query evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    query: str,
    retrieved_ids: list[str],
    relevant_ids: list[str],
    latency_ms: int = 0,
    privacy_violations: int = 0,
) -> QueryResult:
    """Compute all metrics for one (query, retrieved, relevant) triple."""
    return QueryResult(
        query=query,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        recall_at_1=recall_at_k(retrieved_ids, relevant_ids, 1),
        recall_at_3=recall_at_k(retrieved_ids, relevant_ids, 3),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        mrr_at_10=mrr_at_k(retrieved_ids, relevant_ids, 10),
        ndcg_at_10=ndcg_at_k(retrieved_ids, relevant_ids, 10),
        latency_ms=latency_ms,
        privacy_violations=privacy_violations,
    )


# ---------------------------------------------------------------------------
# ANN Recall@k validation
# ---------------------------------------------------------------------------

def ann_recall_at_k(
    exact_results: list[str],
    ann_results: list[str],
    k: int,
) -> float:
    """
    ANN Recall@k vs exact: fraction of exact top-k that ANN also returns in top-k.

    Must be ≥ 0.95 before any ANN backend is activated (per architectural rule 11).
    A value of 1.0 means perfect — ANN returns exactly the same top-k as exact search.
    """
    if not exact_results:
        return 1.0  # nothing to compare
    exact_set = set(exact_results[:k])
    ann_set = set(ann_results[:k])
    overlap = len(exact_set & ann_set)
    return overlap / len(exact_set)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values) -> float:
    lst = list(values)
    return sum(lst) / len(lst) if lst else 0.0
