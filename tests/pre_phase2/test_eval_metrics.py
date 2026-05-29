"""
Sub-phase 2.3 — Evaluation metrics unit tests.

Covers: recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k,
        evaluate(), EvalReport aggregation, BenchmarkSuite, ann_recall_at_k.
"""
from __future__ import annotations

import math

import pytest

from app.services.benchmark.eval_metrics import (
    BenchmarkSuite,
    EvalReport,
    QueryResult,
    ann_recall_at_k,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall_first_result(self):
        assert recall_at_k(["a", "b", "c"], ["a"], k=1) == 1.0

    def test_recall_zero_when_no_hits(self):
        assert recall_at_k(["a", "b"], ["c", "d"], k=5) == 0.0

    def test_partial_recall(self):
        # 1 of 2 relevant in top 3
        assert recall_at_k(["a", "b", "c"], ["a", "x"], k=3) == 0.5

    def test_recall_at_k_clips_to_k(self):
        # relevant item at position 4 should not count for k=3
        assert recall_at_k(["a", "b", "c", "rel"], ["rel"], k=3) == 0.0

    def test_recall_with_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=5) == 0.0

    def test_recall_with_empty_retrieved(self):
        assert recall_at_k([], ["a"], k=5) == 0.0

    def test_recall_1_0_all_relevant_in_top_k(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_recall_ignores_position_beyond_k(self):
        # all relevant items beyond k — recall=0
        assert recall_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_recall_multiple_relevant_partial_hit(self):
        # 2 of 3 relevant in top 5
        result = recall_at_k(["a", "b", "x", "y", "c"], ["a", "b", "d"], k=5)
        assert abs(result - 2/3) < 0.001


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_zero_precision(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_half_precision(self):
        assert precision_at_k(["a", "x"], ["a"], k=2) == 0.5

    def test_precision_at_k_zero(self):
        assert precision_at_k(["a", "b"], ["a"], k=0) == 0.0

    def test_precision_does_not_exceed_one(self):
        assert precision_at_k(["a", "a", "a"], ["a"], k=3) <= 1.0


# ---------------------------------------------------------------------------
# mrr_at_k
# ---------------------------------------------------------------------------

class TestMRRAtK:
    def test_first_result_relevant_mrr_is_1(self):
        assert mrr_at_k(["a", "b", "c"], ["a"], k=10) == 1.0

    def test_second_result_relevant_mrr_is_0_5(self):
        assert mrr_at_k(["x", "a", "b"], ["a"], k=10) == 0.5

    def test_no_relevant_result_mrr_is_0(self):
        assert mrr_at_k(["x", "y", "z"], ["a"], k=10) == 0.0

    def test_relevant_beyond_k_mrr_is_0(self):
        # relevant item at position 5, k=3 → not counted
        assert mrr_at_k(["x", "y", "z", "w", "a"], ["a"], k=3) == 0.0

    def test_multiple_relevant_uses_first_hit(self):
        # positions 2 and 4 relevant; MRR = 1/2
        assert mrr_at_k(["x", "a", "y", "b"], ["a", "b"], k=10) == 0.5

    def test_empty_retrieved(self):
        assert mrr_at_k([], ["a"], k=10) == 0.0

    def test_empty_relevant(self):
        assert mrr_at_k(["a", "b"], [], k=10) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ndcg_when_all_relevant_at_top(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=3) == pytest.approx(1.0)

    def test_ndcg_zero_when_no_relevant(self):
        assert ndcg_at_k(["x", "y"], ["a"], k=5) == 0.0

    def test_ndcg_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], [], k=5) == 0.0

    def test_ndcg_single_relevant_first_position(self):
        # DCG = 1/log2(2) = 1.0; IDCG = 1.0 → NDCG = 1.0
        assert ndcg_at_k(["a"], ["a"], k=5) == pytest.approx(1.0)

    def test_ndcg_single_relevant_second_position(self):
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0 → NDCG < 1.0
        result = ndcg_at_k(["x", "a"], ["a"], k=5)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert result == pytest.approx(expected)

    def test_ndcg_strictly_decreases_with_lower_rank(self):
        n1 = ndcg_at_k(["a", "x", "y"], ["a"], k=5)
        n2 = ndcg_at_k(["x", "a", "y"], ["a"], k=5)
        n3 = ndcg_at_k(["x", "y", "a"], ["a"], k=5)
        assert n1 > n2 > n3

    def test_ndcg_clips_at_k(self):
        # relevant only at position 6, k=5 → ndcg=0
        assert ndcg_at_k(["x", "x", "x", "x", "x", "a"], ["a"], k=5) == 0.0


# ---------------------------------------------------------------------------
# evaluate() helper
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_query_result(self):
        qr = evaluate("What is the DB?", ["a", "b"], ["a"], latency_ms=42)
        assert isinstance(qr, QueryResult)
        assert qr.recall_at_1 == 1.0
        assert qr.latency_ms == 42

    def test_all_metrics_computed(self):
        qr = evaluate("query", ["a", "b", "c", "d", "e"], ["a", "c"])
        assert 0 <= qr.recall_at_1 <= 1
        assert 0 <= qr.recall_at_3 <= 1
        assert 0 <= qr.recall_at_5 <= 1
        assert 0 <= qr.mrr_at_10 <= 1
        assert 0 <= qr.ndcg_at_10 <= 1

    def test_privacy_violations_tracked(self):
        qr = evaluate("q", ["a"], ["a"], privacy_violations=3)
        assert qr.privacy_violations == 3

    def test_zero_privacy_violations_by_default(self):
        qr = evaluate("q", ["a"], ["a"])
        assert qr.privacy_violations == 0


# ---------------------------------------------------------------------------
# EvalReport aggregation
# ---------------------------------------------------------------------------

class TestEvalReport:
    def _make_report(self, recall_5_values: list[float]) -> EvalReport:
        report = EvalReport(scenario_name="test")
        for r5 in recall_5_values:
            # Craft a QueryResult with specified recall@5
            qr = evaluate(
                f"query_{r5}",
                retrieved_ids=["a"] if r5 > 0 else [],
                relevant_ids=["a"],
                latency_ms=10,
            )
            # Directly override for controlled testing
            qr.recall_at_5 = r5
            report.query_results.append(qr)
        return report

    def test_mean_recall_at_5(self):
        report = self._make_report([0.6, 0.8, 1.0])
        assert abs(report.recall_at_5 - 0.8) < 0.001

    def test_empty_report_metrics_are_zero(self):
        report = EvalReport(scenario_name="empty")
        assert report.recall_at_5 == 0.0
        assert report.mrr_at_10 == 0.0

    def test_meets_recall_target_true_when_above_080(self):
        report = EvalReport(scenario_name="t")
        report.query_results.append(evaluate("q", ["a"], ["a"]))  # recall=1.0
        assert report.meets_recall_target is True

    def test_meets_recall_target_false_when_below_080(self):
        report = EvalReport(scenario_name="t")
        report.query_results.append(evaluate("q", [], ["a"]))  # recall=0.0
        assert report.meets_recall_target is False

    def test_privacy_clean_true_when_no_violations(self):
        report = EvalReport(scenario_name="t")
        report.query_results.append(evaluate("q", ["a"], ["a"], privacy_violations=0))
        assert report.privacy_clean is True

    def test_privacy_clean_false_when_violations_exist(self):
        report = EvalReport(scenario_name="t")
        report.query_results.append(evaluate("q", ["a"], ["a"], privacy_violations=1))
        assert report.privacy_clean is False

    def test_latency_p50_median(self):
        report = EvalReport(scenario_name="t")
        for ms in [10, 20, 30, 40, 50]:
            report.query_results.append(evaluate("q", [], [], latency_ms=ms))
        assert report.latency_p50_ms == 30  # median of [10,20,30,40,50]

    def test_latency_p95(self):
        report = EvalReport(scenario_name="t")
        for ms in range(1, 101):
            report.query_results.append(evaluate("q", [], [], latency_ms=ms))
        # p95 ≈ 95th percentile
        assert report.latency_p95_ms >= 90

    def test_total_privacy_violations_sum(self):
        report = EvalReport(scenario_name="t")
        report.query_results.append(evaluate("q1", [], [], privacy_violations=2))
        report.query_results.append(evaluate("q2", [], [], privacy_violations=3))
        assert report.total_privacy_violations == 5

    def test_summary_contains_scenario_name(self):
        report = EvalReport(scenario_name="my_scenario")
        summary = report.summary()
        assert "my_scenario" in summary

    def test_summary_contains_metric_names(self):
        report = EvalReport(scenario_name="t")
        report.query_results.append(evaluate("q", ["a"], ["a"]))
        summary = report.summary()
        assert "Recall" in summary
        assert "MRR" in summary
        assert "NDCG" in summary


# ---------------------------------------------------------------------------
# BenchmarkSuite aggregation
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:
    def _make_suite(self) -> BenchmarkSuite:
        suite = BenchmarkSuite()
        r1 = EvalReport(scenario_name="s1")
        r1.query_results.append(evaluate("q1", ["a"], ["a"]))       # recall=1.0
        r2 = EvalReport(scenario_name="s2")
        r2.query_results.append(evaluate("q2", [], ["b"]))           # recall=0.0
        suite.reports = [r1, r2]
        return suite

    def test_total_queries(self):
        suite = self._make_suite()
        assert suite.total_queries == 2

    def test_overall_recall_at_5_average(self):
        suite = self._make_suite()
        assert abs(suite.overall_recall_at_5 - 0.5) < 0.001

    def test_privacy_clean_true_when_all_clean(self):
        suite = self._make_suite()
        assert suite.privacy_clean is True

    def test_privacy_clean_false_when_any_report_has_violations(self):
        suite = self._make_suite()
        suite.reports[0].query_results[0].privacy_violations = 1
        assert suite.privacy_clean is False

    def test_full_report_string_contains_overall(self):
        suite = self._make_suite()
        report = suite.full_report()
        assert "OVERALL" in report
        assert "s1" in report
        assert "s2" in report


# ---------------------------------------------------------------------------
# ANN Recall@k validation
# ---------------------------------------------------------------------------

class TestANNRecallAtK:
    def test_perfect_overlap_returns_1(self):
        exact = ["a", "b", "c", "d", "e"]
        ann = ["a", "b", "c", "d", "e"]
        assert ann_recall_at_k(exact, ann, k=5) == 1.0

    def test_zero_overlap_returns_0(self):
        exact = ["a", "b", "c"]
        ann = ["x", "y", "z"]
        assert ann_recall_at_k(exact, ann, k=3) == 0.0

    def test_partial_overlap(self):
        exact = ["a", "b", "c", "d"]
        ann = ["a", "b", "x", "y"]
        assert ann_recall_at_k(exact, ann, k=4) == 0.5

    def test_empty_exact_returns_1(self):
        # No exact results → nothing to miss → vacuously perfect
        assert ann_recall_at_k([], ["a", "b"], k=5) == 1.0

    def test_clips_to_k(self):
        exact = ["a", "b", "c", "d", "e"]
        ann = ["a", "b", "c", "x", "y"]  # matches first 3 of exact top-5
        result = ann_recall_at_k(exact, ann, k=5)
        # overlap = {a,b,c} ∩ {a,b,c,x,y} = {a,b,c} → 3/5
        assert result == pytest.approx(3 / 5)

    def test_must_be_above_095_threshold(self):
        exact = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        # ANN returns 9 of 10 correct → 0.9 < 0.95 threshold
        ann = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "z"]
        assert ann_recall_at_k(exact, ann, k=10) == pytest.approx(0.9)
        assert ann_recall_at_k(exact, ann, k=10) < 0.95
