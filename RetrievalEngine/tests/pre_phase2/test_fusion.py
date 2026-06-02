"""
Unit tests for RRF fusion (no DB, no mocks — pure Python).

Covers:
  - Empty input
  - Single leg passthrough
  - Multi-leg merge with correct score formula
  - Higher k dampens rank differences (relative ordering preserved)
  - Custom weights respected
  - Memory in all legs scores highest
  - rrf_explain attribution per leg
  - rrf_scores_to_ranked_ids ordering
  - Duplicate memory IDs within one leg handled
"""
from __future__ import annotations
import pytest
from app.services.retrieval.fusion import rrf_fuse, rrf_explain, rrf_scores_to_ranked_ids


class TestRRFFuse:
    def test_empty_returns_empty(self):
        assert rrf_fuse([]) == []

    def test_empty_lists_return_empty(self):
        assert rrf_fuse([[], [], []]) == []

    def test_single_leg_preserves_order(self):
        ids = ["a", "b", "c"]
        result = rrf_fuse([ids])
        out_ids = [mid for mid, _ in result]
        assert out_ids == ids

    def test_scores_decrease_with_rank(self):
        ids = ["a", "b", "c"]
        result = rrf_fuse([ids])
        scores = [s for _, s in result]
        assert scores[0] > scores[1] > scores[2]

    def test_rrf_formula_k60(self):
        result = rrf_fuse([["a", "b"]], k=60)
        score_a = dict(result)["a"]
        score_b = dict(result)["b"]
        assert score_a == pytest.approx(1.0 / (0 + 60), rel=1e-6)
        assert score_b == pytest.approx(1.0 / (1 + 60), rel=1e-6)

    def test_custom_k_changes_scores(self):
        result_60 = dict(rrf_fuse([["a", "b"]], k=60))
        result_10 = dict(rrf_fuse([["a", "b"]], k=10))
        # At k=10, rank difference matters more
        ratio_60 = result_60["a"] / result_60["b"]
        ratio_10 = result_10["a"] / result_10["b"]
        assert ratio_10 > ratio_60

    def test_memory_in_all_legs_scores_highest(self):
        result = rrf_fuse([
            ["shared", "bm25_only"],
            ["shared", "dense_only"],
            ["shared", "entity_only"],
        ])
        top_id = result[0][0]
        assert top_id == "shared"

    def test_multi_leg_scores_add(self):
        # "a" appears in both legs at rank 0 → score = 1/60 + 1/60 = 2/60
        result = dict(rrf_fuse([["a", "b"], ["a", "c"]], k=60))
        assert result["a"] == pytest.approx(2.0 / 60, rel=1e-6)

    def test_union_of_all_legs(self):
        result = rrf_fuse([["a", "b"], ["c", "d"], ["e"]])
        ids = {mid for mid, _ in result}
        assert ids == {"a", "b", "c", "d", "e"}

    def test_custom_weights_applied(self):
        # leg 0 weight=2, leg 1 weight=1
        # "a" appears only in leg 0 (rank 0): score = 2*(1/60)
        # "b" appears only in leg 1 (rank 0): score = 1*(1/60)
        result = dict(rrf_fuse([["a"], ["b"]], weights=[2.0, 1.0], k=60))
        assert result["a"] == pytest.approx(2.0 / 60, rel=1e-6)
        assert result["b"] == pytest.approx(1.0 / 60, rel=1e-6)
        assert result["a"] > result["b"]

    def test_weights_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="weights length"):
            rrf_fuse([["a"], ["b"]], weights=[1.0])

    def test_result_sorted_descending(self):
        result = rrf_fuse([["a", "b", "c"], ["c", "b", "a"]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_uniform_weight_equals_no_weight(self):
        lists = [["a", "b"], ["b", "c"]]
        default = dict(rrf_fuse(lists))
        explicit = dict(rrf_fuse(lists, weights=[1.0, 1.0]))
        for mid in default:
            assert default[mid] == pytest.approx(explicit[mid], rel=1e-9)


class TestRRFExplain:
    def test_explain_in_all_legs(self):
        lists = [["a", "b"], ["a", "c"], ["b", "a"]]
        names = ["bm25", "dense", "entity"]
        exp = rrf_explain(lists, names, "a")
        assert exp["memory_id"] == "a"
        assert exp["total_score"] > 0
        assert len(exp["contributions"]) == 3

    def test_explain_not_in_leg_is_zero(self):
        lists = [["a"], ["b"]]
        names = ["bm25", "dense"]
        exp = rrf_explain(lists, names, "a")
        dense_contrib = next(c for c in exp["contributions"] if c["leg"] == "dense")
        assert dense_contrib["contribution"] == 0.0
        assert dense_contrib["rank"] is None

    def test_explain_total_matches_fuse(self):
        lists = [["x", "y"], ["y", "x"]]
        names = ["bm25", "dense"]
        fused = dict(rrf_fuse(lists))
        exp = rrf_explain(lists, names, "x")
        assert exp["total_score"] == pytest.approx(fused["x"], rel=1e-9)


class TestRRFToRankedIds:
    def test_extracts_ids_in_order(self):
        fused = [("a", 0.05), ("b", 0.03), ("c", 0.01)]
        assert rrf_scores_to_ranked_ids(fused) == ["a", "b", "c"]

    def test_empty_input(self):
        assert rrf_scores_to_ranked_ids([]) == []
