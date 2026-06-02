"""
Phase 4.6 tests — HaikuTemporalReranker and retrieve_with_details.

All tests are fully offline: no API calls, no disk writes (except temp files).
Haiku API calls are mocked with a deterministic stub.

Test classes:
  TestBuildPrompt              (8)  — prompt construction and content
  TestParseResponse            (10) — JSON / fallback / edge-case parsing
  TestCacheKey                 (4)  — deterministic hash keys
  TestHaikuTemporalRerankerInit  (5) — constructor, cache loading
  TestHaikuTemporalRerankerMock  (12) — full rerank() with mocked _call_haiku
  TestRetrieveWithDetails      (10) — MultiSignalRetriever.retrieve_with_details
  TestCachePersistence         (5)  — disk load/save behavior
  TestEdgeCases                (6)  — empty candidates, 1 candidate, API failure

Total: 60 tests
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from app.services.retrieval.haiku_temporal_reranker import (
    HaikuTemporalReranker,
    _build_prompt,
    _cache_key,
    _parse_response,
)
from app.services.retrieval.multi_signal_retrieval import MultiSignalRetriever


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_candidates(n: int = 5) -> list[dict]:
    return [
        {
            "session_id":       f"session_{i}",
            "memory_text":      f"As of session {i}, Alice works at Hospital {i} as a nurse.",
            "session_position": i,
        }
        for i in range(1, n + 1)
    ]


def _unit_embed(text: str) -> list[float]:
    h = abs(hash(text)) % 1000
    angle = h * (2 * math.pi / 1000)
    v = [math.cos(angle), math.sin(angle), math.cos(2*angle), math.sin(2*angle)]
    norm = math.sqrt(sum(x*x for x in v))
    return [x / norm for x in v]


def _make_retriever(n_mems: int = 6) -> MultiSignalRetriever:
    texts = [f"Memory {i}. Alice works at Hospital as a nurse in session {i}." for i in range(n_mems)]
    rng   = np.random.default_rng(0)
    embs_raw = rng.standard_normal((n_mems, 4)).astype(np.float32)
    norms = np.linalg.norm(embs_raw, axis=1, keepdims=True)
    embs  = embs_raw / np.where(norms == 0, 1.0, norms)
    sessions  = [f"session_{i % 3}" for i in range(n_mems)]
    positions = [i + 1 for i in range(n_mems)]
    return MultiSignalRetriever(
        mem_texts=texts, mem_embs=embs,
        mem_session_keys=sessions, mem_positions=positions,
        entity_store=[], embed_fn=_unit_embed,
        semantic_threshold=0.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TestBuildPrompt
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildPrompt:
    def test_contains_query(self):
        cands = _make_candidates(3)
        prompt = _build_prompt("What does Alice do?", cands)
        assert "What does Alice do?" in prompt

    def test_contains_candidate_count(self):
        cands = _make_candidates(5)
        prompt = _build_prompt("query", cands)
        assert "5" in prompt

    def test_contains_session_positions(self):
        cands = _make_candidates(3)
        prompt = _build_prompt("query", cands)
        assert "pos=1" in prompt
        assert "pos=2" in prompt
        assert "pos=3" in prompt

    def test_contains_memory_text(self):
        cands = [{"session_id": "s0", "memory_text": "unique memory text xyz", "session_position": 5}]
        prompt = _build_prompt("query", cands)
        assert "unique memory text xyz" in prompt

    def test_contains_ranking_rules(self):
        prompt = _build_prompt("query", _make_candidates(2))
        # At least one temporal rule should be present
        assert "CURRENT" in prompt or "current" in prompt.lower()

    def test_numbered_from_one(self):
        cands = _make_candidates(3)
        prompt = _build_prompt("query", cands)
        assert "1." in prompt
        assert "2." in prompt
        assert "3." in prompt

    def test_returns_string(self):
        prompt = _build_prompt("query", _make_candidates(2))
        assert isinstance(prompt, str)

    def test_memory_text_truncated_at_300_chars(self):
        long_text = "A" * 500
        cands = [{"session_id": "s", "memory_text": long_text, "session_position": 1}]
        prompt = _build_prompt("query", cands)
        # The rendered memory text in the prompt should be at most 300 chars
        assert "A" * 301 not in prompt


# ══════════════════════════════════════════════════════════════════════════════
# TestParseResponse
# ══════════════════════════════════════════════════════════════════════════════

class TestParseResponse:
    def test_valid_json_ranked(self):
        result = _parse_response('{"ranked": [3, 1, 2]}', n=3)
        assert result[:3] == [3, 1, 2]

    def test_valid_json_array(self):
        result = _parse_response("[2, 1, 3]", n=3)
        assert result[:3] == [2, 1, 3]

    def test_duplicate_indices_deduplicated(self):
        result = _parse_response('{"ranked": [1, 1, 2, 3]}', n=3)
        assert result.count(1) == 1

    def test_out_of_range_indices_dropped(self):
        result = _parse_response('{"ranked": [0, 1, 4, 2, 3]}', n=3)
        assert 0 not in result
        assert 4 not in result
        assert set(result) == {1, 2, 3}

    def test_missing_indices_appended(self):
        # Only [1, 2] given, 3 is missing
        result = _parse_response('{"ranked": [1, 2]}', n=3)
        assert set(result) == {1, 2, 3}
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3

    def test_regex_fallback_on_invalid_json(self):
        result = _parse_response('ranked: [2, 1, 3] some text', n=3)
        assert set(result) == {1, 2, 3}

    def test_identity_fallback_on_garbage(self):
        result = _parse_response("no numbers here at all", n=3)
        assert result == [1, 2, 3]

    def test_identity_fallback_on_empty(self):
        result = _parse_response("", n=3)
        assert result == [1, 2, 3]

    def test_json_with_extra_text(self):
        result = _parse_response('Here is the ranking: {"ranked": [2, 1, 3]}', n=3)
        assert result[:3] == [2, 1, 3]

    def test_returns_list_of_ints(self):
        result = _parse_response('{"ranked": [1, 2, 3]}', n=3)
        assert all(isinstance(x, int) for x in result)


# ══════════════════════════════════════════════════════════════════════════════
# TestCacheKey
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheKey:
    def test_same_inputs_same_key(self):
        cands = _make_candidates(3)
        k1 = _cache_key("query", cands)
        k2 = _cache_key("query", cands)
        assert k1 == k2

    def test_different_query_different_key(self):
        cands = _make_candidates(3)
        k1 = _cache_key("query A", cands)
        k2 = _cache_key("query B", cands)
        assert k1 != k2

    def test_different_candidates_different_key(self):
        k1 = _cache_key("query", _make_candidates(3))
        k2 = _cache_key("query", _make_candidates(4))
        assert k1 != k2

    def test_returns_24_char_hex(self):
        k = _cache_key("query", _make_candidates(3))
        assert len(k) == 24
        assert all(c in "0123456789abcdef" for c in k)


# ══════════════════════════════════════════════════════════════════════════════
# TestHaikuTemporalRerankerInit
# ══════════════════════════════════════════════════════════════════════════════

class TestHaikuTemporalRerankerInit:
    def test_basic_construction(self):
        r = HaikuTemporalReranker(api_key="test-key")
        assert r is not None

    def test_empty_api_key(self):
        r = HaikuTemporalReranker(api_key="")
        assert r._api_key == ""

    def test_none_api_key_reads_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-xyz")
        r = HaikuTemporalReranker(api_key=None)
        assert r._api_key == "env-key-xyz"

    def test_cache_path_none_empty_cache(self):
        r = HaikuTemporalReranker(api_key="key", cache_path=None)
        assert r.cache_size == 0

    def test_load_existing_cache(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"abc123": [1, 2, 3]}, f)
            p = Path(f.name)
        r = HaikuTemporalReranker(api_key="key", cache_path=p)
        assert r.cache_size == 1
        p.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# TestHaikuTemporalRerankerMock
# ══════════════════════════════════════════════════════════════════════════════

class TestHaikuTemporalRerankerMock:
    def _make_reranker(self, response: str = '{"ranked": [1,2,3,4,5]}') -> HaikuTemporalReranker:
        r = HaikuTemporalReranker(api_key="fake-key")
        r._call_haiku = lambda prompt, n: _parse_response(response, n)
        return r

    def test_returns_session_ids(self):
        r = self._make_reranker('{"ranked": [1, 2, 3]}')
        cands = _make_candidates(3)
        result = r.rerank("query", cands)
        assert all(isinstance(s, str) for s in result)

    def test_correct_reranking_order(self):
        r = self._make_reranker('{"ranked": [3, 1, 2]}')
        cands = _make_candidates(3)
        result = r.rerank("query", cands)
        assert result[0] == "session_3"
        assert result[1] == "session_1"
        assert result[2] == "session_2"

    def test_result_length_matches_candidates(self):
        r = self._make_reranker('{"ranked": [5,4,3,2,1]}')
        cands = _make_candidates(5)
        result = r.rerank("query", cands)
        assert len(result) == 5

    def test_recency_query_gets_later_session_first(self):
        """Simulate Haiku correctly preferring later session for 'currently' query."""
        # Haiku returns [5, 4, 3, 2, 1] — most recent first
        r = self._make_reranker('{"ranked": [5, 4, 3, 2, 1]}')
        cands = _make_candidates(5)
        result = r.rerank("Where does Alice currently work?", cands)
        assert result[0] == "session_5"  # highest session_position first

    def test_first_occurrence_query_gets_earliest_session(self):
        """Simulate Haiku correctly preferring earliest session for 'first' query."""
        # Haiku returns [1, 2, 3, 4, 5] — earliest first
        r = self._make_reranker('{"ranked": [1, 2, 3, 4, 5]}')
        cands = _make_candidates(5)
        result = r.rerank("When did Alice first mention the hospital?", cands)
        assert result[0] == "session_1"  # earliest session_position first

    def test_caching_prevents_second_api_call(self):
        call_count = [0]

        def counting_call(prompt, n):
            call_count[0] += 1
            return list(range(1, n + 1))

        r = HaikuTemporalReranker(api_key="fake-key")
        r._call_haiku = counting_call
        cands = _make_candidates(3)

        r.rerank("same query", cands)
        r.rerank("same query", cands)
        assert call_count[0] == 1  # Only called once

    def test_different_queries_make_two_api_calls(self):
        call_count = [0]

        def counting_call(prompt, n):
            call_count[0] += 1
            return list(range(1, n + 1))

        r = HaikuTemporalReranker(api_key="fake-key")
        r._call_haiku = counting_call
        cands = _make_candidates(3)

        r.rerank("query A", cands)
        r.rerank("query B", cands)
        assert call_count[0] == 2

    def test_no_api_key_returns_identity_order(self):
        r = HaikuTemporalReranker(api_key="")
        cands = _make_candidates(4)
        result = r.rerank("query", cands)
        assert result == ["session_1", "session_2", "session_3", "session_4"]

    def test_max_candidates_limit(self):
        r = HaikuTemporalReranker(api_key="fake-key", max_candidates=3)
        seen_n = [None]

        def capped_call(prompt, n):
            seen_n[0] = n
            return list(range(1, n + 1))

        r._call_haiku = capped_call
        cands = _make_candidates(10)
        r.rerank("query", cands)
        assert seen_n[0] == 3  # capped at max_candidates

    def test_invalid_api_response_falls_back_to_identity(self):
        r = HaikuTemporalReranker(api_key="fake-key")
        r._call_haiku = lambda prompt, n: _parse_response("totally invalid", n)
        cands = _make_candidates(3)
        result = r.rerank("query", cands)
        # All session IDs should be present
        assert set(result) == {"session_1", "session_2", "session_3"}

    def test_clear_cache_resets_count(self):
        r = self._make_reranker()
        r.rerank("query", _make_candidates(3))
        assert r.cache_size == 1
        r.clear_cache()
        assert r.cache_size == 0

    def test_prompt_contains_query_text(self):
        prompts_seen = []

        def capturing_call(prompt, n):
            prompts_seen.append(prompt)
            return list(range(1, n + 1))

        r = HaikuTemporalReranker(api_key="fake-key")
        r._call_haiku = capturing_call
        r.rerank("What is Alice's current job?", _make_candidates(2))
        assert "Alice" in prompts_seen[0]
        assert "current job" in prompts_seen[0]


# ══════════════════════════════════════════════════════════════════════════════
# TestRetrieveWithDetails
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrieveWithDetails:
    def test_returns_list_of_dicts(self):
        r = _make_retriever(n_mems=6)
        result = r.retrieve_with_details("query", top_k=3)
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)

    def test_each_dict_has_required_keys(self):
        r = _make_retriever(n_mems=6)
        result = r.retrieve_with_details("query", top_k=2)
        for d in result:
            assert "session_id" in d
            assert "memory_text" in d
            assert "session_position" in d

    def test_no_duplicate_session_ids(self):
        r = _make_retriever(n_mems=6)
        result = r.retrieve_with_details("query", top_k=5)
        session_ids = [d["session_id"] for d in result]
        assert len(session_ids) == len(set(session_ids))

    def test_memory_text_is_string(self):
        r = _make_retriever(n_mems=6)
        result = r.retrieve_with_details("query", top_k=3)
        for d in result:
            assert isinstance(d["memory_text"], str)
            assert len(d["memory_text"]) > 0

    def test_session_position_is_int(self):
        r = _make_retriever(n_mems=6)
        result = r.retrieve_with_details("query", top_k=3)
        for d in result:
            assert isinstance(d["session_position"], int)

    def test_top_k_respected(self):
        r = _make_retriever(n_mems=6)
        result = r.retrieve_with_details("query", top_k=2)
        assert len(result) <= 2

    def test_session_ids_are_valid_keys(self):
        sessions = ["sess_a", "sess_b", "sess_c", "sess_a", "sess_b", "sess_c"]
        r = _make_retriever(n_mems=6)
        # Patch session keys to known values
        r._session_keys = sessions
        result = r.retrieve_with_details("query", top_k=3)
        valid = {"sess_a", "sess_b", "sess_c"}
        for d in result:
            assert d["session_id"] in valid

    def test_positions_come_from_stored_positions(self):
        positions = [5, 10, 15, 20, 25, 30]
        sessions  = [f"s{i}" for i in range(6)]
        texts = [f"memory {i}" for i in range(6)]
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((6, 4)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        r = MultiSignalRetriever(
            mem_texts=texts, mem_embs=embs,
            mem_session_keys=sessions, mem_positions=positions,
            entity_store=[], embed_fn=_unit_embed,
            semantic_threshold=0.0,
        )
        result = r.retrieve_with_details("query", top_k=6)
        returned_positions = {d["session_id"]: d["session_position"] for d in result}
        for d in result:
            sid = d["session_id"]
            idx = sessions.index(sid)
            assert d["session_position"] == positions[idx]

    def test_with_rephrases(self):
        r = _make_retriever(n_mems=6)
        result = r.retrieve_with_details("query", rephrases=["alt1", "alt2"], top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_empty_returns_list(self):
        # Single memory single session — should return 1 result
        texts = ["single memory"]
        embs  = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        r = MultiSignalRetriever(
            mem_texts=texts, mem_embs=embs,
            mem_session_keys=["only_session"], mem_positions=[1],
            entity_store=[], embed_fn=_unit_embed, semantic_threshold=0.0,
        )
        result = r.retrieve_with_details("query", top_k=5)
        assert len(result) == 1
        assert result[0]["session_id"] == "only_session"


# ══════════════════════════════════════════════════════════════════════════════
# TestCachePersistence
# ══════════════════════════════════════════════════════════════════════════════

class TestCachePersistence:
    def test_cache_written_after_rerank(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            p = Path(f.name)

        r = HaikuTemporalReranker(api_key="fake-key", cache_path=p)
        r._call_haiku = lambda prompt, n: list(range(1, n + 1))
        r.rerank("test query", _make_candidates(3))

        content = json.loads(p.read_text())
        assert len(content) == 1
        p.unlink()

    def test_loaded_cache_used_without_api_call(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cands = _make_candidates(3)
            key = _cache_key("pre-cached query", cands)
            json.dump({key: [3, 2, 1]}, f)
            p = Path(f.name)

        call_count = [0]

        def counting_call(prompt, n):
            call_count[0] += 1
            return list(range(1, n + 1))

        r = HaikuTemporalReranker(api_key="fake-key", cache_path=p)
        r._call_haiku = counting_call
        result = r.rerank("pre-cached query", cands)

        assert call_count[0] == 0  # no API call
        assert result[0] == "session_3"  # from cached [3, 2, 1]
        p.unlink()

    def test_corrupt_cache_starts_fresh(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{")
            p = Path(f.name)

        r = HaikuTemporalReranker(api_key="fake-key", cache_path=p)
        assert r.cache_size == 0  # corrupt → empty cache
        p.unlink()

    def test_no_cache_path_does_not_crash(self):
        r = HaikuTemporalReranker(api_key="fake-key", cache_path=None)
        r._call_haiku = lambda prompt, n: list(range(1, n + 1))
        result = r.rerank("query", _make_candidates(3))
        assert len(result) == 3  # no crash, no file written

    def test_cache_accumulates_across_queries(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            p = Path(f.name)

        r = HaikuTemporalReranker(api_key="fake-key", cache_path=p)
        r._call_haiku = lambda prompt, n: list(range(1, n + 1))

        for i in range(5):
            r.rerank(f"unique query {i}", _make_candidates(3))

        content = json.loads(p.read_text())
        assert len(content) == 5
        p.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# TestEdgeCases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_candidates_returns_empty(self):
        r = HaikuTemporalReranker(api_key="fake-key")
        result = r.rerank("query", [])
        assert result == []

    def test_single_candidate_returns_without_api_call(self):
        call_count = [0]
        r = HaikuTemporalReranker(api_key="fake-key")
        r._call_haiku = lambda p, n: (call_count.__setitem__(0, call_count[0]+1) or [1])
        result = r.rerank("query", _make_candidates(1))
        assert result == ["session_1"]
        assert call_count[0] == 0  # no API call for single candidate

    def test_api_exception_returns_identity(self):
        r = HaikuTemporalReranker(api_key="fake-key")

        def failing_call(prompt, n):
            raise RuntimeError("API failed")

        r._call_haiku = failing_call
        cands = _make_candidates(4)
        # The exception is caught inside _call_haiku which returns identity
        # BUT _call_haiku itself raises here, so we need to test through rerank
        # Actually: rerank() calls _call_haiku but doesn't catch — let's verify _call_haiku
        # In the real implementation, _call_haiku catches all exceptions and returns identity
        # Since we overrode it here to raise, let's test the base class behavior
        r2 = HaikuTemporalReranker(api_key="fake-key")
        # Patch the internal client to raise
        r2._api_key = "fake"
        with patch.object(r2, "_get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API error")
            result = r2._call_haiku("prompt", 4)
        assert result == [1, 2, 3, 4]  # identity fallback

    def test_large_candidate_pool_capped_at_max(self):
        r = HaikuTemporalReranker(api_key="fake-key", max_candidates=5)
        seen_n = [None]

        def capped_call(prompt, n):
            seen_n[0] = n
            return list(range(1, n + 1))

        r._call_haiku = capped_call
        cands = _make_candidates(20)
        result = r.rerank("query", cands)
        assert seen_n[0] == 5
        assert len(result) == 5  # capped, not 20

    def test_retrieve_with_details_positions_in_stored_range(self):
        """All session_positions in result must be from the stored positions list."""
        positions = [1, 2, 3, 4, 5, 6]
        texts = [f"text {i}" for i in range(6)]
        sessions = [f"s{i}" for i in range(6)]
        rng = np.random.default_rng(7)
        embs = rng.standard_normal((6, 4)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        r = MultiSignalRetriever(
            mem_texts=texts, mem_embs=embs,
            mem_session_keys=sessions, mem_positions=positions,
            entity_store=[], embed_fn=_unit_embed, semantic_threshold=0.0,
        )
        result = r.retrieve_with_details("query", top_k=6)
        for d in result:
            assert d["session_position"] in positions

    def test_positions_stored_in_retriever(self):
        """Ensure _positions is stored and accessible."""
        positions = [3, 7, 11]
        texts = ["a", "b", "c"]
        embs = np.eye(3, 4, dtype=np.float32)
        r = MultiSignalRetriever(
            mem_texts=texts, mem_embs=embs,
            mem_session_keys=["s0", "s1", "s2"], mem_positions=positions,
            entity_store=[], embed_fn=_unit_embed, semantic_threshold=0.0,
        )
        assert r._positions == [3, 7, 11]
