"""
Phase 4.4 — Tests for RichMemoryExtractor, build helpers, entity store.

All tests are offline (mock LLM). No API key required.

Test classes:
  TestSessionPositionFromId       — session_position_from_id helper
  TestValidateMemories            — word-length filter
  TestParseResponse               — LLM output parsing (clean/fenced/partial)
  TestExtractEntities             — entity extraction (regex fallback path)
  TestToRecords                   — memory record schema
  TestBuildEntityStore            — entity aggregation + memory_count
  TestCheckMemoryQuality          — quality audit metrics
  TestRichMemoryExtractorNoKey    — fallback extraction (no API key)
  TestRichMemoryExtractorMock     — extraction with mocked LLM
  TestExtractBatch                — batch extraction + checkpoint resume
  TestMemoryWordCounts            — boundary conditions (14w, 15w, 80w, 81w)
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_RE = Path(__file__).resolve().parent.parent
if str(_RE) not in sys.path:
    sys.path.insert(0, str(_RE))

from app.services.extraction.rich_memory_extractor import (
    RichMemoryExtractor,
    _extract_entities,
    _extract_list_from_text,
    _parse_response,
    _to_records,
    _validate_memories,
    build_entity_store,
    check_memory_quality,
    session_position_from_id,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_GOOD_MEMORY = (
    "As of session 4, Caroline has made a firm decision to pursue counseling, "
    "inspired by years of supporting friends through personal challenges."
)
_SHORT_MEMORY  = "Alice works here."                  # 3 words — too short
_LONG_MEMORY   = " ".join([f"word{i}" for i in range(85)])  # 85 words — too long
_PRONOUN_MEMORY = "As of session 3, she decided to pursue nursing."

_VALID_JSON_RESPONSE = json.dumps({
    "state_memories": [_GOOD_MEMORY],
    "episodic_memories": [
        "During session 4, Caroline announced her career change to Melanie, "
        "who responded with strong encouragement and expressed full confidence "
        "in Caroline's counseling abilities."
    ],
})

_FENCED_JSON_RESPONSE = f"```json\n{_VALID_JSON_RESPONSE}\n```"


# ─────────────────────────────────────────────────────────────────────────────
# TestSessionPositionFromId
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionPositionFromId:
    def test_locomo_format(self):
        assert session_position_from_id("c0_session_4") == 4

    def test_locomo_session_10(self):
        assert session_position_from_id("c0_session_10") == 10

    def test_lme_format(self):
        assert session_position_from_id("answer_4be1b6b4_2") == 2

    def test_lme_format_single_digit(self):
        assert session_position_from_id("answer_abc123_1") == 1

    def test_no_trailing_number_fallback(self):
        assert session_position_from_id("no_number") == 1

    def test_empty_string_fallback(self):
        assert session_position_from_id("") == 1

    def test_large_session_number(self):
        assert session_position_from_id("c5_session_272") == 272


# ─────────────────────────────────────────────────────────────────────────────
# TestValidateMemories
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateMemories:
    def test_valid_memory_kept(self):
        result = _validate_memories([_GOOD_MEMORY])
        assert len(result) == 1
        assert result[0] == _GOOD_MEMORY

    def test_short_memory_discarded(self):
        result = _validate_memories([_SHORT_MEMORY])
        assert result == []

    def test_long_memory_truncated_to_80_words(self):
        result = _validate_memories([_LONG_MEMORY])
        assert len(result) == 1
        assert len(result[0].split()) == 80

    def test_mixed_list(self):
        result = _validate_memories([_GOOD_MEMORY, _SHORT_MEMORY, _LONG_MEMORY])
        assert len(result) == 2  # GOOD kept, SHORT discarded, LONG truncated

    def test_non_string_skipped(self):
        result = _validate_memories([None, 42, _GOOD_MEMORY])
        assert len(result) == 1

    def test_exactly_20_words_kept(self):
        mem = " ".join(["word"] * 20)
        result = _validate_memories([mem])
        assert len(result) == 1

    def test_exactly_19_words_discarded(self):
        mem = " ".join(["word"] * 19)
        result = _validate_memories([mem])
        assert result == []

    def test_exactly_14_words_discarded(self):
        mem = " ".join(["word"] * 14)
        result = _validate_memories([mem])
        assert result == []

    def test_exactly_80_words_kept(self):
        mem = " ".join(["word"] * 80)
        result = _validate_memories([mem])
        assert len(result) == 1
        assert len(result[0].split()) == 80

    def test_81_words_truncated_to_80(self):
        mem = " ".join(["word"] * 81)
        result = _validate_memories([mem])
        assert len(result[0].split()) == 80

    def test_empty_list(self):
        result = _validate_memories([])
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# TestParseResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestParseResponse:
    def test_clean_json(self):
        result = _parse_response(_VALID_JSON_RESPONSE)
        assert len(result["state_memories"]) == 1
        assert "Caroline" in result["state_memories"][0]
        assert len(result["episodic_memories"]) == 1

    def test_markdown_fenced_json(self):
        result = _parse_response(_FENCED_JSON_RESPONSE)
        assert len(result["state_memories"]) == 1
        assert len(result["episodic_memories"]) == 1

    def test_empty_arrays_handled(self):
        raw = json.dumps({"state_memories": [], "episodic_memories": []})
        result = _parse_response(raw)
        assert result["state_memories"] == []
        assert result["episodic_memories"] == []

    def test_missing_key_returns_empty(self):
        raw = json.dumps({"state_memories": [_GOOD_MEMORY]})
        result = _parse_response(raw)
        assert len(result["state_memories"]) == 1
        assert result["episodic_memories"] == []

    def test_short_memories_filtered_during_parse(self):
        raw = json.dumps({
            "state_memories": [_SHORT_MEMORY, _GOOD_MEMORY],
            "episodic_memories": [],
        })
        result = _parse_response(raw)
        assert len(result["state_memories"]) == 1  # short discarded
        assert "Caroline" in result["state_memories"][0]

    def test_malformed_json_regex_fallback(self):
        # Broken JSON but extractable via regex
        partial = f'"state_memories": ["{_GOOD_MEMORY}"], "episodic_memories": []'
        result = _parse_response(partial)
        assert len(result["state_memories"]) == 1

    def test_completely_unparseable_returns_empty(self):
        result = _parse_response("Sorry, I cannot extract memories from this.")
        assert result["state_memories"] == []
        assert result["episodic_memories"] == []


# ─────────────────────────────────────────────────────────────────────────────
# TestExtractListFromText
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractListFromText:
    def test_extracts_long_strings(self):
        text = '"state_memories": ["As of session 4, Caroline has decided on counseling as her career path.", "Alice works at a hospital."]'
        result = _extract_list_from_text(text, "state_memories")
        assert len(result) >= 1
        assert any("Caroline" in r for r in result)

    def test_wrong_key_returns_empty(self):
        text = '"state_memories": ["' + _GOOD_MEMORY + '"]'
        result = _extract_list_from_text(text, "episodic_memories")
        assert result == []

    def test_too_short_strings_not_extracted(self):
        # regex requires ≥ 20 chars
        text = '"state_memories": ["short"]'
        result = _extract_list_from_text(text, "state_memories")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# TestExtractEntities
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractEntities:
    def test_extracts_person_name(self):
        result = _extract_entities("As of session 4, Caroline has decided to pursue counseling.")
        assert any("Caroline" in e for e in result)

    def test_extracts_multiple_names(self):
        result = _extract_entities("Caroline and Melanie are close friends who share career advice.")
        names = [e for e in result if e in ("Caroline", "Melanie")]
        assert len(names) >= 1

    def test_no_entities_for_generic_text(self):
        # Pure generic text — no proper nouns
        result = _extract_entities("the weather was nice today in the city during the morning hours")
        assert isinstance(result, list)

    def test_does_not_extract_pronouns(self):
        result = _extract_entities("She decided to pursue nursing as her career.")
        assert "She" not in result
        assert "she" not in result

    def test_returns_list(self):
        result = _extract_entities(_GOOD_MEMORY)
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# TestToRecords
# ─────────────────────────────────────────────────────────────────────────────

class TestToRecords:
    def _parsed(self):
        return {
            "state_memories": [_GOOD_MEMORY],
            "episodic_memories": [
                "During session 4, Caroline announced to Melanie her decision to pursue counseling."
            ],
        }

    def test_record_count(self):
        records = _to_records(self._parsed(), "c0_session_4", 4, None, "LoCoMo")
        assert len(records) == 2

    def test_state_memory_type(self):
        records = _to_records(self._parsed(), "c0_session_4", 4, None, "LoCoMo")
        state = [r for r in records if r["memory_type"] == "state"]
        assert len(state) == 1

    def test_episodic_memory_type(self):
        records = _to_records(self._parsed(), "c0_session_4", 4, None, "LoCoMo")
        episodic = [r for r in records if r["memory_type"] == "episodic"]
        assert len(episodic) == 1

    def test_memory_id_format(self):
        records = _to_records(self._parsed(), "c0_session_4", 4, None, "LoCoMo")
        for rec in records:
            assert rec["memory_id"].startswith("mem_c0_session_4_")

    def test_session_position_stored(self):
        records = _to_records(self._parsed(), "c0_session_4", 4, "2024-02-10", "LoCoMo")
        assert all(r["session_position"] == 4 for r in records)

    def test_session_date_stored(self):
        records = _to_records(self._parsed(), "c0_session_4", 4, "2024-02-10", "LoCoMo")
        assert all(r["session_date"] == "2024-02-10" for r in records)

    def test_dataset_stored(self):
        records = _to_records(self._parsed(), "c0_session_4", 4, None, "LoCoMo")
        assert all(r["dataset"] == "LoCoMo" for r in records)

    def test_entities_list_present(self):
        records = _to_records(self._parsed(), "c0_session_4", 4, None, "LoCoMo")
        assert all(isinstance(r["entities"], list) for r in records)

    def test_empty_parsed_produces_no_records(self):
        parsed = {"state_memories": [], "episodic_memories": []}
        records = _to_records(parsed, "c0_session_4", 4, None, "LoCoMo")
        assert records == []

    def test_memory_id_index_increments(self):
        parsed = {
            "state_memories": [_GOOD_MEMORY, _GOOD_MEMORY],
            "episodic_memories": [],
        }
        records = _to_records(parsed, "c0_session_4", 4, None, "LoCoMo")
        ids = [r["memory_id"] for r in records]
        assert "mem_c0_session_4_state_000" in ids
        assert "mem_c0_session_4_state_001" in ids


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildEntityStore
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildEntityStore:
    def _make_records(self, entities_per_record: list[list[str]]) -> list[dict]:
        records = []
        for i, entities in enumerate(entities_per_record):
            records.append({
                "memory_id": f"mem_session_{i}_state_000",
                "session_id": f"session_{i}",
                "session_position": i + 1,
                "session_date": None,
                "memory": _GOOD_MEMORY,
                "memory_type": "state",
                "entities": entities,
                "dataset": "LoCoMo",
            })
        return records

    def test_entity_aggregation(self):
        records = self._make_records([["Caroline", "Melanie"], ["Caroline"]])
        store = build_entity_store(records)
        caroline = next(e for e in store if e["entity_text"] == "caroline")
        assert caroline["memory_count"] == 2

    def test_memory_count_equals_linked_length(self):
        records = self._make_records([["Alice"], ["Alice"], ["Alice"]])
        store = build_entity_store(records)
        alice = next(e for e in store if e["entity_text"] == "alice")
        assert alice["memory_count"] == len(alice["linked_memory_ids"])

    def test_deduplication_within_entity(self):
        # Same memory_id linked twice should not inflate count
        records = self._make_records([["Bob", "Bob"]])
        store = build_entity_store(records)
        bob = next((e for e in store if e["entity_text"] == "bob"), None)
        assert bob is not None
        assert bob["memory_count"] == 1  # same mem_id deduplicated

    def test_canonical_name_stored(self):
        records = self._make_records([["Caroline"]])
        store = build_entity_store(records)
        caroline = next(e for e in store if e["entity_text"] == "caroline")
        assert caroline["canonical_name"] == "Caroline"

    def test_short_entity_name_skipped(self):
        records = self._make_records([["A", "Caroline"]])
        store = build_entity_store(records)
        keys = [e["entity_text"] for e in store]
        assert "a" not in keys

    def test_sorted_by_memory_count_descending(self):
        records = self._make_records([
            ["Alice"],
            ["Alice", "Bob"],
            ["Alice", "Bob", "Carol"],
            ["Alice"],
        ])
        store = build_entity_store(records)
        counts = [e["memory_count"] for e in store]
        assert counts == sorted(counts, reverse=True)

    def test_empty_records_returns_empty_store(self):
        store = build_entity_store([])
        assert store == []

    def test_linked_memory_ids_contain_correct_ids(self):
        records = self._make_records([["Caroline"], ["Caroline"]])
        store = build_entity_store(records)
        caroline = next(e for e in store if e["entity_text"] == "caroline")
        assert "mem_session_0_state_000" in caroline["linked_memory_ids"]
        assert "mem_session_1_state_000" in caroline["linked_memory_ids"]


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckMemoryQuality
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckMemoryQuality:
    def _make_record(self, memory: str, entities: list[str] | None = None) -> dict:
        return {
            "memory_id": "mem_x",
            "session_id": "session_1",
            "session_position": 1,
            "session_date": None,
            "memory": memory,
            "memory_type": "state",
            "entities": entities or [],
            "dataset": "test",
        }

    def test_empty_returns_zero_count(self):
        q = check_memory_quality([])
        assert q["total_memories"] == 0

    def test_total_count_correct(self):
        records = [self._make_record(_GOOD_MEMORY) for _ in range(5)]
        q = check_memory_quality(records)
        assert q["total_memories"] == 5

    def test_temporal_grounded_detected(self):
        records = [self._make_record("As of session 4, Caroline has decided on counseling.")]
        q = check_memory_quality(records)
        assert q["temporal_grounded_pct"] > 0

    def test_ungrounded_detected(self):
        records = [self._make_record("Caroline works at a hospital and enjoys hiking.")]
        q = check_memory_quality(records)
        assert q["temporal_grounded_pct"] == 0

    def test_pronoun_leakage_detected(self):
        records = [self._make_record("As of session 3, she has decided to become a nurse.")]
        q = check_memory_quality(records)
        assert q["pronoun_pct"] > 0

    def test_no_pronoun_leakage(self):
        records = [self._make_record(_GOOD_MEMORY)]
        q = check_memory_quality(records)
        assert q["pronoun_pct"] == 0

    def test_entity_coverage_with_entities(self):
        records = [self._make_record(_GOOD_MEMORY, entities=["Caroline"])]
        q = check_memory_quality(records)
        assert q["entity_coverage_pct"] == 1.0

    def test_entity_coverage_without_entities(self):
        records = [self._make_record(_GOOD_MEMORY, entities=[])]
        q = check_memory_quality(records)
        assert q["entity_coverage_pct"] == 0.0

    def test_avg_words_computed(self):
        mem = " ".join(["word"] * 30)
        records = [self._make_record(mem)]
        q = check_memory_quality(records)
        assert abs(q["avg_words"] - 30.0) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# TestRichMemoryExtractorNoKey
# ─────────────────────────────────────────────────────────────────────────────

class TestRichMemoryExtractorNoKey:
    """Fallback extraction when no API key is available."""

    def setup_method(self):
        self.extractor = RichMemoryExtractor(api_key="", model="claude-sonnet-4-6")

    def test_fallback_returns_list(self):
        content = "Caroline: I have decided to become a counselor after years of supporting friends."
        result = self.extractor.extract(content, "c0_session_4", 4, dataset="LoCoMo")
        assert isinstance(result, list)

    def test_fallback_memory_ids_contain_session_id(self):
        content = (
            "Alice has been working at City Hospital as a senior nurse "
            "for over five years and loves her dedicated team there every single day."
        )
        result = self.extractor.extract(content, "c0_session_1", 1, dataset="LoCoMo")
        assert len(result) > 0
        for rec in result:
            assert "session" in rec["memory_id"]

    def test_fallback_memory_type_episodic(self):
        # 20+ words to ensure it clears the 15-word minimum in the fallback path
        content = (
            "Alice has been working at City Hospital as a senior nurse "
            "for over five years and loves her dedicated team there every single day."
        )
        result = self.extractor.extract(content, "c0_session_1", 1, dataset="LoCoMo")
        assert len(result) > 0, "Fallback should produce at least one record"
        for rec in result:
            assert rec["memory_type"] == "episodic"

    def test_fallback_required_keys_present(self):
        content = "Bob has been working as an engineer at TechCorp for the past three years."
        result = self.extractor.extract(content, "c0_session_2", 2, dataset="LoCoMo")
        required_keys = {
            "memory_id", "session_id", "session_position", "session_date",
            "memory", "memory_type", "entities", "dataset",
        }
        for rec in result:
            assert required_keys.issubset(set(rec.keys()))

    def test_fallback_session_position_from_id_when_none(self):
        content = "Alice works at City Hospital as a senior nurse and loves her team there."
        result = self.extractor.extract(content, "c0_session_7", dataset="LoCoMo")
        for rec in result:
            assert rec["session_position"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# TestRichMemoryExtractorMock
# ─────────────────────────────────────────────────────────────────────────────

class TestRichMemoryExtractorMock:
    """Extraction with mocked LLM (simulates API calls without network)."""

    def setup_method(self):
        self.extractor = RichMemoryExtractor(api_key="fake_key", model="claude-sonnet-4-6")

    def _mock_call(self, response_text: str):
        """Patch _call_llm to return fixed text."""
        return patch.object(self.extractor, "_call_llm", return_value=response_text)

    def test_clean_json_response(self):
        with self._mock_call(_VALID_JSON_RESPONSE):
            result = self.extractor.extract(
                "Caroline: I am becoming a counselor.",
                "c0_session_4", 4, dataset="LoCoMo",
            )
        assert len(result) == 2
        assert result[0]["memory_type"] == "state"
        assert result[1]["memory_type"] == "episodic"

    def test_fenced_json_response(self):
        with self._mock_call(_FENCED_JSON_RESPONSE):
            result = self.extractor.extract(
                "Some session text.",
                "c0_session_4", 4, dataset="LoCoMo",
            )
        assert len(result) == 2

    def test_memory_ids_unique(self):
        response = json.dumps({
            "state_memories": [_GOOD_MEMORY, _GOOD_MEMORY],
            "episodic_memories": [
                "During session 4, Caroline announced her career decision to Melanie confidently."
            ],
        })
        with self._mock_call(response):
            result = self.extractor.extract(
                "text", "c0_session_4", 4, dataset="LoCoMo",
            )
        ids = [r["memory_id"] for r in result]
        assert len(ids) == len(set(ids)), "Memory IDs must be unique"

    def test_session_position_auto_derived(self):
        with self._mock_call(_VALID_JSON_RESPONSE):
            result = self.extractor.extract(
                "text", "c0_session_7", dataset="LoCoMo",
            )
        assert all(r["session_position"] == 7 for r in result)

    def test_exception_falls_back_gracefully(self):
        with patch.object(self.extractor, "_call_llm", side_effect=Exception("API error")):
            result = self.extractor.extract(
                "Alice works at City Hospital as a senior nurse with a dedicated team.",
                "c0_session_1", 1, dataset="LoCoMo",
            )
        assert isinstance(result, list)

    def test_all_required_keys_present(self):
        with self._mock_call(_VALID_JSON_RESPONSE):
            result = self.extractor.extract(
                "text", "c0_session_4", 4, session_date="2024-02-10", dataset="LoCoMo",
            )
        required = {"memory_id", "session_id", "session_position", "session_date",
                    "memory", "memory_type", "entities", "dataset"}
        for rec in result:
            assert required.issubset(rec.keys())

    def test_session_date_propagated(self):
        with self._mock_call(_VALID_JSON_RESPONSE):
            result = self.extractor.extract(
                "text", "c0_session_4", 4, session_date="2024-02-10", dataset="LoCoMo",
            )
        assert all(r["session_date"] == "2024-02-10" for r in result)

    def test_dataset_propagated(self):
        with self._mock_call(_VALID_JSON_RESPONSE):
            result = self.extractor.extract("text", "s1", 1, dataset="LongMemEval")
        assert all(r["dataset"] == "LongMemEval" for r in result)


# ─────────────────────────────────────────────────────────────────────────────
# TestExtractBatch
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractBatch:
    def setup_method(self):
        self.extractor = RichMemoryExtractor(api_key="fake_key", model="claude-sonnet-4-6")

    def _sessions(self, n: int) -> list[dict]:
        return [
            {
                "session_id": f"c0_session_{i+1}",
                "content": f"Caroline: I discussed topic {i} today with Melanie about career plans.",
                "session_position": i + 1,
                "dataset": "LoCoMo",
            }
            for i in range(n)
        ]

    def test_batch_returns_all_session_ids(self):
        sessions = self._sessions(3)
        with patch.object(self.extractor, "_call_llm", return_value=_VALID_JSON_RESPONSE):
            result = self.extractor.extract_batch(sessions, verbose=False)
        assert set(result.keys()) == {"c0_session_1", "c0_session_2", "c0_session_3"}

    def test_batch_each_value_is_list(self):
        sessions = self._sessions(2)
        with patch.object(self.extractor, "_call_llm", return_value=_VALID_JSON_RESPONSE):
            result = self.extractor.extract_batch(sessions, verbose=False)
        for v in result.values():
            assert isinstance(v, list)

    def test_checkpoint_resume(self):
        sessions = self._sessions(4)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ckpt.json"
            # Pre-populate checkpoint with first 2 sessions
            existing = {
                "c0_session_1": [{"memory_id": "mem_c0_session_1_state_000",
                                   "memory": _GOOD_MEMORY, "memory_type": "state",
                                   "session_id": "c0_session_1", "session_position": 1,
                                   "session_date": None, "entities": [], "dataset": "LoCoMo"}],
                "c0_session_2": [],
            }
            ckpt.write_text(json.dumps(existing))

            call_count = 0

            def fake_call(prompt, max_tokens=1200):
                nonlocal call_count
                call_count += 1
                return _VALID_JSON_RESPONSE

            with patch.object(self.extractor, "_call_llm", side_effect=fake_call):
                result = self.extractor.extract_batch(
                    sessions, verbose=False, checkpoint_path=ckpt, checkpoint_every=1
                )

        # Only sessions 3 and 4 should have been extracted (2 calls)
        assert call_count == 2
        assert "c0_session_1" in result
        assert "c0_session_3" in result

    def test_checkpoint_written_on_completion(self):
        sessions = self._sessions(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ckpt.json"
            with patch.object(self.extractor, "_call_llm", return_value=_VALID_JSON_RESPONSE):
                self.extractor.extract_batch(
                    sessions, verbose=False, checkpoint_path=ckpt, checkpoint_every=100
                )
            assert ckpt.exists()
            saved = json.loads(ckpt.read_text())
            assert len(saved) == 2

    def test_no_api_key_falls_back_for_all(self):
        extractor = RichMemoryExtractor(api_key="", model="claude-sonnet-4-6")
        sessions = self._sessions(2)
        result = extractor.extract_batch(sessions, verbose=False)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestMemoryWordCounts (boundary conditions)
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryWordCounts:
    """Explicit boundary tests for the 15-80 word constraint."""

    def _mem(self, n_words: int) -> str:
        return " ".join(["token"] * n_words)

    def test_14_words_rejected(self):
        assert _validate_memories([self._mem(14)]) == []

    def test_19_words_rejected(self):
        assert _validate_memories([self._mem(19)]) == []

    def test_20_words_accepted(self):
        assert len(_validate_memories([self._mem(20)])) == 1

    def test_80_words_accepted(self):
        result = _validate_memories([self._mem(80)])
        assert len(result) == 1
        assert len(result[0].split()) == 80

    def test_81_words_truncated(self):
        result = _validate_memories([self._mem(81)])
        assert len(result) == 1
        assert len(result[0].split()) == 80

    def test_100_words_truncated_to_80(self):
        result = _validate_memories([self._mem(100)])
        assert len(result[0].split()) == 80

    def test_1_word_rejected(self):
        assert _validate_memories(["Hello"]) == []

    def test_whitespace_only_rejected(self):
        assert _validate_memories(["   "]) == []
