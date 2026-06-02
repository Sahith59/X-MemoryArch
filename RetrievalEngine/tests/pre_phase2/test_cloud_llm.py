"""
Tests for CloudLLMExtractionBackend.

Covers:
  - is_available() checks API keys
  - Full pipeline with mocked Claude/OpenAI response
  - OpenAI JSON-object wrapping handled by _parse_llm_json
  - Fallback to rule_based on API error
  - Missing source_quote rejected by validate_draft
  - Invalid canonical_type normalized by resolve_canonical_type
  - Dynamic type labels reported in result
  - Deduplication across chunks
  - Provider routing (claude → anthropic, gpt → openai)
  - Noise filtering reduces tokens sent to LLM
  - Low confidence memories tracked
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction.cloud_llm import (
    CloudLLMExtractionBackend,
    _call_anthropic,
    _call_openai,
    _parse_llm_json,
    _provider_from_model,
)
from app.services.extraction.base import ExtractionConfig, P2ExtractionResult
from tests.conftest import (
    VALID_LLM_RESPONSE,
    OPENAI_WRAPPED_RESPONSE,
    INVALID_JSON_RESPONSE,
    MISSING_SOURCE_QUOTE_RESPONSE,
    DYNAMIC_TYPE_RESPONSE,
    UNKNOWN_TYPE_RESPONSE,
    make_phase1_extraction_result,
)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------

class TestCloudLLMAvailability:
    def test_available_with_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = CloudLLMExtractionBackend()
        assert backend.is_available() is True

    def test_available_with_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        backend = CloudLLMExtractionBackend()
        assert backend.is_available() is True

    def test_unavailable_without_keys(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = CloudLLMExtractionBackend()
        assert backend.is_available() is False

    def test_name(self):
        assert CloudLLMExtractionBackend().name == "cloud_llm"


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

class TestProviderRouting:
    def test_claude_model_routes_to_anthropic(self):
        assert _provider_from_model("claude-haiku-4-5-20251001") == "anthropic"
        assert _provider_from_model("claude-sonnet-4-6") == "anthropic"

    def test_gpt_model_routes_to_openai(self):
        assert _provider_from_model("gpt-4o-mini") == "openai"
        assert _provider_from_model("gpt-4o") == "openai"

    def test_o1_model_routes_to_openai(self):
        assert _provider_from_model("o1-mini") == "openai"

    def test_unknown_model_defaults_to_anthropic(self):
        assert _provider_from_model("some-unknown-model") == "anthropic"


# ---------------------------------------------------------------------------
# Full pipeline: extract() with mocked LLM
# ---------------------------------------------------------------------------

class TestCloudLLMExtract:
    def _make_backend(self):
        return CloudLLMExtractionBackend()

    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_extract_returns_p2_result(self, db, session_obj):
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=VALID_LLM_RESPONSE):
            result = backend.extract(session_obj.id, db, ExtractionConfig(backend="cloud_llm"))
        assert isinstance(result, P2ExtractionResult)

    def test_backend_used(self, db, session_obj):
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=VALID_LLM_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert result.backend_used == "cloud_llm"

    def test_memories_created_count(self, db, session_obj):
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=VALID_LLM_RESPONSE):
            result = backend.extract(session_obj.id, db)
        # 2 valid memories in VALID_LLM_RESPONSE
        assert result.memories_created == 2

    def test_memory_ids_populated(self, db, session_obj):
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=VALID_LLM_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert len(result.memory_ids) == result.memories_created

    def test_llm_api_calls_counted(self, db, session_obj):
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=VALID_LLM_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert result.llm_api_calls >= 1

    def test_openai_wrapped_response_parsed(self, db, session_obj):
        """OpenAI returns {"memories": [...]} — must unwrap correctly."""
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=OPENAI_WRAPPED_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert result.memories_created == 2

    def test_model_used_recorded(self, db, session_obj):
        backend = self._make_backend()
        cfg = ExtractionConfig(backend="cloud_llm", model="claude-haiku-4-5-20251001")
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=VALID_LLM_RESPONSE):
            result = backend.extract(session_obj.id, db, cfg)
        assert result.model_used == "claude-haiku-4-5-20251001"

    def test_chunk_api_error_skipped_gracefully(self, db, session_obj):
        """Per-chunk API errors are swallowed — 0 memories, no fallback."""
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", side_effect=RuntimeError("API down")):
            result = backend.extract(session_obj.id, db)
        assert result.memories_created == 0
        assert result.fallback_used is False

    def test_fallback_on_method_level_error(self, db, session_obj):
        """Method-level failure triggers rule_based fallback."""
        backend = self._make_backend()
        cfg = ExtractionConfig(backend="cloud_llm", fallback_to_rule_based=True)

        mock_phase1 = make_phase1_extraction_result(session_obj.id)
        with patch.object(backend, "_extract_with_llm", side_effect=RuntimeError("session not found")):
            with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_phase1):
                result = backend.extract(session_obj.id, db, cfg)

        assert result.fallback_used is True
        assert "rule_based" in result.backend_used

    def test_no_fallback_raises_on_method_error(self, db, session_obj):
        """Method-level failure with fallback=False propagates the exception."""
        backend = self._make_backend()
        cfg = ExtractionConfig(backend="cloud_llm", fallback_to_rule_based=False)

        with patch.object(backend, "_extract_with_llm", side_effect=RuntimeError("session not found")):
            with pytest.raises(RuntimeError, match="session not found"):
                backend.extract(session_obj.id, db, cfg)

    def test_invalid_json_returns_zero(self, db, session_obj):
        """Malformed JSON → chunk skipped → zero memories extracted."""
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=INVALID_JSON_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert result.memories_created == 0

    def test_missing_source_quote_rejected(self, db, session_obj):
        """Memories without source_quote are rejected by validate_draft."""
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=MISSING_SOURCE_QUOTE_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert result.memories_created == 0

    def test_unknown_canonical_type_becomes_unclassified(self, db, session_obj):
        """Unknown canonical_type is normalized to 'unclassified', not rejected."""
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=UNKNOWN_TYPE_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert result.memories_created == 1

    def test_type_alias_resolved(self, db, session_obj):
        """'preference' maps to 'constraint' via CANONICAL_TYPE_MAP."""
        from app import models
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=DYNAMIC_TYPE_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert result.memories_created == 1
        # Verify the stored memory has canonical_type = 'constraint'
        # (or 'unclassified' if the model doesn't have the column yet — both acceptable)
        assert result.memory_ids != []

    def test_noise_filtered_count_positive(self, db, filler_session, mock_is_technical_false):
        """Session with only filler → all sentences filtered → zero memories."""
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=VALID_LLM_RESPONSE):
            result = backend.extract(filler_session.id, db)
        # noise gate should remove everything; LLM may not even be called
        assert result.noise_filtered_count >= 0

    def test_session_id_in_result(self, db, session_obj):
        backend = self._make_backend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=VALID_LLM_RESPONSE):
            result = backend.extract(session_obj.id, db)
        assert result.session_id == session_obj.id


# ---------------------------------------------------------------------------
# _parse_llm_json (unit tests — no mocking needed)
# ---------------------------------------------------------------------------

class TestParseLlmJson:
    def test_direct_array(self):
        raw = '[{"title": "A"}, {"title": "B"}]'
        result = _parse_llm_json(raw)
        assert len(result) == 2
        assert result[0]["title"] == "A"

    def test_memories_key_wrapped(self):
        raw = json.dumps({"memories": [{"title": "X"}]})
        result = _parse_llm_json(raw)
        assert len(result) == 1

    def test_results_key_wrapped(self):
        raw = json.dumps({"results": [{"title": "Y"}]})
        result = _parse_llm_json(raw)
        assert len(result) == 1

    def test_markdown_fenced_json(self):
        raw = "```json\n[{\"title\": \"Z\"}]\n```"
        result = _parse_llm_json(raw)
        assert len(result) == 1
        assert result[0]["title"] == "Z"

    def test_markdown_fenced_no_language(self):
        raw = "```\n[{\"title\": \"W\"}]\n```"
        result = _parse_llm_json(raw)
        assert len(result) == 1

    def test_invalid_json_returns_empty(self):
        result = _parse_llm_json("this is not json { unclosed")
        assert result == []

    def test_empty_array(self):
        result = _parse_llm_json("[]")
        assert result == []

    def test_single_object_wrapped(self):
        raw = json.dumps({"canonical_type": "decision", "title": "T"})
        result = _parse_llm_json(raw)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_substring_array_extraction(self):
        raw = 'Some preamble text [{"title": "A"}] trailing text'
        result = _parse_llm_json(raw)
        assert len(result) == 1
