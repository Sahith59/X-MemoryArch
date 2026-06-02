"""
Tests for LocalLLMExtractionBackend (Ollama).

Covers:
  - is_available() pings Ollama /api/tags
  - extract() full pipeline with mocked Ollama HTTP
  - Fallback to rule_based when Ollama unreachable (fallback_to_rule_based=True)
  - Raises RuntimeError when Ollama down and fallback=False
  - Ollama timeout surfaced as RuntimeError
  - _build_ollama_prompt format (system/user delimiters)
  - _ollama_generate HTTP payload structure
  - Same P2ExtractionResult fields as cloud_llm
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction.local_llm import (
    LocalLLMExtractionBackend,
    _build_ollama_prompt,
    _ollama_generate,
    _ollama_is_running,
)
from app.services.extraction.base import ExtractionConfig, P2ExtractionResult
from tests.conftest import (
    VALID_LLM_RESPONSE,
    INVALID_JSON_RESPONSE,
    make_phase1_extraction_result,
)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

class TestLocalLLMAvailability:
    def test_available_when_ollama_running(self, monkeypatch):
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            backend = LocalLLMExtractionBackend()
            assert backend.is_available() is True

    def test_unavailable_when_ollama_down(self):
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=False):
            backend = LocalLLMExtractionBackend()
            assert backend.is_available() is False

    def test_name(self):
        assert LocalLLMExtractionBackend().name == "local_llm"


# ---------------------------------------------------------------------------
# _ollama_is_running
# ---------------------------------------------------------------------------

class TestOllamaIsRunning:
    def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            assert _ollama_is_running("http://localhost:11434") is True

    def test_returns_false_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("httpx.get", return_value=mock_resp):
            assert _ollama_is_running("http://localhost:11434") is False

    def test_returns_false_on_connection_error(self):
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert _ollama_is_running("http://localhost:11434") is False

    def test_returns_false_on_any_exception(self):
        with patch("httpx.get", side_effect=Exception("random")):
            assert _ollama_is_running("http://localhost:11434") is False


# ---------------------------------------------------------------------------
# _build_ollama_prompt
# ---------------------------------------------------------------------------

class TestBuildOllamaPrompt:
    def test_contains_system_delimiter(self):
        prompt = _build_ollama_prompt("system text", "user text")
        assert "<|system|>" in prompt
        assert "system text" in prompt

    def test_contains_user_delimiter(self):
        prompt = _build_ollama_prompt("system text", "user text")
        assert "<|user|>" in prompt
        assert "user text" in prompt

    def test_contains_assistant_delimiter(self):
        prompt = _build_ollama_prompt("system text", "user text")
        assert "<|assistant|>" in prompt

    def test_order_system_before_user(self):
        prompt = _build_ollama_prompt("SYSTEM", "USER")
        assert prompt.index("<|system|>") < prompt.index("<|user|>")

    def test_end_delimiters_present(self):
        prompt = _build_ollama_prompt("s", "u")
        assert "<|end|>" in prompt


# ---------------------------------------------------------------------------
# _ollama_generate
# ---------------------------------------------------------------------------

class TestOllamaGenerate:
    def test_returns_response_field(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": '[{"title": "T"}]'}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            result = _ollama_generate("prompt", "llama3.2", "http://localhost:11434")
        assert result == '[{"title": "T"}]'

    def test_timeout_raises_runtime_error(self):
        import httpx
        with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(RuntimeError, match="timed out"):
                _ollama_generate("prompt", "llama3.2", "http://localhost:11434")

    def test_http_error_raises_runtime_error(self):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        exc = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)
        with patch("httpx.post", side_effect=exc):
            with pytest.raises(RuntimeError, match="Ollama HTTP error"):
                _ollama_generate("prompt", "llama3.2", "http://localhost:11434")

    def test_payload_includes_format_json(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "[]"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            _ollama_generate("prompt", "llama3.2", "http://localhost:11434")
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["format"] == "json"
            assert call_kwargs["json"]["stream"] is False


# ---------------------------------------------------------------------------
# Full extract() pipeline
# ---------------------------------------------------------------------------

class TestLocalLLMExtract:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_extract_returns_p2_result(self, db, session_obj):
        backend = LocalLLMExtractionBackend()
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            with patch("app.services.extraction.local_llm._ollama_generate", return_value=VALID_LLM_RESPONSE):
                result = backend.extract(session_obj.id, db)
        assert isinstance(result, P2ExtractionResult)

    def test_backend_used(self, db, session_obj):
        backend = LocalLLMExtractionBackend()
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            with patch("app.services.extraction.local_llm._ollama_generate", return_value=VALID_LLM_RESPONSE):
                result = backend.extract(session_obj.id, db)
        assert result.backend_used == "local_llm"

    def test_memories_created(self, db, session_obj):
        backend = LocalLLMExtractionBackend()
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            with patch("app.services.extraction.local_llm._ollama_generate", return_value=VALID_LLM_RESPONSE):
                result = backend.extract(session_obj.id, db)
        assert result.memories_created == 2

    def test_model_used_recorded(self, db, session_obj):
        backend = LocalLLMExtractionBackend()
        cfg = ExtractionConfig(backend="local_llm", model="mistral")
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            with patch("app.services.extraction.local_llm._ollama_generate", return_value=VALID_LLM_RESPONSE):
                result = backend.extract(session_obj.id, db, cfg)
        assert result.model_used == "mistral"

    def test_fallback_when_ollama_unreachable(self, db, session_obj):
        backend = LocalLLMExtractionBackend()
        cfg = ExtractionConfig(backend="local_llm", fallback_to_rule_based=True)
        mock_phase1 = make_phase1_extraction_result(session_obj.id)

        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=False):
            with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_phase1):
                result = backend.extract(session_obj.id, db, cfg)

        assert result.fallback_used is True
        assert "rule_based" in result.backend_used

    def test_raises_when_ollama_down_no_fallback(self, db, session_obj):
        backend = LocalLLMExtractionBackend()
        cfg = ExtractionConfig(backend="local_llm", fallback_to_rule_based=False)

        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=False):
            with pytest.raises(RuntimeError, match="Ollama is not reachable"):
                backend.extract(session_obj.id, db, cfg)

    def test_chunk_error_skipped_gracefully(self, db, session_obj):
        """Per-chunk Ollama errors are swallowed — 0 memories, no fallback."""
        backend = LocalLLMExtractionBackend()
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            with patch("app.services.extraction.local_llm._ollama_generate", side_effect=RuntimeError("timeout")):
                result = backend.extract(session_obj.id, db)
        assert result.memories_created == 0
        assert result.fallback_used is False

    def test_fallback_on_extraction_method_error(self, db, session_obj):
        """Method-level failure (not chunk-level) triggers rule_based fallback."""
        backend = LocalLLMExtractionBackend()
        cfg = ExtractionConfig(backend="local_llm", fallback_to_rule_based=True)
        mock_phase1 = make_phase1_extraction_result(session_obj.id)

        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            with patch.object(backend, "_extract_with_ollama", side_effect=RuntimeError("method crash")):
                with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_phase1):
                    result = backend.extract(session_obj.id, db, cfg)

        assert result.fallback_used is True

    def test_invalid_json_from_ollama_returns_zero(self, db, session_obj):
        backend = LocalLLMExtractionBackend()
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            with patch("app.services.extraction.local_llm._ollama_generate", return_value=INVALID_JSON_RESPONSE):
                result = backend.extract(session_obj.id, db)
        assert result.memories_created == 0

    def test_llm_api_calls_counted(self, db, session_obj):
        backend = LocalLLMExtractionBackend()
        with patch("app.services.extraction.local_llm._ollama_is_running", return_value=True):
            with patch("app.services.extraction.local_llm._ollama_generate", return_value=VALID_LLM_RESPONSE):
                result = backend.extract(session_obj.id, db)
        assert result.llm_api_calls >= 1
