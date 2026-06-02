"""
Tests for ExtractionBackend Protocol, registry, and data classes.

Covers:
  - Protocol structural compliance for all 3 backends
  - Registry: register_backend, get_backend, list_backends
  - get_backend_or_fallback behavior
  - resolve_canonical_type mapping
  - ExtractionConfig defaults
  - P2ExtractionResult and ExtractedMemoryDraft dataclasses
"""
from __future__ import annotations

import pytest

from app.services.extraction.base import (
    CANONICAL_TYPE_MAP,
    VALID_CANONICAL_TYPES,
    ExtractionBackend,
    ExtractionConfig,
    ExtractedMemoryDraft,
    P2ExtractionResult,
    _REGISTRY,
    get_backend,
    get_backend_or_fallback,
    list_backends,
    register_backend,
    resolve_canonical_type,
)
from app.services.extraction.rule_based import rule_based_backend
from app.services.extraction.cloud_llm import cloud_llm_backend
from app.services.extraction.local_llm import local_llm_backend


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_rule_based_is_backend(self):
        assert isinstance(rule_based_backend, ExtractionBackend)

    def test_cloud_llm_is_backend(self):
        assert isinstance(cloud_llm_backend, ExtractionBackend)

    def test_local_llm_is_backend(self):
        assert isinstance(local_llm_backend, ExtractionBackend)

    def test_rule_based_name(self):
        assert rule_based_backend.name == "rule_based"

    def test_cloud_llm_name(self):
        assert cloud_llm_backend.name == "cloud_llm"

    def test_local_llm_name(self):
        assert local_llm_backend.name == "local_llm"

    def test_all_backends_have_is_available(self):
        for backend in (rule_based_backend, cloud_llm_backend, local_llm_backend):
            assert callable(backend.is_available)

    def test_all_backends_have_extract(self):
        for backend in (rule_based_backend, cloud_llm_backend, local_llm_backend):
            assert callable(backend.extract)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_three_backends_registered(self):
        assert "rule_based" in _REGISTRY
        assert "cloud_llm" in _REGISTRY
        assert "local_llm" in _REGISTRY

    def test_list_backends_returns_all(self):
        statuses = list_backends()
        names = {s["name"] for s in statuses}
        assert {"rule_based", "cloud_llm", "local_llm"}.issubset(names)

    def test_list_backends_has_available_field(self):
        for status in list_backends():
            assert "available" in status
            assert isinstance(status["available"], bool)

    def test_get_backend_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown extraction backend"):
            get_backend("nonexistent_backend")

    def test_get_backend_unavailable_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(cloud_llm_backend, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="not available"):
            get_backend("cloud_llm")

    def test_get_backend_available_returns_backend(self, monkeypatch):
        monkeypatch.setattr(rule_based_backend, "is_available", lambda: True)
        backend = get_backend("rule_based")
        assert backend is rule_based_backend

    def test_get_backend_or_fallback_preferred_available(self, monkeypatch):
        monkeypatch.setattr(rule_based_backend, "is_available", lambda: True)
        backend, fallback_used = get_backend_or_fallback("rule_based")
        assert backend is rule_based_backend
        assert fallback_used is False

    def test_get_backend_or_fallback_uses_rule_based_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(cloud_llm_backend, "is_available", lambda: False)
        monkeypatch.setattr(rule_based_backend, "is_available", lambda: True)
        backend, fallback_used = get_backend_or_fallback("cloud_llm")
        assert backend is rule_based_backend
        assert fallback_used is True

    def test_register_backend_adds_to_registry(self):
        class FakeBackend:
            @property
            def name(self):
                return "test_fake"
            def is_available(self):
                return True
            def extract(self, session_id, db, config=None):
                pass

        fake = FakeBackend()
        register_backend(fake)
        assert "test_fake" in _REGISTRY
        del _REGISTRY["test_fake"]  # cleanup


# ---------------------------------------------------------------------------
# resolve_canonical_type
# ---------------------------------------------------------------------------

class TestResolveCanonicalType:
    @pytest.mark.parametrize("raw,expected", [
        ("decision", "decision"),
        ("problem", "problem"),
        ("preference", "constraint"),
        ("plan", "task"),
        ("fact", "insight"),
        ("procedure", "how_to"),
        ("bug", "problem"),
        ("architecture", "structure"),
        ("reference", "reference_context"),
        ("code_context", "reference_context"),
        ("setup_instruction", "how_to"),
    ])
    def test_known_type_mapping(self, raw, expected):
        assert resolve_canonical_type(raw) == expected

    def test_unknown_type_returns_unclassified(self):
        assert resolve_canonical_type("random_made_up_type") == "unclassified"

    def test_empty_string_returns_unclassified(self):
        assert resolve_canonical_type("") == "unclassified"

    def test_case_insensitive(self):
        assert resolve_canonical_type("DECISION") == "decision"
        assert resolve_canonical_type("Decision") == "decision"

    def test_spaces_normalized(self):
        assert resolve_canonical_type("open question") == "open_question"

    def test_hyphens_normalized(self):
        assert resolve_canonical_type("open-question") == "open_question"

    def test_all_canonical_map_values_are_valid(self):
        for v in CANONICAL_TYPE_MAP.values():
            assert v in VALID_CANONICAL_TYPES, f"'{v}' is not a valid canonical type"


# ---------------------------------------------------------------------------
# ExtractionConfig defaults
# ---------------------------------------------------------------------------

class TestExtractionConfig:
    def test_defaults(self):
        cfg = ExtractionConfig()
        assert cfg.backend == "rule_based"
        assert cfg.model is None
        assert cfg.noise_gate_threshold == 0.15
        assert cfg.chunk_size_tokens == 2000
        assert cfg.temperature == 0.0
        assert cfg.fallback_to_rule_based is True
        assert cfg.include_project_context is True

    def test_override(self):
        cfg = ExtractionConfig(backend="cloud_llm", model="claude-sonnet-4-6", temperature=0.2)
        assert cfg.backend == "cloud_llm"
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.temperature == 0.2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class TestExtractedMemoryDraft:
    def test_required_fields(self):
        draft = ExtractedMemoryDraft(
            canonical_type="decision",
            type_label="Decision",
            title="Use Redis",
            content="We will use Redis for caching.",
        )
        assert draft.canonical_type == "decision"
        assert draft.importance == 3  # default
        assert draft.confidence == 0.7  # default
        assert draft.source_quote is None
        assert draft.extraction_backend == "rule_based"

    def test_source_message_ids_defaults_empty(self):
        draft = ExtractedMemoryDraft(
            canonical_type="task", type_label="Task", title="T", content="C"
        )
        assert draft.source_message_ids == []


class TestP2ExtractionResult:
    def test_required_fields(self):
        result = P2ExtractionResult(
            session_id="sess-123",
            memories_created=2,
            summary="Extracted 2 memories",
        )
        assert result.session_id == "sess-123"
        assert result.memories_created == 2
        assert result.backend_used == "rule_based"  # default
        assert result.model_used is None
        assert result.fallback_used is False
        assert result.dynamic_type_labels == []

    def test_all_fields(self):
        result = P2ExtractionResult(
            session_id="sess-456",
            memories_created=5,
            summary="Done",
            memory_ids=["a", "b", "c"],
            backend_used="cloud_llm",
            model_used="claude-haiku-4-5-20251001",
            chunks_processed=3,
            noise_filtered_count=10,
            llm_api_calls=3,
            fallback_used=False,
            skipped_duplicates=2,
            duplicate_titles=["Old Decision"],
            low_confidence_queued=1,
            dynamic_type_labels=["User Preference"],
        )
        assert result.chunks_processed == 3
        assert result.noise_filtered_count == 10
        assert result.dynamic_type_labels == ["User Preference"]
