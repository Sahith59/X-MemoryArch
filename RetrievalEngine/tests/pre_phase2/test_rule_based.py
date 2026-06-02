"""
Tests for RuleBasedExtractionBackend.

Covers:
  - is_available() always returns True
  - extract() delegates to Phase 1 extract_memories_from_session()
  - Returns P2ExtractionResult with correct backend metadata
  - Phase 2 column stamping (hasattr guards work silently)
  - Empty extraction (no memories created)
  - Skipped duplicates and low_confidence_queued propagated correctly
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction.rule_based import RuleBasedExtractionBackend
from app.services.extraction.base import ExtractionConfig, P2ExtractionResult
from tests.conftest import make_phase1_extraction_result


class TestRuleBasedAvailability:
    def test_is_always_available(self):
        backend = RuleBasedExtractionBackend()
        assert backend.is_available() is True

    def test_name_is_rule_based(self):
        backend = RuleBasedExtractionBackend()
        assert backend.name == "rule_based"


class TestRuleBasedExtract:
    def test_extract_returns_p2_result(self, db, session_obj):
        backend = RuleBasedExtractionBackend()

        mock_result = make_phase1_extraction_result(session_obj.id)
        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)

        assert isinstance(result, P2ExtractionResult)

    def test_backend_used_is_rule_based(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)
        assert result.backend_used == "rule_based"

    def test_model_used_is_none(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)
        assert result.model_used is None

    def test_llm_api_calls_is_zero(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)
        assert result.llm_api_calls == 0

    def test_chunks_processed_is_zero(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)
        assert result.chunks_processed == 0

    def test_fallback_used_is_false(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)
        assert result.fallback_used is False

    def test_session_id_propagated(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)
        assert result.session_id == session_obj.id

    def test_memories_created_propagated(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mem1 = MagicMock()
        mem1.id = "mem-id-1"
        mock_result = make_phase1_extraction_result(session_obj.id, memories=[mem1])
        mock_result.memories_created = 1

        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)

        assert result.memories_created == 1

    def test_duplicate_info_propagated(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        mock_result.skipped_duplicates = 3
        mock_result.duplicate_titles = ["Old Memory A", "Old Memory B", "Old Memory C"]

        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)

        assert result.skipped_duplicates == 3
        assert "Old Memory A" in result.duplicate_titles

    def test_low_confidence_queued_propagated(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        mock_result.low_confidence_queued = 5

        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)

        assert result.low_confidence_queued == 5

    def test_dynamic_type_labels_empty(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)
        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)
        assert result.dynamic_type_labels == []

    def test_zero_memories_returns_valid_result(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id, memories=[])
        mock_result.memories_created = 0

        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            result = backend.extract(session_obj.id, db)

        assert result.memories_created == 0
        assert result.memory_ids == []

    def test_extract_passes_session_id_and_db(self, db, session_obj):
        backend = RuleBasedExtractionBackend()
        mock_result = make_phase1_extraction_result(session_obj.id)

        with patch("app.services.memory_service.extract_memories_from_session", return_value=mock_result) as mock_fn:
            backend.extract(session_obj.id, db)
            mock_fn.assert_called_once_with(session_id=session_obj.id, db=db)

    def test_phase2_stamp_does_not_crash_without_columns(self, db, session_obj):
        """Phase 2 columns not on model — hasattr guards must prevent AttributeError."""
        backend = RuleBasedExtractionBackend()
        mem = MagicMock()
        mem.id = "mem-stamp-1"
        mem.type = "decision"
        # Simulate model WITHOUT Phase 2 columns
        del mem.extraction_backend
        del mem.canonical_type
        del mem.type_label
        del mem.llm_reasoning
        del mem.contextual_prefix

        mock_result = make_phase1_extraction_result(session_obj.id, memories=[mem])
        mock_result.memories_created = 1

        from app import models
        from unittest.mock import patch as upatch

        with upatch("app.services.memory_service.extract_memories_from_session", return_value=mock_result):
            with upatch.object(db, "query") as mock_query:
                mock_query.return_value.filter.return_value.all.return_value = []
                result = backend.extract(session_obj.id, db)

        assert result.memories_created == 1
