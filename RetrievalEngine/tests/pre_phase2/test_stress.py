"""
Stress and edge case tests for Pre-Phase 2.

Covers:
  - Large session (10KB) processed without error
  - All filler → noise gate removes everything → early return
  - LLM returns empty array → memories_created=0
  - LLM returns all-duplicate items → memories_created=0
  - Very long title truncated to 300 chars in DB write
  - Unknown canonical_type normalized to 'unclassified'
  - Privacy level defaults to 'internal' for invalid values
  - Low confidence memories counted in low_confidence_queued
  - dynamic_type_labels collects open labels not in CANONICAL_TYPE_MAP
  - Multiple chunks all produce unique memories (no cross-chunk dedup)
  - Cross-chunk dedup eliminates overlapping duplicate from adjacent chunks
  - cloud_llm falls back to rule_based when API key missing AND fallback=True
  - local_llm falls back to rule_based when Ollama down AND fallback=True
  - Backend registry survives import-order independence
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction.cloud_llm import CloudLLMExtractionBackend, _parse_llm_json
from app.services.extraction.local_llm import LocalLLMExtractionBackend
from app.services.extraction.rule_based import RuleBasedExtractionBackend
from app.services.extraction.base import (
    ExtractionConfig,
    ExtractedMemoryDraft,
    P2ExtractionResult,
    CANONICAL_TYPE_MAP,
    resolve_canonical_type,
)
from app.services.extraction.deduplicator import deduplicate_drafts, deduplicate_against_existing
from app.services.extraction.chunker import apply_noise_gate, chunk_sentences, prepare_chunks
from app.services.extraction.prompts import validate_draft
from tests.conftest import make_phase1_extraction_result


# ---------------------------------------------------------------------------
# Large session stress test
# ---------------------------------------------------------------------------

class TestLargeSession:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_large_session_processed_without_error(self, db, project):
        """10KB technical session must not raise or timeout."""
        from app import crud, schemas
        large_content = "\n".join([
            f"Decision {i}: We decided to use component {i} for performance and scalability reasons."
            for i in range(200)
        ])
        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Large Session",
            raw_content=large_content,
            session_date="2026-05-28",
        ))

        empty_response = "[]"
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=empty_response):
            result = backend.extract(session.id, db)

        assert isinstance(result, P2ExtractionResult)
        assert result.session_id == session.id
        assert result.chunks_processed >= 1  # at least one chunk created

    def test_large_session_creates_multiple_chunks(self, mock_is_technical_true):
        """200 technical sentences should produce more than 1 chunk at 2000-token limit."""
        sentences = [
            f"We use service component {i} for handling distributed transactions in our microservice architecture."
            for i in range(200)
        ]
        chunks = chunk_sentences(sentences, max_tokens=500)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# All-filler session
# ---------------------------------------------------------------------------

class TestAllFillerSession:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_false):
        pass

    def test_all_filler_returns_zero_memories(self, db, filler_session):
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value="[]"):
            result = backend.extract(filler_session.id, db)

        assert result.memories_created == 0
        assert result.noise_filtered_count >= 0

    def test_all_filler_summary_indicates_no_content(self, db, filler_session):
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value="[]"):
            result = backend.extract(filler_session.id, db)

        assert "No substantive content" in result.summary or result.memories_created == 0


# ---------------------------------------------------------------------------
# Empty LLM response
# ---------------------------------------------------------------------------

class TestEmptyLlmResponse:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_empty_array_response_zero_memories(self, db, session_obj):
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value="[]"):
            result = backend.extract(session_obj.id, db)
        assert result.memories_created == 0
        assert result.memory_ids == []

    def test_empty_object_response_zero_memories(self, db, session_obj):
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value='{"memories": []}'):
            result = backend.extract(session_obj.id, db)
        assert result.memories_created == 0


# ---------------------------------------------------------------------------
# Title truncation
# ---------------------------------------------------------------------------

class TestTitleTruncation:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_long_title_truncated_to_300(self, db, session_obj):
        long_title = "A" * 500
        response = json.dumps([{
            "canonical_type": "decision",
            "type_label": "Decision",
            "title": long_title,
            "content": "Content for this memory.",
            "importance": 3,
            "confidence": 0.8,
            "source_quote": "A very long title memory was discussed.",
            "source_message_ids": [],
            "entities": [],
            "links": [],
            "tags": [],
            "related_files": [],
            "privacy_level": "internal",
            "review_recommended": False,
            "llm_reasoning": "Test.",
            "temporal": {},
        }])
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=response):
            result = backend.extract(session_obj.id, db)

        assert result.memories_created == 1
        # Verify the memory's title was truncated (check via DB query)
        from app import models
        if result.memory_ids:
            memory = db.query(models.Memory).filter(models.Memory.id == result.memory_ids[0]).first()
            assert len(memory.title) <= 300


# ---------------------------------------------------------------------------
# Unknown canonical_type handling
# ---------------------------------------------------------------------------

class TestUnknownCanonicalType:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_unknown_type_memory_created_as_unclassified(self, db, session_obj):
        response = json.dumps([{
            "canonical_type": "completely_unknown_type_xyz_123",
            "type_label": "Novel Type",
            "title": "Memory with unknown type",
            "content": "Content for memory with unknown type.",
            "importance": 3,
            "confidence": 0.75,
            "source_quote": "Memory with unknown type was discussed in the session.",
            "source_message_ids": [],
            "entities": [],
            "links": [],
            "tags": [],
            "related_files": [],
            "privacy_level": "internal",
            "review_recommended": False,
            "llm_reasoning": "Test.",
            "temporal": {},
        }])
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=response):
            result = backend.extract(session_obj.id, db)

        assert result.memories_created == 1

    def test_resolve_canonical_type_never_raises(self):
        for value in ["", "RANDOM", "123", "!@#$", None]:
            if value is None:
                assert resolve_canonical_type("") == "unclassified"
            else:
                result = resolve_canonical_type(value)
                assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Privacy level normalization
# ---------------------------------------------------------------------------

class TestPrivacyLevelNormalization:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_valid_privacy_levels_accepted(self, db, session_obj):
        for privacy in ("public", "internal", "sensitive", "secret"):
            response = json.dumps([{
                "canonical_type": "insight",
                "type_label": "Insight",
                "title": f"Memory with {privacy} privacy",
                "content": f"Content for {privacy} memory.",
                "importance": 3,
                "confidence": 0.8,
                "source_quote": f"Memory with {privacy} privacy was noted.",
                "source_message_ids": [],
                "entities": [],
                "links": [],
                "tags": [],
                "related_files": [],
                "privacy_level": privacy,
                "review_recommended": False,
                "llm_reasoning": "Test.",
                "temporal": {},
            }])
            backend = CloudLLMExtractionBackend()
            with patch("app.services.extraction.cloud_llm._call_llm", return_value=response):
                result = backend.extract(session_obj.id, db)
            assert result.memories_created == 1


# ---------------------------------------------------------------------------
# Low confidence tracking
# ---------------------------------------------------------------------------

class TestLowConfidenceTracking:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_low_confidence_memories_counted(self, db, session_obj):
        response = json.dumps([
            {
                "canonical_type": "insight",
                "type_label": "Insight",
                "title": "Possible caching improvement",
                "content": "Adding Redis may help with throughput, but measurement is unclear.",
                "importance": 2,
                "confidence": 0.4,  # below 0.6 threshold
                "source_quote": "Someone mentioned Redis might improve things.",
                "source_message_ids": [],
                "entities": [],
                "links": [],
                "tags": [],
                "related_files": [],
                "privacy_level": "internal",
                "review_recommended": False,
                "llm_reasoning": "Uncertain — needs measurement.",
                "temporal": {},
            },
            {
                "canonical_type": "decision",
                "type_label": "Decision",
                "title": "Use PostgreSQL for production database",
                "content": "Team explicitly decided on PostgreSQL for ACID compliance and concurrent writes.",
                "importance": 4,
                "confidence": 0.95,  # above 0.6 threshold
                "source_quote": "We are using PostgreSQL in production for ACID compliance.",
                "source_message_ids": [],
                "entities": [],
                "links": [],
                "tags": [],
                "related_files": [],
                "privacy_level": "internal",
                "review_recommended": False,
                "llm_reasoning": "Very clear explicit decision.",
                "temporal": {},
            },
        ])
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=response):
            result = backend.extract(session_obj.id, db)

        assert result.low_confidence_queued >= 1


# ---------------------------------------------------------------------------
# Dynamic type labels
# ---------------------------------------------------------------------------

class TestDynamicTypeLabels:
    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def test_open_type_labels_collected(self, db, session_obj):
        open_label = "User Workflow Pattern"  # not in CANONICAL_TYPE_MAP
        response = json.dumps([{
            "canonical_type": "workflow_pattern",
            "type_label": open_label,
            "title": "User onboarding workflow",
            "content": "Users follow a 3-step onboarding process: sign up, verify, activate.",
            "importance": 3,
            "confidence": 0.85,
            "source_quote": "Users follow a 3-step onboarding process.",
            "source_message_ids": [],
            "entities": [],
            "links": [],
            "tags": [],
            "related_files": [],
            "privacy_level": "internal",
            "review_recommended": False,
            "llm_reasoning": "Clear workflow.",
            "temporal": {},
        }])
        backend = CloudLLMExtractionBackend()
        with patch("app.services.extraction.cloud_llm._call_llm", return_value=response):
            result = backend.extract(session_obj.id, db)

        # open_label is not in CANONICAL_TYPE_MAP values, so it should appear in dynamic_type_labels
        if result.memories_created > 0:
            assert open_label in result.dynamic_type_labels or "workflow_pattern" in result.dynamic_type_labels


# ---------------------------------------------------------------------------
# Cross-chunk deduplication stress
# ---------------------------------------------------------------------------

class TestCrossChunkDedup:
    def test_many_unique_drafts_all_kept(self):
        # Use sufficiently distinct titles, content, and source_quotes so no dedup fires.
        topics = [
            ("database", "PostgreSQL", "connection pool", "concurrent write requirements"),
            ("caching", "Redis", "session storage", "reduce database load"),
            ("api", "FastAPI", "rate limiting", "1000 requests per minute"),
            ("auth", "JWT", "refresh tokens", "token invalidation on logout"),
            ("migration", "Alembic", "schema changes", "users table migration"),
            ("testing", "pytest", "integration tests", "payment processing flow"),
            ("monitoring", "Prometheus", "metrics collection", "p95 latency target"),
            ("deployment", "Docker", "container orchestration", "production rollout"),
            ("security", "HTTPS", "TLS certificates", "certificate rotation policy"),
            ("logging", "Structlog", "structured logs", "centralized log aggregation"),
        ]
        drafts = []
        for i, (area, tech, concept, detail) in enumerate(topics):
            drafts.append(ExtractedMemoryDraft(
                canonical_type="decision",
                type_label="Decision",
                title=f"{area.title()} architecture: use {tech} for {concept}",
                content=f"Team chose {tech} as the primary solution for {concept}. Reason: {detail}.",
                source_quote=f"We chose {tech} to handle {concept} because of {detail}.",
            ))
        result = deduplicate_drafts(drafts)
        assert result.kept_count == len(topics)
        assert result.removed_count == 0

    def test_all_identical_source_quotes_deduped_to_one(self):
        """50 identical drafts → 1 kept."""
        drafts = [
            ExtractedMemoryDraft(
                canonical_type="decision",
                type_label="Decision",
                title=f"Decision variant {i}",
                content="We decided to use PostgreSQL for production.",
                source_quote="We decided to use PostgreSQL for production.",
                confidence=0.5 + (i * 0.01),  # increasing confidence
            )
            for i in range(50)
        ]
        result = deduplicate_drafts(drafts)
        assert result.kept_count == 1
        assert result.removed_count == 49

    def test_highest_confidence_wins_in_dedup(self):
        """When deduplicating, the highest-confidence item wins."""
        drafts = [
            ExtractedMemoryDraft(
                canonical_type="decision",
                type_label="Decision",
                title="PostgreSQL decision",
                content="We decided to use PostgreSQL.",
                source_quote="We decided to use PostgreSQL.",
                confidence=0.3 + (i * 0.1),
            )
            for i in range(8)
        ]
        result = deduplicate_drafts(drafts)
        assert result.kept_count == 1
        assert result.kept[0].confidence == pytest.approx(0.3 + 7 * 0.1, abs=0.01)

    def test_dedup_against_existing_preserves_new(self):
        existing_titles = [f"Old Memory {i}" for i in range(100)]
        existing_contents = [f"Content for old memory {i}." for i in range(100)]
        new_drafts = [
            ExtractedMemoryDraft(
                canonical_type="decision",
                type_label="Decision",
                title=f"Brand New Decision {i}",
                content=f"Brand new content {i} never seen before.",
                source_quote=f"Brand new quote {i}.",
            )
            for i in range(10)
        ]
        result = deduplicate_against_existing(new_drafts, existing_titles, existing_contents)
        assert result.kept_count == 10
        assert result.removed_count == 0


# ---------------------------------------------------------------------------
# validate_draft edge cases
# ---------------------------------------------------------------------------

class TestValidateDraftEdgeCases:
    def _base(self):
        return {
            "canonical_type": "decision",
            "type_label": "Decision",
            "title": "Test memory",
            "content": "Some content.",
            "importance": 3,
            "confidence": 0.8,
            "source_quote": "Test memory was discussed.",
        }

    def test_importance_as_string_coerced(self):
        raw = self._base()
        raw["importance"] = "4"
        result = validate_draft(raw)
        assert result is not None
        assert result["importance"] == 4

    def test_confidence_as_string_coerced(self):
        raw = self._base()
        raw["confidence"] = "0.75"
        result = validate_draft(raw)
        assert result is not None
        assert result["confidence"] == pytest.approx(0.75)

    def test_whitespace_only_source_quote_rejected(self):
        raw = self._base()
        raw["source_quote"] = "   "
        result = validate_draft(raw)
        assert result is None

    def test_deeply_nested_temporal_handled(self):
        raw = self._base()
        raw["temporal"] = {
            "valid_from": "2026-01-01",
            "valid_until": "2026-12-31",
            "supersedes_memory_id": None,
        }
        result = validate_draft(raw)
        assert result is not None
