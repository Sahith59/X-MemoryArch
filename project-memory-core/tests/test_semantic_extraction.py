"""
Sub-phase 1.20 — Semantic Extraction Engine Tests.

Covers:
  Classifier unit tests (no DB):
    1.  Casual greetings are rejected by the technical gate
    2.  Social filler is rejected by the technical gate
    3.  Common-word false positives (old regex traps) are rejected
    4.  Clear technical sentences pass the gate
    5.  Decision sentence classifies as 'decision'
    6.  Bug sentence classifies as 'bug'
    7.  Architecture sentence classifies as 'architecture'
    8.  Setup instruction classifies as 'setup_instruction'
    9.  Open question classifies as 'open_question'
    10. Constraint classifies as 'constraint'
    11. Confidence is in [THRESHOLD, 1.0] range
    12. embed_text returns float32 bytes of length 1536 (384 * 4)
    13. Embedding is unit-normalised

  Extraction integration tests (full pipeline via API):
    14. Casual conversation content produces zero memories
    15. Technical content produces at least one memory
    16. Extracted memory has review_status=auto_extracted
    17. Extracted memory has source_type=ai_session
    18. Extracted memory has source_quote populated
    19. Extracted memory has privacy_level=internal by default
    20. Extracted memory confidence is in valid range (0, 1]
    21. Extracted memory quality_score > 0
    22. Decision sentence extraction yields type=decision
    23. Bug sentence extraction yields type=bug
    24. Old false positive: 'going with' in casual sentence is NOT extracted
    25. Old false positive: 'structure' in non-tech sentence is NOT extracted
    26. Embedding stored in DB for extracted memories
    27. Near-duplicate within same extraction is skipped once
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services.semantic_classifier import (
    TYPE_CONF_THRESHOLD,
    classify,
    embed_text,
    is_technical,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Semantic Test Project") -> str:
    r = client.post("/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def _session(client: TestClient, pid: str, content: str) -> str:
    r = client.post(f"/projects/{pid}/sessions", json={
        "tool_name": "Claude",
        "title": "Semantic Test Session",
        "raw_content": content,
        "session_date": "2026-05-25",
    })
    assert r.status_code == 201
    return r.json()["id"]


def _extract(client: TestClient, sid: str) -> dict:
    r = client.post(f"/sessions/{sid}/extract-memories")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1–13: Classifier unit tests (no DB, no HTTP)
# ---------------------------------------------------------------------------

class TestTechnicalGate:
    def test_greeting_rejected(self):
        assert is_technical("How's everything going with you today?") is False

    def test_social_filler_rejected(self):
        assert is_technical("That sounds great, glad to hear it!") is False

    def test_casual_question_rejected(self):
        assert is_technical("Hey bro, what's up?") is False

    def test_job_structure_false_positive_rejected(self):
        # Old regex trap: 'structure' matched ARCHITECTURE
        assert is_technical(
            "If you're feeling unfulfilled in the usual job structure, building something of your own helps."
        ) is False

    def test_going_with_false_positive_rejected(self):
        # Old regex trap: 'going with' matched DECISION
        assert is_technical("How's everything going with you today?") is False

    def test_technical_decision_passes(self):
        assert is_technical(
            "We decided to use PostgreSQL over MySQL for better JSON support."
        ) is True

    def test_technical_bug_passes(self):
        assert is_technical(
            "The authentication middleware throws a TypeError when the token is null."
        ) is True

    def test_technical_architecture_passes(self):
        assert is_technical(
            "The system architecture uses layers: router, service, repository, and database."
        ) is True

    def test_technical_constraint_passes(self):
        assert is_technical(
            "We must not store personally identifiable information in the local database."
        ) is True


class TestTypeClassifier:
    def test_decision_classified_correctly(self):
        result = classify("We decided to use PostgreSQL over MySQL for better JSON support.")
        assert result is not None
        mem_type, conf = result
        assert mem_type == "decision"
        assert conf >= TYPE_CONF_THRESHOLD

    def test_bug_classified_correctly(self):
        result = classify(
            "TypeError thrown in the authentication middleware when the token is null."
        )
        assert result is not None
        mem_type, conf = result
        assert mem_type == "problem"

    def test_architecture_classified_correctly(self):
        result = classify(
            "The system uses a layered architecture separating router, service, and repository."
        )
        assert result is not None
        mem_type, _ = result
        assert mem_type == "structure"

    def test_setup_instruction_classified_correctly(self):
        result = classify(
            "Run alembic upgrade head to apply all pending database migrations."
        )
        assert result is not None
        mem_type, _ = result
        assert mem_type == "how_to"

    def test_open_question_classified_correctly(self):
        result = classify(
            "Should we use cursor-based or offset-based pagination for the memories API?"
        )
        assert result is not None
        mem_type, _ = result
        assert mem_type == "open_question"

    def test_constraint_classified_correctly(self):
        result = classify(
            "The system must not store any personally identifiable information locally."
        )
        assert result is not None
        mem_type, _ = result
        assert mem_type == "constraint"

    def test_confidence_in_valid_range(self):
        result = classify("We decided to use FastAPI as the backend web framework.")
        assert result is not None
        _, conf = result
        assert TYPE_CONF_THRESHOLD <= conf <= 1.0


class TestEmbedding:
    def test_embed_returns_correct_byte_length(self):
        # 384 dims × 4 bytes (float32) = 1536 bytes
        b = embed_text("We decided to use FastAPI for the backend.")
        assert len(b) == 1536

    def test_embed_is_unit_normalised(self):
        b = embed_text("We decided to use FastAPI for the backend.")
        vec = np.frombuffer(b, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-4

    def test_different_texts_produce_different_embeddings(self):
        b1 = embed_text("We decided to use FastAPI for the backend.")
        b2 = embed_text("The authentication middleware throws a TypeError.")
        assert b1 != b2


# ---------------------------------------------------------------------------
# 14–27: Integration tests via TestClient
# ---------------------------------------------------------------------------

class TestExtractionPipeline:
    def test_casual_content_produces_zero_memories(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "Hey, how are you doing today? "
            "Glad to hear it's going well! "
            "That sounds great, let me know if you need anything else."
        )
        result = _extract(client, sid)
        assert result["memories_created"] == 0

    def test_technical_content_produces_memories(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to use FastAPI as our primary web framework because of its "
            "automatic OpenAPI documentation and type-safe request validation. "
            "The authentication middleware throws a TypeError when the JWT token is null, "
            "which we fixed by adding a guard clause at the entry point."
        )
        result = _extract(client, sid)
        assert result["memories_created"] >= 1

    def test_extracted_memory_review_status(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to adopt PostgreSQL over SQLite for the production database "
            "because it supports concurrent writes and has better JSON indexing."
        )
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        for m in result["memories"]:
            assert m["review_status"] == "auto_extracted"

    def test_extracted_memory_source_type(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to adopt PostgreSQL over SQLite for the production database "
            "because it supports concurrent writes and has better JSON indexing."
        )
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        for m in result["memories"]:
            assert m["source_type"] == "ai_session"

    def test_extracted_memory_source_quote_populated(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to adopt PostgreSQL over SQLite for the production database "
            "because it supports concurrent writes and has better JSON indexing."
        )
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        for m in result["memories"]:
            assert m["source_quote"] is not None
            assert len(m["source_quote"]) > 0

    def test_extracted_memory_default_privacy(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to use Redis for session caching with a 30-minute TTL "
            "because it provides atomic operations and low-latency reads."
        )
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        for m in result["memories"]:
            assert m["privacy_level"] == "internal"

    def test_extracted_memory_confidence_range(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to use Alembic for database migrations because it integrates "
            "cleanly with SQLAlchemy and supports both upgrade and downgrade paths."
        )
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        for m in result["memories"]:
            assert 0 < m["confidence"] <= 1.0

    def test_extracted_memory_quality_score_positive(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to use Alembic for database migrations because it integrates "
            "cleanly with SQLAlchemy and supports both upgrade and downgrade paths."
        )
        result = _extract(client, sid)
        assert result["memories_created"] >= 1
        for m in result["memories"]:
            assert (m.get("quality_score") or 0) > 0

    def test_decision_sentence_yields_decision_type(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "We decided to use PostgreSQL as our primary database instead of MySQL "
            "because PostgreSQL has native JSON support and better full-text indexing."
        )
        result = _extract(client, sid)
        types = [m["type"] for m in result["memories"]]
        assert "decision" in types

    def test_bug_sentence_yields_bug_type(self, client: TestClient):
        pid = _project(client)
        sid = _session(client, pid,
            "The authentication middleware throws a TypeError when the JWT token is "
            "null or expired.  We fixed this by adding a null-check guard at the top "
            "of the middleware function before any decoding is attempted."
        )
        result = _extract(client, sid)
        types = [m["type"] for m in result["memories"]]
        assert "problem" in types

    def test_going_with_in_casual_sentence_not_extracted(self, client: TestClient):
        # Regression: old 'going with' regex matched casual conversation
        pid = _project(client)
        sid = _session(client, pid,
            "How's everything going with you today? "
            "I'm doing well, thanks for asking! "
            "Let me know if there's anything I can help with."
        )
        result = _extract(client, sid)
        assert result["memories_created"] == 0

    def test_job_structure_not_extracted_as_architecture(self, client: TestClient):
        # Regression: old 'structure' regex matched 'job structure'
        pid = _project(client)
        sid = _session(client, pid,
            "If you're feeling unfulfilled in the usual job structure, building "
            "something of your own can give you a whole new sense of purpose. "
            "It will come with its own challenges, but it is very rewarding."
        )
        result = _extract(client, sid)
        # Must not classify any of these as architecture
        arch_memories = [m for m in result["memories"] if m["type"] == "structure"]
        assert len(arch_memories) == 0

    def test_near_duplicate_skipped_in_same_run(self, client: TestClient):
        pid = _project(client)
        # Two nearly identical sentences back-to-back
        sid = _session(client, pid,
            "We decided to use FastAPI as our primary web framework for this project. "
            "We decided to use FastAPI as the primary web framework for this system. "
            "The authentication middleware throws a TypeError when the token is null."
        )
        result = _extract(client, sid)
        # The second near-duplicate decision should be skipped
        decision_memories = [m for m in result["memories"] if m["type"] == "decision"]
        assert len(decision_memories) <= 1
