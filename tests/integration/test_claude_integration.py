"""
Real integration tests for CloudLLMExtractionBackend.

Requires:
  - ANTHROPIC_API_KEY in RetrievalEngine/.env
  - pip install python-dotenv

Run with:
  cd RetrievalEngine
  pytest tests/integration/test_claude_integration.py -v -s

Skipped automatically if ANTHROPIC_API_KEY is not set.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Load .env from RetrievalEngine root
_RE_ROOT = Path(__file__).resolve().parents[2]
_env_file = _RE_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

pytestmark = pytest.mark.integration


# Skip entire module if no Anthropic key
def pytest_configure(config):
    pass


ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
skip_no_key = pytest.mark.skipif(
    not ANTHROPIC_KEY or ANTHROPIC_KEY.startswith("sk-ant-..."),
    reason="ANTHROPIC_API_KEY not set in .env",
)

# Technical session content that should yield extractable memories
TECHNICAL_SESSION = """
We spent the session deciding on our database architecture.
After evaluating SQLite, PostgreSQL, and MongoDB, we decided to use PostgreSQL for production.
The main reasons were ACID compliance, support for concurrent writes, and the team's existing expertise.

We also agreed that all database migrations will be managed with Alembic.
Migration scripts must be reviewed by at least two engineers before merging to main.

For caching, we chose Redis with a 60-second TTL for session data.
The connection pool size is set to 20 to handle peak load.

TODO: Set up pgBouncer for connection pooling before the Q3 release.
TODO: Write load tests for the payment endpoint — targeting p95 < 200ms.

The API rate limit is 1000 requests per minute per user.
Requests exceeding this threshold will receive a 429 response.
""".strip()


@skip_no_key
class TestClaudeIntegration:
    """Live Claude API calls — validates end-to-end extraction pipeline."""

    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        """Keep embedding/entity extraction mocked — only LLM call is real."""
        pass

    def test_extract_returns_p2_result(self, db, project):
        """Real Claude call must return a valid P2ExtractionResult."""
        from app import crud, schemas
        from app.services.extraction.cloud_llm import CloudLLMExtractionBackend
        from app.services.extraction.base import ExtractionConfig, P2ExtractionResult

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Architecture Decision Session",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = CloudLLMExtractionBackend()
        assert backend.is_available(), "Anthropic key set but is_available() is False"

        cfg = ExtractionConfig(
            backend="cloud_llm",
            model="claude-haiku-4-5-20251001",  # cheapest/fastest
        )
        result = backend.extract(session.id, db, cfg)

        assert isinstance(result, P2ExtractionResult)
        assert result.session_id == session.id
        assert result.backend_used == "cloud_llm"
        assert result.model_used == "claude-haiku-4-5-20251001"
        assert result.llm_api_calls >= 1

    def test_memories_extracted_from_technical_content(self, db, project):
        """Claude must extract at least 1 memory from clearly technical content."""
        from app import crud, schemas
        from app.services.extraction.cloud_llm import CloudLLMExtractionBackend
        from app.services.extraction.base import ExtractionConfig

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Real Extraction Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = CloudLLMExtractionBackend()
        cfg = ExtractionConfig(backend="cloud_llm", model="claude-haiku-4-5-20251001")
        result = backend.extract(session.id, db, cfg)

        assert result.memories_created >= 1, (
            f"Expected ≥1 memory from technical content, got {result.memories_created}. "
            f"Check that Claude returned valid JSON with source_quote fields."
        )

    def test_memory_ids_written_to_db(self, db, project):
        """Extracted memories must be persisted in the DB."""
        from app import crud, schemas, models
        from app.services.extraction.cloud_llm import CloudLLMExtractionBackend
        from app.services.extraction.base import ExtractionConfig

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="DB Write Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = CloudLLMExtractionBackend()
        cfg = ExtractionConfig(backend="cloud_llm", model="claude-haiku-4-5-20251001")
        result = backend.extract(session.id, db, cfg)

        if result.memories_created > 0:
            for mid in result.memory_ids:
                memory = db.query(models.Memory).filter(models.Memory.id == mid).first()
                assert memory is not None, f"Memory {mid} not found in DB"
                assert memory.title
                assert memory.content
                assert len(memory.title) <= 300

    def test_chunks_processed_reported(self, db, project):
        """chunks_processed must be ≥ 1 for non-trivial content."""
        from app import crud, schemas
        from app.services.extraction.cloud_llm import CloudLLMExtractionBackend
        from app.services.extraction.base import ExtractionConfig

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Chunk Count Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = CloudLLMExtractionBackend()
        cfg = ExtractionConfig(backend="cloud_llm", model="claude-haiku-4-5-20251001")
        result = backend.extract(session.id, db, cfg)

        assert result.chunks_processed >= 1

    def test_filler_session_yields_few_memories(self, db, filler_session):
        """Filler-only content filtered by noise gate — Claude not even called or returns []."""
        from app.services.extraction.cloud_llm import CloudLLMExtractionBackend
        from app.services.extraction.base import ExtractionConfig

        backend = CloudLLMExtractionBackend()
        cfg = ExtractionConfig(backend="cloud_llm", model="claude-haiku-4-5-20251001")
        result = backend.extract(filler_session.id, db, cfg)

        # Noise gate should remove everything — 0 memories expected
        assert result.memories_created == 0 or result.noise_filtered_count > 0

    def test_result_has_no_fallback(self, db, project):
        """Successful real extraction must not flag fallback."""
        from app import crud, schemas
        from app.services.extraction.cloud_llm import CloudLLMExtractionBackend
        from app.services.extraction.base import ExtractionConfig

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="No Fallback Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = CloudLLMExtractionBackend()
        cfg = ExtractionConfig(backend="cloud_llm", model="claude-haiku-4-5-20251001")
        result = backend.extract(session.id, db, cfg)

        assert result.fallback_used is False
