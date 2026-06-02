"""
Real integration tests for LocalLLMExtractionBackend (Ollama).

Requires:
  - Ollama running on localhost:11434 (or OLLAMA_BASE_URL in .env)
  - At least one model pulled (defaults to llama3.1:latest)
  - OLLAMA_MODEL in .env to override the default model

Run with:
  cd RetrievalEngine
  pytest tests/integration/test_ollama_integration.py -v -s

Skipped automatically if Ollama is not reachable.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Load .env
_RE_ROOT = Path(__file__).resolve().parents[2]
_env_file = _RE_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

pytestmark = pytest.mark.integration

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")


def _ollama_reachable() -> bool:
    try:
        import httpx
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


skip_no_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama not reachable at {OLLAMA_BASE_URL}",
)

TECHNICAL_SESSION = """
We finalized the deployment strategy in today's session.
The team decided to containerize all services using Docker.
Each service will have its own Dockerfile and docker-compose entry.

We are using Kubernetes for orchestration in production.
The staging environment will use docker-compose only.

Security: all container images must be scanned with Trivy before deployment.
Base images must be pinned to specific digest hashes, not just tags.

TODO: Write a CI pipeline step that runs Trivy on every PR.
TODO: Document the rollback procedure for production deployments.

The deployment target is AWS EKS in us-east-1.
Auto-scaling is configured for 2 to 10 replicas based on CPU utilization.
""".strip()


@skip_no_ollama
class TestOllamaIntegration:
    """Live Ollama calls — validates LocalLLMExtractionBackend end-to-end."""

    @pytest.fixture(autouse=True)
    def _patch_ml(self, mock_embed, mock_entities, mock_is_technical_true):
        pass

    def _cfg(self):
        from app.services.extraction.base import ExtractionConfig
        return ExtractionConfig(backend="local_llm", model=OLLAMA_MODEL)

    def test_extract_returns_p2_result(self, db, project):
        """Real Ollama call must return a valid P2ExtractionResult."""
        from app import crud, schemas
        from app.services.extraction.local_llm import LocalLLMExtractionBackend
        from app.services.extraction.base import P2ExtractionResult

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Ollama Extraction Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = LocalLLMExtractionBackend()
        result = backend.extract(session.id, db, self._cfg())

        assert isinstance(result, P2ExtractionResult)
        assert result.session_id == session.id
        assert result.backend_used == "local_llm"
        assert result.model_used == OLLAMA_MODEL

    def test_ollama_extracts_some_memories(self, db, project):
        """Ollama must extract ≥1 memory from clearly technical content."""
        from app import crud, schemas
        from app.services.extraction.local_llm import LocalLLMExtractionBackend

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Ollama Memory Count Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = LocalLLMExtractionBackend()
        result = backend.extract(session.id, db, self._cfg())

        assert result.memories_created >= 1, (
            f"Expected ≥1 memory from technical deployment content, got {result.memories_created}. "
            f"Model: {OLLAMA_MODEL}. Ensure model returns valid JSON with source_quote field."
        )

    def test_memory_ids_in_db(self, db, project):
        """Ollama-extracted memories must be persisted in DB."""
        from app import crud, schemas, models
        from app.services.extraction.local_llm import LocalLLMExtractionBackend

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Ollama DB Write Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = LocalLLMExtractionBackend()
        result = backend.extract(session.id, db, self._cfg())

        if result.memories_created > 0:
            for mid in result.memory_ids:
                memory = db.query(models.Memory).filter(models.Memory.id == mid).first()
                assert memory is not None, f"Memory {mid} not in DB"
                assert memory.title
                assert memory.content

    def test_no_fallback_on_success(self, db, project):
        """Successful Ollama extraction must not flag fallback."""
        from app import crud, schemas
        from app.services.extraction.local_llm import LocalLLMExtractionBackend

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Ollama Fallback Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = LocalLLMExtractionBackend()
        result = backend.extract(session.id, db, self._cfg())

        assert result.fallback_used is False

    def test_llm_api_calls_counted(self, db, project):
        """llm_api_calls must be ≥1 after a real Ollama call."""
        from app import crud, schemas
        from app.services.extraction.local_llm import LocalLLMExtractionBackend

        session = crud.create_session(db, project.id, schemas.SessionCreate(
            tool_name="Claude",
            title="Ollama API Calls Test",
            raw_content=TECHNICAL_SESSION,
            session_date="2026-05-28",
        ))

        backend = LocalLLMExtractionBackend()
        result = backend.extract(session.id, db, self._cfg())

        assert result.llm_api_calls >= 1

    def test_is_available_true_when_running(self):
        """is_available() must return True when Ollama is reachable."""
        from app.services.extraction.local_llm import LocalLLMExtractionBackend
        backend = LocalLLMExtractionBackend()
        assert backend.is_available() is True
