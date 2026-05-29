"""
Shared fixtures for Phase 2 (Pre-Phase 2) extraction tests.

Namespace collision handling:
  Both Phase 1 (project-memory-core/app/) and Phase 2 (RetrievalEngine/app/)
  use `app` as the top-level package. Python can only cache one `app` in
  sys.modules. We resolve this by:

  1. Adding RetrievalEngine/ first — so `app` = Phase 2's package.
  2. Extending app.__path__ and app.services.__path__ to also include
     Phase 1's directories. This is the Python namespace package trick.
  3. After extension, `from app import crud` finds Phase 1's crud.py,
     `from app.services.extraction.base import ...` finds Phase 2's base.py.

This must happen at module level BEFORE any test file is imported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ── sys.path setup ─────────────────────────────────────────────────────────
_RE_ROOT = Path(__file__).resolve().parents[1]   # RetrievalEngine/
_P1_ROOT = Path(__file__).resolve().parents[2] / "project-memory-core"

# Phase 2 (RetrievalEngine/) must be first — `app` maps to Phase 2's package.
if str(_RE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RE_ROOT))

# ── Claim `app` for Phase 2 and extend __path__ to Phase 1 ─────────────────
import app as _p2_app  # Phase 2's app (empty __init__.py)

_p1_app_dir = str(_P1_ROOT / "app")
if _p1_app_dir not in list(_p2_app.__path__):
    _p2_app.__path__.append(_p1_app_dir)

# Extend app.services.__path__ so Phase 1 services resolve correctly.
import app.services as _p2_services  # noqa: E402
_p1_services_dir = str(_P1_ROOT / "app" / "services")
if _p1_services_dir not in list(_p2_services.__path__):
    _p2_services.__path__.append(_p1_services_dir)

# ── Phase 1 modules now accessible via extended namespace ──────────────────
import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.database import Base  # resolves to Phase 1's database.py via extended __path__
from app import crud, schemas  # resolves to Phase 1's crud/schemas
import app.p2_models  # noqa: F401 — registers Phase 2 models (RetrievalRun) with Base.metadata


# ---------------------------------------------------------------------------
# In-memory SQLite engine (StaticPool = all connections share one in-memory DB)
# ---------------------------------------------------------------------------
_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


# ---------------------------------------------------------------------------
# DB fixture — fresh schema per test function
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    Base.metadata.create_all(bind=_engine)
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=_engine)


# ---------------------------------------------------------------------------
# Domain object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project(db):
    return crud.create_project(db, schemas.ProjectCreate(
        name="Test Project",
        description="A project for Phase 2 extraction tests",
        tech_stack=["Python", "FastAPI", "SQLAlchemy"],
        goals=["Build extraction pipeline"],
        domain="software",
    ))


@pytest.fixture
def session_obj(db, project):
    """A session with substantive technical content."""
    return crud.create_session(db, project.id, schemas.SessionCreate(
        tool_name="Claude",
        title="Architecture Planning Session",
        raw_content=TECHNICAL_CONTENT,
        session_date="2026-05-28",
    ))


@pytest.fixture
def filler_session(db, project):
    """A session with only filler/noise content — nothing should be extracted."""
    return crud.create_session(db, project.id, schemas.SessionCreate(
        tool_name="Claude",
        title="Filler Session",
        raw_content=FILLER_CONTENT,
        session_date="2026-05-28",
    ))


@pytest.fixture
def mixed_session(db, project):
    """A session with both filler and technical content."""
    return crud.create_session(db, project.id, schemas.SessionCreate(
        tool_name="Claude",
        title="Mixed Session",
        raw_content=MIXED_CONTENT,
        session_date="2026-05-28",
    ))


# ---------------------------------------------------------------------------
# Mock helpers — prevent loading heavy ML models in unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embed(mocker):
    """Patch embed_text to return a fixed 384-dim zero embedding (bytes)."""
    import struct
    fake_embedding = struct.pack(f"{384}f", *([0.0] * 384))
    return mocker.patch(
        "app.services.semantic_classifier.embed_text",
        return_value=fake_embedding,
    )


@pytest.fixture
def mock_entities(mocker):
    """Patch extract_entities to return empty list (avoids spaCy model load)."""
    return mocker.patch(
        "app.services.entity_extractor.extract_entities",
        return_value=[],
    )


@pytest.fixture
def mock_is_technical_true(mocker):
    """Patch is_technical → True (all sentences pass semantic gate)."""
    return mocker.patch(
        "app.services.semantic_classifier.is_technical",
        return_value=True,
    )


@pytest.fixture
def mock_is_technical_false(mocker):
    """Patch is_technical → False (all sentences fail semantic gate)."""
    return mocker.patch(
        "app.services.semantic_classifier.is_technical",
        return_value=False,
    )


# ---------------------------------------------------------------------------
# Canonical LLM response payloads
# ---------------------------------------------------------------------------

VALID_LLM_MEMORIES = [
    {
        "canonical_type": "decision",
        "type_label": "Architecture Decision",
        "title": "Use PostgreSQL for production database",
        "content": "Chose PostgreSQL over SQLite due to concurrent write requirements and ACID guarantees needed in production.",
        "importance": 4,
        "confidence": 0.92,
        "source_quote": "We decided to use PostgreSQL instead of SQLite for production because of concurrent write requirements.",
        "source_message_ids": [],
        "entities": [
            {"label": "PostgreSQL", "entity_type": "TECH"},
            {"label": "SQLite", "entity_type": "TECH"},
        ],
        "links": [],
        "tags": ["database", "postgresql", "architecture"],
        "related_files": [],
        "privacy_level": "internal",
        "review_recommended": False,
        "llm_reasoning": "Explicit decision with clear technical rationale.",
        "temporal": {},
    },
    {
        "canonical_type": "task",
        "type_label": "Pending Task",
        "title": "Add database migration script for users table",
        "content": "Create Alembic migration for the users table schema changes.",
        "importance": 3,
        "confidence": 0.88,
        "source_quote": "TODO: Add database migration script for the users table.",
        "source_message_ids": [],
        "entities": [],
        "links": [],
        "tags": ["migration", "database", "todo"],
        "related_files": [],
        "privacy_level": "internal",
        "review_recommended": False,
        "llm_reasoning": "Explicit TODO item requiring action.",
        "temporal": {},
    },
]

VALID_LLM_RESPONSE = json.dumps(VALID_LLM_MEMORIES)
OPENAI_WRAPPED_RESPONSE = json.dumps({"memories": VALID_LLM_MEMORIES})
INVALID_JSON_RESPONSE = "This is not valid JSON at all { unclosed"

MISSING_SOURCE_QUOTE_RESPONSE = json.dumps([{
    "canonical_type": "insight",
    "type_label": "Technical Insight",
    "title": "Redis improves session performance",
    "content": "Using Redis for session storage reduces DB load significantly.",
    "importance": 3,
    "confidence": 0.75,
    "source_quote": "",  # empty — should be rejected
    "tags": [],
    "entities": [],
    "links": [],
    "related_files": [],
    "privacy_level": "internal",
    "review_recommended": False,
    "llm_reasoning": "Performance observation.",
    "temporal": {},
}])

DYNAMIC_TYPE_RESPONSE = json.dumps([{
    "canonical_type": "preference",  # alias → constraint in CANONICAL_TYPE_MAP
    "type_label": "User Preference",
    "title": "Team prefers immutable data structures",
    "content": "The team has a strong preference for immutable data over mutable state.",
    "importance": 3,
    "confidence": 0.80,
    "source_quote": "Team prefers immutable data structures wherever possible.",
    "source_message_ids": [],
    "entities": [],
    "links": [],
    "tags": ["preference", "code-style"],
    "related_files": [],
    "privacy_level": "internal",
    "review_recommended": False,
    "llm_reasoning": "Explicit team preference stated.",
    "temporal": {},
}])

UNKNOWN_TYPE_RESPONSE = json.dumps([{
    "canonical_type": "random_unknown_type_xyz",
    "type_label": "Completely Novel Type",
    "title": "Some memory with unknown type",
    "content": "Content for a memory with an unrecognized canonical_type.",
    "importance": 2,
    "confidence": 0.70,
    "source_quote": "Some memory with unknown type came up.",
    "source_message_ids": [],
    "entities": [],
    "links": [],
    "tags": [],
    "related_files": [],
    "privacy_level": "internal",
    "review_recommended": False,
    "llm_reasoning": "Test case.",
    "temporal": {},
}])


# ---------------------------------------------------------------------------
# Sample session text
# ---------------------------------------------------------------------------

TECHNICAL_CONTENT = (
    "We decided to use PostgreSQL instead of SQLite for production because of concurrent write requirements. "
    "The connection pool is configured with pool_size=10 and max_overflow=20. "
    "TODO: Add database migration script for the users table. "
    "The authentication service has a bug where refresh tokens are not invalidated on logout. "
    "We will implement Redis caching for session data to reduce database load. "
    "The API response time target is under 100ms for 95th percentile. "
    "We are using FastAPI with SQLAlchemy ORM and Alembic for migrations. "
    "The team decided to enforce strict type annotations across all service modules. "
    "We should add integration tests for the payment processing flow. "
    "The rate limiter is configured to allow 1000 requests per minute per API key."
)

FILLER_CONTENT = (
    "Sure! I would be happy to help with that. "
    "Great question! Let me think about that. "
    "Of course, here is what I would suggest. "
    "Thank you for the explanation! "
    "That is a great point! "
    "Okay, let me help you with this. "
    "I understand that. "
    "Let me help you with this task. "
    "How can I assist you today? "
    "I hope that helps!"
)

MIXED_CONTENT = FILLER_CONTENT + " " + TECHNICAL_CONTENT


def make_phase1_extraction_result(session_id: str, memories=None):
    """Build a mock Phase 1 ExtractionResult for rule_based tests."""
    mock = MagicMock()
    mock.session_id = session_id
    mock.memories = memories or []
    mock.memories_created = len(mock.memories)
    mock.summary = f"Extracted {mock.memories_created} memories"
    mock.skipped_duplicates = 0
    mock.duplicate_titles = []
    mock.low_confidence_queued = 0
    return mock
