"""
Sub-phase 2.3 — Synthetic benchmark corpus.

Builds deterministic, schema-native evaluation scenarios directly in the test DB.
Each scenario exercises a specific retrieval challenge:

  1. direct_keyword    — BM25 title/content match (baseline correctness)
  2. entity_match      — entity-based retrieval (entity leg)
  3. supersession      — superseded memory excluded by default
  4. privacy_clearance — secret/sensitive memories blocked at lower clearance
  5. valid_until_expiry — expired memories excluded (valid_until < now)
  6. review_rejected   — rejected memories excluded at all clearance levels
  7. contradictory_update — newer memory supersedes older contradictory one
  8. multi_type        — different memory types for different queries
  9. privacy_leakage   — systematic cross-clearance isolation check

Each corpus fixture returns:
  CorpusFixture with:
    - project_id
    - memory_map: {semantic_key → DB memory_id}
    - test_cases: list[TestCase]  (frozen query → expected_key list)

The IDs are assigned at fixture creation time (UUID), making the corpus
reproducible within a test run while staying independent of DB auto-increment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A frozen benchmark query with expected gold-label memory keys."""
    query: str
    expected_keys: list[str]       # keys into CorpusFixture.memory_map
    scenario: str
    max_clearance: str = "internal"
    embed_query: bool = False       # False = offline/deterministic; BM25+entity only
    description: str = ""          # human-readable description of what's tested
    expect_excluded_keys: list[str] = field(default_factory=list)  # must NOT appear


@dataclass
class CorpusFixture:
    """Populated corpus: memory IDs resolved, test cases ready to run."""
    project_id: str
    memory_map: dict[str, str]     # {semantic_key: DB memory_id}
    test_cases: list[TestCase]

    def resolve_expected_ids(self, test_case: TestCase) -> list[str]:
        """Convert expected_keys to actual memory IDs."""
        return [self.memory_map[k] for k in test_case.expected_keys if k in self.memory_map]

    def resolve_excluded_ids(self, test_case: TestCase) -> list[str]:
        """Convert expect_excluded_keys to actual memory IDs."""
        return [self.memory_map[k] for k in test_case.expect_excluded_keys if k in self.memory_map]


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def build_synthetic_corpus(db: Session, project_id: str) -> CorpusFixture:
    """
    Create all synthetic benchmark memories in the DB and return the corpus fixture.

    All memories are created fresh. The corpus is designed to exercise all nine
    scenario types. Tests use embed_query=False so the corpus is fully offline/
    deterministic (BM25 + entity legs only — dense leg requires sentence_transformers).
    """
    from app import models, crud

    now = datetime.now(timezone.utc)
    memory_map: dict[str, str] = {}

    def _make(
        key: str,
        title: str,
        content: str,
        mem_type: str = "decision",
        importance: int = 3,
        privacy_level: str = "internal",
        status: str = "active",
        review_status: str = "verified",
        superseded_by: str | None = None,
        valid_until: datetime | None = None,
        source_quote: str | None = None,
        retrieval_hint: str | None = None,
        search_text: str | None = None,
    ) -> models.Memory:
        m = models.Memory(
            project_id=project_id,
            type=mem_type,
            title=title,
            content=content,
            importance=importance,
            privacy_level=privacy_level,
            status=status,
            review_status=review_status,
            superseded_by=superseded_by,
            valid_until=valid_until,
            source_quote=source_quote or title,
            retrieval_hint=retrieval_hint or title,
            search_text=search_text or f"{title} {content}",
        )
        db.add(m)
        db.flush()  # assigns m.id without committing
        memory_map[key] = m.id
        return m

    # -----------------------------------------------------------------------
    # Scenario 1: Direct keyword match (BM25 baseline correctness)
    # -----------------------------------------------------------------------
    _make(
        "db_decision",
        title="PostgreSQL chosen for production database",
        content="We decided to use PostgreSQL instead of SQLite for production "
                "due to concurrent write requirements and ACID guarantees.",
        importance=4,
        source_quote="We decided to use PostgreSQL for production.",
        retrieval_hint="PostgreSQL selected over SQLite for ACID compliance.",
    )
    _make(
        "auth_bug",
        title="Authentication refresh token bug",
        content="The authentication service has a critical bug where refresh tokens "
                "are not invalidated on user logout, creating a security vulnerability.",
        mem_type="problem",
        importance=5,
        source_quote="Refresh tokens are not invalidated on logout.",
        retrieval_hint="Auth bug: refresh tokens survive logout.",
    )
    _make(
        "api_rate_limit",
        title="API rate limit configuration",
        content="The rate limiter is configured to allow 1000 requests per minute "
                "per API key using token bucket algorithm.",
        mem_type="constraint",
        importance=3,
        source_quote="Rate limiter allows 1000 requests per minute per API key.",
        retrieval_hint="Rate limit: 1000 req/min per API key.",
    )
    _make(
        "redis_caching",
        title="Redis caching for session data",
        content="We will implement Redis caching for session storage to reduce "
                "PostgreSQL database load. Target: 80% cache hit rate.",
        mem_type="decision",
        importance=4,
        source_quote="Implement Redis caching for session storage.",
        retrieval_hint="Redis cache for sessions, reduce DB load.",
    )
    _make(
        "fastapi_stack",
        title="FastAPI with SQLAlchemy and Alembic",
        content="The project uses FastAPI as the web framework, SQLAlchemy ORM "
                "for database access, and Alembic for schema migrations.",
        mem_type="structure",
        importance=3,
        source_quote="FastAPI with SQLAlchemy ORM and Alembic.",
        retrieval_hint="Stack: FastAPI + SQLAlchemy + Alembic.",
    )

    # -----------------------------------------------------------------------
    # Scenario 2: Entity-based retrieval (entity leg)
    # -----------------------------------------------------------------------
    _make(
        "redis_entity_mem",
        title="Redis connection pool settings",
        content="Redis connection pool configured with max_connections=50 "
                "and socket_timeout=5 seconds for resilient session caching.",
        mem_type="structure",
        importance=3,
        search_text="Redis connection pool max_connections socket_timeout session caching",
        source_quote="Redis pool: max_connections=50, socket_timeout=5.",
    )
    _make(
        "postgres_entity_mem",
        title="PostgreSQL connection pool configuration",
        content="PostgreSQL pool configured with pool_size=10 and max_overflow=20 "
                "to handle concurrent requests efficiently.",
        mem_type="structure",
        importance=3,
        search_text="PostgreSQL connection pool pool_size max_overflow concurrent",
        source_quote="PostgreSQL pool_size=10, max_overflow=20.",
    )

    # -----------------------------------------------------------------------
    # Scenario 3: Supersession chain
    # -----------------------------------------------------------------------
    # Old decision — will be superseded
    old_db = _make(
        "old_sqlite_decision",
        title="SQLite used for all environments",
        content="We initially decided to use SQLite for both development and production "
                "for simplicity and zero-configuration setup.",
        importance=2,
        status="superseded",   # directly mark superseded
        search_text="SQLite development production simplicity",
        source_quote="SQLite for all environments initially.",
    )
    # New decision — supersedes old one
    _make(
        "new_postgres_supersedes",
        title="PostgreSQL replaces SQLite in production",
        content="After load testing revealed SQLite write contention, we migrated "
                "production to PostgreSQL. Development retains SQLite.",
        importance=4,
        superseded_by=None,
        search_text="PostgreSQL replaces SQLite production migration load testing contention",
        source_quote="Migrated to PostgreSQL after load testing showed SQLite contention.",
    )
    # Mark old memory as superseded_by new (need new ID first — flush was above)
    old_db.superseded_by = memory_map["new_postgres_supersedes"]

    # -----------------------------------------------------------------------
    # Scenario 4: Privacy clearance transitions
    # -----------------------------------------------------------------------
    _make(
        "public_arch_overview",
        title="System architecture overview",
        content="The system uses a three-tier architecture with a React frontend, "
                "FastAPI backend, and PostgreSQL database.",
        privacy_level="public",
        importance=3,
        search_text="system architecture React FastAPI PostgreSQL three-tier",
        source_quote="Three-tier architecture: React + FastAPI + PostgreSQL.",
    )
    _make(
        "internal_perf_target",
        title="Internal performance SLA",
        content="Internal target: API p95 latency must stay below 200ms. "
                "This is not publicly advertised.",
        privacy_level="internal",
        importance=3,
        search_text="internal performance SLA p95 latency API response time",
        source_quote="API p95 latency target: below 200ms (internal).",
    )
    _make(
        "sensitive_customer_data",
        title="Customer PII handling policy",
        content="Customer email and payment data is encrypted at rest using AES-256. "
                "Access restricted to engineering leads only.",
        privacy_level="sensitive",
        importance=4,
        search_text="customer PII email payment AES-256 encryption sensitive",
        source_quote="Customer PII encrypted with AES-256, restricted access.",
    )
    _make(
        "secret_api_key",
        title="Production API key rotation schedule",
        content="Production API keys are rotated every 30 days. The master key is "
                "stored in HashiCorp Vault at vault.internal/prod/api-keys.",
        privacy_level="secret",
        importance=5,
        search_text="production API key rotation HashiCorp Vault secret schedule",
        source_quote="API keys rotated every 30 days via HashiCorp Vault.",
    )

    # -----------------------------------------------------------------------
    # Scenario 5: valid_until expiry (time-bounded facts)
    # -----------------------------------------------------------------------
    past = now - timedelta(days=30)
    _make(
        "expired_rate_limit",
        title="Old rate limit 100 requests per minute",
        content="Previous rate limit was 100 requests per minute per API key. "
                "This was the original configuration before scaling.",
        valid_until=past,   # expired 30 days ago
        importance=2,
        search_text="rate limit 100 requests minute API key old configuration",
        source_quote="Old rate limit: 100 req/min (now expired).",
    )
    _make(
        "current_rate_limit",
        title="Current rate limit 1000 requests per minute",
        content="Current production rate limit is 1000 requests per minute per API key, "
                "updated after infrastructure upgrade.",
        valid_until=None,   # no expiry — always valid
        importance=3,
        search_text="current rate limit 1000 requests minute production infrastructure",
        source_quote="Current rate limit: 1000 req/min after infrastructure upgrade.",
    )

    # -----------------------------------------------------------------------
    # Scenario 6: review_status = rejected (must never appear)
    # -----------------------------------------------------------------------
    _make(
        "rejected_bad_decision",
        title="Migrate to MongoDB for all storage",
        content="Proposal to migrate from PostgreSQL to MongoDB for flexible schema. "
                "Rejected after review: ACID requirements make this unsuitable.",
        review_status="rejected",
        importance=2,
        search_text="MongoDB migration flexible schema rejected PostgreSQL ACID",
        source_quote="MongoDB migration proposal rejected for ACID requirements.",
    )
    _make(
        "good_decision_nearby",
        title="Keep PostgreSQL with flexible JSONB columns",
        content="Decision to keep PostgreSQL but use JSONB columns for flexible "
                "metadata storage, satisfying both ACID and schema flexibility needs.",
        review_status="verified",
        importance=4,
        search_text="PostgreSQL JSONB flexible metadata schema ACID storage decision",
        source_quote="PostgreSQL with JSONB for flexible schema needs.",
    )

    # -----------------------------------------------------------------------
    # Scenario 7: Contradictory update (newer supersedes older)
    # -----------------------------------------------------------------------
    _make(
        "old_team_size",
        title="Team size is 3 engineers",
        content="The engineering team currently consists of 3 backend engineers "
                "working on the core API and database layer.",
        status="superseded",
        importance=2,
        search_text="team size 3 engineers backend API database",
        source_quote="Team: 3 backend engineers.",
    )
    _make(
        "new_team_size",
        title="Team size updated to 6 engineers",
        content="After Q2 hiring, the engineering team has grown to 6 engineers: "
                "3 backend, 2 frontend, 1 DevOps.",
        importance=3,
        search_text="team size 6 engineers backend frontend DevOps hiring Q2",
        source_quote="Team grew to 6 after Q2 hiring: 3 backend, 2 frontend, 1 DevOps.",
    )

    # -----------------------------------------------------------------------
    # Scenario 8: Multi-type queries
    # -----------------------------------------------------------------------
    _make(
        "open_question_scaling",
        title="How to handle 10x traffic spike",
        content="Open question: if traffic spikes 10x during a product launch, "
                "what is the autoscaling strategy? No decision made yet.",
        mem_type="open_question",
        importance=3,
        search_text="traffic spike autoscaling strategy launch open question 10x",
        source_quote="Open question: autoscaling strategy for 10x traffic spike.",
    )
    _make(
        "workflow_deploy",
        title="Deployment workflow using GitHub Actions",
        content="All deployments go through GitHub Actions CI/CD pipeline. "
                "Push to main triggers staging deploy; manual approval for production.",
        mem_type="workflow_pattern",
        importance=4,
        search_text="deployment GitHub Actions CI/CD pipeline staging production approval",
        source_quote="Deployments via GitHub Actions; manual approval for production.",
    )
    _make(
        "task_migration",
        title="Add Alembic migration for users table",
        content="TODO: Create an Alembic migration script for the users table "
                "schema changes including the new email_verified column.",
        mem_type="task",
        importance=3,
        search_text="Alembic migration users table schema email_verified column TODO",
        source_quote="TODO: Alembic migration for users table with email_verified.",
    )

    # -----------------------------------------------------------------------
    # Scenario 9: Privacy leakage — cross-clearance isolation
    # (Additional secret/sensitive memories to test strict isolation)
    # -----------------------------------------------------------------------
    _make(
        "secret_db_password",
        title="Database master password policy",
        content="The PostgreSQL master password must be 32+ characters, rotated "
                "quarterly, and stored only in HashiCorp Vault. Never in source code.",
        privacy_level="secret",
        importance=5,
        search_text="PostgreSQL master password HashiCorp Vault rotation quarterly secret",
        source_quote="DB master password: 32+ chars, quarterly rotation, Vault only.",
    )
    _make(
        "sensitive_audit_log",
        title="Audit log access policy",
        content="Audit logs contain user actions and are classified as sensitive. "
                "Access requires security team approval and is logged separately.",
        privacy_level="sensitive",
        importance=4,
        search_text="audit log sensitive access security approval user actions",
        source_quote="Audit logs: sensitive, require security team approval.",
    )

    db.commit()

    # -----------------------------------------------------------------------
    # Frozen test cases
    # -----------------------------------------------------------------------
    test_cases: list[TestCase] = [

        # --- Scenario 1: Direct keyword match ---
        TestCase(
            query="PostgreSQL database decision",
            expected_keys=["db_decision"],
            scenario="direct_keyword",
            description="Direct BM25 title match for DB decision",
        ),
        TestCase(
            query="authentication refresh token bug logout",
            expected_keys=["auth_bug"],
            scenario="direct_keyword",
            description="BM25 content match for auth bug",
        ),
        TestCase(
            query="rate limit requests per minute API key",
            expected_keys=["api_rate_limit"],
            scenario="direct_keyword",
            description="BM25 rate limit constraint",
        ),
        TestCase(
            query="Redis session caching database load",
            expected_keys=["redis_caching"],
            scenario="direct_keyword",
            description="BM25 Redis caching decision",
        ),
        TestCase(
            query="FastAPI SQLAlchemy Alembic web framework",
            expected_keys=["fastapi_stack"],
            scenario="direct_keyword",
            description="BM25 tech stack structure",
        ),

        # --- Scenario 2: Entity match ---
        TestCase(
            query="Redis connection pool configuration",
            expected_keys=["redis_entity_mem"],
            scenario="entity_match",
            description="Entity leg: Redis entity → connection pool memory",
        ),
        TestCase(
            query="PostgreSQL pool size overflow connections",
            expected_keys=["postgres_entity_mem"],
            scenario="entity_match",
            description="Entity leg: PostgreSQL entity → pool config memory",
        ),

        # --- Scenario 3: Supersession ---
        TestCase(
            query="SQLite production database migration PostgreSQL",
            expected_keys=["new_postgres_supersedes"],
            expect_excluded_keys=["old_sqlite_decision"],
            scenario="supersession",
            description="Superseded memory excluded; new decision returned",
        ),

        # --- Scenario 4: Privacy clearance ---
        TestCase(
            query="system architecture React FastAPI database",
            expected_keys=["public_arch_overview"],
            expect_excluded_keys=["internal_perf_target", "sensitive_customer_data", "secret_api_key"],
            scenario="privacy_clearance",
            max_clearance="public",
            description="Public clearance: only public memories visible",
        ),
        TestCase(
            query="API performance latency SLA target internal",
            expected_keys=["internal_perf_target"],
            expect_excluded_keys=["sensitive_customer_data", "secret_api_key"],
            scenario="privacy_clearance",
            max_clearance="internal",
            description="Internal clearance: public + internal visible, not higher",
        ),

        # --- Scenario 5: valid_until expiry ---
        TestCase(
            query="rate limit current production requests minute",
            expected_keys=["current_rate_limit"],
            expect_excluded_keys=["expired_rate_limit"],
            scenario="valid_until_expiry",
            description="Expired memory excluded; current rate limit returned",
        ),

        # --- Scenario 6: review_rejected ---
        TestCase(
            query="PostgreSQL JSONB flexible schema ACID storage",
            expected_keys=["good_decision_nearby"],
            expect_excluded_keys=["rejected_bad_decision"],
            scenario="review_rejected",
            description="Rejected memory never returned regardless of match quality",
        ),

        # --- Scenario 7: Contradictory update ---
        TestCase(
            query="engineering team size engineers backend frontend",
            expected_keys=["new_team_size"],
            expect_excluded_keys=["old_team_size"],
            scenario="contradictory_update",
            description="Updated team size supersedes old; newer returned",
        ),

        # --- Scenario 8: Multi-type ---
        TestCase(
            query="traffic spike autoscaling open question launch",
            expected_keys=["open_question_scaling"],
            scenario="multi_type",
            description="Open question type retrieved for uncertainty query",
        ),
        TestCase(
            query="GitHub Actions deployment CI CD pipeline staging",
            expected_keys=["workflow_deploy"],
            scenario="multi_type",
            description="Workflow pattern type retrieved for deployment query",
        ),
        TestCase(
            query="Alembic migration users table TODO email verified",
            expected_keys=["task_migration"],
            scenario="multi_type",
            description="Task type retrieved for TODO query",
        ),

        # --- Scenario 9: Privacy leakage isolation ---
        TestCase(
            query="API key rotation schedule vault password",
            expect_excluded_keys=["secret_api_key", "secret_db_password"],
            expected_keys=[],  # no public/internal match expected
            scenario="privacy_leakage",
            max_clearance="public",
            description="Secret memories must not surface at public clearance",
        ),
        TestCase(
            query="audit log user actions sensitive security approval",
            expect_excluded_keys=["sensitive_audit_log", "secret_api_key"],
            expected_keys=[],
            scenario="privacy_leakage",
            max_clearance="internal",
            description="Sensitive memories must not surface at internal clearance",
        ),
    ]

    return CorpusFixture(
        project_id=project_id,
        memory_map=memory_map,
        test_cases=test_cases,
    )
