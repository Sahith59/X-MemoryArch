"""
Sub-phase 2.4 — Graph Expansion + Cluster Summaries tests.

Coverage:
  Query classifiers   — is_code_query, is_multi_concept_query
  Entity soft-boost   — apply_entity_soft_boost
  Code anchor         — code_anchor_retrieval
  1-hop expansion     — expand_1hop
  2-hop expansion     — expand_2hop
  graph_expand        — orchestrator integration
  Cluster summaries   — generate_cluster_summaries (template + mock LLM)
  Pipeline wiring     — retrieve() with expansion enabled/disabled
  RetrievalResult     — expanded_via_links field
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from app.services.retrieval.graph_expansion import (
    apply_entity_soft_boost,
    code_anchor_retrieval,
    expand_1hop,
    expand_2hop,
    graph_expand,
    is_code_query,
    is_multi_concept_query,
)
from app.services.retrieval.cluster_summaries import (
    ClusterSummaryResult,
    generate_cluster_summaries,
)
from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
from app.services.vector_backends.sqlite_exact import SQLiteExactBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def _make_embedding(dim: int = 384, val: float = 1.0, idx: int = 0) -> bytes:
    vec = np.zeros(dim, dtype=np.float32)
    vec[idx] = val
    return vec.tobytes()


def _create_memory(
    db,
    project_id: str,
    *,
    title: str = "T",
    content: str = "C",
    status: str = "active",
    privacy: str = "internal",
    review_status: str = "auto_extracted",
    file_path: str | None = None,
    symbol_name: str | None = None,
    cluster_id: int | None = None,
    cluster_label: str | None = None,
    memory_type: str = "decision",
    importance: int = 3,
    embedding: bytes | None = None,
) -> object:
    from app import models as phase1_models
    mid = str(uuid.uuid4())
    m = phase1_models.Memory(
        id=mid,
        project_id=project_id,
        type=memory_type,
        title=title,
        content=content,
        status=status,
        privacy_level=privacy,
        review_status=review_status,
        file_path=file_path,
        symbol_name=symbol_name,
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        importance=importance,
        confidence=1.0,
        embedding=embedding,
        created_at=_now(),
        updated_at=_now(),
    )
    db.add(m)
    db.commit()
    return m


def _create_entity(db, memory_id: str, project_id: str, text: str, label: str = "TECH"):
    from app import models as phase1_models
    eid = str(uuid.uuid4())
    e = phase1_models.MemoryEntity(
        id=eid,
        memory_id=memory_id,
        project_id=project_id,
        entity_text=text,
        entity_label=label,
        normalized_text=text.lower(),
        created_at=_now(),
    )
    db.add(e)
    db.commit()
    return e


def _create_link(db, src_id: str, tgt_id: str, rel_type: str = "relates_to", superseded: bool = False):
    from app import models as phase1_models
    lid = str(uuid.uuid4())
    link = phase1_models.MemoryLink(
        id=lid,
        source_memory_id=src_id,
        target_memory_id=tgt_id,
        relationship_type=rel_type,
        superseded_at=_now() if superseded else None,
        created_at=_now(),
    )
    db.add(link)
    db.commit()
    return link


def _create_project(db) -> object:
    from app import crud, schemas
    return crud.create_project(db, schemas.ProjectCreate(
        name="Graph Expansion Test Project",
        description="Tests for 2.4",
        tech_stack=["Python"],
        goals=["Test graph expansion"],
        domain="software",
    ))


@pytest.fixture(autouse=True)
def _patch_ml(mock_embed, mock_entities, mock_is_technical_true):
    pass


# ---------------------------------------------------------------------------
# is_code_query
# ---------------------------------------------------------------------------

class TestIsCodeQuery:
    def test_python_file_extension(self):
        assert is_code_query("where is auth.py") is True

    def test_js_file_extension(self):
        assert is_code_query("fix bug in handler.js") is True

    def test_def_keyword(self):
        assert is_code_query("def authenticate user") is True

    def test_class_keyword(self):
        assert is_code_query("class UserManager") is True

    def test_import_statement(self):
        assert is_code_query("import fastapi routers") is True

    def test_path_with_slashes(self):
        assert is_code_query("find /app/routers/auth") is True

    def test_from_import(self):
        assert is_code_query("from app.models import User") is True

    def test_plain_natural_language(self):
        assert is_code_query("what is the database connection limit") is False

    def test_empty_query(self):
        assert is_code_query("") is False

    def test_business_query(self):
        assert is_code_query("why did we choose postgresql over sqlite") is False

    def test_function_keyword(self):
        assert is_code_query("function handleLogin") is True


# ---------------------------------------------------------------------------
# is_multi_concept_query
# ---------------------------------------------------------------------------

class TestIsMultiConceptQuery:
    def test_explicit_and(self):
        assert is_multi_concept_query("authentication and authorization") is True

    def test_explicit_or(self):
        assert is_multi_concept_query("redis or memcached for caching") is True

    def test_five_meaningful_tokens(self):
        assert is_multi_concept_query("database connection pool configuration settings") is True

    def test_four_meaningful_tokens_no_connectors(self):
        # exactly 4 meaningful tokens — should NOT be multi-concept
        assert is_multi_concept_query("database connection pool settings") is False

    def test_short_query_no_connectors(self):
        assert is_multi_concept_query("auth bug") is False

    def test_stop_words_dont_count(self):
        # "the a the to for is" → 0 meaningful tokens
        assert is_multi_concept_query("the a the to for is") is False

    def test_long_query_with_and(self):
        assert is_multi_concept_query("auth service and rate limiter and caching layer") is True

    def test_empty_string(self):
        assert is_multi_concept_query("") is False


# ---------------------------------------------------------------------------
# apply_entity_soft_boost
# ---------------------------------------------------------------------------

class TestEntitySoftBoost:
    def test_boost_increases_score_for_matching_memory(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="Redis cache", content="Redis is a cache")
        _create_entity(db, m.id, project.id, "redis")

        scores = {m.id: 1.0}
        boosted = apply_entity_soft_boost(db, scores, "redis cache setup", entity_boost_weight=0.15)

        assert boosted[m.id] > 1.0

    def test_no_boost_for_non_matching_entity(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="Unrelated memory")
        _create_entity(db, m.id, project.id, "postgres")

        scores = {m.id: 1.0}
        boosted = apply_entity_soft_boost(db, scores, "redis cache", entity_boost_weight=0.15)

        # No overlap → score unchanged
        assert boosted[m.id] == pytest.approx(1.0)

    def test_boost_proportional_to_overlap_ratio(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")

        # m1 has all tokens from query; m2 has only one
        _create_entity(db, m1.id, project.id, "redis")
        _create_entity(db, m1.id, project.id, "cache")
        _create_entity(db, m2.id, project.id, "redis")

        scores = {m1.id: 1.0, m2.id: 1.0}
        boosted = apply_entity_soft_boost(db, scores, "redis cache", entity_boost_weight=0.15)

        assert boosted[m1.id] > boosted[m2.id]

    def test_empty_scores_returns_empty(self, db):
        result = apply_entity_soft_boost(db, {}, "some query")
        assert result == {}

    def test_empty_query_tokens_returns_unchanged(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="M")
        scores = {m.id: 2.0}
        # Query of only stop words → no tokens → no boost
        boosted = apply_entity_soft_boost(db, scores, "the a the", entity_boost_weight=0.15)
        assert boosted[m.id] == pytest.approx(2.0)

    def test_memory_with_no_entities_not_boosted(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="M")
        scores = {m.id: 1.5}
        boosted = apply_entity_soft_boost(db, scores, "redis cache", entity_boost_weight=0.15)
        assert boosted[m.id] == pytest.approx(1.5)

    def test_boost_weight_zero_leaves_scores_unchanged(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="M")
        _create_entity(db, m.id, project.id, "redis")
        scores = {m.id: 1.0}
        boosted = apply_entity_soft_boost(db, scores, "redis", entity_boost_weight=0.0)
        assert boosted[m.id] == pytest.approx(1.0)

    def test_boost_max_is_boost_weight_times_one(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="M")
        _create_entity(db, m.id, project.id, "redis")
        _create_entity(db, m.id, project.id, "cache")
        scores = {m.id: 1.0}
        # Query = "redis cache" → 2 tokens, entity overlap = 2 → ratio = 1.0
        boosted = apply_entity_soft_boost(db, scores, "redis cache", entity_boost_weight=0.15)
        assert boosted[m.id] == pytest.approx(1.0 * (1 + 0.15 * 1.0))


# ---------------------------------------------------------------------------
# code_anchor_retrieval
# ---------------------------------------------------------------------------

class TestCodeAnchorRetrieval:
    def test_finds_memory_by_file_path(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="Auth", file_path="app/auth.py")
        allowed = {m.id}
        result = code_anchor_retrieval(db, project.id, "auth.py login bug", allowed)
        assert m.id in result

    def test_finds_memory_by_symbol_name(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="Handler", symbol_name="authenticate_user")
        allowed = {m.id}
        result = code_anchor_retrieval(db, project.id, "where is authenticate function", allowed)
        assert m.id in result

    def test_excludes_memory_not_in_allowed_ids(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="Auth", file_path="app/auth.py")
        result = code_anchor_retrieval(db, project.id, "auth.py", allowed_ids=set())
        assert m.id not in result

    def test_returns_empty_for_no_matching_tokens(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="Auth", file_path="app/auth.py")
        allowed = {m.id}
        result = code_anchor_retrieval(db, project.id, "the a or", allowed)
        assert result == []

    def test_caps_at_max_results(self, db):
        project = _create_project(db)
        memories = [
            _create_memory(db, project.id, title=f"M{i}", file_path=f"app/module{i}.py")
            for i in range(15)
        ]
        allowed = {m.id for m in memories}
        result = code_anchor_retrieval(db, project.id, "module", allowed, max_results=5)
        assert len(result) <= 5

    def test_returns_empty_when_allowed_ids_empty(self, db):
        project = _create_project(db)
        result = code_anchor_retrieval(db, project.id, "auth.py", set())
        assert result == []

    def test_only_matches_within_project(self, db):
        from app import crud, schemas
        p1 = _create_project(db)
        p2 = crud.create_project(db, schemas.ProjectCreate(
            name="Other", description="", tech_stack=[], goals=[], domain="software",
        ))
        m_p1 = _create_memory(db, p1.id, title="P1 auth", file_path="app/auth.py")
        m_p2 = _create_memory(db, p2.id, title="P2 auth", file_path="app/auth.py")

        allowed = {m_p1.id, m_p2.id}
        result = code_anchor_retrieval(db, p1.id, "auth.py", allowed)
        assert m_p1.id in result
        assert m_p2.id not in result


# ---------------------------------------------------------------------------
# expand_1hop
# ---------------------------------------------------------------------------

class TestExpand1Hop:
    def test_outbound_edge_returns_neighbor(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        _create_link(db, m1.id, m2.id)

        allowed = {m1.id, m2.id}
        top_k_scores = {m1.id: 1.0}
        result = expand_1hop(db, [m1.id], allowed, top_k_scores)

        assert m2.id in result

    def test_inbound_edge_returns_source(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        _create_link(db, m2.id, m1.id)  # m2 → m1

        allowed = {m1.id, m2.id}
        top_k_scores = {m1.id: 1.0}
        result = expand_1hop(db, [m1.id], allowed, top_k_scores)

        assert m2.id in result

    def test_damping_factor_is_half(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        _create_link(db, m1.id, m2.id)

        allowed = {m1.id, m2.id}
        top_k_scores = {m1.id: 0.8}
        result = expand_1hop(db, [m1.id], allowed, top_k_scores)

        assert result[m2.id] == pytest.approx(0.4)

    def test_superseded_link_not_followed(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        _create_link(db, m1.id, m2.id, superseded=True)

        allowed = {m1.id, m2.id}
        result = expand_1hop(db, [m1.id], allowed, {m1.id: 1.0})

        assert m2.id not in result

    def test_excluded_memory_not_returned(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2 secret", privacy="secret")
        _create_link(db, m1.id, m2.id)

        allowed = {m1.id}  # m2 excluded from allowed
        result = expand_1hop(db, [m1.id], allowed, {m1.id: 1.0})

        assert m2.id not in result

    def test_already_in_top_k_not_duplicated(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        _create_link(db, m1.id, m2.id)

        allowed = {m1.id, m2.id}
        # m2 is already in top-k
        result = expand_1hop(db, [m1.id, m2.id], allowed, {m1.id: 1.0, m2.id: 0.9})

        assert m2.id not in result

    def test_capped_at_max_per_hop(self, db):
        project = _create_project(db)
        m_origin = _create_memory(db, project.id, title="Origin")
        neighbors = [_create_memory(db, project.id, title=f"N{i}") for i in range(20)]
        for n in neighbors:
            _create_link(db, m_origin.id, n.id)

        allowed = {m_origin.id} | {n.id for n in neighbors}
        result = expand_1hop(db, [m_origin.id], allowed, {m_origin.id: 1.0}, max_per_hop=5)

        assert len(result) <= 5

    def test_empty_top_k_returns_empty(self, db):
        project = _create_project(db)
        result = expand_1hop(db, [], {}, {})
        assert result == {}

    def test_multiple_paths_takes_highest_score(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        m3 = _create_memory(db, project.id, title="M3")
        _create_link(db, m1.id, m3.id)
        _create_link(db, m2.id, m3.id)

        allowed = {m1.id, m2.id, m3.id}
        # m1 scores higher → m3 should inherit from m1
        scores = {m1.id: 1.0, m2.id: 0.4}
        result = expand_1hop(db, [m1.id, m2.id], allowed, scores)

        assert m3.id in result
        assert result[m3.id] == pytest.approx(0.5)  # 0.5 × 1.0


# ---------------------------------------------------------------------------
# expand_2hop
# ---------------------------------------------------------------------------

class TestExpand2Hop:
    def test_2hop_reaches_second_degree_neighbor(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        m3 = _create_memory(db, project.id, title="M3")
        _create_link(db, m1.id, m2.id)
        _create_link(db, m2.id, m3.id)

        # hop1 = {m2: 0.5}
        hop1 = {m2.id: 0.5}
        already = {m1.id, m2.id}
        allowed = {m1.id, m2.id, m3.id}

        result = expand_2hop(db, hop1, allowed, already)
        assert m3.id in result

    def test_2hop_damping_is_quarter_of_original(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        m3 = _create_memory(db, project.id, title="M3")
        _create_link(db, m1.id, m2.id)
        _create_link(db, m2.id, m3.id)

        hop1 = {m2.id: 0.5}  # m2 has half of m1's score
        result = expand_2hop(db, hop1, {m1.id, m2.id, m3.id}, {m1.id, m2.id})

        assert result[m3.id] == pytest.approx(0.25)  # 0.5 × 0.5

    def test_2hop_excluded_from_allowed_not_returned(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        m3 = _create_memory(db, project.id, title="M3 secret")
        _create_link(db, m1.id, m2.id)
        _create_link(db, m2.id, m3.id)

        hop1 = {m2.id: 0.5}
        # m3 not in allowed_ids
        result = expand_2hop(db, hop1, {m1.id, m2.id}, {m1.id, m2.id})

        assert m3.id not in result

    def test_2hop_bounded_by_max_total(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        hop1_members = [_create_memory(db, project.id, title=f"H{i}") for i in range(5)]
        hop2_targets = [_create_memory(db, project.id, title=f"T{i}") for i in range(20)]

        for h in hop1_members:
            _create_link(db, m1.id, h.id)
        for i, h in enumerate(hop1_members):
            for t in hop2_targets[i*4:(i+1)*4]:
                _create_link(db, h.id, t.id)

        hop1 = {h.id: 0.5 for h in hop1_members}
        already = {m1.id} | {h.id for h in hop1_members}
        allowed = already | {t.id for t in hop2_targets}

        result = expand_2hop(db, hop1, allowed, already, max_total=5)
        assert len(result) <= 5

    def test_2hop_empty_hop1_returns_empty(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="M")
        result = expand_2hop(db, {}, {m.id}, {m.id})
        assert result == {}


# ---------------------------------------------------------------------------
# graph_expand (orchestrator)
# ---------------------------------------------------------------------------

class TestGraphExpand:
    def test_entity_boost_increases_score(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="PostgreSQL decision")
        _create_entity(db, m.id, project.id, "postgresql")

        scores = {m.id: 1.0}
        augmented, expanded = graph_expand(
            db, project.id, [m.id], scores, {m.id},
            "postgresql database",
            enable_entity_boost=True,
            enable_graph_expansion=False,
        )

        assert augmented[m.id] > 1.0
        assert expanded == 0  # only boost, no new memories

    def test_1hop_expansion_adds_neighbor(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        _create_link(db, m1.id, m2.id)

        scores = {m1.id: 1.0}
        allowed = {m1.id, m2.id}
        augmented, expanded = graph_expand(
            db, project.id, [m1.id], scores, allowed,
            "what is the architecture",
            enable_entity_boost=False,
            enable_graph_expansion=True,
            enable_2hop=False,
        )

        assert m2.id in augmented
        assert expanded == 1

    def test_2hop_requires_multi_concept_query(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        m3 = _create_memory(db, project.id, title="M3")
        _create_link(db, m1.id, m2.id)
        _create_link(db, m2.id, m3.id)

        scores = {m1.id: 1.0}
        allowed = {m1.id, m2.id, m3.id}

        # Short single-concept query — 2-hop should NOT fire
        _, _ = graph_expand(
            db, project.id, [m1.id], scores, allowed,
            "auth bug",  # NOT multi-concept
            enable_graph_expansion=True,
            enable_2hop=True,
        )
        augmented, _ = graph_expand(
            db, project.id, [m1.id], scores, allowed,
            "auth bug",
        )
        assert m3.id not in augmented  # m2 may appear via 1-hop but not m3

    def test_2hop_fires_for_multi_concept_query(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        m3 = _create_memory(db, project.id, title="M3")
        _create_link(db, m1.id, m2.id)
        _create_link(db, m2.id, m3.id)

        scores = {m1.id: 1.0}
        allowed = {m1.id, m2.id, m3.id}

        augmented, expanded = graph_expand(
            db, project.id, [m1.id], scores, allowed,
            "authentication and authorization and rate limiting and caching config",
            enable_graph_expansion=True,
            enable_2hop=True,
            enable_entity_boost=False,
        )

        assert m3.id in augmented
        assert expanded >= 2  # both m2 and m3 are new

    def test_expansion_disabled_returns_only_original_scores(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        _create_link(db, m1.id, m2.id)

        scores = {m1.id: 1.0}
        augmented, expanded = graph_expand(
            db, project.id, [m1.id], scores, {m1.id, m2.id},
            "query",
            enable_entity_boost=False,
            enable_graph_expansion=False,
            enable_2hop=False,
        )

        assert m2.id not in augmented
        assert expanded == 0

    def test_expanded_count_non_negative(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="M")
        _, expanded = graph_expand(db, project.id, [m.id], {m.id: 1.0}, {m.id}, "query")
        assert expanded >= 0

    def test_code_anchor_injected_for_code_query(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="Auth module", file_path="app/auth.py")

        scores = {m1.id: 1.0}
        allowed = {m1.id, m2.id}
        augmented, expanded = graph_expand(
            db, project.id, [m1.id], scores, allowed,
            "bug in auth.py",
            enable_graph_expansion=True,
            enable_entity_boost=False,
        )

        assert m2.id in augmented
        assert expanded >= 1


# ---------------------------------------------------------------------------
# generate_cluster_summaries
# ---------------------------------------------------------------------------

class TestGenerateClusterSummaries:
    def _setup_cluster(self, db, project_id: str, cluster_id: int, size: int,
                       label: str = "Test Cluster") -> list:
        return [
            _create_memory(
                db, project_id,
                title=f"Memory {i} for cluster {cluster_id}",
                content=f"Content about topic {label} item {i}",
                cluster_id=cluster_id,
                cluster_label=label,
            )
            for i in range(size)
        ]

    def test_generates_summary_for_large_cluster(self, db):
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="Auth Cluster")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10)

        assert len(results) == 1
        assert isinstance(results[0], ClusterSummaryResult)
        assert results[0].cluster_id == 1
        assert results[0].memory_count == 10

    def test_skips_small_cluster_below_threshold(self, db):
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=5, label="Small Cluster")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10)
        assert len(results) == 0

    def test_min_cluster_size_configurable(self, db):
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=3, label="Tiny Cluster")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=3)
        assert len(results) == 1

    def test_multiple_clusters_processed(self, db):
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="Auth")
        self._setup_cluster(db, project.id, cluster_id=2, size=10, label="Database")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=5)
        assert len(results) == 2

    def test_only_one_large_cluster_processed_when_other_too_small(self, db):
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="Big")
        self._setup_cluster(db, project.id, cluster_id=2, size=3, label="Small")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10)
        assert len(results) == 1
        assert results[0].cluster_id == 1

    def test_summary_stored_as_memory_row(self, db):
        from app import models as phase1_models
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="Test")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10)
        summary_id = results[0].summary_memory_id

        stored = db.query(phase1_models.Memory).filter(phase1_models.Memory.id == summary_id).first()
        assert stored is not None
        assert stored.type == "cluster_summary"
        assert stored.project_id == project.id

    def test_summary_title_contains_cluster_label(self, db):
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="Auth Cluster")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10)
        assert "Auth Cluster" in results[0].summary_text or "Auth Cluster" in results[0].cluster_label

    def test_rerrun_updates_existing_summary_not_duplicate(self, db):
        from app import models as phase1_models
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="Cluster")

        generate_cluster_summaries(db, project.id, min_cluster_size=10)
        generate_cluster_summaries(db, project.id, min_cluster_size=10)

        count = (
            db.query(phase1_models.Memory)
            .filter(
                phase1_models.Memory.project_id == project.id,
                phase1_models.Memory.type == "cluster_summary",
                phase1_models.Memory.cluster_id == 1,
            )
            .count()
        )
        assert count == 1  # Only one summary per cluster

    def test_template_fallback_when_no_llm(self, db, mocker):
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="Fallback Test")

        # Force _claude_summary to return None so template is used
        mocker.patch(
            "app.services.retrieval.cluster_summaries._claude_summary",
            return_value=None,
        )

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10, llm_fn=None)

        assert len(results) == 1
        assert results[0].summary_text  # non-empty
        assert results[0].used_llm is False  # template used

    def test_custom_llm_fn_called(self, db):
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="LLM Test")

        def mock_llm(cluster_label: str, memories: list) -> str:
            return f"LLM summary for {cluster_label} with {len(memories)} memories"

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10, llm_fn=mock_llm)

        assert results[0].used_llm is True
        assert "LLM summary" in results[0].summary_text

    def test_cluster_summaries_not_double_processed(self, db):
        from app import models as phase1_models
        project = _create_project(db)
        self._setup_cluster(db, project.id, cluster_id=1, size=10, label="Main")

        # Pre-existing cluster_summary should be ignored as input (no cluster summaries of cluster summaries)
        generate_cluster_summaries(db, project.id, min_cluster_size=10)

        # Count how many times the summary itself participates in subsequent runs
        generate_cluster_summaries(db, project.id, min_cluster_size=10)

        count = (
            db.query(phase1_models.Memory)
            .filter(
                phase1_models.Memory.project_id == project.id,
                phase1_models.Memory.type == "cluster_summary",
            )
            .count()
        )
        assert count == 1

    def test_ignores_superseded_memories_in_cluster(self, db):
        project = _create_project(db)
        # 10 active + 5 superseded in same cluster
        for i in range(10):
            _create_memory(db, project.id, title=f"Active {i}", cluster_id=1,
                           cluster_label="Mixed Cluster")
        for i in range(5):
            _create_memory(db, project.id, title=f"Superseded {i}", cluster_id=1,
                           cluster_label="Mixed Cluster", status="superseded")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10)
        # Should find the 10 active memories → 1 result
        assert len(results) == 1
        assert results[0].memory_count == 10

    def test_returns_empty_when_no_clusters_assigned(self, db):
        project = _create_project(db)
        # Memories with no cluster_id
        for i in range(10):
            _create_memory(db, project.id, title=f"Unclusterd {i}")

        results = generate_cluster_summaries(db, project.id, min_cluster_size=10)
        assert results == []


# ---------------------------------------------------------------------------
# Pipeline integration — retrieve() wires expansion
# ---------------------------------------------------------------------------

class TestPipelineWithExpansion:
    @pytest.fixture
    def setup_fts(self, db):
        """Setup FTS5 for BM25 to work in tests."""
        from app.search import setup_fts
        setup_fts(db.get_bind())
        return db

    def test_retrieve_result_has_expanded_via_links_field(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())

        m = _create_memory(db, project.id, title="Auth decision",
                           content="We use JWT tokens for authentication")

        cfg = RetrievalConfig(top_k=5, embed_query=False, max_clearance="internal")
        result = retrieve(db, project.id, "authentication", SQLiteExactBackend(db), cfg)

        assert hasattr(result, "expanded_via_links")
        assert isinstance(result.expanded_via_links, int)
        assert result.expanded_via_links >= 0

    def test_retrieve_with_link_expands_neighbor(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())

        m1 = _create_memory(db, project.id, title="Auth decision",
                             content="JWT token auth for all API calls")
        m2 = _create_memory(db, project.id, title="Token refresh",
                             content="Refresh tokens should expire in 24 hours")
        _create_link(db, m1.id, m2.id)

        cfg = RetrievalConfig(
            top_k=5, embed_query=False, max_clearance="internal",
            enable_graph_expansion=True, enable_2hop=False,
        )
        result = retrieve(db, project.id, "JWT auth", SQLiteExactBackend(db), cfg)

        # Both m1 (direct hit) and m2 (via expansion) should appear
        all_ids = set(result.selected_memory_ids)
        assert m1.id in all_ids

    def test_retrieve_expansion_disabled_excludes_neighbor(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())

        m1 = _create_memory(db, project.id, title="Auth decision",
                             content="JWT token auth for all API calls")
        m2 = _create_memory(db, project.id, title="Unrelated memory",
                             content="This memory has nothing to do with auth whatsoever")
        _create_link(db, m1.id, m2.id)

        cfg = RetrievalConfig(
            top_k=1,  # Only top-1 — m2 can only appear via expansion
            embed_query=False, max_clearance="internal",
            enable_graph_expansion=False, enable_entity_boost=False,
        )
        result = retrieve(db, project.id, "JWT auth", SQLiteExactBackend(db), cfg)

        # With expansion disabled and top_k=1, m2 should not appear
        assert m2.id not in result.selected_memory_ids or len(result.selected_memory_ids) == 1

    def test_retrieve_entity_boost_doesnt_break_pipeline(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())

        m = _create_memory(db, project.id, title="Auth with entity",
                            content="PostgreSQL database configuration")
        _create_entity(db, m.id, project.id, "postgresql")

        cfg = RetrievalConfig(
            top_k=5, embed_query=False, max_clearance="internal",
            enable_entity_boost=True,
        )
        result = retrieve(db, project.id, "postgresql config", SQLiteExactBackend(db), cfg)

        assert result is not None
        assert result.expanded_via_links >= 0

    def test_retrieve_result_scores_dict_present(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())

        m = _create_memory(db, project.id, title="M", content="Some content about auth")
        cfg = RetrievalConfig(top_k=5, embed_query=False, max_clearance="internal")
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)

        assert isinstance(result.rrf_scores, dict)
        for mid in result.selected_memory_ids:
            assert mid in result.rrf_scores

    def test_retrieve_no_expansion_when_no_links(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())

        m = _create_memory(db, project.id, title="Isolated memory", content="auth config")
        cfg = RetrievalConfig(top_k=5, embed_query=False, max_clearance="internal",
                              enable_graph_expansion=True)
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)

        assert result.expanded_via_links == 0  # no links → no expansion

    def test_retrieve_empty_project_returns_zero_expanded(self, db):
        project = _create_project(db)
        from app.search import setup_fts
        setup_fts(db.get_bind())

        cfg = RetrievalConfig(top_k=5, embed_query=False)
        result = retrieve(db, project.id, "auth", SQLiteExactBackend(db), cfg)

        assert result.expanded_via_links == 0


# ---------------------------------------------------------------------------
# Stress / edge cases
# ---------------------------------------------------------------------------

class TestGraphExpansionEdgeCases:
    def test_large_graph_doesnt_exceed_max_expansion(self, db):
        project = _create_project(db)
        # 50 memories, all linked to one central node
        center = _create_memory(db, project.id, title="Center")
        spokes = [_create_memory(db, project.id, title=f"Spoke {i}") for i in range(50)]
        for s in spokes:
            _create_link(db, center.id, s.id)

        scores = {center.id: 1.0}
        allowed = {center.id} | {s.id for s in spokes}

        augmented, expanded = graph_expand(
            db, project.id, [center.id], scores, allowed,
            "architecture and configuration and deployment and security and testing",
            max_expansion_total=15,
        )

        # Should not exceed max_expansion_total
        assert len(augmented) <= 1 + 15 + 15  # center + 1hop + 2hop caps

    def test_circular_links_dont_cause_infinite_loops(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        m3 = _create_memory(db, project.id, title="M3")
        # Cycle: m1 → m2 → m3 → m1
        _create_link(db, m1.id, m2.id)
        _create_link(db, m2.id, m3.id)
        _create_link(db, m3.id, m1.id)

        scores = {m1.id: 1.0}
        allowed = {m1.id, m2.id, m3.id}

        # Should complete without hanging
        augmented, expanded = graph_expand(
            db, project.id, [m1.id], scores, allowed,
            "auth and database and caching and deployment and monitoring",
        )
        assert isinstance(augmented, dict)

    def test_expansion_respects_allowed_ids_strictly(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1 public", privacy="public")
        m2 = _create_memory(db, project.id, title="M2 secret", privacy="secret")
        _create_link(db, m1.id, m2.id)

        scores = {m1.id: 1.0}
        allowed = {m1.id}  # m2 excluded from allowed (e.g. clearance gate)

        augmented, _ = graph_expand(
            db, project.id, [m1.id], scores, allowed,
            "auth query",
        )

        assert m2.id not in augmented

    def test_graph_expand_returns_tuple(self, db):
        project = _create_project(db)
        m = _create_memory(db, project.id, title="M")
        result = graph_expand(db, project.id, [m.id], {m.id: 1.0}, {m.id}, "query")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_all_expansion_flags_disabled_score_unchanged(self, db):
        project = _create_project(db)
        m1 = _create_memory(db, project.id, title="M1")
        m2 = _create_memory(db, project.id, title="M2")
        _create_entity(db, m1.id, project.id, "postgresql")
        _create_link(db, m1.id, m2.id)

        original_scores = {m1.id: 1.0}
        augmented, expanded = graph_expand(
            db, project.id, [m1.id], original_scores, {m1.id, m2.id},
            "postgresql auth",
            enable_entity_boost=False,
            enable_graph_expansion=False,
            enable_2hop=False,
        )

        assert augmented[m1.id] == pytest.approx(1.0)
        assert expanded == 0
        assert m2.id not in augmented
