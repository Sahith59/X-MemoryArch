"""
Phase 4.2 tests — EntityRegistry and QueryEntityExtractor.

Covers:
  EntityRegistry:
    - Name normalization (titles, punctuation)
    - get_or_create (exact + fuzzy + new)
    - Prefix token merging ("Alice" → "Alice Johnson")
    - Multi-word fuzzy merge ("Alice Johnson" / "Alice Jonson")
    - register_session + sessions_for_entity
    - aliases()
    - merge() (explicit)
    - find_by_query (partial name expansion)
    - save / load roundtrip
    - build_from_triples (factory)
    - stats()

  QueryEntityExtractor:
    - extract() with entity found in registry
    - extract() with partial name (prefix expansion)
    - Intent: ENTITY_STATE detection
    - Intent: EPISODE detection
    - Intent: SEMANTIC fallback
    - Pronoun resolution via context
    - Regex fallback (no spaCy) returns registry-confirmed mentions

  Integration:
    - build_from_triples + extract end-to-end
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.services.knowledge_graph.entity_registry import (
    EntityRegistry,
    Entity,
    _normalize_name,
    _edit_distance,
    _is_fuzzy_name_match,
    _is_valid_entity_name,
)
from app.services.knowledge_graph.query_entity_extractor import (
    QueryEntityExtractor,
    QueryEntities,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_registry(*name_session_pairs: tuple[str, str]) -> EntityRegistry:
    """Build a registry from (name, session_id) pairs."""
    reg = EntityRegistry(dataset_name="test")
    for name, sid in name_session_pairs:
        reg.register_session(name, sid)
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_name
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_lowercase(self):
        assert _normalize_name("Alice") == "alice"

    def test_strips_trailing_punctuation(self):
        assert _normalize_name("Alice.") == "alice"
        assert _normalize_name("Alice,") == "alice"

    def test_strips_dr_title(self):
        assert _normalize_name("Dr. Smith") == "smith"
        assert _normalize_name("dr. Smith") == "smith"

    def test_strips_mr_title(self):
        assert _normalize_name("Mr. Johnson") == "johnson"

    def test_strips_mrs_title(self):
        assert _normalize_name("Mrs. Brown") == "brown"

    def test_strips_prof_title(self):
        assert _normalize_name("Prof. Chen") == "chen"

    def test_strips_ms_title(self):
        assert _normalize_name("Ms. Williams") == "williams"

    def test_preserves_multiword(self):
        assert _normalize_name("Alice Johnson") == "alice johnson"

    def test_empty_string(self):
        assert _normalize_name("") == ""

    def test_strips_surrounding_quotes(self):
        assert _normalize_name('"Alice"') == "alice"


# ─────────────────────────────────────────────────────────────────────────────
# _edit_distance
# ─────────────────────────────────────────────────────────────────────────────

class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("alice", "alice") == 0

    def test_single_insertion(self):
        assert _edit_distance("alice", "alic") == 1

    def test_single_substitution(self):
        assert _edit_distance("johnson", "jonson") == 1

    def test_transposition(self):
        assert _edit_distance("alice", "aliec") == 2

    def test_large_diff(self):
        assert _edit_distance("alice", "zzzzzzzzz") > 3


# ─────────────────────────────────────────────────────────────────────────────
# _is_fuzzy_name_match
# ─────────────────────────────────────────────────────────────────────────────

class TestFuzzyNameMatch:
    def test_exact_match(self):
        assert _is_fuzzy_name_match("alice", "alice")

    def test_prefix_single_token_forward(self):
        # "alice" is a prefix of "alice johnson"
        assert _is_fuzzy_name_match("alice", "alice johnson")

    def test_prefix_single_token_reverse(self):
        # "alice johnson" contains "alice" as prefix
        assert _is_fuzzy_name_match("alice johnson", "alice")

    def test_multi_word_fuzzy_last_token(self):
        # "alice johnson" vs "alice jonson" — same first, edit_distance(johnson,jonson)=1
        assert _is_fuzzy_name_match("alice johnson", "alice jonson")

    def test_no_match_different_first_name(self):
        # "alice johnson" vs "bob johnson" — different first name
        assert not _is_fuzzy_name_match("alice johnson", "bob johnson")

    def test_no_match_short_tokens(self):
        # "bob kim" vs "bob lee" — last tokens too short (< 4 chars)
        assert not _is_fuzzy_name_match("bob kim", "bob lee")

    def test_no_match_completely_different(self):
        assert not _is_fuzzy_name_match("alice johnson", "city hospital")


# ─────────────────────────────────────────────────────────────────────────────
# EntityRegistry — get_or_create
# ─────────────────────────────────────────────────────────────────────────────

class TestGetOrCreate:
    def test_creates_new_entity(self):
        reg = EntityRegistry()
        eid = reg.get_or_create("Alice")
        assert eid.startswith("ent_")
        ent = reg.get_entity(eid)
        assert ent is not None
        assert ent.canonical == "Alice"

    def test_returns_same_id_for_same_name(self):
        reg = EntityRegistry()
        eid1 = reg.get_or_create("Alice")
        eid2 = reg.get_or_create("Alice")
        assert eid1 == eid2

    def test_prefix_merge_short_into_long(self):
        reg = EntityRegistry()
        eid1 = reg.get_or_create("Alice")
        # "Alice Johnson" should merge with existing "Alice" entity
        eid2 = reg.get_or_create("Alice Johnson")
        assert eid1 == eid2

    def test_prefix_merge_long_into_short(self):
        reg = EntityRegistry()
        eid1 = reg.get_or_create("Alice Johnson")
        eid2 = reg.get_or_create("Alice")
        assert eid1 == eid2

    def test_canonical_upgrades_to_longer_form(self):
        reg = EntityRegistry()
        reg.get_or_create("Alice")
        eid = reg.get_or_create("Alice Johnson")
        ent = reg.get_entity(eid)
        # Canonical should be upgraded to the more complete form
        assert ent.canonical == "Alice Johnson"

    def test_title_stripped_before_lookup(self):
        reg = EntityRegistry()
        eid1 = reg.get_or_create("Smith")
        eid2 = reg.get_or_create("Dr. Smith")
        assert eid1 == eid2

    def test_empty_name_returns_empty_string(self):
        reg = EntityRegistry()
        assert reg.get_or_create("") == ""

    def test_different_entities_separate_ids(self):
        reg = EntityRegistry()
        eid1 = reg.get_or_create("Alice")
        eid2 = reg.get_or_create("Bob")
        assert eid1 != eid2


# ─────────────────────────────────────────────────────────────────────────────
# EntityRegistry — register_session + sessions_for_entity
# ─────────────────────────────────────────────────────────────────────────────

class TestRegisterSessionAndLookup:
    def test_register_and_retrieve(self):
        reg = make_registry(("Alice", "session_1"), ("Alice", "session_3"))
        sessions = reg.sessions_for_entity("Alice")
        assert "session_1" in sessions
        assert "session_3" in sessions

    def test_no_duplicate_sessions(self):
        reg = EntityRegistry()
        reg.register_session("Alice", "session_1")
        reg.register_session("Alice", "session_1")
        sessions = reg.sessions_for_entity("Alice")
        assert sessions.count("session_1") == 1

    def test_prefix_lookup(self):
        # Register "Alice Johnson", query by "Alice"
        reg = make_registry(("Alice Johnson", "session_5"), ("Alice Johnson", "session_9"))
        sessions = reg.sessions_for_entity("Alice")
        assert "session_5" in sessions
        assert "session_9" in sessions

    def test_unknown_entity_returns_empty(self):
        reg = make_registry(("Alice", "session_1"))
        assert reg.sessions_for_entity("Charlie") == []

    def test_multiple_entities_tracked_separately(self):
        reg = make_registry(
            ("Alice", "session_1"),
            ("Bob",   "session_2"),
            ("Alice", "session_3"),
        )
        alice_sessions = reg.sessions_for_entity("Alice")
        bob_sessions   = reg.sessions_for_entity("Bob")
        assert "session_1" in alice_sessions
        assert "session_3" in alice_sessions
        assert "session_2" in bob_sessions
        assert "session_2" not in alice_sessions


# ─────────────────────────────────────────────────────────────────────────────
# EntityRegistry — aliases
# ─────────────────────────────────────────────────────────────────────────────

class TestAliases:
    def test_aliases_includes_original(self):
        reg = EntityRegistry()
        eid = reg.get_or_create("Alice Johnson")
        assert "Alice Johnson" in reg.aliases(eid)

    def test_aliases_includes_merged_forms(self):
        reg = EntityRegistry()
        reg.get_or_create("Alice Johnson")
        eid = reg.get_or_create("Alice")
        # Both forms should be aliases
        aliases = reg.aliases(eid)
        assert any("alice" in a.lower() for a in aliases)

    def test_unknown_entity_id_returns_empty(self):
        reg = EntityRegistry()
        assert reg.aliases("ent_9999") == []


# ─────────────────────────────────────────────────────────────────────────────
# EntityRegistry — explicit merge
# ─────────────────────────────────────────────────────────────────────────────

class TestMerge:
    def test_merge_combines_sessions(self):
        reg = make_registry(
            ("Alice Smith", "session_1"),
            ("Alice Jones", "session_2"),  # different entity
        )
        eid_smith = reg.get_or_create("Alice Smith")
        eid_jones = reg.get_or_create("Alice Jones")
        # Force merge
        reg.merge(eid_smith, eid_jones)
        sessions = reg.sessions_for_entity("Alice Smith")
        assert "session_1" in sessions
        assert "session_2" in sessions

    def test_merge_removes_dropped_entity(self):
        reg = EntityRegistry()
        eid1 = reg.get_or_create("Alice")
        eid2 = reg.get_or_create("Bob")
        reg.merge(eid1, eid2)
        assert reg.get_entity(eid2) is None

    def test_merge_self_is_noop(self):
        reg = EntityRegistry()
        eid = reg.get_or_create("Alice")
        reg.merge(eid, eid)
        assert reg.get_entity(eid) is not None


# ─────────────────────────────────────────────────────────────────────────────
# EntityRegistry — find_by_query
# ─────────────────────────────────────────────────────────────────────────────

class TestFindByQuery:
    def test_exact_match(self):
        reg = make_registry(("Alice Johnson", "s1"))
        eids = reg.find_by_query("Alice Johnson")
        assert len(eids) == 1

    def test_prefix_expansion(self):
        reg = make_registry(("Alice Johnson", "s1"))
        eids = reg.find_by_query("Alice")
        assert len(eids) >= 1

    def test_reverse_prefix_expansion(self):
        reg = make_registry(("Alice", "s1"))
        eids = reg.find_by_query("Alice Johnson")
        assert len(eids) >= 1

    def test_no_match_returns_empty(self):
        reg = make_registry(("Alice", "s1"))
        eids = reg.find_by_query("Charlie")
        assert eids == []

    def test_title_stripped_in_query(self):
        reg = make_registry(("Smith", "s1"))
        eids = reg.find_by_query("Dr. Smith")
        assert len(eids) == 1


# ─────────────────────────────────────────────────────────────────────────────
# EntityRegistry — save / load roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_sessions(self, tmp_path):
        reg = make_registry(
            ("Alice Johnson", "s1"),
            ("Alice Johnson", "s3"),
            ("Bob Smith",     "s2"),
        )
        path = tmp_path / "registry.json"
        reg.save(path)

        loaded = EntityRegistry.load(path)
        assert loaded.sessions_for_entity("Alice") == reg.sessions_for_entity("Alice")
        assert loaded.sessions_for_entity("Bob")   == reg.sessions_for_entity("Bob")

    def test_roundtrip_preserves_aliases(self, tmp_path):
        reg = EntityRegistry()
        reg.get_or_create("Alice Johnson")
        reg.get_or_create("Alice")
        path = tmp_path / "registry.json"
        reg.save(path)

        loaded = EntityRegistry.load(path)
        eids = loaded.find_by_query("Alice")
        assert len(eids) >= 1

    def test_roundtrip_counter_continues(self, tmp_path):
        reg = EntityRegistry()
        reg.get_or_create("Alice")
        reg.get_or_create("Bob")
        path = tmp_path / "registry.json"
        reg.save(path)

        loaded = EntityRegistry.load(path)
        # New entity after load should not collide with existing IDs
        new_eid = loaded.get_or_create("Charlie")
        assert new_eid not in [e.id for e in reg._entities.values()]


# ─────────────────────────────────────────────────────────────────────────────
# EntityRegistry — build_from_triples
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFromTriples:
    @pytest.fixture
    def sample_triples(self):
        return {
            "session_1": [
                {"subject": "Alice", "relation": "WORKS_AT", "object": "City Hospital",
                 "detail": "as a nurse", "text": "Alice works at City Hospital", "temporal": "current"},
                {"subject": "Alice", "relation": "HOBBY", "object": "hiking",
                 "detail": "", "text": "Alice enjoys hiking", "temporal": "permanent"},
            ],
            "session_2": [
                {"subject": "Alice", "relation": "KNOWS", "object": "Bob",
                 "detail": "", "text": "Alice knows Bob", "temporal": "permanent"},
                {"subject": "Bob", "relation": "LIVES_AT", "object": "Seattle",
                 "detail": "", "text": "Bob lives in Seattle", "temporal": "current"},
            ],
            "session_3": [
                {"subject": "Alice", "relation": "WORKS_AT", "object": "City Hospital",
                 "detail": "", "text": "Alice works at City Hospital", "temporal": "current"},
                {"subject": "Carol", "relation": "STUDIED_AT", "object": "MIT",
                 "detail": "", "text": "Carol studied at MIT", "temporal": "historical"},
            ],
        }

    def test_subjects_registered(self, sample_triples):
        reg = EntityRegistry.build_from_triples(sample_triples)
        alice_sessions = reg.sessions_for_entity("Alice")
        assert "session_1" in alice_sessions
        assert "session_2" in alice_sessions
        assert "session_3" in alice_sessions

    def test_object_of_state_relation_registered(self, sample_triples):
        reg = EntityRegistry.build_from_triples(sample_triples)
        # "City Hospital" is WORKS_AT object → should be in registry as ORG
        hospital_sessions = reg.sessions_for_entity("City Hospital")
        assert "session_1" in hospital_sessions

    def test_object_of_social_relation_registered(self, sample_triples):
        reg = EntityRegistry.build_from_triples(sample_triples)
        # "Bob" appears as KNOWS object → registered
        bob_sessions = reg.sessions_for_entity("Bob")
        assert "session_2" in bob_sessions

    def test_attribute_object_not_indexed(self, sample_triples):
        reg = EntityRegistry.build_from_triples(sample_triples)
        # "hiking" is HOBBY object → should NOT be indexed as entity
        hiking_sessions = reg.sessions_for_entity("hiking")
        assert hiking_sessions == []

    def test_stats_populated(self, sample_triples):
        reg = EntityRegistry.build_from_triples(sample_triples, dataset_name="Test")
        st = reg.stats()
        assert st["total_entities"] > 0
        assert st["sessions_covered"] == 3
        assert st["dataset"] == "Test"


# ─────────────────────────────────────────────────────────────────────────────
# QueryEntityExtractor — intent classification
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentClassification:
    @pytest.fixture
    def extractor(self):
        reg = make_registry(
            ("Alice Johnson", "s1"),
            ("Bob Smith",     "s2"),
        )
        return QueryEntityExtractor(reg)

    def test_entity_state_what_does(self, extractor):
        result = extractor.extract("What does Alice do for work?")
        assert result.intent == "ENTITY_STATE"
        assert result.intent_score > 0.5

    def test_entity_state_where_does(self, extractor):
        result = extractor.extract("Where does Bob live?")
        assert result.intent == "ENTITY_STATE"

    def test_entity_state_possessive(self, extractor):
        result = extractor.extract("What is Alice's job?")
        assert result.intent == "ENTITY_STATE"

    def test_entity_state_currently(self, extractor):
        result = extractor.extract("Is Alice currently still working at the hospital?")
        assert result.intent == "ENTITY_STATE"

    def test_episode_when_did(self, extractor):
        result = extractor.extract("When did Alice visit Paris?")
        assert result.intent == "EPISODE"

    def test_episode_what_hotel(self, extractor):
        result = extractor.extract("What hotel did they stay at during the Paris trip?")
        assert result.intent == "EPISODE"

    def test_episode_during_trip(self, extractor):
        result = extractor.extract("What did Alice do during the conference trip?")
        assert result.intent == "EPISODE"

    def test_semantic_factual(self, extractor):
        result = extractor.extract("What is photosynthesis?")
        assert result.intent == "SEMANTIC"

    def test_semantic_general(self, extractor):
        result = extractor.extract("Tell me about machine learning algorithms.")
        assert result.intent == "SEMANTIC"


# ─────────────────────────────────────────────────────────────────────────────
# QueryEntityExtractor — entity extraction + registry expansion
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryEntityExtraction:
    @pytest.fixture
    def registry(self):
        return make_registry(
            ("Alice Johnson", "s1"),
            ("Alice Johnson", "s3"),
            ("Bob Smith",     "s2"),
        )

    @pytest.fixture
    def extractor(self, registry):
        return QueryEntityExtractor(registry)

    def test_finds_registered_full_name(self, extractor):
        result = extractor.extract("What does Alice Johnson do?")
        assert len(result.entity_ids) >= 1

    def test_expands_partial_name(self, extractor):
        # "Alice" in query → should find "Alice Johnson" in registry
        result = extractor.extract("What does Alice do for work?")
        assert len(result.entity_ids) >= 1
        # Should return canonical "Alice Johnson"
        if result.entities:
            assert "alice" in result.entities[0].lower()

    def test_entity_ids_are_valid(self, extractor):
        result = extractor.extract("Where does Bob live?")
        for eid in result.entity_ids:
            assert extractor._registry.get_entity(eid) is not None

    def test_no_entity_in_factual_query(self, extractor):
        result = extractor.extract("What is the capital of France?")
        # Should find no registered entities
        assert len(result.entity_ids) == 0 or result.intent == "SEMANTIC"


# ─────────────────────────────────────────────────────────────────────────────
# QueryEntityExtractor — pronoun resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestPronounResolution:
    @pytest.fixture
    def extractor(self):
        reg = make_registry(("Alice Johnson", "s1"))
        return QueryEntityExtractor(reg)

    def test_pronoun_resolved_from_context(self, extractor):
        context = "User: What does Alice do? Agent: Alice works as a nurse."
        result = extractor.extract("Where does she live?", context=context)
        # Should find Alice from context
        assert len(result.raw_mentions) >= 1

    def test_no_pronoun_no_resolution(self, extractor):
        result = extractor.extract("What does Bob do?")
        # No context pronoun resolution needed
        assert result.intent is not None


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — end-to-end build + extract
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    @pytest.fixture
    def triple_facts(self):
        return {
            "c1_session_3": [
                {"subject": "Alice", "relation": "WORKS_AT", "object": "City Hospital",
                 "detail": "as a nurse", "text": "Alice works at City Hospital as a nurse",
                 "temporal": "current"},
                {"subject": "Alice", "relation": "HOBBY", "object": "gardening",
                 "detail": "", "text": "Alice enjoys gardening", "temporal": "permanent"},
                {"subject": "Alice", "relation": "KNOWS", "object": "Bob",
                 "detail": "from college", "text": "Alice knows Bob from college",
                 "temporal": "permanent"},
            ],
            "c1_session_7": [
                {"subject": "Alice", "relation": "WORKS_AT", "object": "City Hospital",
                 "detail": "", "text": "Alice works at City Hospital", "temporal": "current"},
                {"subject": "Bob", "relation": "LIVES_AT", "object": "Seattle",
                 "detail": "", "text": "Bob lives in Seattle", "temporal": "current"},
            ],
            "c2_session_1": [
                {"subject": "Carol", "relation": "STUDIES_AT", "object": "MIT",
                 "detail": "in computer science", "text": "Carol studies at MIT in computer science",
                 "temporal": "current"},
                {"subject": "Carol", "relation": "SKILL", "object": "Python",
                 "detail": "", "text": "Carol has the skill of Python", "temporal": "permanent"},
            ],
        }

    def test_entity_state_query_finds_correct_sessions(self, triple_facts):
        registry  = EntityRegistry.build_from_triples(triple_facts, dataset_name="Test")
        extractor = QueryEntityExtractor(registry)

        result = extractor.extract("What does Alice do for work?")
        assert result.intent == "ENTITY_STATE"
        assert len(result.entity_ids) >= 1

        # Retrieve sessions for Alice
        sessions = registry.sessions_for_entity("Alice")
        assert "c1_session_3" in sessions
        assert "c1_session_7" in sessions

    def test_partial_name_lookup_finds_sessions(self, triple_facts):
        registry = EntityRegistry.build_from_triples(triple_facts, dataset_name="Test")
        sessions = registry.sessions_for_entity("Bob")
        assert "c1_session_7" in sessions

    def test_org_entity_lookup(self, triple_facts):
        registry = EntityRegistry.build_from_triples(triple_facts, dataset_name="Test")
        sessions = registry.sessions_for_entity("City Hospital")
        assert "c1_session_3" in sessions
        assert "c1_session_7" in sessions

    def test_save_load_then_extract(self, triple_facts, tmp_path):
        registry = EntityRegistry.build_from_triples(triple_facts, dataset_name="Test")
        path = tmp_path / "registry.json"
        registry.save(path)

        loaded    = EntityRegistry.load(path)
        extractor = QueryEntityExtractor(loaded)
        result    = extractor.extract("Where does Bob live?")
        assert result.intent == "ENTITY_STATE"
        sessions = loaded.sessions_for_entity("Bob")
        assert "c1_session_7" in sessions
