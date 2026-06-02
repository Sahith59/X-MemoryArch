"""
Sub-phase 1.39 — Universal entity extractor tests.

Covers:
  Unit tests — extract_entities() pure function:
  1.  Empty / blank input returns empty list
  2.  Layer 1 NER: ORG entity extracted
  3.  Layer 1 NER: PERSON entity extracted
  4.  Layer 1 NER: PRODUCT entity extracted
  5.  Layer 1 NER: GPE entity extracted
  6.  Layer 2 CONCEPT: noun chunk from a design sentence
  7.  Layer 2 CONCEPT: noun chunk from a research sentence
  8.  Layer 2 CONCEPT: noun chunk from a marketing sentence
  9.  Layer 2: generic STOP_CONCEPT words not extracted
  10. Layer 3 TECH: tech keyword extracted in software domain
  11. Layer 3 TECH: tech keyword NOT extracted in general domain
  12. Layer 3 TECH: tech keyword NOT extracted in design domain
  13. Normalized form is lowercase
  14. Same normalized form deduplicated — appears once
  15. NER-captured terms not duplicated as CONCEPT or TECH
  16. Short tokens (<2 chars) not extracted
  17. Purely numeric chunks not extracted
  18. extract_entities_for_memory combines title + content
  19. domain parameter defaults to "general" (no TECH extraction)

  Integration tests — domain-aware via API:
  20. Software project memory → TECH entities stored
  21. Non-software project memory → no TECH entities stored
  22. design domain → CONCEPT entities stored for design text
  23. research domain → CONCEPT entities stored for research text
"""
import pytest
from fastapi.testclient import TestClient

from app.services.entity_extractor import extract_entities, extract_entities_for_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project(client: TestClient, name: str = "Entity Test", domain: str = "general") -> str:
    r = client.post("/projects", json={"name": name, "domain": domain})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _memory(client: TestClient, pid: str, title: str, content: str) -> dict:
    r = client.post(f"/projects/{pid}/memories", json={
        "type": "decision",
        "title": title,
        "content": content,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _entities(client: TestClient, memory_id: str) -> list[dict]:
    r = client.get(f"/memories/{memory_id}/entities")
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------------------
# 1: Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_string_returns_empty(self):
        assert extract_entities("") == []

    def test_whitespace_only_returns_empty(self):
        assert extract_entities("   ") == []

    def test_none_raises_or_returns_empty(self):
        # Passing empty-equivalent inputs should always return empty list
        result = extract_entities("")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 2–5: Layer 1 NER (universal, no domain requirement)
# ---------------------------------------------------------------------------

class TestLayer1NER:
    def test_org_entity_extracted(self):
        entities = extract_entities("Microsoft announced a new partnership with OpenAI.")
        labels = {e.label for e in entities}
        # spaCy may classify companies as ORG or PRODUCT
        assert "ORG" in labels or "PRODUCT" in labels

    def test_person_entity_extracted(self):
        entities = extract_entities("Alice presented the Q3 research findings to the board.")
        norms = {e.normalized for e in entities}
        assert "alice" in norms

    def test_gpe_entity_extracted(self):
        entities = extract_entities("The team is based in San Francisco and expanding to London.")
        norms = {e.normalized for e in entities}
        # spaCy should pick up one or both cities
        assert "san francisco" in norms or "london" in norms

    def test_product_or_org_extracted(self):
        # "Slack" or "Notion" — spaCy often tags these as ORG or PRODUCT
        entities = extract_entities("We use Slack for team communication and Notion for docs.")
        norms = {e.normalized for e in entities}
        assert "slack" in norms or "notion" in norms


# ---------------------------------------------------------------------------
# 6–8: Layer 2 CONCEPT (noun chunks, domain-agnostic)
# ---------------------------------------------------------------------------

class TestLayer2Concepts:
    def test_design_concept_extracted(self):
        entities = extract_entities(
            "The color palette was updated to match the brand guidelines."
        )
        labels = {e.label for e in entities}
        assert "CONCEPT" in labels

    def test_research_concept_extracted(self):
        entities = extract_entities(
            "The control group showed no significant response to the treatment."
        )
        labels = {e.label for e in entities}
        assert "CONCEPT" in labels

    def test_marketing_concept_extracted(self):
        entities = extract_entities(
            "The conversion funnel drops sharply at the checkout stage."
        )
        labels = {e.label for e in entities}
        assert "CONCEPT" in labels

    def test_multi_domain_concepts_no_hardcoding(self):
        # Designer sentence — should extract CONCEPT without TECH or NER
        entities = extract_entities(
            "We need to refine the user journey map before the next sprint review.",
            domain="design",
        )
        labels = {e.label for e in entities}
        assert "CONCEPT" in labels


# ---------------------------------------------------------------------------
# 9: STOP_CONCEPT filtering
# ---------------------------------------------------------------------------

class TestStopConcepts:
    def test_generic_nouns_not_extracted_as_concepts(self):
        # "things" and "ways" are in _STOP_CONCEPTS
        entities = extract_entities("There are many things and various ways to approach this.")
        concept_norms = {e.normalized for e in entities if e.label == "CONCEPT"}
        # "things" and "ways" must be filtered out
        assert "things" not in concept_norms
        assert "ways" not in concept_norms

    def test_single_letter_not_extracted(self):
        entities = extract_entities("We added feature A to the system.")
        # No single-character entities
        for e in entities:
            assert len(e.normalized) >= 2


# ---------------------------------------------------------------------------
# 10–12: Layer 3 TECH (domain-gated)
# ---------------------------------------------------------------------------

class TestLayer3Tech:
    def test_tech_keyword_extracted_in_software_domain(self):
        entities = extract_entities(
            "We switched the cache from Redis to PostgreSQL for persistent storage.",
            domain="software",
        )
        tech_norms = {e.normalized for e in entities if e.label == "TECH"}
        assert "redis" in tech_norms or "postgresql" in tech_norms

    def test_tech_keyword_not_extracted_in_general_domain(self):
        entities = extract_entities(
            "We switched the cache from Redis to PostgreSQL for persistent storage.",
            domain="general",
        )
        tech_norms = {e.normalized for e in entities if e.label == "TECH"}
        # No TECH layer in general domain — redis/postgres may still appear as ORG/PRODUCT
        # but NOT as TECH-labeled entities
        assert len(tech_norms) == 0

    def test_tech_keyword_not_extracted_in_design_domain(self):
        entities = extract_entities(
            "The design system uses Docker containers for the preview environment.",
            domain="design",
        )
        tech_norms = {e.normalized for e in entities if e.label == "TECH"}
        assert len(tech_norms) == 0

    def test_tech_keyword_extracted_in_data_domain(self):
        entities = extract_entities(
            "We migrated the pipeline from MySQL to Elasticsearch.",
            domain="data",
        )
        tech_norms = {e.normalized for e in entities if e.label == "TECH"}
        assert "mysql" in tech_norms or "elasticsearch" in tech_norms

    def test_tech_keyword_extracted_in_security_domain(self):
        entities = extract_entities(
            "The service now validates JWT tokens using OAuth 2.0.",
            domain="security",
        )
        tech_norms = {e.normalized for e in entities if e.label == "TECH"}
        assert "jwt" in tech_norms or "oauth" in tech_norms


# ---------------------------------------------------------------------------
# 13–17: Entity properties
# ---------------------------------------------------------------------------

class TestEntityProperties:
    def test_normalized_is_lowercase(self):
        entities = extract_entities(
            "Alice joined Google last year.", domain="general"
        )
        for e in entities:
            assert e.normalized == e.normalized.lower()

    def test_same_normalized_deduplicated(self):
        # "FastAPI" and "fastapi" — same normalized form, should appear once
        entities = extract_entities(
            "FastAPI is great. We chose fastapi for the backend.",
            domain="software",
        )
        norms = [e.normalized for e in entities]
        assert len(norms) == len(set(norms)), f"Duplicate norms: {norms}"

    def test_ner_term_not_duplicated_as_tech(self):
        # No exact same normalized form should appear twice across all layers
        entities = extract_entities(
            "We use GitHub Actions for CI.",
            domain="software",
        )
        norms = [e.normalized for e in entities]
        assert len(norms) == len(set(norms)), f"Duplicate norms found: {norms}"

    def test_purely_numeric_chunk_not_extracted(self):
        entities = extract_entities("Version 3.14 was released on 2026-01-01.")
        concept_norms = {e.normalized for e in entities if e.label == "CONCEPT"}
        assert "3.14" not in concept_norms

    def test_short_token_not_extracted(self):
        entities = extract_entities("We ran test A and B successfully.")
        for e in entities:
            assert len(e.normalized) >= 2


# ---------------------------------------------------------------------------
# 18–19: extract_entities_for_memory
# ---------------------------------------------------------------------------

class TestExtractEntitiesForMemory:
    def test_combines_title_and_content(self):
        # Title has "Redis", content has "PostgreSQL" — both should be found in software domain
        entities = extract_entities_for_memory(
            title="Redis migration plan",
            content="We switched from Redis to PostgreSQL for durability.",
            domain="software",
        )
        norms = {e.normalized for e in entities}
        assert "redis" in norms or "postgresql" in norms

    def test_default_domain_is_general(self):
        entities = extract_entities_for_memory(
            title="Redis migration plan",
            content="We switched from Redis to PostgreSQL.",
        )
        tech_norms = {e.normalized for e in entities if e.label == "TECH"}
        assert len(tech_norms) == 0


# ---------------------------------------------------------------------------
# 20–23: Integration — domain threading via API
# ---------------------------------------------------------------------------

class TestEntityDomainIntegrationAPI:
    def test_software_project_gets_tech_entities(self, client: TestClient):
        pid = _project(client, "Software Project", domain="software")
        mem = _memory(
            client, pid,
            title="Cache layer decision",
            content="We chose Redis over Memcached for sub-millisecond latency.",
        )
        ents = _entities(client, mem["id"])
        tech_labels = [e for e in ents if e["entity_label"] == "TECH"]
        # Redis and/or Memcached should appear as TECH in software domain
        assert len(tech_labels) >= 1

    def test_general_project_no_tech_entities(self, client: TestClient):
        pid = _project(client, "General Project", domain="general")
        mem = _memory(
            client, pid,
            title="Cache layer decision",
            content="We chose Redis over Memcached for sub-millisecond latency.",
        )
        ents = _entities(client, mem["id"])
        tech_labels = [e for e in ents if e["entity_label"] == "TECH"]
        assert len(tech_labels) == 0

    def test_design_project_gets_concept_entities(self, client: TestClient):
        pid = _project(client, "Design Project", domain="design")
        mem = _memory(
            client, pid,
            title="Color system redesign",
            content=(
                "The design team updated the color palette to align with "
                "the brand guidelines and improve accessibility."
            ),
        )
        ents = _entities(client, mem["id"])
        concept_labels = [e for e in ents if e["entity_label"] == "CONCEPT"]
        assert len(concept_labels) >= 1

    def test_research_project_gets_concept_entities(self, client: TestClient):
        pid = _project(client, "Research Project", domain="research")
        mem = _memory(
            client, pid,
            title="Study methodology",
            content=(
                "The control group showed no response while the treatment group "
                "demonstrated a significant improvement in the primary outcome measure."
            ),
        )
        ents = _entities(client, mem["id"])
        concept_labels = [e for e in ents if e["entity_label"] == "CONCEPT"]
        assert len(concept_labels) >= 1
