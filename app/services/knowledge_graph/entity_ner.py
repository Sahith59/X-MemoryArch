"""
Phase 3.7 — Lightweight NER for knowledge graph construction.

Two-layer approach for maximum portability:
  Layer 1: spaCy en_core_web_sm (used if available — best quality)
  Layer 2: Regex + keyword patterns (always-available fallback)

The benchmark runs in the system Python environment where spaCy may not be
installed. This module detects and adapts gracefully, so A8r works without
any venv switching.

Public API:
    extract_entities(text) -> list[EntityMention]
    EntityMention: (text, normalized, entity_type)

Entity types: PERSON | ORG | TECH | PRODUCT | CONCEPT | GPE | OTHER
"""
from __future__ import annotations

import re
from typing import NamedTuple


class EntityMention(NamedTuple):
    text:        str   # surface form
    normalized:  str   # lowercase, stripped — used as graph node key
    entity_type: str   # PERSON | ORG | TECH | PRODUCT | CONCEPT | GPE | OTHER


# ── spaCy layer (optional) ────────────────────────────────────────────────────

_spacy_nlp = None
_spacy_available: bool | None = None

def _try_load_spacy():
    global _spacy_nlp, _spacy_available
    if _spacy_available is not None:
        return _spacy_available
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
        _spacy_available = True
    except Exception:
        _spacy_available = False
    return _spacy_available

# spaCy label → our EntityType mapping
_SPACY_LABEL_MAP = {
    "PERSON":     "PERSON",
    "ORG":        "ORG",
    "PRODUCT":    "PRODUCT",
    "GPE":        "GPE",        # geopolitical: countries, cities, states
    "LOC":        "GPE",
    "FAC":        "ORG",        # facilities (hospitals, airports) → treat as ORG
    "LANGUAGE":   "TECH",
    "EVENT":      "CONCEPT",
    "WORK_OF_ART":"CONCEPT",
    "LAW":        "CONCEPT",
}


def _extract_spacy(text: str) -> list[EntityMention]:
    """Extract entities using spaCy en_core_web_sm."""
    doc = _spacy_nlp(text[:5000])  # cap at 5K chars for speed
    seen: set[str] = set()
    results: list[EntityMention] = []
    for ent in doc.ents:
        norm = ent.text.lower().strip()
        if len(norm) < 2 or norm in seen:
            continue
        etype = _SPACY_LABEL_MAP.get(ent.label_, "OTHER")
        seen.add(norm)
        results.append(EntityMention(text=ent.text.strip(), normalized=norm, entity_type=etype))
    return results


# ── Regex fallback layer ──────────────────────────────────────────────────────

# Known technology terms (subset — catches most in benchmark datasets)
_TECH_TERMS: frozenset[str] = frozenset({
    # Databases
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
    "qdrant", "faiss", "chroma", "pinecone", "neo4j", "dynamodb", "cassandra",
    # Frameworks / Languages
    "python", "javascript", "typescript", "golang", "rust", "java", "kotlin",
    "fastapi", "flask", "django", "react", "vue", "angular", "nextjs", "express",
    "sqlalchemy", "pydantic", "celery", "graphql", "grpc", "openapi",
    # Tools / Platforms
    "docker", "kubernetes", "terraform", "ansible", "nginx", "apache",
    "github", "gitlab", "jenkins", "circleci", "aws", "gcp", "azure", "vercel",
    "anthropic", "openai", "claude", "chatgpt", "gemini", "ollama",
    "huggingface", "pytorch", "tensorflow", "sklearn",
    # Auth / Security
    "jwt", "oauth", "oauth2", "saml", "ldap", "ssl", "tls",
    # Protocols
    "rest", "graphql", "websocket", "http", "https", "grpc",
    # Xmem-specific
    "gte-large", "gte_large", "bm25", "faiss", "qdrant", "sqlite-vec",
})

# Patterns for explicit relations (PERSON works at/uses/knows ENTITY)
_WORKS_AT_PATTERNS = [
    re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:works?|worked|working)\s+(?:at|for|with)\s+([A-Z][A-Za-z\s&]+)', re.IGNORECASE),
    re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is|was|are|were)\s+(?:a|an)?\s*(?:employee|member|part|staff)\s+(?:of|at)\s+([A-Z][A-Za-z\s&]+)', re.IGNORECASE),
    re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:joined|joins|joining)\s+([A-Z][A-Za-z\s&]+)', re.IGNORECASE),
]

_USES_PATTERNS = [
    re.compile(r'\b(we|they|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:use|uses|used|using|chose|choose|decided|picked)\s+([A-Za-z][\w.-]+(?:\s+[A-Za-z][\w.-]+)?)', re.IGNORECASE),
    re.compile(r'\bswitch(?:ed|ing)?\s+to\s+([A-Za-z][\w.-]+)', re.IGNORECASE),
    re.compile(r'\bmigrat(?:e|ed|ing)\s+to\s+([A-Za-z][\w.-]+)', re.IGNORECASE),
]


def _extract_regex(text: str) -> list[EntityMention]:
    """Regex + pattern-based entity extraction — spaCy-free fallback."""
    seen: set[str] = set()
    results: list[EntityMention] = []

    def _add(surface: str, etype: str):
        norm = surface.lower().strip().strip("'\".,;:")
        if len(norm) < 2 or norm in seen:
            return
        # Skip pure stopwords
        if norm in {"the", "a", "an", "this", "that", "it", "we", "they",
                    "he", "she", "his", "her", "our", "their", "its", "i",
                    "you", "me", "us", "him", "who", "what", "when", "where",
                    "how", "why", "is", "are", "was", "were", "be", "been"}:
            return
        seen.add(norm)
        results.append(EntityMention(text=surface.strip(), normalized=norm, entity_type=etype))

    # ── Known tech terms ──────────────────────────────────────────────────────
    text_lower = text.lower()
    for term in _TECH_TERMS:
        # Match as whole word
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            # Find the actual casing in the text
            m = re.search(r'\b' + re.escape(term) + r'\b', text_lower)
            if m:
                surface = text[m.start():m.end()]
                _add(surface, "TECH")

    # ── Named entities: single and multi-word capitalized sequences ──────────
    # Pattern 1: single capitalized word preceded by a lowercase word (mid-sentence names)
    #   "did Caroline go" → "Caroline"; "met Melanie" → "Melanie"
    single_mid = re.findall(
        r'(?<=[a-z\']\s)([A-Z][a-z]{2,20})(?=[\s,.\'\!?]|$)',
        text,
    )
    for name in single_mid:
        norm = name.lower()
        if norm not in _TECH_TERMS and name not in {
            "The", "This", "That", "These", "Those", "They", "Their",
            "What", "When", "Where", "Why", "How", "Which",
        }:
            _add(name, "PERSON")

    # Pattern 2: single capitalized first name at sentence start (followed by verb/space)
    first_names = re.findall(
        r'(?:^|(?<=[.!?\n]))\s*([A-Z][a-z]{2,20})(?=[\s,\'\!])',
        text,
    )
    for name in first_names:
        norm = name.lower()
        if norm not in _TECH_TERMS and name not in {
            "The", "This", "That", "What", "When", "Where", "Why", "How",
            "Which", "Who", "He", "She", "They", "We", "It",
        }:
            _add(name, "PERSON")

    # Pattern 3: multi-word capitalized sequences (ORG / GPE / PERSON)
    multi_cap = re.findall(
        r'([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})',
        text,
    )
    for phrase in multi_cap:
        phrase = phrase.strip()
        phrase_lower = phrase.lower()
        if phrase_lower in _TECH_TERMS:
            continue
        words = phrase.split()
        org_suffixes = {"hospital", "university", "institute", "corp", "inc",
                        "ltd", "llc", "bank", "company", "clinic", "school",
                        "college", "lab", "labs", "group", "ventures",
                        "foundation", "services", "technologies", "center",
                        "centre", "academy", "community", "committee"}
        gpe_indicators = {"city", "new", "los", "san", "north", "south",
                          "east", "west", "central", "port", "mount"}
        if any(w.lower() in org_suffixes for w in words):
            etype = "ORG"
        elif any(w.lower() in gpe_indicators for w in words):
            etype = "GPE"
        else:
            etype = "PERSON"
        _add(phrase, etype)

    return results


# ── Explicit relation extraction ──────────────────────────────────────────────

class RelationMention(NamedTuple):
    subject:       str   # normalized entity text
    relation_type: str   # WORKS_AT | USES | KNOWS | DECIDED_ON
    object_:       str   # normalized entity text
    confidence:    float


def extract_relations(text: str) -> list[RelationMention]:
    """
    Extract explicit typed relations from text using regex patterns.
    Used to create high-confidence (non-CO_OCCURS) edges in the graph.
    """
    results: list[RelationMention] = []

    for pat in _WORKS_AT_PATTERNS:
        for m in pat.finditer(text):
            subj = m.group(1).strip().lower()
            obj  = m.group(2).strip().rstrip(".").lower()
            if len(subj) > 1 and len(obj) > 1:
                results.append(RelationMention(
                    subject=subj, relation_type="WORKS_AT",
                    object_=obj, confidence=0.85,
                ))

    for pat in _USES_PATTERNS:
        for m in pat.finditer(text):
            tech = m.group(m.lastindex or 1).strip().lower()
            if tech in _TECH_TERMS or len(tech) > 2:
                subj = "project"   # generic subject when not specified
                if m.group(1).lower() not in {"we", "they"}:
                    subj = m.group(1).strip().lower()
                results.append(RelationMention(
                    subject=subj, relation_type="USES",
                    object_=tech, confidence=0.80,
                ))

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> list[EntityMention]:
    """
    Extract named entities from text using the best available method.

    Uses spaCy en_core_web_sm if available (best quality).
    Falls back to regex + keyword patterns if spaCy is not installed.

    Returns deduplicated list of EntityMention(text, normalized, entity_type).
    """
    if not text or not text.strip():
        return []

    if _try_load_spacy():
        try:
            spacy_results = _extract_spacy(text)
            # Supplement with tech terms that spaCy misses
            regex_results = _extract_regex(text)
            # Merge: prefer spaCy, add regex TECH terms not found by spaCy
            spacy_norms = {e.normalized for e in spacy_results}
            merged = list(spacy_results)
            for e in regex_results:
                if e.entity_type == "TECH" and e.normalized not in spacy_norms:
                    merged.append(e)
            return merged
        except Exception:
            pass

    return _extract_regex(text)
