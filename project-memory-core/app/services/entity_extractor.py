"""
Sub-phase 1.39 — Universal entity extraction.

Two-layer approach (domain-agnostic by default):

  Layer 1 — spaCy NER (already universal)
    Extracts named entities across all domains: ORG, PRODUCT, PERSON,
    GPE, EVENT, WORK_OF_ART, LAW, MONEY, LANGUAGE.

  Layer 2 — spaCy noun chunks (NEW — replaces hardcoded TECH_KEYWORDS)
    Extracts meaningful noun phrases from any domain:
    "color palette", "user journey", "sprint velocity", "control group",
    "conversion funnel", "PostgreSQL index", "brand guidelines" — all domains,
    zero hardcoding. Label: CONCEPT.

  Layer 3 — TECH keyword supplement (PRESERVED but domain-gated)
    Still extracted for projects where domain is in TECHNICAL_DOMAINS
    (software, data, security). These are high-precision signals for
    software projects that spaCy's general model consistently misses.
    Label: TECH.

The domain parameter gates Layer 3. Layers 1 and 2 always run.
"""
from __future__ import annotations
from typing import NamedTuple

_nlp = None

# spaCy entity labels worth keeping — expanded for universal coverage
_USEFUL_NER_LABELS = {
    "ORG",          # organizations, companies, teams
    "PRODUCT",      # named products, software, services
    "PERSON",       # people
    "LANGUAGE",     # programming or natural languages
    "GPE",          # geopolitical entities — countries, cities
    "EVENT",        # named events, conferences, releases
    "WORK_OF_ART",  # documents, books, publications, artworks
    "LAW",          # regulations, policies, legal references
    "MONEY",        # financial amounts and currencies
}

# Domains where the TECH keyword supplement adds value
_TECHNICAL_DOMAINS = {"software", "data", "security"}

# Generic noun phrases that add no retrieval value — filtered from noun chunks
_STOP_CONCEPTS: set[str] = {
    "thing", "things", "way", "ways", "lot", "lots", "kind", "kinds",
    "type", "types", "user", "users", "people", "time", "times",
    "day", "days", "bit", "bits", "part", "parts", "place", "places",
    "point", "points", "use", "work", "change", "changes", "result",
    "results", "end", "start", "number", "case", "cases", "example",
    "examples", "set", "sets", "list", "item", "items", "step", "steps",
    "note", "notes", "idea", "ideas", "fact", "facts", "issue", "issues",
    "problem", "problems", "need", "needs", "reason", "reasons",
    "option", "options", "question", "questions", "answer", "answers",
    "process", "area", "areas", "side", "field", "fields", "line",
    "lines", "view", "views", "level", "levels", "term", "terms",
    "sense", "sense", "course", "version", "addition", "approach",
}

# Common dev tools / databases / frameworks en_core_web_sm consistently misses
# Applied ONLY when project domain is in _TECHNICAL_DOMAINS
_TECH_KEYWORDS: set[str] = {
    # Databases
    "redis", "postgresql", "postgres", "mysql", "mongodb", "sqlite",
    "elasticsearch", "cassandra", "dynamodb", "firestore", "cockroachdb",
    "supabase", "planetscale", "chromadb", "pinecone", "weaviate",
    "qdrant", "milvus",
    # Frameworks / runtimes
    "fastapi", "flask", "django", "express", "nextjs", "next.js", "nuxt",
    "uvicorn", "gunicorn", "celery", "kafka", "rabbitmq",
    # Languages (supplement for when spaCy misses them)
    "typescript", "javascript", "python", "golang", "rust", "kotlin", "swift",
    # Infrastructure / cloud
    "docker", "kubernetes", "k8s", "nginx", "terraform", "ansible",
    "aws", "gcp", "azure", "heroku", "vercel", "netlify", "railway",
    # Auth / protocols
    "jwt", "oauth", "cors", "grpc", "graphql", "websocket", "websockets",
    # ORM / migration / schema
    "sqlalchemy", "alembic", "prisma", "drizzle", "pydantic",
    # Version control / CI
    "github", "gitlab", "circleci", "jenkins",
    # AI tools
    "cursor", "claude", "chatgpt", "gemini", "copilot",
    # Testing
    "pytest", "jest", "vitest",
}


class Entity(NamedTuple):
    text: str          # original surface form
    label: str         # NER label, "TECH", or "CONCEPT"
    normalized: str    # lowercased, stripped


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def _is_meaningful_chunk(chunk_text: str, root_pos: str) -> bool:
    """Filter noun chunks to keep only meaningful multi-word or specific-enough ones."""
    norm = chunk_text.strip().lower()
    if len(norm) < 3:
        return False
    if norm in _STOP_CONCEPTS:
        return False
    if root_pos not in {"NOUN", "PROPN"}:
        return False
    # Skip purely numeric chunks
    if norm.replace(".", "").replace(",", "").isdigit():
        return False
    return True


def extract_entities(text: str, domain: str = "general") -> list[Entity]:
    """
    Extract named entities and concepts from text.

    Args:
        text:   Input text to extract from.
        domain: Project domain — used to gate TECH keyword supplement.
                Pass "general" (default) to skip TECH extraction.

    Returns:
        Deduplicated list of Entity tuples. Same normalized form appears once.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(text)

    seen: set[str] = set()
    entities: list[Entity] = []

    # ── Layer 1: spaCy NER (universal) ──────────────────────────────────────
    for ent in doc.ents:
        if ent.label_ not in _USEFUL_NER_LABELS:
            continue
        norm = ent.text.lower().strip()
        if len(norm) < 2 or norm in seen:
            continue
        # Defer to TECH label for known tech terms (only in tech domains)
        if domain in _TECHNICAL_DOMAINS and (
            norm in _TECH_KEYWORDS
            or any(part in _TECH_KEYWORDS for part in norm.split())
        ):
            continue
        seen.add(norm)
        entities.append(Entity(text=ent.text.strip(), label=ent.label_, normalized=norm))

    # ── Layer 2: noun chunk extraction (universal — all domains) ────────────
    for chunk in doc.noun_chunks:
        # Use the root word's POS to filter quality
        root_pos = chunk.root.pos_
        chunk_text = chunk.text.strip()
        norm = chunk_text.lower().strip()

        if not _is_meaningful_chunk(chunk_text, root_pos):
            continue
        if norm in seen:
            continue
        # In technical domains, defer single-word tech keywords to Layer 3 (TECH label)
        if domain in _TECHNICAL_DOMAINS and norm in _TECH_KEYWORDS:
            continue
        seen.add(norm)
        entities.append(Entity(text=chunk_text, label="CONCEPT", normalized=norm))

    # ── Layer 3: TECH keyword supplement (technical domains only) ───────────
    if domain in _TECHNICAL_DOMAINS:
        for raw_token in text.split():
            stripped = raw_token.lower().strip(".,;:!?\"'()[]{}—-")
            candidates = [stripped] + stripped.split("-")
            for token in candidates:
                if token in _TECH_KEYWORDS and token not in seen:
                    seen.add(token)
                    original = raw_token.strip(".,;:!?\"'()[]{}—-")
                    entities.append(Entity(text=original, label="TECH", normalized=token))
                    break

    return entities


def extract_entities_for_memory(
    title: str,
    content: str,
    domain: str = "general",
) -> list[Entity]:
    """Extract entities from combined title + content of a memory."""
    combined = f"{title}. {content}" if title else content
    return extract_entities(combined, domain=domain)
