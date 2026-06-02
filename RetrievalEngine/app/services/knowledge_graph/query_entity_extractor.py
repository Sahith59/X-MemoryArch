"""
Phase 4.2 — Query-side entity extraction and intent classification.

At retrieval time, given a raw query string:
  1. Extract named entity mentions (spaCy if available, regex fallback)
  2. Expand to canonical names via EntityRegistry (handles partial names)
  3. Classify query intent: ENTITY_STATE | EPISODE | SEMANTIC

Intent drives Phase 4.3 retrieval blend:
  ENTITY_STATE → heavy graph walk (entity + state relation lookup)
  EPISODE      → episode walk (actor + location/event match)
  SEMANTIC     → pure dense search (no graph walk)

Usage:
    from app.services.knowledge_graph.entity_registry import EntityRegistry
    from app.services.knowledge_graph.query_entity_extractor import QueryEntityExtractor

    registry = EntityRegistry.load("benchmark_cache/entity_registry_LoCoMo.json")
    extractor = QueryEntityExtractor(registry)
    result = extractor.extract("What does Alice do for work?")
    # QueryEntities(entities=["Alice Johnson"], entity_ids=["ent_0001"],
    #               intent="ENTITY_STATE", intent_score=0.9)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.knowledge_graph.entity_registry import EntityRegistry


# ── Intent patterns ────────────────────────────────────────────────────────────

_ENTITY_STATE_PATTERNS: list[re.Pattern] = [
    # "what does X do/work/study/like/prefer/know/have"
    re.compile(
        r'\bwhat\s+(?:does|do|is|are|was|were)\s+\w+(?:\s+\w+)?\s+'
        r'(?:do|work|study|know|like|prefer|dislike|hate|have|own|manage|report|live)\b',
        re.IGNORECASE,
    ),
    # "where does/did X live/work/go/study/stay"
    re.compile(
        r'\bwhere\s+(?:does|do|did|is|are|was|were)\s+\w+(?:\s+\w+)?\s+'
        r'(?:live|work|go|study|stay|move|be\s+from|come\s+from)\b',
        re.IGNORECASE,
    ),
    # "who does X know/work with/report to/manage"
    re.compile(
        r'\bwho\s+(?:does|do|did|is|are|was|were)\s+\w+(?:\s+\w+)?\s+'
        r'(?:know|work\s+with|report\s+to|manage|live\s+with|partner)\b',
        re.IGNORECASE,
    ),
    # "what is X's job/role/name/home/hobby/skill/relationship/partner/boss"
    re.compile(
        r"\bwhat\s+(?:is|was|are|were)\s+\w+(?:\'s)?\s+"
        r'(?:job|role|position|title|name|home|address|hobby|skill|partner|'
        r'boss|manager|relationship|company|employer|school|university)\b',
        re.IGNORECASE,
    ),
    # "where is X from / currently working / currently living"
    re.compile(
        r'\bwhere\s+is\s+\w+(?:\s+\w+)?\s+(?:from|currently|now|based)\b',
        re.IGNORECASE,
    ),
    # "does X still work at / live in / study at"
    re.compile(
        r'\bdoes\s+\w+(?:\s+\w+)?\s+still\s+(?:work|live|study|go|attend)\b',
        re.IGNORECASE,
    ),
    # "X's [attribute]" pattern: "Alice's job", "Bob's current role"
    re.compile(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\'s\s+"
        r'(?:job|role|home|partner|boss|manager|school|company|employer|skill|hobby)\b',
        re.IGNORECASE,
    ),
    # Explicit state keywords in query
    re.compile(
        r'\b(?:currently|nowadays|these\s+days|right\s+now|at\s+the\s+moment|'
        r'still\s+at|still\s+work|still\s+live)\b',
        re.IGNORECASE,
    ),
]

_EPISODE_PATTERNS: list[re.Pattern] = [
    # "when did X visit/go/travel/attend/stay/buy/participate"
    re.compile(
        r'\bwhen\s+did\s+\w+(?:\s+\w+)?\s+'
        r'(?:visit|go|travel|attend|stay|buy|participate|take|meet|see)\b',
        re.IGNORECASE,
    ),
    # "where did X stay/eat/visit/go/travel/attend"
    re.compile(
        r'\bwhere\s+did\s+\w+(?:\s+\w+)?\s+'
        r'(?:stay|eat|visit|go|travel|attend|take|meet)\b',
        re.IGNORECASE,
    ),
    # "what did X do during/when/at/in [event/place]"
    re.compile(
        r'\bwhat\s+did\s+\w+(?:\s+\w+)?\s+do\s+(?:during|when|at|in)\b',
        re.IGNORECASE,
    ),
    # "during the X trip / X's trip to"
    re.compile(
        r'\bduring\s+(?:the\s+)?\w+(?:\'s)?\s+trip\b',
        re.IGNORECASE,
    ),
    # "trip to [place]"
    re.compile(r'\btrip\s+to\b', re.IGNORECASE),
    # "what hotel / what restaurant / what flight / what conference"
    re.compile(
        r'\bwhat\s+(?:hotel|restaurant|flight|conference|event|place|venue|concert)\b',
        re.IGNORECASE,
    ),
]

# ── Entity extraction patterns (regex fallback when spaCy not available) ──────

# Capitalized single name or "First Last" following common query words
_QUERY_NAME_PATTERN = re.compile(
    r'(?:^|\s)'
    r'([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)'
    r'(?=[\'s\s,?.]|$)',
)

# Pronouns that indicate a named entity should be resolved from context
_THIRD_PERSON_PRONOUNS = frozenset({
    "he", "she", "they", "him", "her", "his", "hers", "their", "them",
})


@dataclass
class QueryEntities:
    entities: list[str]       # canonical entity names resolved from registry
    entity_ids: list[str]     # corresponding entity_ids
    intent: str               # ENTITY_STATE | EPISODE | SEMANTIC
    intent_score: float       # 0.0-1.0 confidence in intent
    raw_mentions: list[str] = field(default_factory=list)  # raw names before registry lookup


class QueryEntityExtractor:
    """
    Extracts entity mentions from a query and classifies retrieval intent.
    Integrates with EntityRegistry for alias expansion.
    """

    def __init__(self, registry: "EntityRegistry"):
        self._registry = registry
        self._spacy_nlp = None
        self._spacy_available: bool | None = None

    def extract(self, query: str, context: str = "") -> QueryEntities:
        """
        Extract entities + classify intent from a query string.

        Args:
            query:   The retrieval query.
            context: Recent conversation context for pronoun resolution
                     (last N turns, whitespace-separated). Optional.

        Returns:
            QueryEntities with resolved entity names, IDs, and intent.
        """
        # Step 1: classify intent
        intent, intent_score = self._classify_intent(query)

        # Step 2: extract raw entity mentions
        raw_mentions = self._extract_raw_mentions(query)

        # Step 3: resolve pronouns using context
        if context:
            resolved = self._resolve_pronouns(raw_mentions, query, context)
            raw_mentions = resolved

        # Step 4: expand via registry (partial name → canonical, alias → canonical)
        canonical_names: list[str] = []
        entity_ids: list[str] = []
        seen_ids: set[str] = set()

        for mention in raw_mentions:
            eids = self._registry.find_by_query(mention)
            for eid in eids:
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    entity_ids.append(eid)
                    ent = self._registry.get_entity(eid)
                    canonical_names.append(ent.canonical if ent else mention)

        return QueryEntities(
            entities=canonical_names,
            entity_ids=entity_ids,
            intent=intent,
            intent_score=intent_score,
            raw_mentions=raw_mentions,
        )

    def _classify_intent(self, query: str) -> tuple[str, float]:
        """
        Classify query intent. Returns (intent_type, confidence_score).

        Priority: ENTITY_STATE > EPISODE > SEMANTIC.
        Score = fraction of matching patterns (capped at 1.0).
        """
        entity_state_hits = sum(
            1 for p in _ENTITY_STATE_PATTERNS if p.search(query)
        )
        episode_hits = sum(
            1 for p in _EPISODE_PATTERNS if p.search(query)
        )

        if entity_state_hits > 0:
            score = min(1.0, entity_state_hits * 0.3 + 0.5)
            return ("ENTITY_STATE", score)

        if episode_hits > 0:
            score = min(1.0, episode_hits * 0.3 + 0.5)
            return ("EPISODE", score)

        return ("SEMANTIC", 0.5)

    def _extract_raw_mentions(self, query: str) -> list[str]:
        """
        Extract named entity surface forms from query.
        Uses spaCy if available, regex fallback otherwise.
        """
        if self._try_load_spacy():
            return self._extract_spacy(query)
        return self._extract_regex(query)

    def _try_load_spacy(self) -> bool:
        if self._spacy_available is not None:
            return self._spacy_available
        try:
            import spacy
            self._spacy_nlp = spacy.load("en_core_web_sm")
            self._spacy_available = True
        except Exception:
            self._spacy_available = False
        return self._spacy_available

    def _extract_spacy(self, query: str) -> list[str]:
        doc = self._spacy_nlp(query)
        results: list[str] = []
        seen: set[str] = set()
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC"):
                text = ent.text.strip()
                lower = text.lower()
                if lower not in seen and len(text) >= 2:
                    seen.add(lower)
                    results.append(text)
        return results

    def _extract_regex(self, query: str) -> list[str]:
        """
        Regex fallback: find capitalized names and cross-check against registry.
        Prefers registry-confirmed matches to reduce false positives.
        """
        candidates: list[str] = []
        # Skip words that are question words or common sentence starters
        _SKIP_WORDS = frozenset({
            "what", "where", "when", "who", "why", "how", "which",
            "does", "did", "is", "are", "was", "were", "will", "would",
            "could", "should", "has", "have", "had",
            "the", "a", "an", "this", "that", "it", "they", "he", "she",
            "his", "her", "their", "its", "we", "you", "i",
        })

        for m in _QUERY_NAME_PATTERN.finditer(query):
            mention = m.group(1).strip()
            tokens  = mention.split()
            # Two-word capture where first token is a question/aux word → take second only.
            # e.g. "Would Caroline" → "Caroline"
            if len(tokens) == 2 and tokens[0].lower() in _SKIP_WORDS:
                mention = tokens[1]
            if mention.lower() not in _SKIP_WORDS and len(mention) >= 2:
                candidates.append(mention)

        # Prefer mentions that have a match in the registry
        registry_confirmed: list[str] = []
        unconfirmed: list[str] = []
        for c in candidates:
            if self._registry.find_by_query(c):
                registry_confirmed.append(c)
            else:
                unconfirmed.append(c)

        # Return registry-confirmed first; add unconfirmed only if nothing confirmed
        if registry_confirmed:
            return registry_confirmed
        return unconfirmed

    def _resolve_pronouns(
        self, mentions: list[str], query: str, context: str
    ) -> list[str]:
        """
        If query contains third-person pronouns and no direct entity mention
        was found, extract the last mentioned person from context.
        """
        query_lower = query.lower()
        has_pronoun = any(p in query_lower.split() for p in _THIRD_PERSON_PRONOUNS)

        if not has_pronoun or mentions:
            return mentions  # no pronoun resolution needed

        # Find last capitalized name in context (simple heuristic)
        context_names = _QUERY_NAME_PATTERN.findall(context)
        if not context_names:
            return mentions

        last_name = context_names[-1].strip()
        if last_name and last_name not in mentions:
            return [last_name] + mentions
        return mentions
