"""
Phase 4.2 — Entity Registry with alias normalization and fuzzy name matching.

Builds a canonical entity map from Phase 4.1 extracted triple facts.
Foundation for Phase 4.3 entity graph walk retrieval.

Core operation for Phase 4.3:
  sessions_for_entity("Alice") → ["c1_session_3", "c1_session_7", ...]
  — returns all sessions where Alice appears as a subject

Normalization pipeline (get_or_create):
  1. Lowercase + strip honorifics (Dr., Mr., Mrs., Ms., Prof.)
  2. Exact alias match → return existing entity
  3. Prefix token match ("alice" → "alice johnson") → merge
  4. Multi-word name: first token exact + last token edit_distance ≤ 2 → merge
  5. No match → create new entity

Usage:
    registry = EntityRegistry.build_from_triples(triple_facts, dataset_name="LoCoMo")
    registry.save("benchmark_cache/entity_registry_LoCoMo.json")
    reg = EntityRegistry.load("benchmark_cache/entity_registry_LoCoMo.json")
    session_ids = reg.sessions_for_entity("Alice")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from app.services.extraction.triple_extractor import STATE_RELATIONS, SOCIAL_RELATIONS

# Relations whose objects are named entities (worth indexing in registry)
_INDEX_OBJECT_RELATIONS: frozenset[str] = STATE_RELATIONS | SOCIAL_RELATIONS

# Honorifics to strip during normalization
_HONORIFICS: frozenset[str] = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sir", "lord", "lady", "rev",
    "dr.", "mr.", "mrs.", "ms.", "prof.",
})

# Common words that should not be registered as entity names.
# These appear as spurious subjects when the LLM extraction fallback
# picks up sentence-starting words ("Here are...", "Do you...", "Can I...").
_NON_ENTITY_WORDS: frozenset[str] = frozenset({
    # Generic conversation roles / extraction artifacts
    "unknown", "here", "do", "can", "it", "now", "yes", "no",
    "ok", "okay", "sure", "great", "thanks", "there", "please",
    "note", "hi", "hello", "so", "well", "also", "then",
    "this", "that", "these", "those",
    # Pronouns
    "they", "we", "i", "me", "he", "she", "him", "her", "his", "their", "them",
    # Conjunctions / prepositions that appear sentence-initial in LME
    "however", "as", "and", "to", "or", "but", "yet", "speaking",
    "by", "with", "in", "on", "at", "from", "for", "of", "about",
    # Articles / determiners (the prefix-merge guard catches "the", but belt-and-suspenders)
    "the", "a", "an",
    # Other common sentence starters misidentified as subjects
    "what", "where", "when", "who", "why", "how", "which",
    "if", "while", "although", "because", "since", "after", "before",
    "additionally", "furthermore", "therefore", "moreover", "finally",
    "overall", "generally", "basically", "essentially",
    # Second-person + relative pronouns
    "you", "your", "yours",
    # Relationship words extracted as subjects
    "sister", "brother", "mother", "father", "friend", "colleague",
    "manager", "boss", "partner",
    # Additional LME fallback artifacts
    "regarding", "remember", "both", "are", "have", "has",
    "let", "just", "like", "some", "any", "all",
})


def _is_valid_entity_name(norm: str) -> bool:
    """Return False for strings that are clearly not entity names."""
    if not norm:
        return False
    first_token = norm.split()[0]
    # Single-character tokens are not entity names
    if len(first_token) < 2:
        return False
    # Stopwords / common sentence starters
    if norm in _NON_ENTITY_WORDS or first_token in _NON_ENTITY_WORDS:
        return False
    # Digit-starting strings are not entity names
    if norm[0].isdigit():
        return False
    return True


def _normalize_name(name: str) -> str:
    """
    Lowercase, strip punctuation, remove leading honorifics.
    Returns empty string if input is empty or becomes empty after stripping.
    """
    name = name.lower().strip(" .,;:\"'()[]")
    parts = name.split()
    if parts and parts[0].rstrip(".") in _HONORIFICS:
        parts = parts[1:]
    return " ".join(parts)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 3:
        return abs(la - lb)  # fast path
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[lb]


def _is_fuzzy_name_match(norm_a: str, norm_b: str) -> bool:
    """
    Returns True if two normalized names likely refer to the same entity.

    Rules (conservative — avoids false positives):
      1. Exact match.
      2. One is a single-token that is the first token of the other.
         "alice" → "alice johnson"; safe because we require the longer
         form to start with the shorter, not just contain it.
      3. Both are multi-word, same first token, last token edit_distance ≤ 2
         (require last token ≥ 4 chars to avoid "bob kim" / "bob lee" merges).
    """
    if norm_a == norm_b:
        return True

    toks_a = norm_a.split()
    toks_b = norm_b.split()

    # Rule 2: single-token prefix match — require ≥ 4 chars to avoid "the", "by" etc.
    if len(toks_a) == 1 and len(toks_a[0]) >= 4 and len(toks_b) > 1 and toks_b[0] == toks_a[0]:
        return True
    if len(toks_b) == 1 and len(toks_b[0]) >= 4 and len(toks_a) > 1 and toks_a[0] == toks_b[0]:
        return True

    # Rule 3: multi-word, same first token, fuzzy last token
    if len(toks_a) >= 2 and len(toks_b) >= 2 and toks_a[0] == toks_b[0]:
        last_a, last_b = toks_a[-1], toks_b[-1]
        if len(last_a) >= 4 and len(last_b) >= 4:
            if _edit_distance(last_a, last_b) <= 2:
                return True

    return False


@dataclass
class Entity:
    id: str
    canonical: str       # most complete/common surface form
    aliases: list[str]   # all surface forms seen (includes canonical)
    entity_type: str     # PERSON | ORG | GPE | OTHER
    session_ids: list[str]   # sessions where this entity appears as subject or relevant object


class EntityRegistry:
    """
    Maps entity names → canonical entity IDs, handles aliases and fuzzy matching.
    Primary output: sessions_for_entity(name) → list[session_id]
    """

    def __init__(self, dataset_name: str = ""):
        self._entities: dict[str, Entity] = {}           # entity_id → Entity
        self._alias_map: dict[str, str] = {}             # normalized_alias → entity_id
        self._dataset_name = dataset_name
        self._counter = 0

    # ── Core operations ───────────────────────────────────────────────────────

    def get_or_create(self, name: str, entity_type: str = "PERSON") -> str:
        """
        Normalize name, find or create entity. Returns entity_id.
        Merges into existing entity if fuzzy match found.
        """
        norm = _normalize_name(name)
        if not norm or not _is_valid_entity_name(norm):
            return ""

        # Exact / alias match
        if norm in self._alias_map:
            return self._alias_map[norm]

        # Fuzzy match against all existing normalized aliases
        match_id = self._find_fuzzy_match(norm)
        if match_id:
            ent = self._entities[match_id]
            # Register this alias
            self._alias_map[norm] = match_id
            if name not in ent.aliases:
                ent.aliases.append(name)
            # Upgrade canonical if new form is longer (more complete)
            if len(name) > len(ent.canonical):
                ent.canonical = name
            return match_id

        # Create new entity
        return self._create(name, norm, entity_type)

    def register_session(self, name: str, session_id: str, entity_type: str = "PERSON"):
        """Get or create entity for name, then add session_id to its session set."""
        eid = self.get_or_create(name, entity_type=entity_type)
        if not eid:
            return
        ent = self._entities[eid]
        if session_id not in ent.session_ids:
            ent.session_ids.append(session_id)

    def sessions_for_entity(self, name: str) -> list[str]:
        """
        Return session_ids where this entity appears.
        Looks up by exact alias first, then prefix/fuzzy match.
        Primary API for Phase 4.3 graph walk.
        """
        eids = self.find_by_query(name)
        if not eids:
            return []
        # Union all session_ids across matched entities (handles ambiguous cases)
        seen: set[str] = set()
        result: list[str] = []
        for eid in eids:
            ent = self._entities.get(eid)
            if ent:
                for sid in ent.session_ids:
                    if sid not in seen:
                        seen.add(sid)
                        result.append(sid)
        return result

    def find_by_query(self, query_name: str) -> list[str]:
        """
        Fuzzy lookup for query-time entity expansion.
        Returns list of entity_ids — may be multiple if the name is ambiguous.

        Examples:
          "Alice"         → matches "alice johnson" → ["ent_0001"]
          "Alice Johnson" → exact match             → ["ent_0001"]
          "Dr. Smith"     → strips title → "smith"  → ["ent_0042"]
        """
        norm = _normalize_name(query_name)
        if not norm:
            return []

        # Exact / alias match
        if norm in self._alias_map:
            return [self._alias_map[norm]]

        # Prefix token match: "alice" finds "alice johnson" in registry
        query_tokens = norm.split()
        matches: list[str] = []
        seen_ids: set[str] = set()
        for alias_norm, eid in self._alias_map.items():
            alias_tokens = alias_norm.split()
            if len(alias_tokens) > len(query_tokens):
                if alias_tokens[: len(query_tokens)] == query_tokens:
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        matches.append(eid)
            elif len(alias_tokens) < len(query_tokens):
                # Shorter alias is a prefix of query — e.g., "alice" in registry, "alice johnson" in query
                if query_tokens[: len(alias_tokens)] == alias_tokens:
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        matches.append(eid)

        # If still no match, try edit distance on single-token names
        if not matches and len(query_tokens) == 1 and len(norm) >= 4:
            for alias_norm, eid in self._alias_map.items():
                alias_tokens = alias_norm.split()
                if len(alias_tokens) == 1 and len(alias_norm) >= 4:
                    if _edit_distance(norm, alias_norm) <= 1:
                        if eid not in seen_ids:
                            seen_ids.add(eid)
                            matches.append(eid)

        return matches

    def aliases(self, entity_id: str) -> list[str]:
        """Return all known surface forms for an entity."""
        ent = self._entities.get(entity_id)
        return list(ent.aliases) if ent else []

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def merge(self, keep_id: str, drop_id: str):
        """Merge drop_id into keep_id. All aliases and sessions from drop become part of keep."""
        if keep_id == drop_id or keep_id not in self._entities or drop_id not in self._entities:
            return
        keep = self._entities[keep_id]
        drop = self._entities[drop_id]

        # Absorb aliases
        for alias in drop.aliases:
            if alias not in keep.aliases:
                keep.aliases.append(alias)
            norm = _normalize_name(alias)
            self._alias_map[norm] = keep_id

        # Absorb sessions
        existing_sids = set(keep.session_ids)
        for sid in drop.session_ids:
            if sid not in existing_sids:
                keep.session_ids.append(sid)
                existing_sids.add(sid)

        # Upgrade canonical if drop has a longer form
        if len(drop.canonical) > len(keep.canonical):
            keep.canonical = drop.canonical

        del self._entities[drop_id]

    def stats(self) -> dict:
        total_sessions: set[str] = set()
        person_count = org_count = gpe_count = 0
        for ent in self._entities.values():
            total_sessions.update(ent.session_ids)
            if ent.entity_type == "PERSON":
                person_count += 1
            elif ent.entity_type == "ORG":
                org_count += 1
            elif ent.entity_type == "GPE":
                gpe_count += 1
        return {
            "total_entities": len(self._entities),
            "total_aliases":  len(self._alias_map),
            "sessions_covered": len(total_sessions),
            "by_type": {"PERSON": person_count, "ORG": org_count, "GPE": gpe_count},
            "dataset": self._dataset_name,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        data = {
            "meta": self.stats(),
            "entities": {eid: asdict(ent) for eid, ent in self._entities.items()},
            "alias_map": self._alias_map,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EntityRegistry":
        path = Path(path)
        data = json.loads(path.read_text())
        reg = cls(dataset_name=data.get("meta", {}).get("dataset", ""))
        for eid, edata in data["entities"].items():
            reg._entities[eid] = Entity(**edata)
        reg._alias_map = data["alias_map"]
        # Restore counter so new entities don't collide
        if reg._entities:
            nums = []
            for eid in reg._entities:
                try:
                    nums.append(int(eid.split("_")[-1]))
                except ValueError:
                    pass
            reg._counter = max(nums) + 1 if nums else 0
        return reg

    # ── Class-level factory ───────────────────────────────────────────────────

    @classmethod
    def build_from_triples(
        cls,
        triple_facts: dict[str, list[dict]],
        dataset_name: str = "",
    ) -> "EntityRegistry":
        """
        Build entity registry from Phase 4.1 triple_facts cache.

        Indexes:
          - All triple subjects (always PERSON type)
          - Objects of STATE_RELATIONS and SOCIAL_RELATIONS (ORG/GPE/PERSON)

        For each subject and relevant object, registers which sessions
        they appear in. This powers Phase 4.3 graph walk retrieval.
        """
        registry = cls(dataset_name=dataset_name)

        for session_id, triples in triple_facts.items():
            for t in triples:
                subj = (t.get("subject") or "").strip()
                rel  = (t.get("relation") or "")
                obj  = (t.get("object")  or "").strip()

                if subj:
                    registry.register_session(subj, session_id, entity_type="PERSON")

                if rel in _INDEX_OBJECT_RELATIONS and obj:
                    # Determine entity type from relation context
                    if rel in ("WORKS_AT", "STUDIES_AT", "MEMBER_OF"):
                        otype = "ORG"
                    elif rel == "LIVES_AT":
                        otype = "GPE"
                    elif rel in SOCIAL_RELATIONS:
                        otype = "PERSON"
                    else:
                        otype = "OTHER"
                    # Only index objects that look like named entities (not short attribute values)
                    if len(obj) >= 3 and len(obj.split()) <= 6:
                        registry.register_session(obj, session_id, entity_type=otype)

        return registry

    # ── Private helpers ───────────────────────────────────────────────────────

    def _create(self, name: str, norm: str, entity_type: str) -> str:
        self._counter += 1
        eid = f"ent_{self._counter:04d}"
        ent = Entity(
            id=eid,
            canonical=name,
            aliases=[name],
            entity_type=entity_type,
            session_ids=[],
        )
        self._entities[eid] = ent
        self._alias_map[norm] = eid
        return eid

    def _find_fuzzy_match(self, norm: str) -> str | None:
        """Scan existing aliases for a fuzzy match. Returns entity_id or None."""
        for alias_norm, eid in self._alias_map.items():
            if _is_fuzzy_name_match(norm, alias_norm):
                return eid
        return None
