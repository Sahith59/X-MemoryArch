"""
Triple Fact Extractor — Phase 4.1 (entity-centric knowledge graph representation).

Architectural pivot: instead of atomic sentence facts, extract structured
entity-relationship triples:  {subject, relation, object, detail, text, temporal}

Root cause this fixes:
  Old (broken): "Alice works as a nurse at City Hospital"
    → embedding of this sentence vs "what does Alice do?" → cosine similarity miss
  New (fixed):  {subject: Alice, relation: WORKS_AT, object: City Hospital, detail: as a nurse}
    text: "Alice works at City Hospital as a nurse"
    → subject-first, relation-explicit, concise → better cosine match for entity queries

Phase 4.1 validates pure text representation improvement (same GTE-large, same A6r
pipeline). Phase 4.3 will add the direct graph walk on top of this foundation.

See PHASE4_NOTES.md for hypothesis, expected results, and full rationale.

Usage:
    extractor = TripleExtractor(api_key=ANTHROPIC_KEY)
    triples = extractor.extract(session_content, entity_roster={"Alice": "PERSON"})
    # Returns list[dict] — each dict has {subject, relation, object, detail, text, temporal}
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# ── Relation taxonomy ─────────────────────────────────────────────────────────

# State relations: only ONE can be active per (subject, relation) at a time.
# When a new one arrives for the same subject, it supersedes the old.
STATE_RELATIONS: frozenset[str] = frozenset({
    "WORKS_AT", "LIVES_AT", "STUDIES_AT", "PARTNER_OF", "MEMBER_OF",
    "HAS_ROLE", "REPORTS_TO", "MANAGES", "OWNS",
})

# Attribute relations: stable properties — don't supersede.
ATTRIBUTE_RELATIONS: frozenset[str] = frozenset({
    "HOBBY", "SKILL", "PERSONALITY", "PREFERENCE", "DISLIKES",
    "FEARS", "HEALTH_CONDITION", "DIETARY_RESTRICTION",
})

# Event/episode relations: historical facts — always append.
EVENT_RELATIONS: frozenset[str] = frozenset({
    "VISITED", "ATTENDED", "PARTICIPATED_IN", "PURCHASED",
    "DISCUSSED", "DECIDED", "PLANNED", "ACHIEVED",
})

# Social/structural relations.
SOCIAL_RELATIONS: frozenset[str] = frozenset({
    "KNOWS", "FRIEND_OF", "FAMILY_OF", "WORKS_WITH",
    "LIVES_WITH", "PARENT_OF", "SIBLING_OF",
})

ALL_RELATIONS: frozenset[str] = (
    STATE_RELATIONS | ATTRIBUTE_RELATIONS | EVENT_RELATIONS | SOCIAL_RELATIONS
    | {"HAS_ATTRIBUTE"}  # fallback
)

# Natural-language rendering for each relation (used to build the .text field).
RELATION_VERBS: dict[str, str] = {
    # State
    "WORKS_AT":           "works at",
    "LIVES_AT":           "lives in",
    "STUDIES_AT":         "studies at",
    "PARTNER_OF":         "is in a relationship with",
    "MEMBER_OF":          "is a member of",
    "HAS_ROLE":           "has the role of",
    "REPORTS_TO":         "reports to",
    "MANAGES":            "manages",
    "OWNS":               "owns",
    # Attributes
    "HOBBY":              "enjoys",
    "SKILL":              "has the skill of",
    "PERSONALITY":        "is known for being",
    "PREFERENCE":         "prefers",
    "DISLIKES":           "dislikes",
    "FEARS":              "is afraid of",
    "HEALTH_CONDITION":   "has health condition",
    "DIETARY_RESTRICTION": "has dietary restriction",
    # Events
    "VISITED":            "visited",
    "ATTENDED":           "attended",
    "PARTICIPATED_IN":    "participated in",
    "PURCHASED":          "purchased",
    "DISCUSSED":          "discussed",
    "DECIDED":            "decided to",
    "PLANNED":            "planned to",
    "ACHIEVED":           "achieved",
    # Social
    "KNOWS":              "knows",
    "FRIEND_OF":          "is friends with",
    "FAMILY_OF":          "is related to",
    "WORKS_WITH":         "works with",
    "LIVES_WITH":         "lives with",
    "PARENT_OF":          "is the parent of",
    "SIBLING_OF":         "is the sibling of",
    # Fallback
    "HAS_ATTRIBUTE":      "has attribute",
}

_RELATION_LIST = ", ".join(sorted(ALL_RELATIONS))

# ── Extraction prompt ──────────────────────────────────────────────────────────

_ROSTER_HEADER = """\
KNOWN ENTITIES — resolve ALL pronouns to these names:
{roster}

"""

_EXTRACTION_PROMPT = """\
Extract entity-relationship facts from this conversation as a JSON array.

Each fact must be a JSON object with these exact fields:
  "subject"  — person or org name, NO pronouns (he/she/they → use the actual name)
  "relation" — exactly one from the ALLOWED RELATIONS list below
  "object"   — concrete noun phrase: what the relation points to
  "detail"   — optional qualifier (e.g. "as a nurse", "since 2022") or empty string ""
  "temporal" — one of: "current" | "historical" | "episodic" | "permanent"

ALLOWED RELATIONS:
  State (updatable, only one active at a time):
    WORKS_AT, LIVES_AT, STUDIES_AT, PARTNER_OF, MEMBER_OF, HAS_ROLE,
    REPORTS_TO, MANAGES, OWNS
  Attributes (stable traits):
    HOBBY, SKILL, PERSONALITY, PREFERENCE, DISLIKES, FEARS,
    HEALTH_CONDITION, DIETARY_RESTRICTION
  Events (append-only):
    VISITED, ATTENDED, PARTICIPATED_IN, PURCHASED, DISCUSSED,
    DECIDED, PLANNED, ACHIEVED
  Social:
    KNOWS, FRIEND_OF, FAMILY_OF, WORKS_WITH, LIVES_WITH,
    PARENT_OF, SIBLING_OF
  Fallback:
    HAS_ATTRIBUTE

TEMPORAL VALUES:
  "current"   — present state (Alice works at X right now)
  "historical"— past fact (Alice used to work at X)
  "episodic"  — specific one-time event (Alice visited Paris in March)
  "permanent" — inherent trait or stable relationship (Alice is friends with Bob)

MANDATORY RULES:
1. subject MUST be a proper noun — NO pronouns. Resolve she/he/they → actual name.
2. Extract ≥1 triple per named person mentioned in the text.
3. Extract ≥1 triple per significant event or activity mentioned.
4. Preserve exact dates, places, and amounts in the object or detail field.
5. Output ONLY the JSON array. No explanation, no markdown fences.

{roster_section}Conversation:
{content}

JSON array of triples:"""

_PRONOUN_RE = re.compile(r'\b(she|he|they|her|his|their|him|them)\b', re.IGNORECASE)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class MemoryTriple:
    subject:    str
    relation:   str
    object:     str
    detail:     str
    text:       str      # natural-language rendering for GTE embedding
    temporal:   str      # current | historical | episodic | permanent
    session_id: str = ""


def triple_to_text(subject: str, relation: str, object_: str, detail: str) -> str:
    """
    Generate the natural-language text field from triple components.
    This is what gets embedded by GTE-large.

    Format: "{subject} {verb} {object} {detail}"
    e.g.  : "Alice works at City Hospital as a nurse"
    """
    verb = RELATION_VERBS.get(relation, "is related to")
    text = f"{subject} {verb} {object_}"
    if detail:
        text += f" {detail}"
    return text.strip()


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_triples(raw: str, session_id: str = "") -> list[MemoryTriple]:
    """
    Parse Claude's response into MemoryTriple objects.
    Handles:
      1. Clean JSON array output
      2. JSON wrapped in markdown fences
      3. Partial/malformed JSON (recovers individual objects with regex)
    """
    # Strip markdown fences if present
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)

    triples: list[MemoryTriple] = []

    # Attempt 1: clean JSON parse
    try:
        items = json.loads(text)
        if isinstance(items, list):
            for item in items:
                t = _item_to_triple(item, session_id)
                if t:
                    triples.append(t)
            return triples
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: extract individual JSON objects via regex (partial response recovery)
    obj_re = re.compile(r'\{[^{}]+\}', re.DOTALL)
    for match in obj_re.finditer(text):
        try:
            item = json.loads(match.group())
            t = _item_to_triple(item, session_id)
            if t:
                triples.append(t)
        except (json.JSONDecodeError, ValueError):
            continue

    return triples


def _item_to_triple(item: dict, session_id: str) -> MemoryTriple | None:
    """Validate and convert a parsed dict to a MemoryTriple."""
    subject  = str(item.get("subject", "")).strip()
    relation = str(item.get("relation", "HAS_ATTRIBUTE")).strip().upper()
    object_  = str(item.get("object", "")).strip()
    detail   = str(item.get("detail", "")).strip()
    temporal = str(item.get("temporal", "current")).strip().lower()

    # Skip triples with pronouns in subject position
    if not subject or _PRONOUN_RE.match(subject):
        return None
    # Skip empty objects
    if not object_:
        return None
    # Normalize relation to known set
    if relation not in ALL_RELATIONS:
        relation = "HAS_ATTRIBUTE"
    # Normalize temporal
    if temporal not in {"current", "historical", "episodic", "permanent"}:
        temporal = "current"

    text = triple_to_text(subject, relation, object_, detail)
    return MemoryTriple(
        subject=subject,
        relation=relation,
        object=object_,
        detail=detail,
        text=text,
        temporal=temporal,
        session_id=session_id,
    )


# ── Extractor class ───────────────────────────────────────────────────────────

class TripleExtractor:
    """
    Extracts structured entity-relationship triples from conversational sessions.

    Output per triple:
      subject  — proper noun (person or org)
      relation — typed relation from 30-type taxonomy
      object   — concrete noun phrase
      detail   — optional qualifier
      text     — natural-language rendering for GTE embedding
      temporal — current | historical | episodic | permanent

    Phase 4.1 focus: the .text field. Each triple's text is subject-first and
    relation-explicit, which improves cosine similarity for entity-state queries.
    Phase 4.3 will use subject/relation/object directly for graph-walk retrieval.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5"):
        self._api_key = api_key or self._load_key()
        self._model   = model
        self._client  = None

    def _load_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            env_file = Path(__file__).resolve().parent.parent.parent.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, _, v = line.partition("=")
                        if k.strip() == "ANTHROPIC_API_KEY":
                            return v.strip()
        return key

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call_haiku(self, prompt: str, max_tokens: int = 800) -> str:
        client = self._get_client()
        resp   = client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _build_roster_section(self, entity_roster: dict[str, str]) -> str:
        if not entity_roster:
            return ""
        lines = [f"  - {name} ({etype})" for name, etype in entity_roster.items()]
        return _ROSTER_HEADER.format(roster="\n".join(lines))

    def extract(
        self,
        content: str,
        entity_roster: dict[str, str] | None = None,
        session_id: str = "",
        n_per_chunk: int = 6,
    ) -> list[MemoryTriple]:
        """
        Extract structured triples from one conversation session.

        Args:
            content:       Full session text.
            entity_roster: {name: type} for pronoun resolution.
            session_id:    Identifier attached to each returned triple.
            n_per_chunk:   Target triples per half-chunk (total ≈ 2 × n_per_chunk).

        Returns:
            List of MemoryTriple. All subjects are proper nouns.
        """
        if not self._api_key:
            return self._fallback_extract(content, session_id)

        roster_section = self._build_roster_section(entity_roster or {})

        # Split into two halves for full topic coverage (same technique as
        # ComprehensiveExtractor — prevents dominant-topic bias)
        words    = content.split()
        midpoint = len(words) // 2
        chunks   = [
            " ".join(words[:midpoint]),
            " ".join(words[midpoint:]),
        ]

        all_triples: list[MemoryTriple] = []
        seen_texts:  set[str] = set()

        for chunk in chunks:
            if len(chunk.split()) < 20:
                continue
            prompt = _EXTRACTION_PROMPT.format(
                roster_section=roster_section,
                content=chunk[:3500],
            )
            try:
                raw     = self._call_haiku(prompt, max_tokens=900)
                triples = _parse_triples(raw, session_id)
                for t in triples:
                    key = t.text.lower()[:80]
                    if key not in seen_texts:
                        seen_texts.add(key)
                        all_triples.append(t)
                time.sleep(0.05)
            except Exception:
                # Fallback: convert content to HAS_ATTRIBUTE triples via sentence split
                fallback = self._fallback_extract(chunk, session_id)
                for t in fallback:
                    key = t.text.lower()[:80]
                    if key not in seen_texts:
                        seen_texts.add(key)
                        all_triples.append(t)

        return all_triples

    def extract_batch(
        self,
        sessions: list[dict],
        entity_roster_fn=None,
        n_per_chunk: int = 6,
        verbose: bool = True,
        checkpoint_path: "Path | None" = None,
        checkpoint_every: int = 20,
    ) -> dict[str, list[dict]]:
        """
        Extract triples for multiple sessions.

        Returns:
            {session_id: [triple_dict, ...]}
            Each triple_dict has keys: subject, relation, object, detail, text, temporal.
        """
        import json as _json
        try:
            from app.services.knowledge_graph.entity_ner import extract_entities
            _has_ner = True
        except ImportError:
            _has_ner = False

        results: dict[str, list[dict]] = {}
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            results = _json.loads(Path(checkpoint_path).read_text())
            if verbose and results:
                print(f"  Resumed from checkpoint: {len(results)} sessions already done")

        progressive_roster: dict[str, str] = {}
        remaining   = [s for s in sessions if s["session_id"] not in results]
        total       = len(sessions)
        done_offset = total - len(remaining)

        for i, session in enumerate(remaining):
            sid     = session["session_id"]
            content = session["content"]

            if entity_roster_fn:
                roster = entity_roster_fn(sid)
            elif _has_ner:
                current_entities = {
                    e.text: e.entity_type
                    for e in extract_entities(content)
                    if e.entity_type in ("PERSON", "ORG")
                }
                progressive_roster.update(current_entities)
                roster = dict(progressive_roster)
            else:
                roster = {}

            triples = self.extract(
                content,
                entity_roster=roster,
                session_id=sid,
                n_per_chunk=n_per_chunk,
            )
            results[sid] = [asdict(t) for t in triples]

            # Update progressive roster from extracted subjects
            for t in triples:
                if t.subject and t.subject not in progressive_roster:
                    progressive_roster[t.subject] = "PERSON"

            global_idx = done_offset + i + 1
            if verbose and global_idx % 10 == 0:
                n_triples = sum(len(v) for v in results.values())
                print(
                    f"  {global_idx}/{total} sessions  "
                    f"({n_triples:,} triples total, "
                    f"roster: {len(progressive_roster)} entities)",
                    flush=True,
                )

            if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
                Path(checkpoint_path).write_text(_json.dumps(results))

        if checkpoint_path is not None:
            Path(checkpoint_path).write_text(_json.dumps(results))

        return results

    def _fallback_extract(self, content: str, session_id: str = "") -> list[MemoryTriple]:
        """
        No-API fallback: sentence-split content and wrap each sentence as
        a HAS_ATTRIBUTE triple. Gives the A10 benchmark something to embed
        even without an API key, though quality will be lower.
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content.strip())
        triples   = []
        for s in sentences:
            s = s.strip()
            if len(s) < 30:
                continue
            # Try to extract a subject from the sentence start
            name_match = re.match(r'^([A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20})?)', s)
            subject    = name_match.group(1) if name_match else "Unknown"
            triples.append(MemoryTriple(
                subject=subject,
                relation="HAS_ATTRIBUTE",
                object=s[:100],
                detail="",
                text=s[:120],
                temporal="current",
                session_id=session_id,
            ))
            if len(triples) >= 8:
                break
        return triples


# ── Quality checker ───────────────────────────────────────────────────────────

def check_triple_quality(triples: list[dict]) -> dict:
    """
    Audit extracted triples for quality issues.

    Checks:
      - pronoun_subject: subject contains a pronoun
      - unknown_relation: relation not in ALL_RELATIONS
      - empty_object: object is empty
      - fallback_relation: uses HAS_ATTRIBUTE (indicates taxonomy miss)
    """
    issues: dict[str, list] = {
        "pronoun_subject":  [],
        "unknown_relation": [],
        "empty_object":     [],
        "fallback_relation":[],
    }
    for t in triples:
        subj = t.get("subject", "")
        rel  = t.get("relation", "")
        obj  = t.get("object", "")
        if _PRONOUN_RE.match(subj):
            issues["pronoun_subject"].append(subj)
        if rel not in ALL_RELATIONS:
            issues["unknown_relation"].append(rel)
        if not obj:
            issues["empty_object"].append(subj)
        if rel == "HAS_ATTRIBUTE":
            issues["fallback_relation"].append(subj)

    total = max(1, len(triples))
    return {
        "total_triples":  len(triples),
        "quality_score": 1.0 - (
            len(issues["pronoun_subject"]) * 0.4 +
            len(issues["empty_object"])    * 0.3 +
            len(issues["unknown_relation"])* 0.2
        ) / total,
        "fallback_pct": len(issues["fallback_relation"]) / total,
        "issues": {k: len(v) for k, v in issues.items()},
        "examples": {k: v[:2] for k, v in issues.items() if v},
    }
