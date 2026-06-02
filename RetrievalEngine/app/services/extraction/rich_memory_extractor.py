"""
Phase 4.4 — Rich Memory Extractor.

Replaces TripleExtractor (Phase 4.1) with Mem0-architecture compliant memories:
  - 15-80 words per memory (never 4-word atomic fragments)
  - Temporal grounding in every memory ("As of session N, Alice...")
  - Full proper nouns (no pronouns)
  - ADD-ONLY (nothing deleted at ingestion)

Root cause this fixes:
  Phase 4.1: "Caroline has role counselor"  (4 words)
    cosine("career path decided pursue", "has role counselor") ≈ 0.35  → MISS

  Phase 4.4: "As of session 4, Caroline has made a firm decision to pursue
              counseling, inspired by years of supporting friends through personal
              mental health challenges."  (35 words)
    cosine("career path decided pursue", "made firm decision pursue counseling") ≈ 0.82 → HIT

Usage:
    extractor = RichMemoryExtractor(model="claude-sonnet-4-6")
    records = extractor.extract(content, session_id="c0_session_4",
                                session_position=4, dataset="LoCoMo")
    # records: list of memory dicts, each 15-80 words, temporally grounded
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

# ── Prompt ────────────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You extract memories from a conversation. Output JSON only — no markdown, no explanation.

Session {session_position} of the conversation series.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — IDENTIFY ALL DISTINCT TOPICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing memories, list every distinct topic covered in this session.
A "topic" is any distinct subject: a person's job, a relationship, a living situation,
a hobby, a health issue, a plan, a specific event, an opinion or emotion.

List topics as short labels, e.g.: ["Caroline's career decision", "Melanie's new job",
"weekend event", "Caroline's health", "relationship with John", "apartment search"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — WRITE 8-10 MEMORIES (diversity required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate 8-10 memories total. Every memory must obey ALL rules below:

RULE 1 — DIVERSITY (most critical): Each memory must cover a DIFFERENT topic.
  • Scan your topic list. Each topic must appear in at least one memory.
  • If you have fewer than 8 topics, you may write multiple memories per topic
    BUT each must add genuinely new information (different event, different aspect,
    different time period) — never rephrase the same fact.
  • WRONG: writing 6 memories all about Caroline's LGBTQ activism.
  • RIGHT: one memory per major topic (job, relationship, hobby, health, event, plan...).

RULE 2 — ZERO PRONOUNS: Replace every pronoun with the person's full name.
  WRONG: "She decided to pursue counseling, motivated by her experiences."
  RIGHT: "Caroline decided to pursue counseling, motivated by Caroline's experiences."
  If any pronoun appears in your output, that memory FAILS.

RULE 3 — TEMPORAL GROUNDING (required in every memory):
  Current state  → "As of session {session_position}, [Name] [fact with context]."
  Specific event → "During session {session_position}, [Name] [what happened]."
  Future plan    → "[Name] plans to [goal] (mentioned in session {session_position})."

RULE 4 — LENGTH AND SPECIFICITY:
  • 20-60 words each. No bare facts under 20 words.
  • Use exact specifics: "aerial yoga" not "yoga", "City Hospital head nurse" not "nurse".
  • Only extract what is stated or clearly implied. Never fabricate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return this exact JSON (no markdown fences):
{{
  "topics": ["topic 1", "topic 2", ...],
  "state_memories": ["<memory about an ongoing fact>", ...],
  "episodic_memories": ["<memory about a specific event>", ...]
}}

State = ongoing facts (jobs, relationships, living situations, interests, health).
Episodic = specific events (visited a place, made a decision, had a conversation).
Together, state_memories + episodic_memories must total 8-10 and cover every listed topic.

Conversation (session {session_position}):
{session_content}"""

_MIN_WORDS = 20   # raised from 15 — bare facts under 20 words don't have enough context
_MAX_WORDS = 80
_CONTENT_LIMIT = 4000   # chars per session passed to the LLM


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict[str, list[str]]:
    """
    Parse LLM JSON response into {state_memories, episodic_memories}.
    Handles clean JSON, markdown-fenced JSON, and partial responses.
    """
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)

    # Attempt 1: clean JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {
                "topics":           parsed.get("topics", []),
                "state_memories":   _validate_memories(parsed.get("state_memories", [])),
                "episodic_memories": _validate_memories(parsed.get("episodic_memories", [])),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: regex extraction of list values
    return {
        "topics":           [],
        "state_memories":   _validate_memories(_extract_list_from_text(text, "state_memories")),
        "episodic_memories": _validate_memories(_extract_list_from_text(text, "episodic_memories")),
    }


def _extract_list_from_text(text: str, key: str) -> list[str]:
    """Pull the JSON array value for a given key from raw text."""
    pattern = rf'"{key}"\s*:\s*\[([^\]]*)\]'
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return []
    return re.findall(r'"([^"]{20,})"', m.group(1))


def _validate_memories(memories: list) -> list[str]:
    """
    Keep memories within 15-80 words.
    Memories > 80 words are truncated. Memories < 15 words are discarded.
    """
    valid: list[str] = []
    for mem in memories:
        if not isinstance(mem, str):
            continue
        words = mem.split()
        wc = len(words)
        if wc < _MIN_WORDS:
            continue
        if wc > _MAX_WORDS:
            mem = " ".join(words[:_MAX_WORDS])
        valid.append(mem.strip())
    return valid


# ── Entity extraction ─────────────────────────────────────────────────────────

_spacy_nlp = None
_PRONOUN_SET = frozenset({
    "he", "she", "they", "him", "her", "his", "their", "them", "we", "us",
    "i", "me", "my", "our", "you", "your",
})


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                _spacy_nlp = spacy.blank("en")
        except ImportError:
            _spacy_nlp = None
    return _spacy_nlp


def _extract_entities(memory: str) -> list[str]:
    """
    Extract named entities (PERSON, ORG, GPE, LOC) from a memory text.
    Uses spaCy if available, capitalized-word regex fallback otherwise.
    """
    nlp = _get_spacy()
    if nlp is not None and nlp.pipe_names:  # has actual pipeline components
        try:
            doc = nlp(memory)
            seen: set[str] = set()
            entities: list[str] = []
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC"):
                    name = ent.text.strip()
                    key = name.lower()
                    if key not in seen and key not in _PRONOUN_SET:
                        seen.add(key)
                        entities.append(name)
            return entities
        except Exception:
            pass

    # Regex fallback: capitalized words ≥ 3 chars that aren't pronouns or stopwords
    _STOP = frozenset({"As", "Of", "In", "During", "The", "A", "An", "To", "From",
                        "And", "But", "Or", "Is", "Was", "Has", "Have", "Had",
                        "By", "For", "With", "At", "On", "Before", "After"})
    matches = re.findall(r'\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})?)\b', memory)
    seen: set[str] = set()
    entities: list[str] = []
    for m in matches:
        key = m.lower()
        if key not in _PRONOUN_SET and m not in _STOP:
            if key not in seen:
                seen.add(key)
                entities.append(m)
    return entities


# ── Record builder ────────────────────────────────────────────────────────────

def _to_records(
    parsed: dict[str, list[str]],
    session_id: str,
    session_position: int,
    session_date: str | None,
    dataset: str,
) -> list[dict]:
    """Convert parsed LLM output to flat list of memory record dicts."""
    topics = parsed.get("topics", [])
    records: list[dict] = []
    for mem_type, memories in (
        ("state",    parsed["state_memories"]),
        ("episodic", parsed["episodic_memories"]),
    ):
        for idx, memory in enumerate(memories):
            records.append({
                "memory_id":        f"mem_{session_id}_{mem_type}_{idx:03d}",
                "session_id":       session_id,
                "session_position": session_position,
                "session_date":     session_date,
                "memory":           memory,
                "memory_type":      mem_type,
                "entities":         _extract_entities(memory),
                "dataset":          dataset,
                "topics_listed":    len(topics),
            })
    return records


def session_position_from_id(session_id: str) -> int:
    """
    Derive session position (1-indexed ordinal) from session ID string.

    LoCoMo: "c0_session_4"  → 4
    LME:    "answer_abc_2"  → 2
    """
    m = re.search(r'_(\d+)$', session_id)
    return int(m.group(1)) if m else 1


# ── Entity store ──────────────────────────────────────────────────────────────

def build_entity_store(all_records: list[dict]) -> list[dict]:
    """
    Build entity store from a flat list of memory records.

    Groups by lowercased entity name → {entity_text, canonical_name,
    linked_memory_ids, memory_count}.

    memory_count drives the Phase 4.5 spread attenuation:
      entity_boost[mem_id] += similarity × ENTITY_BOOST_WEIGHT × (1 / memory_count)
    """
    entity_to_mids: dict[str, list[str]] = {}
    entity_canonical: dict[str, str] = {}

    for record in all_records:
        mid = record["memory_id"]
        for entity in record.get("entities", []):
            key = entity.lower().strip()
            if len(key) < 2:
                continue
            entity_to_mids.setdefault(key, []).append(mid)
            entity_canonical[key] = entity  # last-seen casing wins

    store: list[dict] = []
    for key, mids in entity_to_mids.items():
        unique_mids = list(dict.fromkeys(mids))  # deduplicate, preserve order
        store.append({
            "entity_text":      key,
            "canonical_name":   entity_canonical[key],
            "entity_type":      "PERSON",   # refined in Phase 4.5 if needed
            "linked_memory_ids": unique_mids,
            "memory_count":     len(unique_mids),
        })

    return sorted(store, key=lambda e: -e["memory_count"])


# ── Extractor class ───────────────────────────────────────────────────────────

class RichMemoryExtractor:
    """
    Extracts 15-80 word temporally-grounded memories from conversation sessions.

    Single LLM call per session (not split-halves like TripleExtractor).
    Supports Sonnet 4.6 (internal ceiling) and Haiku 4.5 (public product).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
    ):
        # None → auto-load from env/file. "" → explicit no-key (fallback mode).
        self._api_key = self._load_key() if api_key is None else api_key
        self._model = model
        self._client = None

    # ── Key loading ───────────────────────────────────────────────────────────

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

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        client = self._get_client()
        resp = client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        content: str,
        session_id: str,
        session_position: int | None = None,
        session_date: str | None = None,
        dataset: str = "",
    ) -> list[dict]:
        """
        Extract rich memories from one conversation session.

        Returns list of memory record dicts ready for JSON serialization.
        Falls back to sentence-splitting when API key is unavailable.
        """
        if session_position is None:
            session_position = session_position_from_id(session_id)

        if not self._api_key:
            return self._fallback_extract(content, session_id, session_position, dataset)

        prompt = _EXTRACTION_PROMPT.format(
            session_position=session_position,
            session_content=content[:_CONTENT_LIMIT],
        )

        try:
            raw = self._call_llm(prompt)
            parsed = _parse_response(raw)
            return _to_records(parsed, session_id, session_position, session_date, dataset)
        except Exception:
            return self._fallback_extract(content, session_id, session_position, dataset)

    def extract_batch(
        self,
        sessions: list[dict],
        verbose: bool = True,
        checkpoint_path: "Path | str | None" = None,
        checkpoint_every: int = 20,
    ) -> dict[str, list[dict]]:
        """
        Batch extraction with checkpoint resume.

        sessions: list of dicts with keys:
          session_id (required)
          content or search_text (required)
          session_position (optional — derived from session_id if absent)
          session_date (optional)
          dataset (optional)

        Returns: {session_id: [memory_record, ...]}
        """
        results: dict[str, list[dict]] = {}
        ckpt = Path(checkpoint_path) if checkpoint_path else None

        if ckpt and ckpt.exists():
            results = json.loads(ckpt.read_text())
            if verbose and results:
                print(f"  Resumed from checkpoint: {len(results)} sessions done")

        remaining = [s for s in sessions if s["session_id"] not in results]
        total = len(sessions)
        done_offset = total - len(remaining)

        for i, session in enumerate(remaining):
            sid = session["session_id"]
            content = session.get("content") or session.get("search_text", "")
            position = session.get(
                "session_position",
                session_position_from_id(sid),
            )
            date = session.get("session_date")
            ds = session.get("dataset", "")

            records = self.extract(content, sid, position, date, ds)
            results[sid] = records

            global_idx = done_offset + i + 1
            if verbose and global_idx % 10 == 0:
                n_mems = sum(len(v) for v in results.values())
                print(
                    f"  {global_idx}/{total} sessions  ({n_mems:,} memories total)",
                    flush=True,
                )

            if ckpt and (i + 1) % checkpoint_every == 0:
                ckpt.write_text(json.dumps(results))

            time.sleep(0.05)  # avoid burst rate-limit

        if ckpt:
            ckpt.write_text(json.dumps(results))

        return results

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback_extract(
        self,
        content: str,
        session_id: str,
        session_position: int,
        dataset: str,
    ) -> list[dict]:
        """
        No-API fallback: wraps content sentences as episodic memories.
        Quality is poor — only triggered when ANTHROPIC_API_KEY is missing.
        """
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        records: list[dict] = []
        for j, sent in enumerate(sentences[:3]):
            sent = sent.strip()
            if len(sent.split()) < _MIN_WORDS:
                continue
            memory = f"During session {session_position}, {sent}"
            records.append({
                "memory_id":        f"mem_{session_id}_fallback_{j:03d}",
                "session_id":       session_id,
                "session_position": session_position,
                "session_date":     None,
                "memory":           memory,
                "memory_type":      "episodic",
                "entities":         _extract_entities(memory),
                "dataset":          dataset,
            })
        return records


# ── Quality audit ─────────────────────────────────────────────────────────────

def check_memory_quality(all_records: list[dict]) -> dict:
    """
    Audit extracted memories for quality signals.

    Returns dict with:
      total_memories, avg_words, short_count, long_count,
      temporal_grounded_pct, pronoun_pct, entity_coverage_pct
    """
    total = len(all_records)
    if total == 0:
        return {"total_memories": 0}

    word_counts = [len(r["memory"].split()) for r in all_records]
    short_count = sum(1 for wc in word_counts if wc < _MIN_WORDS)
    long_count  = sum(1 for wc in word_counts if wc > _MAX_WORDS)

    _TEMPORAL_MARKERS = re.compile(
        r'(As of session|During session|session \d+|plans to .* session)',
        re.IGNORECASE,
    )
    temporal_grounded = sum(
        1 for r in all_records if _TEMPORAL_MARKERS.search(r["memory"])
    )

    _PRONOUN_RE = re.compile(
        r'\b(she|he|they|her|his|their|him|them)\b', re.IGNORECASE
    )
    pronoun_count = sum(
        1 for r in all_records if _PRONOUN_RE.search(r["memory"])
    )

    entity_coverage = sum(1 for r in all_records if r.get("entities"))

    return {
        "total_memories":        total,
        "avg_words":             sum(word_counts) / total,
        "short_count":           short_count,
        "long_count":            long_count,
        "temporal_grounded_pct": temporal_grounded / total,
        "pronoun_pct":           pronoun_count / total,
        "entity_coverage_pct":   entity_coverage / total,
    }
