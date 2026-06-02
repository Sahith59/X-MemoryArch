"""
Phase 3.8.6 — Memory Tier Router.

Classifies a retrieval query and routes it to the best retrieval tier:

  FACT_TIER   → A6r pipeline (multi-query RRF over extracted facts + reranker)
               Best for: conversational queries about person states, relationships,
               ongoing facts. "What does Alice do for work?", "Where does he live?"

  CHUNK_TIER  → A4mvr pipeline (paragraph chunk max-sim + reranker)
               Best for: episodic / narrative / document queries. "What hotel did they
               stay at?", "What happened during the trip?", "Describe the meeting."

  MERGE_TIER  → RRF blend of FACT_TIER + CHUNK_TIER candidates, then rerank.
               Used when confidence is below MERGE_THRESHOLD — hedges both bets.

Routing is purely rule-based (no ML model required). Rules are derived from
empirical analysis of LoCoMo vs LME queries:

  LoCoMo (conversational): queries about people's state → FACT_TIER wins (+5% vs CHUNK_TIER)
  LME (episodic):          queries about specific events → CHUNK_TIER wins (+4.5% vs FACT_TIER)
  SQuAD (passages):        factual → CHUNK_TIER wins (paragraph max-sim excels)

The goal of A9r is to pick the per-query winner rather than committing one approach
to the whole dataset. Even routing 60% of queries correctly should improve aggregate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class MemoryTier(str, Enum):
    FACT = "fact"       # → A6r path
    CHUNK = "chunk"     # → A4mvr path
    MERGE = "merge"     # → RRF(A6r, A4mvr) then rerank


@dataclass
class RouterResult:
    tier: MemoryTier
    confidence: float
    score_fact: float
    score_chunk: float
    signals_fired: list[str]


# ── Routing threshold ────────────────────────────────────────────────────────

# If |score_fact - score_chunk| < MARGIN → use MERGE_TIER (uncertain)
ROUTE_MARGIN = 1.5

# ── Signal patterns ───────────────────────────────────────────────────────────

# FACT_TIER signals: queries about ongoing person states, relationships, attributes.
# These map well to extracted atomic facts ("Alice is a nurse at City Hospital").
_FACT_PATTERNS: list[tuple[str, float]] = [
    # Person-state: what does/is/are + pronoun/name
    (r"\bwhat does\b", 2.0),
    (r"\bwhat is (?:her|his|their|its|the user.s)\b", 2.0),
    (r"\bwhere does\b", 2.0),
    (r"\bwhere (is|are) (she|he|they|the user)\b", 2.0),
    (r"\bwho is\b", 1.5),
    (r"\bwho does\b", 1.5),
    # Ongoing-state verbs (present tense)
    (r"\b(work|live|study|go to school|partner|dating|married|boyfriend|girlfriend)\b", 1.5),
    (r"\b(hobby|hobbies|interest|like to|loves? to|enjoy|prefer|favorite)\b", 1.5),
    # Relationship vocabulary
    (r"\b(friend|colleague|coworker|sibling|parent|child|boss|manager|team)\b", 1.0),
    # Current state words
    (r"\b(currently|still|anymore|nowadays|at the moment|these days|right now)\b", 1.5),
    (r"\b(current|present|now|today.s|latest)\b.{0,15}\b(job|work|address|home|school|partner|role)\b", 2.0),
    # Person pronoun as subject
    (r"^(what|where|who|how)\b.{0,40}\b(she|he|her|him|they|the user|this person)\b", 1.0),
]

# CHUNK_TIER signals: queries about specific events, narratives, details.
# These map well to paragraph chunks (the answer is buried in one specific passage).
_CHUNK_PATTERNS: list[tuple[str, float]] = [
    # Event/narrative verbs
    (r"\bwhat (happened|occurred|took place)\b", 2.5),
    (r"\bwhen did\b", 2.0),
    (r"\bwhat was the\b", 1.5),          # "what was the hotel/cost/outcome"
    (r"\bwhat were the\b", 1.5),
    (r"\bhow (much|many|long|far|often|frequently)\b", 2.0),
    # Specific-detail vocabulary
    (r"\b(hotel|restaurant|flight|booking|reservation|trip|vacation|tour)\b", 2.0),
    (r"\b(meeting|conference|event|party|ceremony|wedding|appointment)\b", 1.5),
    (r"\b(cost|price|amount|total|budget|fee|charge|salary|pay)\b", 2.0),
    (r"\b(address|location|place|venue|city|country|destination)\b", 1.0),
    # Narrative / descriptive
    (r"\b(describe|explain|summarize|tell me about|overview of)\b", 2.0),
    (r"\b(what did they (do|say|discuss|decide|plan|agree))\b", 1.5),
    (r"\b(during|after|before|while|when) (the|their|his|her)\b", 1.5),
    # Past-event narrative structure
    (r"\bwhat (led|caused|resulted|happened after|happened before)\b", 2.0),
    # "What was [noun]" structure (specific-fact lookup)
    (r"\bwhat was (?:the |a |their |his |her )?(?:name|reason|outcome|result|reaction|response|decision|plan)\b", 2.0),
]

# Compile all patterns once at import time
_COMPILED_FACT = [(re.compile(p, re.IGNORECASE), w) for p, w in _FACT_PATTERNS]
_COMPILED_CHUNK = [(re.compile(p, re.IGNORECASE), w) for p, w in _CHUNK_PATTERNS]


# ── Main router ──────────────────────────────────────────────────────────────

class MemoryTierRouter:
    """
    Route a memory retrieval query to FACT_TIER, CHUNK_TIER, or MERGE_TIER.

    Usage:
        router = MemoryTierRouter()
        result = router.route("What does Alice do for work?")
        # RouterResult(tier=MemoryTier.FACT, confidence=0.87, ...)
    """

    def route(self, query: str) -> RouterResult:
        if not query or not query.strip():
            return RouterResult(
                tier=MemoryTier.MERGE, confidence=0.5,
                score_fact=0.0, score_chunk=0.0, signals_fired=[]
            )

        score_fact = 0.0
        score_chunk = 0.0
        signals: list[str] = []

        # Score against FACT patterns
        for pattern, weight in _COMPILED_FACT:
            if pattern.search(query):
                score_fact += weight
                signals.append(f"fact:{pattern.pattern[:30]}")

        # Score against CHUNK patterns
        for pattern, weight in _COMPILED_CHUNK:
            if pattern.search(query):
                score_chunk += weight
                signals.append(f"chunk:{pattern.pattern[:30]}")

        # Length heuristic: very short queries (<= 6 words) tend to be factual
        word_count = len(query.split())
        if word_count <= 6:
            score_fact += 0.5

        # Long queries (>= 14 words) tend to be narrative/broad
        if word_count >= 14:
            score_chunk += 0.5

        # Decide tier
        margin = abs(score_fact - score_chunk)
        if margin < ROUTE_MARGIN:
            tier = MemoryTier.MERGE
            confidence = 0.5 + margin / (2 * ROUTE_MARGIN) * 0.3  # 0.5–0.65 range
        elif score_fact > score_chunk:
            tier = MemoryTier.FACT
            confidence = min(0.95, 0.65 + score_fact / (score_fact + score_chunk + 1e-6) * 0.3)
        else:
            tier = MemoryTier.CHUNK
            confidence = min(0.95, 0.65 + score_chunk / (score_fact + score_chunk + 1e-6) * 0.3)

        return RouterResult(
            tier=tier,
            confidence=round(confidence, 3),
            score_fact=round(score_fact, 2),
            score_chunk=round(score_chunk, 2),
            signals_fired=signals,
        )

    def route_batch(self, queries: list[str]) -> list[RouterResult]:
        return [self.route(q) for q in queries]

    def distribution(self, queries: list[str]) -> dict[str, int]:
        """Returns tier distribution for a list of queries. Useful for debugging."""
        results = self.route_batch(queries)
        dist: dict[str, int] = {t.value: 0 for t in MemoryTier}
        for r in results:
            dist[r.tier.value] += 1
        return dist
