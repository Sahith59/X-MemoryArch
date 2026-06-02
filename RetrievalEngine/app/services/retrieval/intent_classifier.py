"""
Sub-phase 2.5 — Query intent classifier.

Classifies query intent heuristically first (keyword matching, regex).
Falls back to an optional LLM callable when confidence is below threshold.

Intents:
  temporal     — queries about time, recency, history, change over time
  factual      — what/who/define/describe fact lookups
  code         — file paths, symbols, function names, imports
  exploratory  — how/why/explain/overview/architecture questions
  general      — fallback when no strong signal

Usage:
  from app.services.retrieval.intent_classifier import classify_intent, IntentLabel
  intent, confidence = classify_intent("what changed in auth.py last week")
  # → (IntentLabel.temporal, 0.85)
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Callable


class IntentLabel(str, Enum):
    temporal = "temporal"
    factual = "factual"
    code = "code"
    exploratory = "exploratory"
    general = "general"


# ---------------------------------------------------------------------------
# Heuristic keyword signals
# ---------------------------------------------------------------------------

_TEMPORAL_PATTERNS = re.compile(
    r"\b(when|latest|recent|last|before|after|ago|yesterday|today|"
    r"history|changed|updated|old|new|current|previously|since|"
    r"earliest|newest|expire|expir|valid until|outdated|deprecated|"
    r"what\s+was|used\s+to|no\s+longer|was\s+changed|was\s+updated)\b",
    re.IGNORECASE,
)

_FACTUAL_PATTERNS = re.compile(
    r"\b(what\s+is|what\s+are|who\s+is|define|definition|describe|"
    r"tell\s+me\s+(about|what)|show\s+me|list\s+(all|the)|"
    r"what\s+does|means|meaning|value\s+of|configuration\s+for|"
    r"settings?\s+for|parameter|endpoint)\b",
    re.IGNORECASE,
)

_CODE_PATTERNS = re.compile(
    r"(?:"
    r"[a-zA-Z_][\w]*\.[a-z]{1,5}\b"   # file.ext
    r"|def\s+\w+"                       # Python def
    r"|class\s+\w+"                     # class
    r"|function\s+\w+"                  # JS function
    r"|import\s+[\w.]+"                # import
    r"|from\s+[\w.]+\s+import"         # from … import
    r"|/[\w./]+"                        # path
    r"|#\w+"                            # anchor
    r")",
    re.IGNORECASE,
)

_EXPLORATORY_PATTERNS = re.compile(
    r"\b(how\s+(does|do|did|to|can|should)|why\s+(is|are|was|were|do|did)|"
    r"explain|overview|summarize|understand|learn|architecture|"
    r"design\s+(of|for|behind)|approach|strategy|reasoning|decision\s+behind|"
    r"trade.?off|alternatives?|options?|what\s+led|how\s+we|"
    r"philosophy|principles?|pattern)\b",
    re.IGNORECASE,
)

_LLM_PROMPT_TEMPLATE = """You are a query intent classifier. Classify the query into exactly one of these intents:
temporal, factual, code, exploratory, general

Query: {query}

Respond with ONLY the intent label (one word), nothing else."""


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_intent(
    query: str,
    llm_fn: Callable[[str], str] | None = None,
    llm_confidence_threshold: float = 0.6,
) -> tuple[IntentLabel, float]:
    """
    Classify the intent of a retrieval query.

    Returns:
        (IntentLabel, confidence)  — confidence in [0.0, 1.0]

    Strategy:
      1. Check for code signals first (most distinctive).
      2. Check temporal, factual, exploratory in priority order.
      3. Combine hits into a confidence score.
      4. If confidence < llm_confidence_threshold AND llm_fn provided, ask LLM.
    """
    if not query or not query.strip():
        return IntentLabel.general, 0.5

    lower = query.lower().strip()

    # Count signal hits per intent
    scores: dict[IntentLabel, float] = {
        IntentLabel.temporal: 0.0,
        IntentLabel.factual: 0.0,
        IntentLabel.code: 0.0,
        IntentLabel.exploratory: 0.0,
        IntentLabel.general: 0.3,  # baseline prior for fallback
    }

    temporal_hits = len(_TEMPORAL_PATTERNS.findall(lower))
    factual_hits = len(_FACTUAL_PATTERNS.findall(lower))
    code_hits = len(_CODE_PATTERNS.findall(query))
    exploratory_hits = len(_EXPLORATORY_PATTERNS.findall(lower))

    # Weight by hit count (diminishing returns after 2 hits)
    scores[IntentLabel.temporal] += _hit_score(temporal_hits)
    scores[IntentLabel.factual] += _hit_score(factual_hits)
    scores[IntentLabel.code] += _hit_score(code_hits) * 1.2  # code is distinctive
    scores[IntentLabel.exploratory] += _hit_score(exploratory_hits)

    # Pick the winner
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    # Normalise to [0, 1]
    confidence = min(1.0, best_score / 1.0) if best_score > 0 else 0.3

    # If confidence is high enough, return heuristic result
    if confidence >= llm_confidence_threshold or llm_fn is None:
        return best_intent, round(min(confidence, 1.0), 3)

    # LLM fallback for low-confidence cases
    try:
        prompt = _LLM_PROMPT_TEMPLATE.format(query=query)
        llm_response = llm_fn(prompt).strip().lower()
        # Map response to IntentLabel
        for intent in IntentLabel:
            if intent.value in llm_response:
                return intent, 0.80
    except Exception:
        pass

    return best_intent, round(min(confidence, 1.0), 3)


# ---------------------------------------------------------------------------
# Intent → ranking weight presets
# ---------------------------------------------------------------------------

def get_intent_weights(intent: IntentLabel) -> dict[str, float]:
    """
    Return the 5-signal ranking weight vector for a given intent.

    Signals: rrf, decay, importance, quality, entity
    All weights sum to 1.0.
    """
    _WEIGHTS = {
        IntentLabel.temporal: {
            "rrf": 0.40, "decay": 0.30, "importance": 0.10,
            "quality": 0.10, "entity": 0.10,
        },
        IntentLabel.factual: {
            "rrf": 0.50, "decay": 0.05, "importance": 0.15,
            "quality": 0.20, "entity": 0.10,
        },
        IntentLabel.code: {
            "rrf": 0.55, "decay": 0.05, "importance": 0.10,
            "quality": 0.10, "entity": 0.20,
        },
        IntentLabel.exploratory: {
            "rrf": 0.45, "decay": 0.10, "importance": 0.15,
            "quality": 0.20, "entity": 0.10,
        },
        IntentLabel.general: {
            "rrf": 0.50, "decay": 0.10, "importance": 0.15,
            "quality": 0.15, "entity": 0.10,
        },
    }
    return _WEIGHTS[intent]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hit_score(n_hits: int) -> float:
    """Convert hit count to a confidence contribution (diminishing returns)."""
    if n_hits == 0:
        return 0.0
    if n_hits == 1:
        return 0.65
    if n_hits == 2:
        return 0.85
    return 0.95  # 3+ hits → very confident
