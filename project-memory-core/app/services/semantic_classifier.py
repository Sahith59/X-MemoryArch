"""
Sub-phases 1.20 + 1.40 — Semantic classifier using all-MiniLM-L6-v2.

Sub-phase 1.40 changes:
  - TECHNICAL_EXEMPLARS → SUBSTANTIVE_EXEMPLARS covering all 16 domains.
    The gate now accepts design, research, business, marketing, and personal
    work content in addition to software engineering.
  - classify() now accepts a domain parameter and uses domain-weighted
    centroids (70% domain-group, 30% global) for more precise type assignment.
  - All exemplar sentences are encoded in a single batch at startup to avoid
    redundant model calls. Cost: ~1 s extra startup time (one-time).

Public API (unchanged from sub-phase 1.20):
    is_technical(sentence)              -> bool
    classify(sentence, domain)          -> (type, conf) or None
    embed_text(text)                    -> bytes

Design decisions:
  - Model loaded lazily — server startup stays fast.
  - Centroids computed once and cached in module scope.
  - Cosine similarity on unit-normed vectors == dot product (fast).
  - All numpy operations stay in this module; callers never touch ndarray.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

# Stage 1 gate: max cosine similarity against any SUBSTANTIVE exemplar vector.
# Centroid averaging loses fine-grained domain specificity; max-sim preserves it.
# 0.30 calibrated against real filler (max 0.18) vs shortest real content (min 0.25+).
TECH_GATE_THRESHOLD: float = 0.30

# Stage 2 classifier: cosine similarity against the winning type centroid.
# Multi-domain exemplar blending dilutes per-type centroids; 0.23 is the
# calibrated floor that rejects ambiguous sentences without losing real content.
TYPE_CONF_THRESHOLD: float = 0.23

# ---------------------------------------------------------------------------
# Module-level lazy singletons
# ---------------------------------------------------------------------------

_model = None

_substantive_centroid: Optional[np.ndarray] = None      # gate centroid  (384,) — kept for backward compat
_substantive_vecs: Optional[np.ndarray] = None          # (n_exemplars, 384) — used by max-sim gate
_type_centroids: Optional[dict[str, np.ndarray]] = None  # global per-type (384,) each
_domain_type_centroids: Optional[dict[str, dict[str, np.ndarray]]] = None  # per-group

# Keep old name as alias so any external code that reads the private var works.
_tech_centroid: Optional[np.ndarray] = None  # set to _substantive_centroid after load


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for semantic extraction. "
                "Run: pip install sentence-transformers"
            ) from exc
        logger.info("Loading all-MiniLM-L6-v2 (first extraction call — one-time cost)…")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Semantic model ready.")
    return _model


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _ensure_centroids() -> None:
    """
    Lazy one-time computation of all centroids.

    Encodes every unique exemplar sentence in a single model.encode() call
    to minimise startup overhead, then derives:
      _substantive_centroid       — gate (all-domain)
      _type_centroids             — per type, global (all domains blended)
      _domain_type_centroids      — per type per domain-group (for blending)
    """
    global _substantive_centroid, _substantive_vecs, _type_centroids, _domain_type_centroids, _tech_centroid

    if _substantive_centroid is not None:
        return  # already computed

    from app.services.exemplars import SUBSTANTIVE_EXEMPLARS, EXEMPLARS, DOMAIN_GROUP

    model = _get_model()
    logger.info("Computing universal exemplar centroids (one-time startup cost)…")

    # Collect every unique sentence across gate + type banks
    all_exs: list[str] = list({
        s for s in SUBSTANTIVE_EXEMPLARS + [
            ex
            for domain_map in EXEMPLARS.values()
            for exs in domain_map.values()
            for ex in exs
        ]
    })
    vecs = model.encode(all_exs, normalize_embeddings=True, show_progress_bar=False)
    vec_map: dict[str, np.ndarray] = dict(zip(all_exs, vecs))

    # ── Substantive gate vectors — (n_exemplars, 384) matrix for max-sim ────
    sub_vecs = np.array([vec_map[s] for s in SUBSTANTIVE_EXEMPLARS])
    _substantive_vecs = sub_vecs  # used by is_technical() via max-sim

    # ── Substantive gate centroid — kept for backward compat ─────────────────
    _substantive_centroid = _norm(sub_vecs.mean(axis=0))
    _tech_centroid = _substantive_centroid  # backward compat alias

    # ── Global per-type centroids (all domain examples flattened) ────────────
    _type_centroids = {}
    for mem_type, domain_map in EXEMPLARS.items():
        all_type_exs = [ex for exs in domain_map.values() for ex in exs]
        _type_centroids[mem_type] = _norm(
            np.array([vec_map[s] for s in all_type_exs]).mean(axis=0)
        )

    # ── Per-domain-group per-type centroids ──────────────────────────────────
    canonical_groups = set(DOMAIN_GROUP.values())  # software, design, research, business, marketing, general
    _domain_type_centroids = {g: {} for g in canonical_groups}
    for mem_type, domain_map in EXEMPLARS.items():
        for group in canonical_groups:
            # Domain-group exemplars + general exemplars (for coverage when group-specific set is small)
            group_exs = domain_map.get(group, []) + domain_map.get("general", [])
            if not group_exs:
                continue
            _domain_type_centroids[group][mem_type] = _norm(
                np.array([vec_map[s] for s in group_exs]).mean(axis=0)
            )

    logger.info("Centroids ready: gate + %d types × %d domain groups.",
                len(_type_centroids), len(canonical_groups))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_technical(sentence: str) -> bool:
    """
    Return True when the sentence is substantive work content (any domain).
    Returns False for social filler, greetings, motivational platitudes, etc.

    Uses max-cosine-similarity against all SUBSTANTIVE_EXEMPLARS (not centroid)
    so domain-specific sentences that are far from the average still pass.
    """
    _ensure_centroids()
    model = _get_model()
    vec = model.encode(sentence, normalize_embeddings=True, show_progress_bar=False)
    # _substantive_vecs is (n, 384); @ vec gives (n,) dot products == cosine sims
    return float(np.max(_substantive_vecs @ vec)) >= TECH_GATE_THRESHOLD


def classify(sentence: str, domain: str = "general") -> Optional[tuple[str, float]]:
    """
    Classify a sentence into a memory type.

    With domain provided, blends 70% domain-specific centroid with 30% global
    centroid for each type before picking argmax. Falls back to global centroid
    when a domain-specific one is unavailable.

    Returns (memory_type_str, confidence) or None when below TYPE_CONF_THRESHOLD.
    """
    _ensure_centroids()

    from app.services.exemplars import DOMAIN_GROUP
    model = _get_model()

    vec = model.encode(sentence, normalize_embeddings=True, show_progress_bar=False)

    group = DOMAIN_GROUP.get(domain, "general")
    # For general domain, use global centroids (pure average across all domains).
    # Domain blending is only applied when the project has a specific known domain.
    if group == "general":
        domain_centroids: dict[str, np.ndarray] = {}
    else:
        domain_centroids = (_domain_type_centroids or {}).get(group, {})

    best_type: Optional[str] = None
    best_score: float = -1.0

    for mem_type, global_c in (_type_centroids or {}).items():
        if mem_type in domain_centroids:
            blended = 0.7 * domain_centroids[mem_type] + 0.3 * global_c
            n = float(np.linalg.norm(blended))
            centroid = blended / n if n > 0 else blended
        else:
            centroid = global_c

        score = float(np.dot(vec, centroid))
        if score > best_score:
            best_score = score
            best_type = mem_type

    if best_score < TYPE_CONF_THRESHOLD or best_type is None:
        return None
    return best_type, best_score


def classify_best(sentence: str, domain: str = "general") -> tuple[str, float]:
    """
    Return the best-matching type and its score regardless of TYPE_CONF_THRESHOLD.
    Used by the low-confidence queue to capture the classifier's top guess even
    when confidence is too low to auto-store.
    """
    _ensure_centroids()
    from app.services.exemplars import DOMAIN_GROUP
    model = _get_model()
    vec = model.encode(sentence, normalize_embeddings=True, show_progress_bar=False)
    group = DOMAIN_GROUP.get(domain, "general")
    domain_centroids: dict[str, np.ndarray] = {} if group == "general" else \
        (_domain_type_centroids or {}).get(group, {})
    best_type: str = "unclassified"
    best_score: float = -1.0
    for mem_type, global_c in (_type_centroids or {}).items():
        if mem_type in domain_centroids:
            blended = 0.7 * domain_centroids[mem_type] + 0.3 * global_c
            n = float(np.linalg.norm(blended))
            centroid = blended / n if n > 0 else blended
        else:
            centroid = global_c
        score = float(np.dot(vec, centroid))
        if score > best_score:
            best_score = score
            best_type = mem_type
    return best_type, best_score


def embed_text(text: str) -> bytes:
    """
    Produce a Phase-2-ready embedding for the given text.
    Stored as raw float32 bytes. Reconstruct with:
        vec = np.frombuffer(embedding_bytes, dtype=np.float32)
    """
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)
    return vec.tobytes()


# ---------------------------------------------------------------------------
# Batch API — used by the extraction pipeline to encode all sentences in one
# model call instead of N individual calls (3× faster for long sessions).
# ---------------------------------------------------------------------------

def batch_encode(texts: list[str]) -> "np.ndarray":
    """
    Encode a list of texts in a single model call.
    Returns a (len(texts), 384) float32 array, unit-normed row-wise.
    """
    _ensure_centroids()
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=32)


def is_technical_vec(vec: "np.ndarray") -> bool:
    """Gate check using a pre-computed unit-normed vector."""
    _ensure_centroids()
    return float(np.max(_substantive_vecs @ vec)) >= TECH_GATE_THRESHOLD


def _run_classifier(vec: "np.ndarray", domain_centroids: dict, ) -> tuple[Optional[str], float]:
    best_type: Optional[str] = None
    best_score: float = -1.0
    for mem_type, global_c in (_type_centroids or {}).items():
        if mem_type in domain_centroids:
            blended = 0.7 * domain_centroids[mem_type] + 0.3 * global_c
            n = float(np.linalg.norm(blended))
            centroid = blended / n if n > 0 else blended
        else:
            centroid = global_c
        score = float(np.dot(vec, centroid))
        if score > best_score:
            best_score = score
            best_type = mem_type
    return best_type, best_score


def classify_vec(vec: "np.ndarray", domain: str = "general") -> Optional[tuple[str, float]]:
    """Classify using a pre-computed unit-normed vector."""
    _ensure_centroids()
    from app.services.exemplars import DOMAIN_GROUP
    group = DOMAIN_GROUP.get(domain, "general")
    domain_centroids = {} if group == "general" else (_domain_type_centroids or {}).get(group, {})
    best_type, best_score = _run_classifier(vec, domain_centroids)
    if best_score < TYPE_CONF_THRESHOLD or best_type is None:
        return None
    return best_type, best_score


def classify_best_vec(vec: "np.ndarray", domain: str = "general") -> tuple[str, float]:
    """classify_best using a pre-computed unit-normed vector."""
    _ensure_centroids()
    from app.services.exemplars import DOMAIN_GROUP
    group = DOMAIN_GROUP.get(domain, "general")
    domain_centroids = {} if group == "general" else (_domain_type_centroids or {}).get(group, {})
    best_type, best_score = _run_classifier(vec, domain_centroids)
    return best_type or "unclassified", best_score


def vec_to_bytes(vec: "np.ndarray") -> bytes:
    """Convert a unit-normed float32 vector to storage bytes (avoids a re-encode)."""
    return vec.astype(np.float32).tobytes()
