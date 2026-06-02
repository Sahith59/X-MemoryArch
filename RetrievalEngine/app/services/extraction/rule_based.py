"""
Pre-Phase 2 — RuleBasedExtractionBackend.

Wraps Phase 1's existing memory_service.extract_memories_from_session()
behind the ExtractionBackend protocol. This backend:

  - Is ALWAYS available (offline, no API keys, no external deps).
  - Uses Phase 1's 5-stage pipeline unchanged.
  - Adds Phase 2 fields (extraction_backend, canonical_type, type_label) to
    each Memory row after extraction.
  - canonical_type = memory.type  (Phase 1 type IS the canonical type)
  - type_label = memory.type      (same — no LLM-assigned label)
  - llm_reasoning = None          (rule-based has no reasoning to report)

This backend is the offline fallback for cloud_llm and local_llm.
It is NEVER deleted or removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy.orm import Session

from .base import (
    ExtractionBackend,
    ExtractionConfig,
    P2ExtractionResult,
    register_backend,
)

# Add Phase 1 to sys.path
_P1_ROOT = Path(__file__).resolve().parents[4] / "project-memory-core"
if str(_P1_ROOT) not in sys.path:
    sys.path.insert(0, str(_P1_ROOT))


class RuleBasedExtractionBackend:
    """
    Wraps Phase 1's rule-based extraction pipeline.

    Phase 1 extraction steps (unchanged):
      1. Regex noise gate (NOISE_RE patterns)
      2. Semantic substantive gate (all-MiniLM-L6-v2, threshold 0.30)
      3. Type classifier (cosine vs per-type exemplar centroids)
      4. Structured field extraction (type_metadata via regex)
      5. Embedding + storage

    Phase 2 additions:
      - Sets extraction_backend = 'rule_based' on each Memory row.
      - Sets canonical_type = memory.type (already canonical).
      - Sets type_label = memory.type (same as canonical — no open label).
    """

    @property
    def name(self) -> str:
        return "rule_based"

    def is_available(self) -> bool:
        """Always True — offline, no external dependencies."""
        try:
            from app.services import memory_service  # noqa: F401
            return True
        except ImportError:
            return False

    def extract(
        self,
        session_id: str,
        db: Session,
        config: ExtractionConfig | None = None,
    ) -> P2ExtractionResult:
        """
        Run Phase 1 extraction and wrap result in P2ExtractionResult.

        Also stamps extraction_backend, canonical_type, type_label on
        each Memory row created by Phase 1.
        """
        from app.services.memory_service import extract_memories_from_session
        from app import models

        # Run Phase 1 extraction
        phase1_result = extract_memories_from_session(session_id=session_id, db=db)

        # Stamp Phase 2 columns on the created memories
        memory_ids = [m.id for m in phase1_result.memories]
        if memory_ids:
            rows = db.query(models.Memory).filter(
                models.Memory.id.in_(memory_ids)
            ).all()
            for row in rows:
                _stamp_phase2_fields(db, row, backend="rule_based")
            db.commit()

        return P2ExtractionResult(
            session_id=session_id,
            memories_created=phase1_result.memories_created,
            summary=phase1_result.summary,
            memory_ids=memory_ids,
            backend_used="rule_based",
            model_used=None,
            chunks_processed=0,      # rule-based doesn't chunk
            noise_filtered_count=0,  # Phase 1 handles this internally
            llm_api_calls=0,
            fallback_used=False,
            skipped_duplicates=phase1_result.skipped_duplicates,
            duplicate_titles=phase1_result.duplicate_titles,
            low_confidence_queued=phase1_result.low_confidence_queued,
            dynamic_type_labels=[],
        )


def _stamp_phase2_fields(db: Session, row, backend: str) -> None:
    """
    Safely add Phase 2 columns to an existing Memory row.

    Handles the case where the migration hasn't been applied yet
    by checking for attribute existence before writing.
    """
    try:
        if hasattr(row, "extraction_backend"):
            row.extraction_backend = backend
        if hasattr(row, "canonical_type"):
            row.canonical_type = row.type  # Phase 1 type IS canonical
        if hasattr(row, "type_label"):
            row.type_label = row.type      # same for rule-based
        if hasattr(row, "llm_reasoning"):
            row.llm_reasoning = None
        if hasattr(row, "contextual_prefix"):
            row.contextual_prefix = None
    except Exception:
        # Migration not applied — silently skip stamping
        pass


# Register singleton
_instance = RuleBasedExtractionBackend()
register_backend(_instance)


# Convenience alias used in tests and router
rule_based_backend: ExtractionBackend = _instance  # type: ignore[assignment]
