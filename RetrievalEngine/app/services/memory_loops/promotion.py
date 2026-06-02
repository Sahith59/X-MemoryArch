"""
Phase 3.8 — PromotionLoop.

Frequently-accessed, high-confidence memories deserve more weight in retrieval.
This loop boosts their importance score and marks them as verified so they
surface earlier in reranking.

Trigger: access_count >= ACCESS_THRESHOLD AND confidence >= CONFIDENCE_THRESHOLD
         AND status='active' AND tier='working'

Action (safe, auto-executed):
  - importance: min(current + 1, 5)
  - review_status: 'verified'

No human approval required — this is non-destructive.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.services.memory_loops.base import BaseLoop, LoopResult

logger = logging.getLogger(__name__)

ACCESS_THRESHOLD = 5        # minimum retrieval hits to qualify
CONFIDENCE_THRESHOLD = 0.80  # minimum confidence score


class PromotionLoop(BaseLoop):
    name = "promotion"

    # ── Trigger ────────────────────────────────────────────────────────────

    def run(self, project_id: str, dry_run: bool = False) -> LoopResult:
        from app.models import Memory  # Phase 1 model via namespace extension

        result = LoopResult(loop_name=self.name, project_id=project_id)

        candidates = (
            self.db.query(Memory)
            .filter(
                Memory.project_id == project_id,
                Memory.status == "active",
                Memory.tier == "working",
                Memory.access_count >= ACCESS_THRESHOLD,
                Memory.confidence >= CONFIDENCE_THRESHOLD,
            )
            .all()
        )

        for mem in candidates:
            # Skip if we already promoted this memory (idempotency)
            if self._already_proposed(project_id, "promote_importance", mem.id):
                result.actions_skipped += 1
                continue

            # Skip if already at max importance and already verified
            if mem.importance >= 5 and mem.review_status == "verified":
                result.actions_skipped += 1
                continue

            new_importance = min(5, (mem.importance or 3) + 1)
            proposed = {
                "changes": {
                    "importance": new_importance,
                    "review_status": "verified",
                },
                "old_values": {
                    "importance": mem.importance,
                    "review_status": mem.review_status,
                },
                "trigger": {
                    "access_count": mem.access_count,
                    "confidence": mem.confidence,
                },
            }
            action = self._propose(
                project_id=project_id,
                action_type="promote_importance",
                proposed=proposed,
                reason=(
                    f"Accessed {mem.access_count} times with confidence "
                    f"{mem.confidence:.2f} — promoting importance "
                    f"{mem.importance}→{new_importance}"
                ),
                target_memory_id=mem.id,
                dry_run=dry_run,
            )
            result.actions_proposed += 1
            if action and action.executed:
                result.actions_executed += 1
            elif action and not action.executed:
                result.actions_pending += 1

        if not dry_run:
            self.db.commit()

        logger.info(result.summary())
        return result

    # ── Handler ────────────────────────────────────────────────────────────

    def _handle_promote_importance(self, action: "LoopAction") -> dict:
        from app.models import Memory

        proposed = action.get_proposed_action()
        changes = proposed.get("changes", {})

        mem = self.db.query(Memory).filter(Memory.id == action.target_memory_id).first()
        if mem is None:
            return {"error": "memory not found"}

        old_importance = mem.importance
        old_review = mem.review_status

        if "importance" in changes:
            mem.importance = changes["importance"]
        if "review_status" in changes:
            mem.review_status = changes["review_status"]

        return {
            "old_importance": old_importance,
            "new_importance": mem.importance,
            "old_review_status": old_review,
            "new_review_status": mem.review_status,
        }
