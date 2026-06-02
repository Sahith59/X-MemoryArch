"""
Phase 3.8 — StalenessLoop.

Detects memories that are likely outdated and flags them for human review.
Two triggers fire independently:

  1. Expired valid_until — the memory has a temporal validity window that has passed.
  2. Superseded — another memory explicitly supersedes this one (superseded_by is set
     OR a MemoryLink with relationship_type='supersedes' points to this memory).

Action (safe, auto-executed): set review_status = 'needs_review'
Action (destructive, requires approval): propose_supersede — mark status='archived'

The loop never archives anything autonomously. It only flags.
If the human later approves a propose_supersede action, _handle_propose_supersede
archives the memory.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.services.memory_loops.base import BaseLoop, LoopResult

logger = logging.getLogger(__name__)


class StalenessLoop(BaseLoop):
    name = "staleness"

    def run(self, project_id: str, dry_run: bool = False) -> LoopResult:
        from app.models import Memory, MemoryLink  # Phase 1 via namespace extension

        result = LoopResult(loop_name=self.name, project_id=project_id)
        now = datetime.now(timezone.utc)

        # ── Trigger 1: expired valid_until ────────────────────────────────
        expired = (
            self.db.query(Memory)
            .filter(
                Memory.project_id == project_id,
                Memory.status == "active",
                Memory.valid_until != None,
                Memory.valid_until < now,
            )
            .all()
        )

        for mem in expired:
            result = self._flag_stale(
                mem, project_id,
                reason=f"valid_until={mem.valid_until.isoformat()} has passed",
                result=result,
                dry_run=dry_run,
            )

        # ── Trigger 2: superseded_by field is set ─────────────────────────
        superseded_by_field = (
            self.db.query(Memory)
            .filter(
                Memory.project_id == project_id,
                Memory.status == "active",
                Memory.superseded_by != None,
                # Don't double-flag what we already caught via valid_until
                or_(Memory.valid_until == None, Memory.valid_until >= now),
            )
            .all()
        )

        for mem in superseded_by_field:
            result = self._flag_stale(
                mem, project_id,
                reason=f"superseded_by={mem.superseded_by}",
                result=result,
                dry_run=dry_run,
            )

        # ── Trigger 3: MemoryLink supersedes relationship ─────────────────
        # Find memories that are the TARGET of a "supersedes" link from another memory
        supersede_links = (
            self.db.query(MemoryLink)
            .filter(MemoryLink.relationship_type == "supersedes")
            .all()
        )
        superseded_ids = {link.target_memory_id for link in supersede_links}

        if superseded_ids:
            link_superseded = (
                self.db.query(Memory)
                .filter(
                    Memory.project_id == project_id,
                    Memory.status == "active",
                    Memory.id.in_(superseded_ids),
                    # Skip already-caught by triggers 1+2
                    Memory.valid_until == None,
                    Memory.superseded_by == None,
                )
                .all()
            )

            for mem in link_superseded:
                result = self._flag_stale(
                    mem, project_id,
                    reason="superseded via MemoryLink from a newer memory",
                    result=result,
                    dry_run=dry_run,
                )

        if not dry_run:
            self.db.commit()

        logger.info(result.summary())
        return result

    # ── Internal ───────────────────────────────────────────────────────────

    def _flag_stale(
        self,
        mem: "Memory",
        project_id: str,
        reason: str,
        result: LoopResult,
        dry_run: bool,
    ) -> LoopResult:
        """Flag one memory as needs_review. Safe — auto-executed."""
        if self._already_proposed(project_id, "flag_stale", mem.id):
            result.actions_skipped += 1
            return result

        # If memory is already flagged for review, skip
        if mem.review_status == "needs_review":
            result.actions_skipped += 1
            return result

        proposed = {
            "changes": {"review_status": "needs_review"},
            "old_values": {"review_status": mem.review_status},
            "stale_reason": reason,
        }
        action = self._propose(
            project_id=project_id,
            action_type="flag_stale",
            proposed=proposed,
            reason=reason,
            target_memory_id=mem.id,
            dry_run=dry_run,
        )
        result.actions_proposed += 1
        if action and action.executed:
            result.actions_executed += 1
        else:
            result.actions_pending += 1
        return result

    # ── Handlers ───────────────────────────────────────────────────────────

    def _handle_flag_stale(self, action: "LoopAction") -> dict:
        from app.models import Memory

        mem = self.db.query(Memory).filter(Memory.id == action.target_memory_id).first()
        if mem is None:
            return {"error": "memory not found"}

        old = mem.review_status
        mem.review_status = "needs_review"
        return {"old_review_status": old, "new_review_status": "needs_review"}

    def _handle_propose_supersede(self, action: "LoopAction") -> dict:
        """Archives the target memory. Only called after human_approved=True."""
        from app.models import Memory

        mem = self.db.query(Memory).filter(Memory.id == action.target_memory_id).first()
        if mem is None:
            return {"error": "memory not found"}

        old_status = mem.status
        mem.status = "archived"
        return {"old_status": old_status, "new_status": "archived"}
