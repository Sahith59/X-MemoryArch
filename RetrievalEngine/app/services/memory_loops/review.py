"""
Phase 3.8 — ReviewLoop.

Surfaces memories that have been repeatedly skipped by retrieval and flags
them for potential archival. Two triggers:

  1. Never-accessed (stale cold) — memories created > COLD_AGE_DAYS days ago
     with access_count == 0. Likely noise or low-relevance facts.

  2. Surfaced but never selected — memories that appear in retrieval candidate
     pools (surfaced_memory_ids in retrieval_runs) but were never in the final
     selection (selected_memory_ids) across SKIP_THRESHOLD or more runs.
     This is the "repeatedly skipped by reranker" signal from the plan.

Trigger 1 action (safe, auto-executed): flag review_status = 'needs_review'
Trigger 2 action (safe, auto-executed): flag review_status = 'needs_review'
Archival action (destructive, requires approval): 'archive' — set status='archived'

Note: The ReviewLoop NEVER archives autonomously. It only flags. A separate
human-reviewed step (execute_approved_action) is needed to actually archive.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.services.memory_loops.base import BaseLoop, LoopResult

logger = logging.getLogger(__name__)

COLD_AGE_DAYS = 30        # how old a never-accessed memory must be to trigger
SKIP_THRESHOLD = 5        # surfaced N times but never selected → low value
MIN_RUNS_FOR_SKIP = 10    # need at least this many retrieval_runs before skip signal is valid


class ReviewLoop(BaseLoop):
    name = "review"

    def run(self, project_id: str, dry_run: bool = False) -> LoopResult:
        from app.models import Memory  # Phase 1 via namespace extension
        from app.p2_models import RetrievalRun

        result = LoopResult(loop_name=self.name, project_id=project_id)
        now = datetime.now(timezone.utc)
        cold_cutoff = now - timedelta(days=COLD_AGE_DAYS)

        # ── Trigger 1: never-accessed cold memories ───────────────────────
        cold_mems = (
            self.db.query(Memory)
            .filter(
                Memory.project_id == project_id,
                Memory.status == "active",
                Memory.access_count == 0,
                Memory.created_at <= cold_cutoff,
            )
            .all()
        )

        for mem in cold_mems:
            result = self._flag_for_review(
                mem, project_id,
                reason=(
                    f"Never accessed in {COLD_AGE_DAYS}+ days since creation "
                    f"({mem.created_at.date() if mem.created_at else 'unknown'})"
                ),
                result=result,
                dry_run=dry_run,
                trigger="cold_never_accessed",
            )

        # ── Trigger 2: surfaced but never selected ────────────────────────
        runs = (
            self.db.query(RetrievalRun)
            .filter(
                RetrievalRun.project_id == project_id,
                RetrievalRun.surfaced_memory_ids != None,
            )
            .all()
        )

        if len(runs) >= MIN_RUNS_FOR_SKIP:
            surfaced_count: Counter[str] = Counter()
            selected_count: Counter[str] = Counter()

            for run in runs:
                for mid in run.get_surfaced_ids():
                    surfaced_count[mid] += 1
                for mid in run.get_selected_ids():
                    selected_count[mid] += 1

            always_skipped = {
                mid for mid, cnt in surfaced_count.items()
                if cnt >= SKIP_THRESHOLD and selected_count.get(mid, 0) == 0
            }

            if always_skipped:
                skip_mems = (
                    self.db.query(Memory)
                    .filter(
                        Memory.project_id == project_id,
                        Memory.status == "active",
                        Memory.id.in_(always_skipped),
                    )
                    .all()
                )

                for mem in skip_mems:
                    result = self._flag_for_review(
                        mem, project_id,
                        reason=(
                            f"Surfaced {surfaced_count[mem.id]} times but never selected "
                            f"by retrieval — likely low-value for this project"
                        ),
                        result=result,
                        dry_run=dry_run,
                        trigger="surfaced_never_selected",
                    )

        if not dry_run:
            self.db.commit()

        logger.info(result.summary())
        return result

    # ── Internal ───────────────────────────────────────────────────────────

    def _flag_for_review(
        self,
        mem: "Memory",
        project_id: str,
        reason: str,
        result: LoopResult,
        dry_run: bool,
        trigger: str,
    ) -> LoopResult:
        if self._already_proposed(project_id, "flag_low_value", mem.id):
            result.actions_skipped += 1
            return result

        if mem.review_status == "needs_review":
            result.actions_skipped += 1
            return result

        proposed = {
            "changes": {"review_status": "needs_review"},
            "old_values": {"review_status": mem.review_status},
            "trigger": trigger,
        }
        action = self._propose(
            project_id=project_id,
            action_type="flag_low_value",
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

    def _handle_flag_low_value(self, action: "LoopAction") -> dict:
        from app.models import Memory

        mem = self.db.query(Memory).filter(Memory.id == action.target_memory_id).first()
        if mem is None:
            return {"error": "memory not found"}

        old = mem.review_status
        mem.review_status = "needs_review"
        return {"old_review_status": old, "new_review_status": "needs_review"}

    def _handle_archive(self, action: "LoopAction") -> dict:
        """Archives the target memory. Requires human_approved=True."""
        from app.models import Memory

        mem = self.db.query(Memory).filter(Memory.id == action.target_memory_id).first()
        if mem is None:
            return {"error": "memory not found"}

        old_status = mem.status
        mem.status = "archived"
        return {"old_status": old_status, "new_status": "archived"}
