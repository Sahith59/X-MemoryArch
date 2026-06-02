"""
Phase 3.8 — LoopOrchestrator.

Runs all five memory loops for a given project in sequence. Handles errors per
loop in isolation — one failing loop doesn't abort the others. Provides a
summary across all loops.

Usage:
    from app.services.memory_loops import LoopOrchestrator

    orchestrator = LoopOrchestrator(db)
    summary = orchestrator.run_all("proj-123")
    print(summary.total_proposed, summary.total_executed, summary.total_pending)

    # Dry run (no DB writes):
    summary = orchestrator.run_all("proj-123", dry_run=True)

    # Run individual loops:
    result = orchestrator.run_loop("promotion", "proj-123")

    # Execute a human-approved pending action:
    success = orchestrator.approve_and_execute("action-uuid-here", "proj-123")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.services.memory_loops.base import LoopResult, LoopSafetyError
from app.services.memory_loops.promotion import PromotionLoop
from app.services.memory_loops.staleness import StalenessLoop
from app.services.memory_loops.deduplication import DeduplicationLoop
from app.services.memory_loops.review import ReviewLoop
from app.services.memory_loops.feedback import FeedbackLoop

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorSummary:
    project_id: str
    loop_results: list[LoopResult] = field(default_factory=list)
    failed_loops: list[str] = field(default_factory=list)
    run_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_proposed(self) -> int:
        return sum(r.actions_proposed for r in self.loop_results)

    @property
    def total_executed(self) -> int:
        return sum(r.actions_executed for r in self.loop_results)

    @property
    def total_pending(self) -> int:
        return sum(r.actions_pending for r in self.loop_results)

    @property
    def total_skipped(self) -> int:
        return sum(r.actions_skipped for r in self.loop_results)

    @property
    def total_errors(self) -> int:
        return sum(len(r.errors) for r in self.loop_results)

    def summary(self) -> str:
        lines = [
            f"[orchestrator] project={self.project_id}",
            f"  proposed={self.total_proposed} executed={self.total_executed} "
            f"pending={self.total_pending} skipped={self.total_skipped}",
        ]
        for r in self.loop_results:
            lines.append(f"  {r.summary()}")
        if self.failed_loops:
            lines.append(f"  FAILED loops: {', '.join(self.failed_loops)}")
        return "\n".join(lines)


class LoopOrchestrator:
    """Runs all Phase 3.8 memory maintenance loops."""

    # Default run order: read-only / safe loops first, then learning loops
    DEFAULT_LOOP_ORDER = [
        "promotion",
        "staleness",
        "deduplication",
        "review",
        "feedback",
    ]

    def __init__(
        self,
        db: Session,
        feedback_weights_path: Path | None = None,
    ) -> None:
        self.db = db
        self._loops: dict[str, object] = {
            "promotion":    PromotionLoop(db),
            "staleness":    StalenessLoop(db),
            "deduplication": DeduplicationLoop(db),
            "review":       ReviewLoop(db),
            "feedback":     FeedbackLoop(db, weights_output_path=feedback_weights_path),
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def run_all(
        self,
        project_id: str,
        dry_run: bool = False,
        loops: list[str] | None = None,
    ) -> OrchestratorSummary:
        """
        Run all loops (or a specified subset) for project_id.

        Args:
            project_id: The project to scan.
            dry_run:     If True, compute triggers without writing to DB.
            loops:       Optional list of loop names to run. Default: all.
        """
        order = loops or self.DEFAULT_LOOP_ORDER
        summary = OrchestratorSummary(project_id=project_id)

        for loop_name in order:
            if loop_name not in self._loops:
                logger.warning("Unknown loop '%s' — skipping", loop_name)
                continue
            try:
                result = self._loops[loop_name].run(project_id, dry_run=dry_run)
                summary.loop_results.append(result)
            except LoopSafetyError as exc:
                logger.error("[%s] SAFETY VIOLATION: %s", loop_name, exc)
                summary.failed_loops.append(f"{loop_name}:safety_error")
            except Exception as exc:
                logger.exception("[%s] Unexpected error: %s", loop_name, exc)
                summary.failed_loops.append(loop_name)

        logger.info(summary.summary())
        return summary

    def run_loop(self, loop_name: str, project_id: str, dry_run: bool = False) -> LoopResult:
        """Run a single named loop."""
        if loop_name not in self._loops:
            raise ValueError(f"Unknown loop: '{loop_name}'. Valid: {list(self._loops)}")
        return self._loops[loop_name].run(project_id, dry_run=dry_run)

    def approve_and_execute(self, action_id: str, loop_name: str) -> bool:
        """
        Human-approve a pending loop action and execute it immediately.

        Args:
            action_id: The LoopAction.id to approve and execute.
            loop_name: Which loop owns this action (used to find the right handler).

        Returns:
            True if executed successfully.
        """
        from app.p2_models import LoopAction

        action = self.db.query(LoopAction).filter(LoopAction.id == action_id).first()
        if action is None:
            logger.warning("approve_and_execute: action %s not found", action_id)
            return False
        if action.executed:
            logger.warning("approve_and_execute: action %s already executed", action_id)
            return False
        if action.human_approved == False:
            logger.warning("approve_and_execute: action %s was rejected", action_id)
            return False

        # Set approved
        action.human_approved = True
        self.db.flush()

        if loop_name not in self._loops:
            logger.error("approve_and_execute: unknown loop '%s'", loop_name)
            return False

        loop = self._loops[loop_name]
        executed = loop.execute_approved_action(action_id)
        if executed:
            self.db.commit()
        return executed

    def reject_action(self, action_id: str) -> bool:
        """Mark a pending action as rejected (human declined)."""
        from app.p2_models import LoopAction

        action = self.db.query(LoopAction).filter(LoopAction.id == action_id).first()
        if action is None:
            return False
        if action.executed:
            return False  # too late to reject
        action.human_approved = False
        self.db.commit()
        return True

    def pending_actions(self, project_id: str, loop_name: str | None = None) -> list:
        """Return all pending (unreviewed) loop actions for a project."""
        from app.p2_models import LoopAction

        q = self.db.query(LoopAction).filter(
            LoopAction.project_id == project_id,
            LoopAction.human_approved.is_(None),  # strictly pending (NULL)
            LoopAction.executed == False,
        )
        if loop_name:
            q = q.filter(LoopAction.loop_name == loop_name)
        return q.order_by(LoopAction.created_at).all()
