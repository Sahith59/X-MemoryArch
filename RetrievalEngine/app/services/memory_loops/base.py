"""
Phase 3.8 — BaseLoop: safety enforcement + shared infrastructure.

All five memory loops inherit from BaseLoop. It enforces the project's
non-negotiable safety constraints and provides a consistent interface for
proposing, approving, and executing loop actions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class LoopSafetyError(Exception):
    """Raised when a loop attempts a forbidden operation."""


@dataclass
class LoopResult:
    loop_name: str
    project_id: str
    actions_proposed: int = 0
    actions_executed: int = 0       # safe actions auto-executed this run
    actions_pending: int = 0        # destructive actions awaiting human approval
    actions_skipped: int = 0        # already-processed or no trigger fired
    errors: list[str] = field(default_factory=list)
    run_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        return (
            f"[{self.loop_name}] proposed={self.actions_proposed} "
            f"executed={self.actions_executed} pending={self.actions_pending} "
            f"skipped={self.actions_skipped} errors={len(self.errors)}"
        )


# Action types that require human_approved=True before the loop executes them.
# Anything not in this set is "safe" and auto-executed by the loop.
_DESTRUCTIVE_ACTION_TYPES: frozenset[str] = frozenset({
    "propose_merge",    # DeduplicationLoop: merge two memories into one
    "propose_supersede",# StalenessLoop: explicitly supersede a memory
    "archive",          # ReviewLoop: set status='archived'
})


class BaseLoop:
    """
    Base class for all memory maintenance loops.

    Subclasses implement run(project_id) and call _propose() / _execute_safe()
    to create and act on LoopAction rows. Safety constraints are checked here —
    subclasses never need to re-implement them.
    """

    name: str = "base"

    def __init__(self, db: Session) -> None:
        self.db = db

    # ── Public interface ───────────────────────────────────────────────────

    def run(self, project_id: str, dry_run: bool = False) -> LoopResult:
        """
        Scan memories for this project and propose / execute actions.

        Args:
            project_id: The project to scan.
            dry_run:     If True, compute triggers and return result without
                         writing any rows to the DB. Useful for testing.
        """
        raise NotImplementedError

    def execute_approved_action(self, action_id: str) -> bool:
        """
        Execute a loop action that a human has approved.

        Returns True if the action was executed, False if already executed,
        not approved, or not found.
        """
        from app.p2_models import LoopAction  # avoid circular at module level

        action = self.db.query(LoopAction).filter(LoopAction.id == action_id).first()
        if action is None:
            logger.warning("execute_approved_action: action %s not found", action_id)
            return False
        if action.executed:
            return False
        if not action.human_approved:
            logger.warning("execute_approved_action: action %s not approved", action_id)
            return False

        self._apply_action(action)
        return True

    # ── Safety enforcement ─────────────────────────────────────────────────

    @staticmethod
    def _assert_safe(action_type: str, changes: dict[str, Any]) -> None:
        """
        Raise LoopSafetyError if the proposed action violates any invariant.

        Called before every _propose() call — cannot be bypassed by subclasses.
        """
        # Invariant 1: privacy_level is immutable
        if "privacy_level" in changes:
            raise LoopSafetyError(
                f"Loop action '{action_type}' attempted to modify privacy_level. "
                "Autonomous loops must never change privacy_level."
            )
        # Invariant 2: no hard deletes
        if action_type == "delete":
            raise LoopSafetyError(
                "Loops cannot hard-delete memories. Use action_type='archive' "
                "which requires human approval."
            )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _propose(
        self,
        project_id: str,
        action_type: str,
        proposed: dict[str, Any],
        reason: str,
        target_memory_id: str | None = None,
        secondary_memory_id: str | None = None,
        dry_run: bool = False,
    ) -> "LoopAction | None":
        """
        Create a LoopAction row.

        Safe actions (not in _DESTRUCTIVE_ACTION_TYPES) are immediately
        auto-approved and executed by _execute_safe_immediately().
        Destructive actions stay pending (human_approved=None).
        """
        from app.p2_models import LoopAction

        # Extract changes dict from proposed for safety check
        changes = proposed.get("changes", {})
        self._assert_safe(action_type, changes)

        if dry_run:
            return None  # don't touch DB in dry_run mode

        is_destructive = action_type in _DESTRUCTIVE_ACTION_TYPES

        action = LoopAction(
            project_id=project_id,
            loop_name=self.name,
            action_type=action_type,
            target_memory_id=target_memory_id,
            secondary_memory_id=secondary_memory_id,
            proposed_action=json.dumps(proposed),
            reason=reason,
            # Safe actions get pre-approved; destructive wait for human
            human_approved=(None if is_destructive else True),
            executed=False,
        )
        self.db.add(action)
        self.db.flush()  # get action.id without committing

        if not is_destructive:
            # Execute immediately
            self._apply_action(action)

        return action

    def _apply_action(self, action: "LoopAction") -> None:
        """
        Write the actual DB change described in action.proposed_action.

        Dispatches to _handle_{action_type}(). Marks action as executed.
        """
        from app.p2_models import LoopAction

        handler = getattr(self, f"_handle_{action.action_type}", None)
        if handler is None:
            logger.error(
                "No handler for action_type '%s' in loop '%s'",
                action.action_type, self.name,
            )
            return

        try:
            result = handler(action)
            action.executed = True
            action.executed_at = datetime.now(timezone.utc)
            action.set_result(result or {})
        except Exception as exc:
            logger.exception(
                "Error executing action %s (type=%s): %s",
                action.id, action.action_type, exc,
            )
            action.set_result({"error": str(exc)})

    def _already_proposed(
        self,
        project_id: str,
        action_type: str,
        target_memory_id: str,
        secondary_memory_id: str | None = None,
    ) -> bool:
        """Return True if a non-rejected LoopAction for this target already exists."""
        from app.p2_models import LoopAction

        q = self.db.query(LoopAction).filter(
            LoopAction.project_id == project_id,
            LoopAction.loop_name == self.name,
            LoopAction.action_type == action_type,
            LoopAction.target_memory_id == target_memory_id,
            # Use isnot(False) so that NULL (pending) rows are included.
            # `!= False` uses != which evaluates NULL to NULL in SQL (not TRUE).
            LoopAction.human_approved.isnot(False),
        )
        if secondary_memory_id is not None:
            q = q.filter(LoopAction.secondary_memory_id == secondary_memory_id)
        return q.first() is not None
