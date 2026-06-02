"""
Phase 3.8 — Agent-Mediated Memory Loops.

Five autonomous maintenance loops that keep the memory corpus healthy over time.
All loops obey hard safety constraints enforced at the BaseLoop level.

Safety invariants (non-negotiable):
  1. No loop ever modifies privacy_level
  2. No loop ever hard-deletes a memory (status='archived' is the most destructive write)
  3. Destructive actions (merge, supersede, archive) require human_approved=True
  4. Every proposed and executed action is logged to the loop_actions table

Loops:
  PromotionLoop     — high-confidence, frequently-accessed memories → boosted importance
  StalenessLoop     — expired valid_until or superseded memories → flagged for review
  DeduplicationLoop — near-duplicate pairs (cosine ≥ 0.92) → proposed merge (human confirms)
  ReviewLoop        — never-accessed old memories → flagged for archival review
  FeedbackLoop      — retrieval_runs + gold labels → learned RRF weights

Usage:
    from app.services.memory_loops import LoopOrchestrator
    orchestrator = LoopOrchestrator(db)
    results = orchestrator.run_all(project_id="proj-123")
"""
from app.services.memory_loops.base import BaseLoop, LoopResult, LoopSafetyError
from app.services.memory_loops.promotion import PromotionLoop
from app.services.memory_loops.staleness import StalenessLoop
from app.services.memory_loops.deduplication import DeduplicationLoop
from app.services.memory_loops.review import ReviewLoop
from app.services.memory_loops.feedback import FeedbackLoop
from app.services.memory_loops.orchestrator import LoopOrchestrator

__all__ = [
    "BaseLoop",
    "LoopResult",
    "LoopSafetyError",
    "PromotionLoop",
    "StalenessLoop",
    "DeduplicationLoop",
    "ReviewLoop",
    "FeedbackLoop",
    "LoopOrchestrator",
]
