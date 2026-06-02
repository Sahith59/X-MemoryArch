"""
Phase 3.8 — Comprehensive end-to-end test suite for memory loops.

Covers:
  - Safety enforcement (privacy_level immutable, no hard deletes, human approval required)
  - PromotionLoop: trigger fires, action created, idempotency
  - StalenessLoop: expired valid_until, superseded_by, MemoryLink supersedes
  - DeduplicationLoop: cosine sim above threshold, pairs found, sub-threshold skipped
  - ReviewLoop: never-accessed cold memories, surfaced-but-never-selected
  - FeedbackLoop: weight learning from retrieval_runs, export to file
  - LoopOrchestrator: run_all, approve_and_execute, reject_action, pending_actions
  - Human approval workflow: approve → execute, reject → blocked
  - Dry-run mode: no DB writes
  - Idempotency: running loop twice doesn't double-create actions
"""
from __future__ import annotations

import json
import struct
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

# ── conftest already sets up namespace extension + in-memory SQLite ──────


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_embedding(dim: int = 1024, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


def _near_duplicate_embedding(original_bytes: bytes, noise: float = 0.01) -> bytes:
    """Return an embedding very similar (but not identical) to the original."""
    vec = np.frombuffer(original_bytes, dtype=np.float32).copy()
    rng = np.random.default_rng(42)
    vec += rng.standard_normal(vec.shape).astype(np.float32) * noise
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


def _create_memory(
    db,
    project,
    *,
    content: str = "Alice works at City Hospital",
    confidence: float = 0.9,
    importance: int = 3,
    access_count: int = 0,
    tier: str = "working",
    status: str = "active",
    review_status: str | None = None,
    valid_until: datetime | None = None,
    superseded_by: str | None = None,
    embedding: bytes | None = None,
    created_at: datetime | None = None,
):
    from app import crud, schemas

    mem = crud.create_memory(db, project.id, schemas.MemoryCreate(
        type="insight",
        title=content[:50],
        content=content,
        importance=importance,
        confidence=confidence,
        privacy_level="internal",
    ))
    # Apply fields not in MemoryCreate schema (crud.create_memory commits, so
    # these become an UPDATE on flush)
    mem.access_count = access_count
    mem.tier = tier
    mem.status = status
    if review_status is not None:
        mem.review_status = review_status
    if valid_until is not None:
        mem.valid_until = valid_until
    if superseded_by:
        mem.superseded_by = superseded_by
    if embedding is not None:
        mem.embedding = embedding
    if created_at is not None:
        mem.created_at = created_at
    db.flush()
    return mem


def _create_retrieval_run(
    db,
    project,
    *,
    query: str = "test query",
    intent: str | None = None,
    surfaced_ids: list[str] | None = None,
    selected_ids: list[str] | None = None,
    gold_ids: list[str] | None = None,
):
    from app.p2_models import RetrievalRun

    run = RetrievalRun(
        project_id=project.id,
        query=query,
        intent=intent,
    )
    if surfaced_ids:
        run.set_surfaced_ids(surfaced_ids)
    if selected_ids:
        run.set_selected_ids(selected_ids)
    if gold_ids:
        run.set_gold_ids(gold_ids)
    db.add(run)
    db.flush()
    return run


# ════════════════════════════════════════════════════════════════════════════
# Safety Enforcement
# ════════════════════════════════════════════════════════════════════════════

class TestSafetyEnforcement:
    def test_privacy_level_change_raises(self, db, project):
        from app.services.memory_loops.base import BaseLoop, LoopSafetyError

        loop = BaseLoop(db)
        with pytest.raises(LoopSafetyError, match="privacy_level"):
            loop._assert_safe("some_action", {"privacy_level": "public"})

    def test_hard_delete_raises(self, db, project):
        from app.services.memory_loops.base import BaseLoop, LoopSafetyError

        loop = BaseLoop(db)
        with pytest.raises(LoopSafetyError, match="hard-delete"):
            loop._assert_safe("delete", {})

    def test_safe_action_passes(self, db, project):
        from app.services.memory_loops.base import BaseLoop

        loop = BaseLoop(db)
        # Should not raise
        loop._assert_safe("promote_importance", {"importance": 4})

    def test_promotion_never_sets_privacy(self, db, project):
        """PromotionLoop must not touch privacy_level even if memory has it."""
        from app.services.memory_loops.promotion import PromotionLoop

        mem = _create_memory(db, project, access_count=10, confidence=0.95)
        original_privacy = mem.privacy_level
        db.commit()

        loop = PromotionLoop(db)
        loop.run(project.id)

        db.refresh(mem)
        assert mem.privacy_level == original_privacy  # unchanged

    def test_loops_never_hard_delete(self, db, project):
        """No loop action should set status to anything other than 'archived'."""
        from app.p2_models import LoopAction
        from app.services.memory_loops import LoopOrchestrator

        old_mem = _create_memory(
            db, project,
            valid_until=datetime.now(timezone.utc) - timedelta(days=1),
        )
        db.commit()

        orchestrator = LoopOrchestrator(db)
        orchestrator.run_all(project.id)

        # Verify no LoopAction has action_type='delete'
        delete_actions = db.query(LoopAction).filter(
            LoopAction.action_type == "delete"
        ).count()
        assert delete_actions == 0


# ════════════════════════════════════════════════════════════════════════════
# PromotionLoop
# ════════════════════════════════════════════════════════════════════════════

class TestPromotionLoop:
    def test_qualifies_and_promotes(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.promotion import PromotionLoop, ACCESS_THRESHOLD, CONFIDENCE_THRESHOLD

        mem = _create_memory(
            db, project,
            access_count=ACCESS_THRESHOLD,
            confidence=CONFIDENCE_THRESHOLD,
            importance=2,
        )
        db.commit()

        loop = PromotionLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed == 1
        assert result.actions_executed == 1

        db.refresh(mem)
        assert mem.importance == 3  # boosted from 2
        assert mem.review_status == "verified"

        # LoopAction recorded
        action = db.query(LoopAction).filter(
            LoopAction.target_memory_id == mem.id,
            LoopAction.action_type == "promote_importance",
        ).first()
        assert action is not None
        assert action.executed is True
        assert action.human_approved is True  # auto-approved (safe)

    def test_below_threshold_not_promoted(self, db, project):
        from app.services.memory_loops.promotion import PromotionLoop, ACCESS_THRESHOLD

        mem = _create_memory(
            db, project,
            access_count=ACCESS_THRESHOLD - 1,  # below threshold
            confidence=0.95,
        )
        db.commit()

        loop = PromotionLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed == 0

    def test_low_confidence_not_promoted(self, db, project):
        from app.services.memory_loops.promotion import PromotionLoop, CONFIDENCE_THRESHOLD

        mem = _create_memory(
            db, project,
            access_count=10,
            confidence=CONFIDENCE_THRESHOLD - 0.01,  # just below threshold
        )
        db.commit()

        loop = PromotionLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed == 0

    def test_importance_capped_at_5(self, db, project):
        from app.services.memory_loops.promotion import PromotionLoop

        mem = _create_memory(
            db, project,
            access_count=10,
            confidence=0.95,
            importance=5,  # already max — but review_status not verified
            review_status=None,
        )
        db.commit()

        loop = PromotionLoop(db)
        result = loop.run(project.id)

        db.refresh(mem)
        assert mem.importance == 5  # stays at 5, not 6

    def test_idempotency(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.promotion import PromotionLoop

        mem = _create_memory(db, project, access_count=10, confidence=0.95)
        db.commit()

        loop = PromotionLoop(db)
        loop.run(project.id)
        result2 = loop.run(project.id)  # second run

        assert result2.actions_skipped >= 1
        action_count = db.query(LoopAction).filter(
            LoopAction.target_memory_id == mem.id,
            LoopAction.action_type == "promote_importance",
        ).count()
        assert action_count == 1  # not double-created

    def test_dry_run_no_db_writes(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.promotion import PromotionLoop

        mem = _create_memory(db, project, access_count=10, confidence=0.95, importance=2)
        db.commit()

        loop = PromotionLoop(db)
        result = loop.run(project.id, dry_run=True)

        assert result.actions_proposed == 1
        # No LoopAction rows written
        assert db.query(LoopAction).count() == 0
        # Memory unchanged
        db.refresh(mem)
        assert mem.importance == 2


# ════════════════════════════════════════════════════════════════════════════
# StalenessLoop
# ════════════════════════════════════════════════════════════════════════════

class TestStalenessLoop:
    def test_expired_valid_until_flagged(self, db, project):
        from app.services.memory_loops.staleness import StalenessLoop

        expired = _create_memory(
            db, project,
            valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        db.commit()

        loop = StalenessLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed >= 1
        db.refresh(expired)
        assert expired.review_status == "needs_review"

    def test_future_valid_until_not_flagged(self, db, project):
        from app.services.memory_loops.staleness import StalenessLoop

        valid = _create_memory(
            db, project,
            valid_until=datetime.now(timezone.utc) + timedelta(days=30),
        )
        db.commit()

        loop = StalenessLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed == 0
        db.refresh(valid)
        assert valid.review_status != "needs_review"

    def test_superseded_by_field_flagged(self, db, project):
        from app.services.memory_loops.staleness import StalenessLoop

        newer = _create_memory(db, project, content="Alice is now at General Hospital")
        db.flush()

        old = _create_memory(db, project, content="Alice works at City Hospital")
        old.superseded_by = newer.id
        db.commit()

        loop = StalenessLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed >= 1
        db.refresh(old)
        assert old.review_status == "needs_review"

    def test_memory_link_supersedes_flagged(self, db, project):
        from app.models import MemoryLink
        from app.services.memory_loops.staleness import StalenessLoop

        old = _create_memory(db, project, content="Old fact about Alice")
        db.flush()
        newer = _create_memory(db, project, content="Updated fact about Alice")
        db.flush()

        link = MemoryLink(
            source_memory_id=newer.id,
            target_memory_id=old.id,
            relationship_type="supersedes",
        )
        db.add(link)
        db.commit()

        loop = StalenessLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed >= 1
        db.refresh(old)
        assert old.review_status == "needs_review"

    def test_already_needs_review_skipped(self, db, project):
        from app.services.memory_loops.staleness import StalenessLoop

        mem = _create_memory(
            db, project,
            valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
            review_status="needs_review",
        )
        db.commit()

        loop = StalenessLoop(db)
        result = loop.run(project.id)

        assert result.actions_skipped >= 1
        assert result.actions_proposed == 0

    def test_idempotency(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.staleness import StalenessLoop

        mem = _create_memory(
            db, project,
            valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        db.commit()

        loop = StalenessLoop(db)
        loop.run(project.id)
        loop.run(project.id)  # second run

        # Only one action per memory
        count = db.query(LoopAction).filter(
            LoopAction.target_memory_id == mem.id,
            LoopAction.loop_name == "staleness",
        ).count()
        assert count == 1

    def test_propose_supersede_requires_approval(self, db, project):
        """propose_supersede action_type is destructive — must not auto-execute."""
        from app.p2_models import LoopAction
        from app.services.memory_loops.base import _DESTRUCTIVE_ACTION_TYPES

        assert "propose_supersede" in _DESTRUCTIVE_ACTION_TYPES


# ════════════════════════════════════════════════════════════════════════════
# DeduplicationLoop
# ════════════════════════════════════════════════════════════════════════════

class TestDeduplicationLoop:
    def test_near_duplicate_pair_proposed(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.deduplication import DeduplicationLoop

        emb = _make_embedding(dim=1024, seed=1)
        dup_emb = _near_duplicate_embedding(emb, noise=0.001)  # very similar

        mem_a = _create_memory(db, project, content="Alice works at City Hospital", embedding=emb)
        mem_b = _create_memory(db, project, content="Alice is employed at City Hospital", embedding=dup_emb)
        db.commit()

        loop = DeduplicationLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed >= 1
        assert result.actions_pending >= 1  # requires human approval

        action = db.query(LoopAction).filter(
            LoopAction.loop_name == "deduplication",
            LoopAction.action_type == "propose_merge",
        ).first()
        assert action is not None
        assert action.human_approved is None  # pending — not auto-approved

    def test_distant_pair_not_proposed(self, db, project):
        from app.services.memory_loops.deduplication import DeduplicationLoop

        emb_a = _make_embedding(dim=1024, seed=1)
        emb_b = _make_embedding(dim=1024, seed=99)  # completely different

        _create_memory(db, project, content="Alice at hospital", embedding=emb_a)
        _create_memory(db, project, content="Bob likes hiking", embedding=emb_b)
        db.commit()

        loop = DeduplicationLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed == 0

    def test_propose_merge_does_not_auto_execute(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.deduplication import DeduplicationLoop

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)

        _create_memory(db, project, embedding=emb)
        _create_memory(db, project, embedding=dup)
        db.commit()

        loop = DeduplicationLoop(db)
        loop.run(project.id)

        # All dedup actions must be pending (human_approved=None)
        actions = db.query(LoopAction).filter(
            LoopAction.action_type == "propose_merge"
        ).all()
        for a in actions:
            assert a.human_approved is None
            assert a.executed is False

    def test_human_approve_executes_merge(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops import LoopOrchestrator
        from app.models import Memory

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)

        mem_a = _create_memory(db, project, content="Alice at City Hospital", confidence=0.9, embedding=emb)
        mem_b = _create_memory(db, project, content="Alice employed at City Hospital", confidence=0.7, embedding=dup)
        db.commit()

        orchestrator = LoopOrchestrator(db)
        orchestrator.run_all(project.id, loops=["deduplication"])

        # Find the pending merge action
        action = db.query(LoopAction).filter(
            LoopAction.action_type == "propose_merge",
            LoopAction.human_approved == None,
        ).first()
        assert action is not None

        # Human approves
        success = orchestrator.approve_and_execute(action.id, "deduplication")
        assert success is True

        # Lower-confidence memory should be archived
        db.refresh(mem_a)
        db.refresh(mem_b)
        # One should be archived (the discard — lower confidence mem_b)
        assert mem_b.status == "archived" or mem_a.status == "archived"
        # Not both
        assert not (mem_a.status == "archived" and mem_b.status == "archived")

    def test_different_dim_memories_skipped(self, db, project):
        """Memories with different embedding dimensions should not be compared."""
        from app.services.memory_loops.deduplication import DeduplicationLoop

        emb_384 = _make_embedding(dim=384, seed=1)
        emb_1024 = _make_embedding(dim=1024, seed=1)  # same seed but different dim

        _create_memory(db, project, content="A", embedding=emb_384)
        _create_memory(db, project, content="B", embedding=emb_1024)
        db.commit()

        loop = DeduplicationLoop(db)
        result = loop.run(project.id)
        # Different dims → no pairs compared → 0 proposed
        assert result.actions_proposed == 0

    def test_idempotency(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.deduplication import DeduplicationLoop

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)

        _create_memory(db, project, embedding=emb)
        _create_memory(db, project, embedding=dup)
        db.commit()

        loop = DeduplicationLoop(db)
        loop.run(project.id)
        result2 = loop.run(project.id)

        assert result2.actions_skipped >= 1

        count = db.query(LoopAction).filter(LoopAction.action_type == "propose_merge").count()
        assert count == 1  # not double-created


# ════════════════════════════════════════════════════════════════════════════
# ReviewLoop
# ════════════════════════════════════════════════════════════════════════════

class TestReviewLoop:
    def test_cold_never_accessed_flagged(self, db, project):
        from app.services.memory_loops.review import ReviewLoop, COLD_AGE_DAYS

        cold_mem = _create_memory(
            db, project,
            access_count=0,
            created_at=datetime.now(timezone.utc) - timedelta(days=COLD_AGE_DAYS + 1),
        )
        db.commit()

        loop = ReviewLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed >= 1
        db.refresh(cold_mem)
        assert cold_mem.review_status == "needs_review"

    def test_recently_created_not_flagged(self, db, project):
        from app.services.memory_loops.review import ReviewLoop

        new_mem = _create_memory(
            db, project,
            access_count=0,
            created_at=datetime.now(timezone.utc) - timedelta(days=5),
        )
        db.commit()

        loop = ReviewLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed == 0

    def test_accessed_memory_not_cold_flagged(self, db, project):
        from app.services.memory_loops.review import ReviewLoop, COLD_AGE_DAYS

        accessed = _create_memory(
            db, project,
            access_count=3,  # accessed — not cold
            created_at=datetime.now(timezone.utc) - timedelta(days=COLD_AGE_DAYS + 5),
        )
        db.commit()

        loop = ReviewLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed == 0

    def test_surfaced_never_selected_flagged(self, db, project):
        from app.services.memory_loops.review import ReviewLoop, SKIP_THRESHOLD, MIN_RUNS_FOR_SKIP
        from app.p2_models import RetrievalRun

        target_mem = _create_memory(db, project, content="Low value memory")
        other_mem = _create_memory(db, project, content="High value memory")
        db.flush()

        # Create enough runs where target is surfaced but never selected
        for i in range(MIN_RUNS_FOR_SKIP + 2):
            run = RetrievalRun(
                project_id=project.id,
                query=f"query {i}",
            )
            run.set_surfaced_ids([target_mem.id, other_mem.id])
            run.set_selected_ids([other_mem.id])  # target consistently skipped
            db.add(run)
        db.commit()

        loop = ReviewLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed >= 1
        db.refresh(target_mem)
        assert target_mem.review_status == "needs_review"

    def test_archive_requires_human_approval(self):
        from app.services.memory_loops.base import _DESTRUCTIVE_ACTION_TYPES
        assert "archive" in _DESTRUCTIVE_ACTION_TYPES

    def test_idempotency(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.review import ReviewLoop, COLD_AGE_DAYS

        _create_memory(
            db, project,
            access_count=0,
            created_at=datetime.now(timezone.utc) - timedelta(days=COLD_AGE_DAYS + 1),
        )
        db.commit()

        loop = ReviewLoop(db)
        loop.run(project.id)
        result2 = loop.run(project.id)

        assert result2.actions_skipped >= 1
        count = db.query(LoopAction).filter(
            LoopAction.action_type == "flag_low_value"
        ).count()
        assert count == 1


# ════════════════════════════════════════════════════════════════════════════
# FeedbackLoop
# ════════════════════════════════════════════════════════════════════════════

class TestFeedbackLoop:
    def test_no_runs_skips_learning(self, db, project):
        from app.services.memory_loops.feedback import FeedbackLoop

        loop = FeedbackLoop(db)
        result = loop.run(project.id)

        assert result.actions_proposed == 0
        assert result.actions_skipped >= 0

    def test_learns_from_labeled_runs(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops.feedback import FeedbackLoop, MIN_RUNS
        import tempfile

        mems = [_create_memory(db, project, content=f"fact {i}") for i in range(5)]
        db.flush()
        gold_id = mems[0].id

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "weights.json"
            loop = FeedbackLoop(db, weights_output_path=weights_path)

            # Create enough labeled runs
            for i in range(MIN_RUNS + 2):
                run = _create_retrieval_run(
                    db, project,
                    query=f"q{i}",
                    surfaced_ids=[m.id for m in mems],
                    selected_ids=[mems[1].id],
                    gold_ids=[gold_id],
                )
            db.commit()

            result = loop.run(project.id)

            assert result.actions_proposed == 1
            assert result.actions_executed == 1  # update_weights is safe, auto-executed

            # Weights file should exist
            assert weights_path.exists()
            data = json.loads(weights_path.read_text())
            assert "weights" in data
            assert "_aggregate" in data["weights"]

    def test_load_weights_missing_file_returns_empty(self):
        from app.services.memory_loops.feedback import FeedbackLoop

        result = FeedbackLoop.load_weights(Path("/nonexistent/path/weights.json"))
        assert result == {}

    def test_weights_have_correct_structure(self, db, project):
        from app.services.memory_loops.feedback import FeedbackLoop, MIN_RUNS

        import tempfile
        mems = [_create_memory(db, project, content=f"f{i}") for i in range(3)]
        db.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "w.json"
            loop = FeedbackLoop(db, weights_output_path=weights_path)

            for i in range(MIN_RUNS + 1):
                run = _create_retrieval_run(
                    db, project,
                    intent="factual",
                    surfaced_ids=[m.id for m in mems],
                    gold_ids=[mems[0].id],
                )
            db.commit()

            loop.run(project.id)

            weights = FeedbackLoop.load_weights(weights_path)
            assert "_aggregate" in weights
            agg = weights["_aggregate"]
            assert "alpha" in agg
            assert 0.0 <= agg["alpha"] <= 1.0
            assert "recall_at_k" in agg
            assert "n_runs" in agg


# ════════════════════════════════════════════════════════════════════════════
# LoopOrchestrator
# ════════════════════════════════════════════════════════════════════════════

class TestLoopOrchestrator:
    def test_run_all_returns_summary(self, db, project):
        from app.services.memory_loops import LoopOrchestrator

        orchestrator = LoopOrchestrator(db)
        summary = orchestrator.run_all(project.id)

        assert summary.project_id == project.id
        assert len(summary.loop_results) == 5  # all 5 loops ran
        assert len(summary.failed_loops) == 0

    def test_run_subset_of_loops(self, db, project):
        from app.services.memory_loops import LoopOrchestrator

        orchestrator = LoopOrchestrator(db)
        summary = orchestrator.run_all(project.id, loops=["promotion", "staleness"])

        assert len(summary.loop_results) == 2

    def test_run_single_loop(self, db, project):
        from app.services.memory_loops import LoopOrchestrator

        mem = _create_memory(db, project, access_count=10, confidence=0.95)
        db.commit()

        orchestrator = LoopOrchestrator(db)
        result = orchestrator.run_loop("promotion", project.id)

        assert result.actions_proposed == 1

    def test_reject_action(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops import LoopOrchestrator
        from app.services.memory_loops.deduplication import DeduplicationLoop

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)
        _create_memory(db, project, embedding=emb)
        _create_memory(db, project, embedding=dup)
        db.commit()

        orchestrator = LoopOrchestrator(db)
        orchestrator.run_all(project.id, loops=["deduplication"])

        action = db.query(LoopAction).filter(
            LoopAction.action_type == "propose_merge",
            LoopAction.human_approved == None,
        ).first()
        assert action is not None

        success = orchestrator.reject_action(action.id)
        assert success is True

        db.refresh(action)
        assert action.human_approved is False

    def test_rejected_action_not_executable(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops import LoopOrchestrator

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)
        _create_memory(db, project, embedding=emb)
        _create_memory(db, project, embedding=dup)
        db.commit()

        orchestrator = LoopOrchestrator(db)
        orchestrator.run_all(project.id, loops=["deduplication"])

        action = db.query(LoopAction).filter(
            LoopAction.action_type == "propose_merge"
        ).first()
        orchestrator.reject_action(action.id)

        # Cannot approve after rejection
        success = orchestrator.approve_and_execute(action.id, "deduplication")
        assert success is False

    def test_pending_actions_query(self, db, project):
        from app.services.memory_loops import LoopOrchestrator

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)
        _create_memory(db, project, embedding=emb)
        _create_memory(db, project, embedding=dup)
        db.commit()

        orchestrator = LoopOrchestrator(db)
        orchestrator.run_all(project.id, loops=["deduplication"])

        pending = orchestrator.pending_actions(project.id)
        assert len(pending) >= 1
        for a in pending:
            assert a.human_approved is None
            assert a.executed is False

    def test_dry_run_all(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops import LoopOrchestrator

        _create_memory(db, project, access_count=10, confidence=0.95)
        _create_memory(
            db, project,
            valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        db.commit()

        orchestrator = LoopOrchestrator(db)
        summary = orchestrator.run_all(project.id, dry_run=True)

        # No DB writes
        assert db.query(LoopAction).count() == 0
        # But proposed counts are non-zero (triggers fired)
        assert summary.total_proposed > 0

    def test_unknown_loop_name_raises(self, db, project):
        from app.services.memory_loops import LoopOrchestrator

        orchestrator = LoopOrchestrator(db)
        with pytest.raises(ValueError, match="Unknown loop"):
            orchestrator.run_loop("nonexistent_loop", project.id)

    def test_one_loop_error_doesnt_abort_others(self, db, project, monkeypatch):
        """If one loop crashes, the rest should still run."""
        from app.services.memory_loops import LoopOrchestrator
        from app.services.memory_loops.promotion import PromotionLoop

        def failing_run(project_id, dry_run=False):
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(PromotionLoop, "run", failing_run)

        orchestrator = LoopOrchestrator(db)
        summary = orchestrator.run_all(project.id)

        # promotion failed, but other loops still ran
        assert "promotion" in " ".join(summary.failed_loops)
        # At least 4 other loops ran successfully
        assert len(summary.loop_results) >= 4

    def test_summary_totals_are_aggregate(self, db, project):
        from app.services.memory_loops import LoopOrchestrator
        from app.services.memory_loops.review import COLD_AGE_DAYS

        _create_memory(db, project, access_count=10, confidence=0.95)  # promotion
        _create_memory(
            db, project,
            valid_until=datetime.now(timezone.utc) - timedelta(hours=1),
        )  # staleness
        db.commit()

        orchestrator = LoopOrchestrator(db)
        summary = orchestrator.run_all(project.id)

        # Totals are the sum of individual results
        assert summary.total_proposed == sum(r.actions_proposed for r in summary.loop_results)
        assert summary.total_executed == sum(r.actions_executed for r in summary.loop_results)
        assert summary.total_pending == sum(r.actions_pending for r in summary.loop_results)


# ════════════════════════════════════════════════════════════════════════════
# Human Approval Workflow
# ════════════════════════════════════════════════════════════════════════════

class TestHumanApprovalWorkflow:
    def test_pending_action_not_executed_until_approved(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops import LoopOrchestrator
        from app.models import Memory

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)
        mem_a = _create_memory(db, project, embedding=emb)
        mem_b = _create_memory(db, project, embedding=dup)
        db.commit()

        orchestrator = LoopOrchestrator(db)
        orchestrator.run_all(project.id, loops=["deduplication"])

        # Memories should still be active (merge not executed)
        db.refresh(mem_a)
        db.refresh(mem_b)
        assert mem_a.status == "active"
        assert mem_b.status == "active"

        # Pending action exists
        pending = orchestrator.pending_actions(project.id, loop_name="deduplication")
        assert len(pending) == 1

    def test_approve_then_execute_changes_memory(self, db, project):
        from app.p2_models import LoopAction
        from app.services.memory_loops import LoopOrchestrator
        from app.models import Memory

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)
        mem_a = _create_memory(db, project, confidence=0.9, embedding=emb)
        mem_b = _create_memory(db, project, confidence=0.7, embedding=dup)
        db.commit()

        orchestrator = LoopOrchestrator(db)
        orchestrator.run_all(project.id, loops=["deduplication"])

        pending = orchestrator.pending_actions(project.id, loop_name="deduplication")
        assert len(pending) == 1

        action = pending[0]
        success = orchestrator.approve_and_execute(action.id, "deduplication")
        assert success is True

        # One memory should now be archived
        db.refresh(mem_a)
        db.refresh(mem_b)
        statuses = {mem_a.status, mem_b.status}
        assert "archived" in statuses
        assert "active" in statuses

    def test_cannot_execute_same_action_twice(self, db, project):
        from app.services.memory_loops import LoopOrchestrator

        emb = _make_embedding(1024, 1)
        dup = _near_duplicate_embedding(emb, noise=0.001)
        _create_memory(db, project, embedding=emb)
        _create_memory(db, project, embedding=dup)
        db.commit()

        orchestrator = LoopOrchestrator(db)
        orchestrator.run_all(project.id, loops=["deduplication"])

        pending = orchestrator.pending_actions(project.id)
        action = pending[0]

        first = orchestrator.approve_and_execute(action.id, "deduplication")
        second = orchestrator.approve_and_execute(action.id, "deduplication")

        assert first is True
        assert second is False  # already executed


# ════════════════════════════════════════════════════════════════════════════
# LoopAction model
# ════════════════════════════════════════════════════════════════════════════

class TestLoopActionModel:
    def test_proposed_action_roundtrip(self, db, project):
        from app.p2_models import LoopAction

        payload = {"changes": {"importance": 4}, "old_values": {"importance": 2}}
        action = LoopAction(
            project_id=project.id,
            loop_name="promotion",
            action_type="promote_importance",
            proposed_action=json.dumps(payload),
            reason="test",
            human_approved=True,
        )
        db.add(action)
        db.commit()

        loaded = db.query(LoopAction).filter(LoopAction.id == action.id).first()
        assert loaded.get_proposed_action() == payload

    def test_result_roundtrip(self, db, project):
        from app.p2_models import LoopAction

        action = LoopAction(
            project_id=project.id,
            loop_name="promotion",
            action_type="promote_importance",
            proposed_action=json.dumps({}),
            reason="test",
        )
        db.add(action)
        db.flush()

        result_data = {"old_importance": 2, "new_importance": 3}
        action.set_result(result_data)
        db.commit()

        db.refresh(action)
        assert action.get_result() == result_data

    def test_surfaced_and_gold_ids_roundtrip(self, db, project):
        from app.p2_models import RetrievalRun

        run = RetrievalRun(project_id=project.id, query="test")
        run.set_surfaced_ids(["id1", "id2", "id3"])
        run.set_gold_ids(["id1"])
        db.add(run)
        db.commit()

        db.refresh(run)
        assert run.get_surfaced_ids() == ["id1", "id2", "id3"]
        assert run.get_gold_ids() == ["id1"]
