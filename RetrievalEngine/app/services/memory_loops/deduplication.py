"""
Phase 3.8 — DeduplicationLoop.

Finds near-duplicate memory pairs using embedding cosine similarity and proposes
merges for human review. Merges are NEVER executed autonomously — the loop only
proposes; a human must approve each merge before it executes.

Why cosine similarity? Memories extracted from similar sessions often contain
semantically identical facts ("Alice works at City Hospital" vs "Alice is employed
at City Hospital"). These pollute the reranker's candidate pool and inflate noise.
Deduplication improves precision by giving the reranker a cleaner set of candidates.

Trigger: cosine_similarity(mem_a.embedding, mem_b.embedding) >= SIMILARITY_THRESHOLD
         for any two active memories in the same project.

Action (destructive, requires human approval):
  propose_merge — propose that mem_b content be merged into mem_a, then archived.
  Human confirms which memory to keep and approves.

Performance: O(N²) similarity computation. For N ≤ 2000 (typical project), this
is fast with numpy matmul. For N > 5000, a batch limit applies.
"""
from __future__ import annotations

import logging
import struct

from sqlalchemy.orm import Session

from app.services.memory_loops.base import BaseLoop, LoopResult

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.92   # pairs above this are near-duplicates
MAX_MEMORIES = 5000           # safety limit: skip if project has > this many


class DeduplicationLoop(BaseLoop):
    name = "deduplication"

    def run(self, project_id: str, dry_run: bool = False) -> LoopResult:
        import numpy as np
        from app.models import Memory  # Phase 1 via namespace extension

        result = LoopResult(loop_name=self.name, project_id=project_id)

        # Load active memories with embeddings
        memories = (
            self.db.query(Memory)
            .filter(
                Memory.project_id == project_id,
                Memory.status == "active",
                Memory.embedding != None,
            )
            .limit(MAX_MEMORIES)
            .all()
        )

        if len(memories) < 2:
            logger.debug("[deduplication] project %s: fewer than 2 memories with embeddings", project_id)
            return result

        # Decode embeddings and filter out zero-dim / corrupt ones
        valid: list[tuple[Memory, np.ndarray]] = []
        for mem in memories:
            vec = _decode_embedding(mem.embedding)
            if vec is not None and vec.size > 0:
                valid.append((mem, vec))

        if len(valid) < 2:
            return result

        # Normalize and compute pairwise cosine similarity
        ids = [m.id for m, _ in valid]
        dim = valid[0][1].shape[0]

        # Handle mixed-dim embeddings (e.g. 384-dim old vs 1024-dim new) —
        # only compare memories with the same embedding dimension.
        dim_groups: dict[int, list[tuple[Memory, np.ndarray]]] = {}
        for mem, vec in valid:
            d = vec.shape[0]
            dim_groups.setdefault(d, []).append((mem, vec))

        pairs_found = 0
        for d, group in dim_groups.items():
            if len(group) < 2:
                continue

            mems = [m for m, _ in group]
            vecs = np.stack([v for _, v in group]).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            vecs_norm = vecs / norms

            sims = vecs_norm @ vecs_norm.T  # (N, N) pairwise cosine

            n = len(mems)
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(sims[i, j])
                    if sim < SIMILARITY_THRESHOLD:
                        continue

                    mem_a, mem_b = mems[i], mems[j]

                    # Idempotency: skip if already proposed (either direction)
                    if self._already_proposed(project_id, "propose_merge", mem_a.id, mem_b.id) or \
                       self._already_proposed(project_id, "propose_merge", mem_b.id, mem_a.id):
                        result.actions_skipped += 1
                        continue

                    # Keep the higher-confidence / higher-importance memory as primary
                    keep, discard = _pick_primary(mem_a, mem_b)

                    proposed = {
                        "similarity": round(sim, 4),
                        "keep_memory_id": keep.id,
                        "keep_title": keep.title,
                        "discard_memory_id": discard.id,
                        "discard_title": discard.title,
                        "changes": {
                            # After human approval: discard gets status='archived'
                            # The human may also choose to merge content manually.
                        },
                        "suggested_action": (
                            f"Review both memories. Keep '{keep.title}' (higher confidence/"
                            f"importance). Archive '{discard.title}' after verifying no "
                            f"unique information is lost."
                        ),
                    }

                    action = self._propose(
                        project_id=project_id,
                        action_type="propose_merge",
                        proposed=proposed,
                        reason=(
                            f"Cosine similarity {sim:.3f} ≥ {SIMILARITY_THRESHOLD} — "
                            f"likely duplicate pair"
                        ),
                        target_memory_id=keep.id,
                        secondary_memory_id=discard.id,
                        dry_run=dry_run,
                    )
                    result.actions_proposed += 1
                    result.actions_pending += 1
                    pairs_found += 1

        if not dry_run:
            self.db.commit()

        logger.info(result.summary())
        return result

    # ── Handler (executes only after human_approved=True) ─────────────────

    def _handle_propose_merge(self, action: "LoopAction") -> dict:
        """
        Execute a human-approved dedup merge.
        Archives the secondary (discard) memory. Primary (keep) is unchanged.
        """
        from app.models import Memory

        discard = self.db.query(Memory).filter(Memory.id == action.secondary_memory_id).first()
        keep = self.db.query(Memory).filter(Memory.id == action.target_memory_id).first()

        if discard is None or keep is None:
            return {"error": "one or both memories not found"}

        old_status = discard.status
        discard.status = "archived"
        discard.superseded_by = keep.id

        return {
            "kept_memory_id": keep.id,
            "archived_memory_id": discard.id,
            "old_status": old_status,
        }


# ── Helpers ────────────────────────────────────────────────────────────────

def _decode_embedding(raw: bytes) -> "np.ndarray | None":
    """Decode raw bytes (float32 LE) into a numpy array."""
    try:
        import numpy as np
        return np.frombuffer(raw, dtype=np.float32).copy()
    except Exception:
        return None


def _pick_primary(a: "Memory", b: "Memory") -> tuple["Memory", "Memory"]:
    """
    Choose which memory to keep (primary) and which to discard.
    Prefer: higher confidence → higher importance → higher access_count.
    Returns (keep, discard).
    """
    score_a = (a.confidence or 0.0, a.importance or 0, a.access_count or 0)
    score_b = (b.confidence or 0.0, b.importance or 0, b.access_count or 0)
    if score_b > score_a:
        return b, a
    return a, b
