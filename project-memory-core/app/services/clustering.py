"""
Sub-phase 1.27 — DBSCAN cluster assignment on stored embeddings.

compute_clusters() is a pure function: given a list of memory objects
(only needs .id, .title, .embedding), it returns ClusterAssignment objects
for every memory that has an embedding.

Design choices:
  eps=0.40      → cosine distance threshold (≥0.60 cosine similarity to cluster)
  min_samples=2 → minimum 2 members to form a cluster (avoids singletons)
  metric='cosine' → appropriate for unit-normalised sentence embeddings
  Memories without embeddings: excluded from DBSCAN, caller sets cluster_id=None
  Noise points (DBSCAN label -1): cluster_label=None
  Cluster labels: top-3 informative title words per cluster (no LLM)
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass

import numpy as np

_DBSCAN_EPS = 0.40         # cosine distance threshold
_DBSCAN_MIN_SAMPLES = 2    # minimum cluster size

_STOPWORDS = frozenset({
    "a", "an", "the", "to", "for", "of", "in", "on", "at", "with",
    "and", "or", "but", "not", "is", "are", "was", "were", "be", "been",
    "by", "from", "this", "that", "it", "we", "they", "use", "used",
    "using", "our", "add", "new", "set", "get", "via", "per", "as",
    "into", "out", "up", "its", "also", "which", "when", "how",
})


@dataclass
class ClusterAssignment:
    memory_id: str
    cluster_id: int            # -1 = noise/outlier
    cluster_label: str | None  # None for noise; top-words string for real clusters


def _generate_label(titles: list[str]) -> str:
    """Build a human-readable cluster label from member titles."""
    word_counts: Counter[str] = Counter()
    for title in titles:
        for w in title.lower().split():
            w = w.strip(".,;:!?\"'()-/\\")
            if len(w) > 3 and w not in _STOPWORDS:
                word_counts[w] += 1
    top = [w for w, _ in word_counts.most_common(3)]
    return " / ".join(top) if top else "general"


def compute_clusters(
    memories: list,  # objects with .id (str), .title (str), .embedding (bytes | None)
) -> list[ClusterAssignment]:
    """
    Pure function — no DB access.

    Returns ClusterAssignment only for embedded memories (embedding != None).
    If fewer than 2 embedded memories exist, they are all returned as noise (-1).
    """
    from sklearn.cluster import DBSCAN

    embedded = [
        (m.id, m.title, m.embedding)
        for m in memories
        if m.embedding is not None
    ]

    if len(embedded) < 2:
        return [
            ClusterAssignment(memory_id=mid, cluster_id=-1, cluster_label=None)
            for mid, _, _ in embedded
        ]

    ids = [x[0] for x in embedded]
    titles = [x[1] for x in embedded]
    vecs = np.array(
        [np.frombuffer(x[2], dtype=np.float32) for x in embedded],
        dtype=np.float32,
    )

    # Normalize to unit length for cosine metric
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    labels: np.ndarray = DBSCAN(
        eps=_DBSCAN_EPS, min_samples=_DBSCAN_MIN_SAMPLES, metric="cosine"
    ).fit_predict(vecs)

    # Build human-readable label for each cluster
    cluster_titles: dict[int, list[str]] = {}
    for lbl, title in zip(labels, titles):
        if int(lbl) >= 0:
            cluster_titles.setdefault(int(lbl), []).append(title)

    label_map: dict[int, str] = {
        cid: _generate_label(ttls)
        for cid, ttls in cluster_titles.items()
    }

    return [
        ClusterAssignment(
            memory_id=mid,
            cluster_id=int(lbl),
            cluster_label=label_map.get(int(lbl)) if int(lbl) >= 0 else None,
        )
        for mid, lbl in zip(ids, labels)
    ]
