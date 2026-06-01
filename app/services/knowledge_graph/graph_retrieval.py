"""
Phase 3.7 — Knowledge graph retrieval (2-hop BFS walk).

Given a query string, walks the entity graph to find memory IDs that are
connected via entity relationships — even if those memories don't directly
contain the query terms.

Algorithm:
  1. Extract entities from query text (NER)
  2. Find matching EntityNode rows (exact normalized match)
  3. BFS up to 2 hops:
       Level 0: matched query entity nodes
       Level 1: all nodes directly connected via EntityEdge
       Level 2: all nodes connected to Level 1 nodes
  4. Collect source_memory_id from all edges on the walk path
  5. Return as set of memory IDs → added to RRF candidate pool

Why 2-hop is the right depth:
  - 1-hop: only finds memories mentioning the same entity → too narrow
  - 2-hop: finds memories about entities RELATED to the query entity → cross-session chains
  - 3-hop+: too noisy (everything is connected through common entities like "the project")

Scoring for RRF integration:
  Each candidate memory gets a graph score based on:
    - hop distance (closer = better)
    - edge confidence (explicit relation > co-occurrence)
  Final score: sum over all paths of (hop_penalty × confidence)

Usage:
    from app.services.knowledge_graph.graph_retrieval import GraphRetriever
    retriever = GraphRetriever(db, project_id)
    memory_ids = retriever.walk(query, max_hops=2, top_k=40)
"""
from __future__ import annotations

from collections import deque, defaultdict
from sqlalchemy.orm import Session

from app.services.knowledge_graph.entity_ner import extract_entities


class GraphRetriever:
    """
    Stateful graph retriever for a single (db, project_id) pair.

    Build the lookup tables once per corpus (shared across all queries in a
    benchmark run), then call walk() per query.
    """

    def __init__(self, db: Session, project_id: str):
        import app.models as _models
        self._db         = db
        self._project_id = project_id
        self._models     = _models

        # Pre-load graph into memory for fast BFS (avoid per-query DB roundtrips)
        # node_id → {neighbour_node_id: (relation_type, confidence, memory_id)}
        self._adj: dict[str, list[tuple[str, str, float, str]]] = defaultdict(list)
        # normalized_label → list of node_ids (for query entity matching)
        self._label_to_nodes: dict[str, list[str]] = defaultdict(list)

        self._load_graph()

    def _load_graph(self):
        """Load all nodes and edges for this project into memory."""
        # Load nodes
        nodes = self._db.query(self._models.EntityNode).filter(
            self._models.EntityNode.project_id == self._project_id
        ).all()
        for node in nodes:
            self._label_to_nodes[node.normalized_label].append(node.id)
            # Also index partial matches (first word of multi-word entities)
            words = node.normalized_label.split()
            if len(words) > 1:
                self._label_to_nodes[words[0]].append(node.id)

        # Load edges
        edges = self._db.query(self._models.EntityEdge).filter(
            self._models.EntityEdge.project_id == self._project_id
        ).all()
        for edge in edges:
            self._adj[edge.source_node_id].append((
                edge.target_node_id,
                edge.relation_type,
                edge.confidence,
                edge.source_memory_id or "",
            ))

    def walk(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 40,
        filter_relation: str | None = None,  # e.g. "WORKS_AT" to restrict walk
    ) -> list[tuple[str, float]]:
        """
        Walk the graph starting from entities found in the query.

        Args:
            query:           User query text.
            max_hops:        Maximum BFS depth (default 2).
            top_k:           Return at most this many (memory_id, score) pairs.
            filter_relation: If set, only walk edges of this relation type.

        Returns:
            list of (memory_id, graph_score) sorted by score descending.
            graph_score is used as the RRF numerator when merging with dense results.
        """
        if not self._label_to_nodes:
            return []   # empty graph

        # ── Step 1: Extract query entities and find seed nodes ─────────────────
        query_entities = extract_entities(query)
        seed_node_ids: set[str] = set()

        for ent in query_entities:
            # Exact match on normalized label
            for nid in self._label_to_nodes.get(ent.normalized, []):
                seed_node_ids.add(nid)
            # Substring match: "hospital" matches "city hospital"
            for label, nids in self._label_to_nodes.items():
                if ent.normalized in label or label in ent.normalized:
                    seed_node_ids.update(nids)

        # Also try raw query words as entity names (handles short/informal queries)
        query_words = [w.lower().strip("?.,!") for w in query.split() if len(w) > 3]
        for word in query_words:
            for nid in self._label_to_nodes.get(word, []):
                seed_node_ids.add(nid)

        if not seed_node_ids:
            return []

        # ── Step 2: BFS walk up to max_hops ───────────────────────────────────
        # (node_id, current_hop, accumulated_score)
        queue: deque[tuple[str, int, float]] = deque()
        for nid in seed_node_ids:
            queue.append((nid, 0, 1.0))

        visited_nodes: set[str] = set(seed_node_ids)
        # memory_id → best graph score seen
        memory_scores: dict[str, float] = {}

        while queue:
            node_id, hop, score = queue.popleft()

            if hop >= max_hops:
                continue

            for (neighbour_id, rel_type, confidence, memory_id) in self._adj.get(node_id, []):
                # Optionally filter by relation type
                if filter_relation and rel_type != filter_relation:
                    continue

                # Score penalty: 0.5 per hop, multiplied by edge confidence
                # hop=0 → score=1.0; hop=1 → score=0.5×conf; hop=2 → score=0.25×conf²
                edge_score = score * 0.5 * confidence

                # Collect memory IDs from edges (provenance: which memory established this edge)
                if memory_id:
                    existing = memory_scores.get(memory_id, 0.0)
                    memory_scores[memory_id] = max(existing, edge_score)

                if neighbour_id not in visited_nodes:
                    visited_nodes.add(neighbour_id)
                    queue.append((neighbour_id, hop + 1, edge_score))

        if not memory_scores:
            return []

        # Sort by score descending, cap at top_k
        ranked = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def is_empty(self) -> bool:
        """True if the graph has no nodes (graph not yet built for this project)."""
        return len(self._label_to_nodes) == 0


def build_and_retrieve(
    db: Session,
    project_id: str,
    memories: list,
    query: str,
    max_hops: int = 2,
    top_k: int = 40,
) -> list[tuple[str, float]]:
    """
    Convenience: build the graph and immediately walk it for one query.
    Used in one-shot contexts (e.g. testing). For repeated queries over the
    same corpus, build GraphRetriever once and call .walk() per query.
    """
    from app.services.knowledge_graph.graph_builder import build_graph
    build_graph(db, project_id, memories)
    retriever = GraphRetriever(db, project_id)
    return retriever.walk(query, max_hops=max_hops, top_k=top_k)
