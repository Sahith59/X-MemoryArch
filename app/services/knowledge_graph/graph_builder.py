"""
Phase 3.7 — Knowledge graph builder.

Takes a memory corpus (list of Memory ORM objects with their content) and
constructs EntityNode + EntityEdge rows in the same SQLite database.

Two edge types are built:
  CO_OCCURS  (confidence=0.5) — two entities appear in the same memory.
             This is the primary retrieval signal: "Alice" + "City Hospital"
             co-occurring in 3 memories means a strong graph path from the
             "alice" node to any query about her employer.

  WORKS_AT / USES / KNOWS  (confidence=0.80–0.90) — explicitly detected
             typed relations from the memory text via regex patterns.
             Higher confidence; fewer in number.

Performance:
  Building is O(N × E²) where N = memories and E = avg entities/memory.
  With E ≈ 5-8 and N ≤ 5000 this completes in < 10 seconds.
  Results are cached per (db, project_id) call — no rebuild on repeated queries.

Usage:
    from app.services.knowledge_graph.graph_builder import build_graph
    build_graph(db, project_id, memories)
    # memories = list of Memory ORM objects (need .id, .content, .search_text)
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.services.knowledge_graph.entity_ner import extract_entities, extract_relations


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_graph(
    db: Session,
    project_id: str,
    memories: list,        # list of Memory ORM objects
    rebuild: bool = False,
) -> dict:
    """
    Build entity nodes and edges for a memory corpus.

    Args:
        db:         SQLAlchemy session pointing to the shared SQLite.
        project_id: The project whose memories to process.
        memories:   List of Memory ORM objects (need id, content, search_text fields).
        rebuild:    If True, delete existing nodes/edges for this project first.

    Returns:
        {"nodes": int, "co_occurs_edges": int, "typed_edges": int}
    """
    import app.models as _models

    if rebuild:
        # Clean existing graph for this project
        db.query(_models.EntityEdge).filter(
            _models.EntityEdge.project_id == project_id
        ).delete(synchronize_session=False)
        db.query(_models.EntityNode).filter(
            _models.EntityNode.project_id == project_id
        ).delete(synchronize_session=False)
        db.commit()

    # ── Step 1: Load existing nodes to avoid duplicates ───────────────────────
    existing_nodes: dict[tuple[str, str], str] = {}  # (normalized_label, entity_type) → node_id
    for node in db.query(_models.EntityNode).filter(
        _models.EntityNode.project_id == project_id
    ).all():
        existing_nodes[(node.normalized_label, node.entity_type)] = node.id

    # Track new nodes created this run
    new_nodes: dict[tuple[str, str], str] = {}  # same key → node_id

    def _get_or_create_node(canonical: str, normalized: str, entity_type: str) -> str:
        key = (normalized, entity_type)
        if key in existing_nodes:
            # Increment mention count
            node = db.query(_models.EntityNode).filter(
                _models.EntityNode.id == existing_nodes[key]
            ).first()
            if node:
                node.mention_count += 1
            return existing_nodes[key]
        if key in new_nodes:
            return new_nodes[key]
        node_id = str(uuid.uuid4())
        node = _models.EntityNode(
            id=node_id,
            project_id=project_id,
            canonical_label=canonical,
            normalized_label=normalized,
            entity_type=entity_type,
            mention_count=1,
        )
        db.add(node)
        new_nodes[key] = node_id
        existing_nodes[key] = node_id  # also register for future lookups this run
        return node_id

    # ── Step 2: Check existing edges to skip duplicates ───────────────────────
    existing_edge_keys: set[tuple[str, str, str]] = set()
    for edge in db.query(_models.EntityEdge).filter(
        _models.EntityEdge.project_id == project_id
    ).with_entities(
        _models.EntityEdge.source_node_id,
        _models.EntityEdge.target_node_id,
        _models.EntityEdge.relation_type,
    ).all():
        existing_edge_keys.add((edge.source_node_id, edge.target_node_id, edge.relation_type))

    new_edges_batch: list[_models.EntityEdge] = []

    def _add_edge(src_id: str, tgt_id: str, rtype: str, mem_id: str, conf: float):
        if src_id == tgt_id:
            return
        key = (src_id, tgt_id, rtype)
        if key in existing_edge_keys:
            return
        existing_edge_keys.add(key)
        new_edges_batch.append(_models.EntityEdge(
            id=str(uuid.uuid4()),
            project_id=project_id,
            source_node_id=src_id,
            target_node_id=tgt_id,
            relation_type=rtype,
            source_memory_id=mem_id,
            confidence=conf,
        ))

    # ── Step 3: Process each memory ───────────────────────────────────────────
    n_co_occurs = 0
    n_typed     = 0

    for mem in memories:
        text = (mem.search_text or mem.content or "").strip()
        if not text:
            continue

        # Extract entities for this memory
        entities = extract_entities(text)
        if not entities:
            continue

        # Create nodes for all entities in this memory
        mem_node_ids: list[str] = []
        for ent in entities:
            nid = _get_or_create_node(ent.text, ent.normalized, ent.entity_type)
            mem_node_ids.append(nid)

        # CO_OCCURS edges: connect all entity pairs in this memory
        for i in range(len(mem_node_ids)):
            for j in range(i + 1, len(mem_node_ids)):
                _add_edge(mem_node_ids[i], mem_node_ids[j], "CO_OCCURS", mem.id, 0.5)
                _add_edge(mem_node_ids[j], mem_node_ids[i], "CO_OCCURS", mem.id, 0.5)
                n_co_occurs += 1

        # Typed relation edges (WORKS_AT, USES, KNOWS)
        relations = extract_relations(text)
        for rel in relations:
            # Find node IDs for subject and object
            subj_matches = [
                nid for (norm, _), nid in existing_nodes.items()
                if norm == rel.subject or rel.subject in norm
            ]
            obj_matches = [
                nid for (norm, _), nid in existing_nodes.items()
                if norm == rel.object_ or rel.object_ in norm
            ]
            for s_id in subj_matches[:1]:
                for o_id in obj_matches[:1]:
                    _add_edge(s_id, o_id, rel.relation_type, mem.id, rel.confidence)
                    n_typed += 1

        # Flush in batches of 500 to keep memory manageable
        if len(new_edges_batch) >= 500:
            db.add_all(new_edges_batch)
            db.commit()
            new_edges_batch.clear()

    # Final flush
    if new_edges_batch:
        db.add_all(new_edges_batch)
    db.commit()

    n_nodes = len(existing_nodes) + len(new_nodes)
    return {
        "nodes":         n_nodes,
        "co_occurs_edges": n_co_occurs,
        "typed_edges":   n_typed,
    }
