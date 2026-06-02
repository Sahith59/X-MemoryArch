import uuid
import json
from datetime import datetime, timezone
from sqlalchemy import String, Text, Integer, Float, DateTime, Date, ForeignKey, Index, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Stored as JSON strings e.g. '["Python", "FastAPI"]'
    tech_stack: Mapped[str | None] = mapped_column(Text, nullable=True)
    goals: Mapped[str | None] = mapped_column(Text, nullable=True)
    repo_path: Mapped[str | None] = mapped_column(String, nullable=True)
    domain: Mapped[str] = mapped_column(String, nullable=False, default="general")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    sessions: Mapped[list["AISession"]] = relationship(
        "AISession", back_populates="project", cascade="all, delete-orphan"
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="project", cascade="all, delete-orphan"
    )

    # Helpers so callers never touch raw JSON strings
    def get_tech_stack(self) -> list[str]:
        return json.loads(self.tech_stack) if self.tech_stack else []

    def get_goals(self) -> list[str]:
        return json.loads(self.goals) if self.goals else []


class AISession(Base):
    __tablename__ = "ai_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"), nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_date: Mapped[str | None] = mapped_column(String, nullable=True)  # ISO date string
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    project: Mapped["Project"] = relationship("Project", back_populates="sessions")
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="source_session", cascade="all, delete-orphan"
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message", cascade="all, delete-orphan",
        foreign_keys="Message.session_id",
        order_by="Message.message_index",
    )


class Message(Base):
    """Sub-phase 1.31 — individual messages within a session.

    When a session is created with raw_content (backward compat), one Message
    row is auto-created with role='unknown' and content_type='mixed'.
    When structured messages are provided, each gets its own row.
    """
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("ai_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False, default="unknown")
    # user / assistant / system / tool / context / unknown
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False, default="text")
    # text / code_block / error_trace / command / file_diff / mixed
    message_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    has_code: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    has_error: Mapped[bool] = mapped_column(Integer, nullable=False, default=False)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    # JSON catch-all for tool-specific extras (citations, file refs, tool outputs, diffs)
    # Named extra_metadata to avoid conflict with SQLAlchemy's reserved .metadata attribute
    extra_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_messages_session_id", "session_id"),
        Index("ix_messages_project_id", "project_id"),
        Index("ix_messages_role", "role"),
    )

    def get_extra_metadata(self) -> dict:
        return json.loads(self.extra_metadata) if self.extra_metadata else {}


class HandoffEvent(Base):
    """Sub-phase 1.34 — records a context handoff between AI tools.

    Captures when a developer transfers context from one tool to another
    (e.g. Claude → ChatGPT → Cursor), optionally linking to the ContextPacket
    that was used. Status tracks whether the handoff was completed or failed.
    """
    __tablename__ = "handoff_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    # Nullable — packet may not have been formally created
    context_packet_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("context_packets.id", ondelete="SET NULL"), nullable=True
    )
    source_tool: Mapped[str] = mapped_column(String, nullable=False)
    target_tool: Mapped[str] = mapped_column(String, nullable=False)
    # pending / completed / failed
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    # Optional note explaining why the handoff happened
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    # When the handoff actually occurred (caller-supplied or defaults to now)
    handoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        Index("ix_handoff_events_project_id", "project_id"),
        Index("ix_handoff_events_status", "status"),
        Index("ix_handoff_events_source_tool", "source_tool"),
        Index("ix_handoff_events_target_tool", "target_tool"),
    )


class ContextPacket(Base):
    """Sub-phase 1.33 — stored handoff packets for cross-tool context transfer.

    A ContextPacket bundles selected memories and sessions into a structured
    markdown document that can be pasted into any AI tool to instantly restore
    project context. token_estimate lets the caller know how much of the
    target tool's context window will be consumed.
    """
    __tablename__ = "context_packets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    # The AI tool this packet is intended for (e.g. "Claude", "ChatGPT", "Cursor")
    target_tool: Mapped[str] = mapped_column(String, nullable=False)
    # Why this packet is being created — guides what to include
    intent: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON lists of IDs to include
    included_memory_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    included_session_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Auto-generated markdown content assembled at creation time
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        Index("ix_context_packets_project_id", "project_id"),
    )

    def get_included_memory_ids(self) -> list[str]:
        return json.loads(self.included_memory_ids) if self.included_memory_ids else []

    def get_included_session_ids(self) -> list[str]:
        return json.loads(self.included_session_ids) if self.included_session_ids else []


class MemorySuggestion(Base):
    """Sub-phase 1.32 — staging table for memory review queue.

    Option B: extraction writes memories directly AND creates parallel
    MemorySuggestion records with status=approved / created_by=rule_based.
    New manual or LLM path creates pending suggestions; approval endpoint
    promotes them to real Memory rows.
    """
    __tablename__ = "memory_suggestions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    source_session_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("ai_sessions.id", ondelete="SET NULL"), nullable=True
    )
    # JSON list of message IDs this suggestion was derived from
    source_message_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    # pending / approved / rejected / edited
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    # rule_based / llm / manual
    created_by: Mapped[str] = mapped_column(String, nullable=False, default="manual")
    # ID of the Memory row created when this suggestion was approved
    memory_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        Index("ix_memory_suggestions_project_id", "project_id"),
        Index("ix_memory_suggestions_status", "status"),
        Index("ix_memory_suggestions_session_id", "source_session_id"),
    )

    def get_source_message_ids(self) -> list[str]:
        return json.loads(self.source_message_ids) if self.source_message_ids else []


class MemoryChangelog(Base):
    __tablename__ = "memory_changelog"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    memory_id: Mapped[str] = mapped_column(String, ForeignKey("memories.id"), nullable=False)
    field: Mapped[str] = mapped_column(String, nullable=False)
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class MemoryLink(Base):
    __tablename__ = "memory_links"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    source_memory_id: Mapped[str] = mapped_column(
        String, ForeignKey("memories.id"), nullable=False
    )
    target_memory_id: Mapped[str] = mapped_column(
        String, ForeignKey("memories.id"), nullable=False
    )
    relationship_type: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Sub-phase 1.45: timestamp for supersedes links — when was the old memory superseded?
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String, ForeignKey("projects.id"), nullable=False)
    source_session_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("ai_sessions.id"), nullable=True
    )
    type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[int] = mapped_column(Integer, default=3)       # 1–5
    confidence: Mapped[float] = mapped_column(Float, default=1.0)     # 0.0–1.0
    # Stored as JSON strings
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    related_files: Mapped[str | None] = mapped_column(Text, nullable=True)
    related_tools: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    # Sub-phase 1.10: structured fields, code anchoring, temporal invalidation
    type_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON dict
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    commit_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    superseded_by: Mapped[str | None] = mapped_column(String, nullable=True)  # ID of newer memory
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Sub-phase 1.15: quality layer + retrieval-ready fields
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_status: Mapped[str | None] = mapped_column(String, nullable=True)   # draft/verified/needs_review/auto_extracted/rejected
    search_text: Mapped[str | None] = mapped_column(Text, nullable=True)        # flattened field for Phase-2 vector embedding
    source_type: Mapped[str | None] = mapped_column(String, nullable=True)      # ai_session/manual/code_scan/import
    source_quote: Mapped[str | None] = mapped_column(Text, nullable=True)       # verbatim sentence that triggered extraction
    symbol_name: Mapped[str | None] = mapped_column(String, nullable=True)      # function/class/variable name
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    branch_name: Mapped[str | None] = mapped_column(String, nullable=True)
    # Sub-phase 1.18: privacy control
    privacy_level: Mapped[str] = mapped_column(String, default="internal")
    # Sub-phase 1.20: semantic embedding (all-MiniLM-L6-v2, float32 bytes, 384-dim)
    # Phase-2 loads with: np.frombuffer(embedding, dtype=np.float32)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # Sub-phase 1.23: memory tier — "working" (hot) | "archival" (cold)
    tier: Mapped[str] = mapped_column(String, default="working")
    # Sub-phase 1.24: embedding model identity for Phase-2 compatibility checks
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Sub-phase 1.25: access tracking — feeds tier recalculation and Phase-2 retrieval ranking
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Sub-phase 1.26: query-facing retrieval hint — synthesized at write time, no model inference
    retrieval_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Sub-phase 1.27: DBSCAN cluster assignment — pre-computed on stored embeddings
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cluster_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Sub-phase 1.28: Ebbinghaus decay score — pre-computed freshness signal for Phase-2 ranking
    decay_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Sub-phase 1.31: message-level source tracing (JSON list of message IDs)
    source_message_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Sub-phase 2.7: LLM-generated context prefix prepended before embedding (Anthropic 67% uplift)
    contextual_prefix: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    project: Mapped["Project"] = relationship("Project", back_populates="memories")
    source_session: Mapped["AISession | None"] = relationship("AISession", back_populates="memories")

    def get_tags(self) -> list[str]:
        return json.loads(self.tags) if self.tags else []

    def get_related_files(self) -> list[str]:
        return json.loads(self.related_files) if self.related_files else []

    def get_related_tools(self) -> list[str]:
        return json.loads(self.related_tools) if self.related_tools else []

    def get_type_metadata(self) -> dict | None:
        return json.loads(self.type_metadata) if self.type_metadata else None

    def get_source_message_ids(self) -> list[str]:
        return json.loads(self.source_message_ids) if self.source_message_ids else []

    entities: Mapped[list["MemoryEntity"]] = relationship(
        "MemoryEntity", back_populates="memory", cascade="all, delete-orphan"
    )


class MemoryEntity(Base):
    """Sub-phase 1.22 — named entities extracted at write time via spaCy NER."""
    __tablename__ = "memory_entities"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    memory_id: Mapped[str] = mapped_column(
        String, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    entity_text: Mapped[str] = mapped_column(String, nullable=False)      # surface form
    entity_label: Mapped[str] = mapped_column(String, nullable=False)     # ORG/PRODUCT/PERSON/LANGUAGE/TECH
    normalized_text: Mapped[str] = mapped_column(String, nullable=False)  # lowercased
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    memory: Mapped["Memory"] = relationship("Memory", back_populates="entities")

    __table_args__ = (
        Index("ix_memory_entities_memory_id", "memory_id"),
        Index("ix_memory_entities_project_id", "project_id"),
        Index("ix_memory_entities_normalized_text", "normalized_text"),
    )


# ---------------------------------------------------------------------------
# Phase 3.7 — Bi-temporal Knowledge Graph
# ---------------------------------------------------------------------------

class EntityNode(Base):
    """Phase 3.7 — Deduplicated entity node in the knowledge graph.

    Each unique entity (PERSON, ORG, TECH, PRODUCT, CONCEPT) across all memories
    in a project maps to exactly one EntityNode. Multiple MemoryEntity rows with
    the same normalized surface form are resolved to a single node here.

    canonical_label: The most common or most specific surface form seen.
    entity_type:     PERSON | ORG | TECH | PRODUCT | CONCEPT | GPE | OTHER
    """
    __tablename__ = "entity_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    canonical_label: Mapped[str] = mapped_column(String, nullable=False)   # best surface form e.g. "Alice"
    normalized_label: Mapped[str] = mapped_column(String, nullable=False)  # lowercased key e.g. "alice"
    entity_type: Mapped[str] = mapped_column(String, nullable=False)       # PERSON/ORG/TECH/PRODUCT/CONCEPT/GPE/OTHER
    mention_count: Mapped[int] = mapped_column(Integer, default=1)         # how many memories mention this entity
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        Index("ix_entity_nodes_project_id", "project_id"),
        Index("ix_entity_nodes_normalized_label", "normalized_label"),
        # Uniqueness: one node per (project, normalized_label, entity_type)
        Index("ix_entity_nodes_unique_key", "project_id", "normalized_label", "entity_type", unique=True),
    )


class EntityEdge(Base):
    """Phase 3.7 — Directed, bi-temporal relationship edge between two entity nodes.

    Bi-temporal fields:
      valid_from:  when the relationship became true (from memory content / date)
      valid_until: when it stopped being true (NULL = currently valid)

    Relation types:
      CO_OCCURS    — both entities mentioned in the same memory (always built)
      WORKS_AT     — person works at an organization
      USES         — person/project uses a technology
      KNOWS        — person knows another person
      DECIDED_ON   — project decided to adopt X
      MEMBER_OF    — person is member of a group/org

    source_memory_id: provenance — the memory that established this edge.
    confidence: 1.0 for explicit relations, 0.5 for co-occurrence.
    """
    __tablename__ = "entity_edges"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    source_node_id: Mapped[str] = mapped_column(
        String, ForeignKey("entity_nodes.id", ondelete="CASCADE"), nullable=False
    )
    target_node_id: Mapped[str] = mapped_column(
        String, ForeignKey("entity_nodes.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String, nullable=False)      # CO_OCCURS / WORKS_AT / USES / KNOWS / etc.
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # NULL = currently true
    source_memory_id: Mapped[str | None] = mapped_column(String, nullable=True)  # provenance
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_entity_edges_project_id", "project_id"),
        Index("ix_entity_edges_source_node", "source_node_id"),
        Index("ix_entity_edges_target_node", "target_node_id"),
        Index("ix_entity_edges_relation_type", "relation_type"),
        Index("ix_entity_edges_source_memory", "source_memory_id"),
    )
