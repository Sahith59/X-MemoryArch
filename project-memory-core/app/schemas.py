from __future__ import annotations
import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Domain registry (Sub-phase 1.37)
# ---------------------------------------------------------------------------

VALID_DOMAINS: dict[str, str] = {
    # Technical
    "software":   "Software engineering, development, DevOps, data engineering",
    "data":       "Data science, data analysis, BI, analytics, ML/AI",
    "security":   "Cybersecurity, infosec, penetration testing, compliance",
    # Design & Creative
    "design":     "UX/UI design, product design, visual design, brand identity",
    "creative":   "Writing, content creation, art, music, video, photography",
    # Business & Professional
    "product":    "Product management, roadmapping, user research, OKRs",
    "business":   "Business strategy, operations, entrepreneurship, consulting",
    "marketing":  "Marketing, growth, SEO, content strategy, social media, ads",
    "sales":      "Sales, account management, CRM, revenue operations",
    "finance":    "Finance, accounting, investment analysis, budgeting, FP&A",
    "legal":      "Legal, compliance, contracts, policy, regulatory affairs",
    "hr":         "Human resources, recruiting, people ops, culture, L&D",
    # Knowledge Work
    "research":   "Academic research, scientific research, literature review",
    "education":  "Teaching, curriculum design, learning, training, coaching",
    # Personal
    "personal":   "Personal productivity, goals, life management, journaling",
    # Catch-all
    "general":    "Mixed or undefined domain — balanced extraction, no domain bias",
}

TECHNICAL_DOMAINS = {"software", "data", "security"}


class MemoryType(str, Enum):
    decision = "decision"
    problem = "problem"          # was: bug
    task = "task"
    insight = "insight"
    structure = "structure"      # was: architecture
    reference_context = "reference_context"  # was: code_context
    how_to = "how_to"            # was: setup_instruction
    open_question = "open_question"
    constraint = "constraint"
    conversation_note = "conversation_note"
    workflow_pattern = "workflow_pattern"
    failed_approach = "failed_approach"
    unclassified = "unclassified"  # gate-passed but below type confidence threshold


class MemoryStatus(str, Enum):
    active = "active"
    resolved = "resolved"
    stale = "stale"
    archived = "archived"
    superseded = "superseded"


class ReviewStatus(str, Enum):
    draft = "draft"
    verified = "verified"
    needs_review = "needs_review"
    auto_extracted = "auto_extracted"
    rejected = "rejected"


class SourceType(str, Enum):
    ai_session = "ai_session"
    manual = "manual"
    code_scan = "code_scan"
    import_ = "import"


class PrivacyLevel(str, Enum):
    public = "public"          # safe to export to any AI tool
    internal = "internal"      # share within team / personal tools
    sensitive = "sensitive"    # only include when explicitly requested
    secret = "secret"          # never export automatically


class MemoryTier(str, Enum):
    working = "working"    # hot — recently relevant or high importance
    archival = "archival"  # cold — older, low importance; still queryable


class LinkType(str, Enum):
    supersedes = "supersedes"
    related_to = "related_to"
    conflicts_with = "conflicts_with"
    influenced_by = "influenced_by"
    derived_from = "derived_from"
    blocks = "blocks"
    resolves = "resolves"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_list(v: Any) -> list[str]:
    """Accept a list[str] or a JSON string; always return list[str]."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _dump_json_list(v: list[str] | None) -> str | None:
    if v is None:
        return None
    return json.dumps(v)


# ---------------------------------------------------------------------------
# Type-specific metadata models (Sub-phase 1.10)
# ---------------------------------------------------------------------------

class DecisionMetadata(BaseModel):
    rationale: str | None = None
    alternatives_considered: list[str] = Field(default_factory=list)
    consequences: str | None = None
    decision_status: str | None = None  # e.g. "accepted", "superseded", "under_review"


class ProblemMetadata(BaseModel):
    """Formerly BugMetadata — now universal: code bugs, UX problems, process issues, etc."""
    error_message: str | None = None   # or: issue description
    root_cause: str | None = None
    fix_applied: str | None = None     # or: resolution applied
    environment: str | None = None     # or: context where problem occurs
    affected_files: list[str] = Field(default_factory=list)  # or: affected areas


class StructureMetadata(BaseModel):
    """Formerly ArchitectureMetadata — universal: system design, org structure, content structure, etc."""
    pattern: str | None = None
    rationale: str | None = None
    components_affected: list[str] = Field(default_factory=list)


class HowToMetadata(BaseModel):
    """Formerly SetupInstructionMetadata — universal: any step-by-step instructions."""
    command: str | None = None         # or: first action step
    prerequisites: list[str] = Field(default_factory=list)
    platform: str | None = None        # or: applicable context / environment


class OpenQuestionMetadata(BaseModel):
    context: str | None = None
    options_considered: list[str] = Field(default_factory=list)
    deadline: str | None = None  # ISO date string


class ConstraintMetadata(BaseModel):
    source: str | None = None  # e.g. "legal", "performance", "team policy"
    impact: str | None = None


class WorkflowPatternMetadata(BaseModel):
    trigger: str | None = None        # "when starting a new endpoint"
    steps: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class FailedApproachMetadata(BaseModel):
    approach_tried: str | None = None       # what was attempted
    reason_failed: str | None = None        # why it didn't work
    symptoms: list[str] = Field(default_factory=list)   # observable failures
    alternative_chosen: str | None = None   # what replaced it
    avoid_because: str | None = None        # one-line reason to never repeat this


# ---------------------------------------------------------------------------
# Project schemas
# ---------------------------------------------------------------------------

def _validate_domain(v: str | None) -> str:
    if v is None:
        return "general"
    if v not in VALID_DOMAINS:
        raise ValueError(
            f"Invalid domain '{v}'. Valid domains: {sorted(VALID_DOMAINS.keys())}"
        )
    return v


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    repo_path: str | None = None
    domain: str = Field(default="general")

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        return _validate_domain(v)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    tech_stack: list[str] | None = None
    goals: list[str] | None = None
    repo_path: str | None = None
    domain: str | None = None

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_domain(v)


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str | None
    tech_stack: list[str]
    goals: list[str]
    repo_path: str | None
    domain: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def parse_json_fields(cls, data: Any) -> Any:
        if hasattr(data, "__dict__"):
            # ORM object — use the helper methods on the model
            return {
                "id": data.id,
                "name": data.name,
                "description": data.description,
                "tech_stack": data.get_tech_stack(),
                "goals": data.get_goals(),
                "repo_path": data.repo_path,
                "domain": data.domain or "general",
                "created_at": data.created_at,
                "updated_at": data.updated_at,
            }
        return data


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Message schemas (Sub-phase 1.31)
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"
    context = "context"
    unknown = "unknown"


class MessageContentType(str, Enum):
    text = "text"
    code_block = "code_block"
    error_trace = "error_trace"
    command = "command"
    file_diff = "file_diff"
    mixed = "mixed"


class MessageCreate(BaseModel):
    role: MessageRole = MessageRole.unknown
    content: str = Field(..., min_length=1)
    content_type: MessageContentType = MessageContentType.text
    message_index: int = Field(default=0, ge=0)
    language: str | None = None
    metadata: dict | None = None


class MessageResponse(BaseModel):
    id: str
    project_id: str
    session_id: str
    role: str
    content: str
    content_type: str
    message_index: int
    token_estimate: int
    has_code: bool
    has_error: bool
    language: str | None
    metadata: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def parse_metadata(cls, data: Any) -> Any:
        if hasattr(data, "extra_metadata"):
            return {
                "id": data.id,
                "project_id": data.project_id,
                "session_id": data.session_id,
                "role": data.role,
                "content": data.content,
                "content_type": data.content_type,
                "message_index": data.message_index,
                "token_estimate": data.token_estimate,
                "has_code": bool(data.has_code),
                "has_error": bool(data.has_error),
                "language": data.language,
                "metadata": data.get_extra_metadata(),
                "created_at": data.created_at,
            }
        return data


# AISession schemas
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    tool_name: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=300)
    raw_content: str = Field(..., min_length=1)
    session_date: str | None = None
    # Optional structured messages — if provided, stored per-message
    # If omitted, raw_content is stored as a single mixed message (backward compat)
    messages: list[MessageCreate] | None = None


class SessionUpdate(BaseModel):
    tool_name: str | None = Field(default=None, min_length=1, max_length=100)
    title: str | None = Field(default=None, min_length=1, max_length=300)
    raw_content: str | None = None
    summary: str | None = None
    session_date: str | None = None


class SessionResponse(BaseModel):
    id: str
    project_id: str
    tool_name: str
    title: str
    raw_content: str
    summary: str | None
    session_date: str | None
    message_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Memory schemas
# ---------------------------------------------------------------------------

class MemoryCreate(BaseModel):
    type: MemoryType
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1)
    importance: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    related_tools: list[str] = Field(default_factory=list)
    status: MemoryStatus = MemoryStatus.active
    source_session_id: str | None = None
    # Sub-phase 1.10 fields
    type_metadata: dict | None = None
    file_path: str | None = None
    commit_sha: str | None = None
    # Sub-phase 1.15 fields
    review_status: ReviewStatus | None = None
    source_type: SourceType | None = None
    source_quote: str | None = None
    symbol_name: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    branch_name: str | None = None
    # Sub-phase 1.18 fields
    privacy_level: PrivacyLevel = PrivacyLevel.internal
    # Sub-phase 1.23: tier (auto-computed if omitted)
    tier: MemoryTier | None = None

    @field_validator("type_metadata", mode="before")
    @classmethod
    def validate_type_metadata(cls, v: Any) -> dict | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("type_metadata must be a JSON object (dict)")
        return v


class MemoryUpdate(BaseModel):
    type: MemoryType | None = None
    title: str | None = Field(default=None, min_length=1, max_length=300)
    content: str | None = None
    importance: int | None = Field(default=None, ge=1, le=5)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] | None = None
    related_files: list[str] | None = None
    related_tools: list[str] | None = None
    status: MemoryStatus | None = None
    # Sub-phase 1.10 fields
    type_metadata: dict | None = None
    file_path: str | None = None
    commit_sha: str | None = None
    superseded_by: str | None = None
    valid_until: datetime | None = None
    # Sub-phase 1.15 fields
    review_status: ReviewStatus | None = None
    source_type: SourceType | None = None
    source_quote: str | None = None
    symbol_name: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    branch_name: str | None = None
    # Sub-phase 1.18 fields
    privacy_level: PrivacyLevel | None = None
    # Sub-phase 1.23: tier override (auto-computed if omitted)
    tier: MemoryTier | None = None

    @field_validator("type_metadata", mode="before")
    @classmethod
    def validate_type_metadata(cls, v: Any) -> dict | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("type_metadata must be a JSON object (dict)")
        return v


class MemoryResponse(BaseModel):
    id: str
    project_id: str
    source_session_id: str | None
    type: str
    title: str
    content: str
    importance: int
    confidence: float
    tags: list[str]
    related_files: list[str]
    related_tools: list[str]
    status: str
    # Sub-phase 1.10 fields
    type_metadata: dict | None
    file_path: str | None
    commit_sha: str | None
    superseded_by: str | None
    valid_until: datetime | None
    # Sub-phase 1.15 fields
    quality_score: float | None
    review_status: str | None
    search_text: str | None
    source_type: str | None
    source_quote: str | None
    symbol_name: str | None
    line_start: int | None
    line_end: int | None
    branch_name: str | None
    # Sub-phase 1.18
    privacy_level: str
    # Sub-phase 1.23
    tier: str
    # Sub-phase 1.24
    embedding_model: str | None
    embedding_dim: int | None
    # Sub-phase 1.25
    access_count: int
    last_accessed_at: datetime | None
    # Sub-phase 1.26
    retrieval_hint: str | None
    # Sub-phase 1.27
    cluster_id: int | None
    cluster_label: str | None
    # Sub-phase 1.28
    decay_score: float | None
    # Sub-phase 1.31
    source_message_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def parse_json_fields(cls, data: Any) -> Any:
        if hasattr(data, "__dict__"):
            return {
                "id": data.id,
                "project_id": data.project_id,
                "source_session_id": data.source_session_id,
                "type": data.type,
                "title": data.title,
                "content": data.content,
                "importance": data.importance,
                "confidence": data.confidence,
                "tags": data.get_tags(),
                "related_files": data.get_related_files(),
                "related_tools": data.get_related_tools(),
                "status": data.status,
                "type_metadata": data.get_type_metadata(),
                "file_path": data.file_path,
                "commit_sha": data.commit_sha,
                "superseded_by": data.superseded_by,
                "valid_until": data.valid_until,
                "quality_score": data.quality_score,
                "review_status": data.review_status,
                "search_text": data.search_text,
                "source_type": data.source_type,
                "source_quote": data.source_quote,
                "symbol_name": data.symbol_name,
                "line_start": data.line_start,
                "line_end": data.line_end,
                "branch_name": data.branch_name,
                "privacy_level": data.privacy_level or "internal",
                "tier": data.tier or "working",
                "embedding_model": data.embedding_model,
                "embedding_dim": data.embedding_dim,
                "access_count": data.access_count or 0,
                "last_accessed_at": data.last_accessed_at,
                "retrieval_hint": data.retrieval_hint,
                "cluster_id": data.cluster_id,
                "cluster_label": data.cluster_label,
                "decay_score": data.decay_score,
                "source_message_ids": data.get_source_message_ids(),
                "created_at": data.created_at,
                "updated_at": data.updated_at,
            }
        return data


# ---------------------------------------------------------------------------
# HandoffEvent schemas (Sub-phase 1.34)
# ---------------------------------------------------------------------------

class HandoffStatus(str, Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"


class HandoffEventCreate(BaseModel):
    source_tool: str = Field(..., min_length=1, max_length=100)
    target_tool: str = Field(..., min_length=1, max_length=100)
    context_packet_id: str | None = None
    status: HandoffStatus = HandoffStatus.pending
    note: str | None = Field(default=None, max_length=1000)
    handoff_at: datetime | None = None


class HandoffEventUpdate(BaseModel):
    status: HandoffStatus | None = None
    note: str | None = Field(default=None, max_length=1000)


class HandoffEventResponse(BaseModel):
    id: str
    project_id: str
    context_packet_id: str | None
    source_tool: str
    target_tool: str
    status: str
    note: str | None
    handoff_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# ContextPacket schemas (Sub-phase 1.33)
# ---------------------------------------------------------------------------

class ContextPacketCreate(BaseModel):
    target_tool: str = Field(..., min_length=1, max_length=100)
    intent: str | None = Field(default=None, max_length=500)
    included_memory_ids: list[str] = Field(default_factory=list)
    included_session_ids: list[str] = Field(default_factory=list)


class ContextPacketResponse(BaseModel):
    id: str
    project_id: str
    target_tool: str
    intent: str | None
    included_memory_ids: list[str]
    included_session_ids: list[str]
    content: str
    token_estimate: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def parse_packet_fields(cls, data: Any) -> Any:
        if hasattr(data, "__dict__"):
            return {
                "id": data.id,
                "project_id": data.project_id,
                "target_tool": data.target_tool,
                "intent": data.intent,
                "included_memory_ids": data.get_included_memory_ids(),
                "included_session_ids": data.get_included_session_ids(),
                "content": data.content,
                "token_estimate": data.token_estimate,
                "created_at": data.created_at,
                "updated_at": data.updated_at,
            }
        return data


# ---------------------------------------------------------------------------
# MemorySuggestion schemas (Sub-phase 1.32)
# ---------------------------------------------------------------------------

class SuggestionStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    edited = "edited"


class SuggestionCreatedBy(str, Enum):
    rule_based = "rule_based"
    llm = "llm"
    manual = "manual"


class SuggestionCreate(BaseModel):
    suggested_type: MemoryType
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_session_id: str | None = None
    source_message_ids: list[str] = Field(default_factory=list)
    created_by: SuggestionCreatedBy = SuggestionCreatedBy.manual


class SuggestionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    content: str | None = None
    suggested_type: MemoryType | None = None
    status: SuggestionStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SuggestionResponse(BaseModel):
    id: str
    project_id: str
    source_session_id: str | None
    source_message_ids: list[str]
    suggested_type: str
    title: str
    content: str
    confidence: float
    status: str
    created_by: str
    memory_id: str | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def parse_suggestion_fields(cls, data: Any) -> Any:
        if hasattr(data, "__dict__"):
            return {
                "id": data.id,
                "project_id": data.project_id,
                "source_session_id": data.source_session_id,
                "source_message_ids": data.get_source_message_ids(),
                "suggested_type": data.suggested_type,
                "title": data.title,
                "content": data.content,
                "confidence": data.confidence,
                "status": data.status,
                "created_by": data.created_by,
                "memory_id": data.memory_id,
                "reviewed_at": data.reviewed_at,
                "created_at": data.created_at,
                "updated_at": data.updated_at,
            }
        return data


class ApproveResult(BaseModel):
    suggestion: SuggestionResponse
    memory: "MemoryResponse"


# ---------------------------------------------------------------------------
# Extraction response
# ---------------------------------------------------------------------------

class ExtractionResult(BaseModel):
    session_id: str
    memories_created: int
    summary: str
    memories: list[MemoryResponse]
    skipped_duplicates: int = 0
    duplicate_titles: list[str] = Field(default_factory=list)
    low_confidence_queued: int = 0  # passed gate but below type threshold — needs review


class ManualClassifyRequest(BaseModel):
    type: MemoryType


class EmbedAllResult(BaseModel):
    backfilled: int       # memories that had no embedding and were just embedded
    already_embedded: int # memories that already had an embedding


# ---------------------------------------------------------------------------
# Deduplication schemas (Sub-phase 1.44)
# ---------------------------------------------------------------------------

class NearDuplicate(BaseModel):
    memory: "MemoryResponse"
    similarity: float  # cosine similarity 0.0–1.0

class DuplicateCheckResult(BaseModel):
    is_duplicate: bool
    threshold: float
    near_duplicates: list[NearDuplicate]  # sorted by similarity desc


# ---------------------------------------------------------------------------
# Auto-supersede schemas (Sub-phase 1.45)
# ---------------------------------------------------------------------------

class AutoSupersedeResult(BaseModel):
    memory: "MemoryResponse"
    superseded_count: int
    superseded_memories: list["MemoryResponse"]


class RetierResult(BaseModel):
    updated_count: int
    working_count: int
    archival_count: int


# ---------------------------------------------------------------------------
# Cluster schemas (Sub-phase 1.27)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Auto-link schemas (Sub-phase 1.29)
# ---------------------------------------------------------------------------

class AutolinkResult(BaseModel):
    project_id: str
    links_created: int


# ---------------------------------------------------------------------------
# Cluster schemas (Sub-phase 1.27)
# ---------------------------------------------------------------------------

class ClusterInfo(BaseModel):
    cluster_id: int
    cluster_label: str
    memory_count: int
    memory_ids: list[str]


class ClusterResult(BaseModel):
    project_id: str
    total_clustered: int    # memories with cluster_id >= 0
    total_noise: int        # memories with cluster_id == -1 (outliers)
    total_unclustered: int  # memories with cluster_id == None (no embedding)
    clusters: list[ClusterInfo]


# ---------------------------------------------------------------------------
# Changelog + supersede schemas (Sub-phase 1.14)
# ---------------------------------------------------------------------------

class MemoryChangelogResponse(BaseModel):
    id: str
    memory_id: str
    field: str
    old_value: str | None
    new_value: str | None
    changed_at: datetime

    model_config = {"from_attributes": True}


class SupersedeResult(BaseModel):
    old_memory: MemoryResponse
    new_memory: MemoryResponse


# ---------------------------------------------------------------------------
# Search result (Sub-phase 1.12)
# ---------------------------------------------------------------------------

class SearchResult(MemoryResponse):
    relevance_score: float


# ---------------------------------------------------------------------------
# Memory link schemas (Sub-phase 1.11)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Health dashboard (Sub-phase 1.16)
# ---------------------------------------------------------------------------

class QualityBucket(BaseModel):
    excellent: int  # quality_score >= 0.75
    good: int       # 0.50 <= quality_score < 0.75
    fair: int       # 0.25 <= quality_score < 0.50
    poor: int       # quality_score < 0.25 or None


class ReviewStatusBreakdown(BaseModel):
    auto_extracted: int
    verified: int
    needs_review: int
    rejected: int
    draft: int
    unset: int


class HealthIssue(BaseModel):
    severity: str       # "warning" | "info"
    code: str           # machine-readable code e.g. "high_unreviewed_count"
    message: str        # human-readable description


# ---------------------------------------------------------------------------
# Current Truth Resolver (Sub-phase 1.17)
# ---------------------------------------------------------------------------

class ConflictPair(BaseModel):
    memory_a: MemoryResponse
    memory_b: MemoryResponse
    similarity_score: float     # title similarity ratio (0.0–1.0)
    conflict_type: str          # "title_overlap" | "content_overlap"


class ConflictReport(BaseModel):
    project_id: str
    total_conflicts: int
    pairs: list[ConflictPair]


class ResolveConflictsResult(BaseModel):
    resolved_count: int
    pairs_resolved: list[ConflictPair]


class ProjectHealthReport(BaseModel):
    project_id: str
    generated_at: datetime

    # Volume
    total_memories: int
    active_count: int
    resolved_count: int
    stale_count: int
    archived_count: int
    superseded_count: int

    # Quality
    avg_quality_score: float | None
    quality_buckets: QualityBucket

    # Review
    review_breakdown: ReviewStatusBreakdown

    # Traceability
    pct_with_session_ref: float   # 0.0–1.0
    pct_with_source_quote: float  # 0.0–1.0
    pct_with_source_type: float   # 0.0–1.0

    # Type distribution (dict of type_name → count)
    type_distribution: dict[str, int]

    # Confidence
    avg_confidence: float | None
    low_confidence_count: int   # confidence < 0.60

    # Actionable issues
    issues: list[HealthIssue]


# ---------------------------------------------------------------------------
# Entity schemas (Sub-phase 1.22)
# ---------------------------------------------------------------------------

class MemoryEntityResponse(BaseModel):
    id: str
    memory_id: str
    project_id: str
    entity_text: str
    entity_label: str
    normalized_text: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectEntitySummary(BaseModel):
    normalized_text: str
    entity_label: str
    count: int           # number of memories in this project mentioning the entity
    memory_ids: list[str]


# ---------------------------------------------------------------------------
# Memory link schemas (Sub-phase 1.11)
# ---------------------------------------------------------------------------

class MemoryLinkCreate(BaseModel):
    target_memory_id: str
    relationship_type: LinkType
    note: str | None = None


class MemoryLinkResponse(BaseModel):
    id: str
    source_memory_id: str
    target_memory_id: str
    relationship_type: str
    note: str | None
    superseded_at: datetime | None = None  # set only for relationship_type="supersedes"
    created_at: datetime

    model_config = {"from_attributes": True}
