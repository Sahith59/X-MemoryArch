"""
Pre-Phase 2 — ExtractionBackend Protocol.

All extraction backends (rule_based, cloud_llm, local_llm) implement this
interface. The router calls backend.extract() without knowing which backend
is active. Backends are swappable per-project.

Architectural invariants:
  - Rule-based backend is NEVER deleted — it's the offline fallback.
  - LLM backends ALWAYS run the noise gate first (0.15 threshold).
  - canonical_type is STABLE and BOUNDED (13 Phase 1 families).
  - type_label is OPEN (LLM-assigned, for display/nuance only).
  - Every memory produced has extraction_backend recorded on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# canonical_type → Phase 1 MemoryType mapping
# ---------------------------------------------------------------------------
# LLM extraction prompt offers 13 canonical families. Map them to the
# exact string values Phase 1's MemoryType enum uses in the DB.
# Retrieval filters, analytics, and ranking always use canonical_type.

CANONICAL_TYPE_MAP: dict[str, str] = {
    # Exact matches (LLM family name == Phase 1 DB value)
    "decision": "decision",
    "problem": "problem",
    "constraint": "constraint",
    "open_question": "open_question",
    "workflow_pattern": "workflow_pattern",
    "failed_approach": "failed_approach",
    "insight": "insight",
    "how_to": "how_to",
    "task": "task",
    "structure": "structure",
    "reference_context": "reference_context",
    "conversation_note": "conversation_note",
    # LLM prompt aliases → Phase 1 canonical
    "preference": "constraint",
    "plan": "task",
    "procedure": "how_to",
    "fact": "insight",
    "reference": "reference_context",
    "architecture": "structure",
    "bug": "problem",
    "issue": "problem",
    "setup_instruction": "how_to",
    "code_context": "reference_context",
}

VALID_CANONICAL_TYPES: frozenset[str] = frozenset(CANONICAL_TYPE_MAP.values())


def resolve_canonical_type(raw_type: str) -> str:
    """Map an LLM-assigned type string to a stable Phase 1 canonical type.

    Falls back to 'unclassified' for unknown strings — never raises.
    """
    if not raw_type:
        return "unclassified"
    normalized = raw_type.lower().strip().replace(" ", "_").replace("-", "_")
    return CANONICAL_TYPE_MAP.get(normalized, "unclassified")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """Per-run configuration passed to every backend."""
    backend: str = "rule_based"             # rule_based | cloud_llm | local_llm
    model: str | None = None                # "claude-haiku-4-5-20251001", "llama3.2", etc.
    noise_gate_threshold: float = 0.15      # LLM backends use 0.15; rule_based uses its own 0.30
    chunk_size_tokens: int = 2000           # max tokens per LLM chunk
    temperature: float = 0.0               # deterministic extraction
    max_memories_per_chunk: int = 20        # safety cap
    fallback_to_rule_based: bool = True     # fallback on LLM failure
    include_project_context: bool = True    # send project summary to LLM
    max_existing_memories_context: int = 10 # how many existing memory titles to send as context


# ---------------------------------------------------------------------------
# Intermediate extraction representation (before DB write)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedMemoryDraft:
    """
    Intermediate memory representation produced by an LLM extraction call.
    Validated and written to DB by the backend's _write_drafts() method.
    """
    canonical_type: str          # stable Phase 1 family (for DB .type field)
    type_label: str              # open LLM label (stored in memories.type_label)
    title: str
    content: str
    importance: int = 3          # 1–5
    confidence: float = 0.7      # 0.0–1.0

    # Provenance
    source_quote: str | None = None
    source_message_ids: list[str] = field(default_factory=list)

    # Entities — [{label, entity_type: TECH|ORG|PRODUCT|PERSON|OTHER}]
    entities: list[dict] = field(default_factory=list)

    # Links — [{target_memory_id|null, relation_type, confidence}]
    links: list[dict] = field(default_factory=list)

    tags: list[str] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    type_metadata: dict | None = None
    privacy_level: str = "internal"
    review_recommended: bool = False
    llm_reasoning: str | None = None

    # Temporal fields (LLM can detect these)
    valid_from: str | None = None
    valid_until: str | None = None
    supersedes_memory_id: str | None = None

    # Set by backend before writing to DB
    extraction_backend: str = "rule_based"


# ---------------------------------------------------------------------------
# Extended ExtractionResult for Phase 2
# ---------------------------------------------------------------------------

@dataclass
class P2ExtractionResult:
    """
    Extended result returned by all Phase 2 backends.
    Superset of Phase 1's ExtractionResult schema.
    """
    session_id: str
    memories_created: int
    summary: str
    memory_ids: list[str] = field(default_factory=list)

    # Phase 2 additions
    backend_used: str = "rule_based"
    model_used: str | None = None
    chunks_processed: int = 0
    noise_filtered_count: int = 0       # sentences removed by noise gate
    llm_api_calls: int = 0
    fallback_used: bool = False         # True if fell back to rule_based
    skipped_duplicates: int = 0
    duplicate_titles: list[str] = field(default_factory=list)
    low_confidence_queued: int = 0
    dynamic_type_labels: list[str] = field(default_factory=list)  # open labels not in canonical map


# ---------------------------------------------------------------------------
# ExtractionBackend Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ExtractionBackend(Protocol):
    """
    Protocol all extraction backends must satisfy.

    Backends are instantiated once and called per extraction request.
    They are stateless with respect to individual sessions — all session
    state comes from the DB via the db: Session argument.
    """

    @property
    def name(self) -> str:
        """Unique stable identifier: 'rule_based' | 'cloud_llm' | 'local_llm'."""
        ...

    def is_available(self) -> bool:
        """
        Check whether this backend can currently be used.

        rule_based: always True (offline, no deps beyond Phase 1).
        cloud_llm: True when ANTHROPIC_API_KEY or OPENAI_API_KEY is set.
        local_llm: True when Ollama server responds on OLLAMA_BASE_URL.
        """
        ...

    def extract(
        self,
        session_id: str,
        db: Session,
        config: ExtractionConfig | None = None,
    ) -> P2ExtractionResult:
        """
        Extract memories from an AISession and write them to the DB.

        Args:
            session_id: UUID of the AISession row to extract from.
            db:         Live SQLAlchemy session pointing at Phase 1 database.
            config:     Extraction configuration; defaults used if None.

        Returns:
            P2ExtractionResult with counts, IDs, and telemetry.

        Contract:
          - MUST record extraction_backend on every Memory row it creates.
          - MUST set canonical_type to a value from VALID_CANONICAL_TYPES.
          - MUST NOT raise on partial failures — partial results + error in summary.
          - If fallback_to_rule_based=True, LLM backends catch API errors and
            retry via rule_based before raising.
        """
        ...


# ---------------------------------------------------------------------------
# Backend registry — populated by importing each backend module
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ExtractionBackend] = {}


def register_backend(backend: ExtractionBackend) -> None:
    """Register a backend instance by its name."""
    _REGISTRY[backend.name] = backend


def get_backend(name: str) -> ExtractionBackend:
    """
    Return the named backend.

    Raises KeyError if the backend name is unknown.
    Raises RuntimeError if the backend is not available.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown extraction backend '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    backend = _REGISTRY[name]
    if not backend.is_available():
        raise RuntimeError(
            f"Extraction backend '{name}' is not available. "
            "Check API keys, Ollama connectivity, or use 'rule_based'."
        )
    return backend


def get_backend_or_fallback(name: str) -> tuple[ExtractionBackend, bool]:
    """
    Return (backend, fallback_used).

    If the requested backend is unavailable, falls back to rule_based.
    """
    if name in _REGISTRY and _REGISTRY[name].is_available():
        return _REGISTRY[name], False
    if "rule_based" in _REGISTRY:
        return _REGISTRY["rule_based"], True
    raise RuntimeError("rule_based backend is not registered — cannot fall back.")


def list_backends() -> list[dict]:
    """Return status of all registered backends."""
    return [
        {
            "name": b.name,
            "available": b.is_available(),
        }
        for b in _REGISTRY.values()
    ]
