"""
Pre-Phase 2 — CloudLLMExtractionBackend.

Supports Claude (Anthropic) and OpenAI as extraction LLMs.
Both use the same hybrid pipeline:

  Step 1: Noise gate (threshold 0.15) — removes filler, cuts token cost 30-40%.
  Step 2: Intelligent chunking (~2000 tokens per chunk).
  Step 3: LLM extraction per chunk → structured JSON.
  Step 4: Deduplication across chunks + against existing DB memories.
  Step 5: Write to DB (canonical_type, type_label, llm_reasoning stamped).

Supported models:
  Anthropic: claude-haiku-4-5-20251001 (cost), claude-sonnet-4-6 (quality)
  OpenAI:    gpt-4o-mini

Error handling:
  - JSON parse failure → log warning, skip chunk (partial extraction).
  - API error → if fallback_to_rule_based=True, run RuleBasedBackend instead.
  - Invalid canonical_type from LLM → resolve_canonical_type() normalises.
  - Missing source_quote → validate_draft() rejects the memory.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from .base import (
    ExtractionBackend,
    ExtractionConfig,
    ExtractedMemoryDraft,
    P2ExtractionResult,
    register_backend,
    resolve_canonical_type,
)
from .chunker import prepare_chunks
from .deduplicator import deduplicate_drafts, deduplicate_against_existing
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
    build_project_summary,
    validate_draft,
    REQUIRED_FIELDS,
)

_P1_ROOT = Path(__file__).resolve().parents[4] / "project-memory-core"
if str(_P1_ROOT) not in sys.path:
    sys.path.insert(0, str(_P1_ROOT))

logger = logging.getLogger(__name__)

# Default models
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Provider-level API callers
# ---------------------------------------------------------------------------

def _call_anthropic(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int = 4096,
) -> str:
    """Call Claude API and return the raw text response."""
    import anthropic  # imported lazily — not required for rule_based path

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def _call_openai(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int = 4096,
) -> str:
    """Call OpenAI API in JSON mode and return the raw text response."""
    from openai import OpenAI  # imported lazily

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or "[]"


def _provider_from_model(model: str) -> str:
    """Infer provider from model name."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    return "anthropic"  # default


def _call_llm(
    system: str,
    user: str,
    model: str,
    provider: str,
    temperature: float,
) -> str:
    """Dispatch to the correct provider."""
    if provider == "openai":
        return _call_openai(system, user, model, temperature)
    return _call_anthropic(system, user, model, temperature)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> list[dict]:
    """
    Parse LLM output as a JSON array.

    Handles:
      - Direct JSON array: [...]
      - JSON object with 'memories' key: {"memories": [...]}
      - Markdown-wrapped JSON: ```json [...] ```
    """
    raw = raw.strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    # Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # OpenAI JSON mode wraps in object — try known keys
            for key in ("memories", "results", "items", "extractions"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
            # Single memory object
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try extracting a JSON array substring
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM extraction output as JSON. Raw (first 200): %s", raw[:200])
    return []


# ---------------------------------------------------------------------------
# Draft → DB writer
# ---------------------------------------------------------------------------

def _write_drafts_to_db(
    drafts: list[ExtractedMemoryDraft],
    session_id: str,
    project_id: str,
    db: Session,
    backend_name: str,
) -> list[str]:
    """
    Write validated ExtractedMemoryDraft objects to the Phase 1 DB.

    Returns list of created Memory IDs.
    """
    from app import crud, schemas, models
    from app.services.entity_extractor import extract_entities
    from app.crud import create_memory_entities

    created_ids: list[str] = []

    for draft in drafts:
        try:
            canonical = resolve_canonical_type(draft.canonical_type)
            # Validate canonical_type maps to a real MemoryType
            try:
                mem_type = schemas.MemoryType(canonical)
            except ValueError:
                mem_type = schemas.MemoryType.unclassified
                canonical = "unclassified"

            mem_data = schemas.MemoryCreate(
                type=mem_type,
                title=draft.title[:300],
                content=draft.content,
                importance=max(1, min(5, draft.importance)),
                confidence=max(0.0, min(1.0, draft.confidence)),
                tags=draft.tags or [],
                related_files=draft.related_files or [],
                related_tools=[],
                status=schemas.MemoryStatus.active,
                source_session_id=session_id,
                type_metadata=draft.type_metadata,
                review_status=(
                    schemas.ReviewStatus.needs_review
                    if draft.review_recommended
                    else schemas.ReviewStatus.auto_extracted
                ),
                source_type=schemas.SourceType.ai_session,
                source_quote=draft.source_quote,
                privacy_level=schemas.PrivacyLevel(
                    draft.privacy_level
                    if draft.privacy_level in ("public", "internal", "sensitive", "secret")
                    else "internal"
                ),
            )

            # Generate embedding for the memory
            from app.services.semantic_classifier import embed_text
            embedding = embed_text(f"{draft.title}. {draft.content}")

            memory = crud.create_memory(
                db=db,
                project_id=project_id,
                data=mem_data,
                embedding=embedding,
            )

            # Stamp Phase 2 fields
            _stamp_p2_fields(db, memory, draft, backend_name)

            # Extract and store entities
            entity_texts = [draft.title + " " + draft.content]
            entities = extract_entities(" ".join(entity_texts))
            if entities:
                create_memory_entities(db, memory.id, project_id, entities)

            # Store source_message_ids
            if draft.source_message_ids:
                import json as _json
                memory.source_message_ids = _json.dumps(draft.source_message_ids)
                db.commit()

            created_ids.append(memory.id)

        except Exception as exc:
            logger.warning("Failed to write draft '%s': %s", draft.title, exc)
            continue

    return created_ids


def _stamp_p2_fields(db: Session, memory, draft: ExtractedMemoryDraft, backend_name: str) -> None:
    """Stamp Phase 2 columns on a newly created Memory row."""
    try:
        if hasattr(memory, "extraction_backend"):
            memory.extraction_backend = backend_name
        if hasattr(memory, "canonical_type"):
            memory.canonical_type = resolve_canonical_type(draft.canonical_type)
        if hasattr(memory, "type_label"):
            memory.type_label = draft.type_label or draft.canonical_type
        if hasattr(memory, "llm_reasoning"):
            memory.llm_reasoning = draft.llm_reasoning
        db.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CloudLLMExtractionBackend
# ---------------------------------------------------------------------------

class CloudLLMExtractionBackend:
    """
    LLM extraction backend using Claude or OpenAI.

    Uses hybrid pipeline: noise gate → chunking → LLM → dedup → write.
    Falls back to rule_based on API error if config.fallback_to_rule_based.
    """

    @property
    def name(self) -> str:
        return "cloud_llm"

    def is_available(self) -> bool:
        """True if at least one cloud API key is configured."""
        return bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

    def extract(
        self,
        session_id: str,
        db: Session,
        config: ExtractionConfig | None = None,
    ) -> P2ExtractionResult:
        cfg = config or ExtractionConfig(backend="cloud_llm")
        model = cfg.model or os.environ.get(
            "CLOUD_LLM_COST_MODEL", _DEFAULT_ANTHROPIC_MODEL
        )
        provider = _provider_from_model(model)

        try:
            return self._extract_with_llm(session_id, db, cfg, model, provider)
        except Exception as exc:
            logger.error("Cloud LLM extraction failed (%s): %s", model, exc)
            if cfg.fallback_to_rule_based:
                logger.info("Falling back to rule_based backend for session %s", session_id)
                from .rule_based import RuleBasedExtractionBackend
                result = RuleBasedExtractionBackend().extract(session_id, db, config)
                result.fallback_used = True
                result.backend_used = "cloud_llm→rule_based"
                return result
            raise

    def _extract_with_llm(
        self,
        session_id: str,
        db: Session,
        cfg: ExtractionConfig,
        model: str,
        provider: str,
    ) -> P2ExtractionResult:
        from app import crud, models

        session = crud.get_session(db, session_id)
        project = crud.get_project(db, session.project_id)

        # Build project context for the LLM
        project_summary = build_project_summary(project) if cfg.include_project_context else ""

        # Fetch existing memory titles for dedup context
        existing_memories = db.query(models.Memory).filter(
            models.Memory.project_id == session.project_id,
            models.Memory.status == "active",
        ).limit(cfg.max_existing_memories_context).all()
        existing_titles = [m.title for m in existing_memories]
        existing_contents = [m.content for m in existing_memories]

        # Step 1 + 2: Noise gate → chunks
        chunks, gate_result = prepare_chunks(
            session.raw_content,
            threshold=cfg.noise_gate_threshold,
            max_tokens=cfg.chunk_size_tokens,
        )

        if not chunks:
            return P2ExtractionResult(
                session_id=session_id,
                memories_created=0,
                summary="No substantive content found after noise filtering.",
                backend_used="cloud_llm",
                model_used=model,
                noise_filtered_count=gate_result.filtered_count,
            )

        # Step 3: LLM extraction per chunk
        all_drafts: list[ExtractedMemoryDraft] = []
        api_calls = 0

        for chunk in chunks:
            user_prompt = build_extraction_user_prompt(
                chunk_text=chunk.text,
                project_summary=project_summary,
                existing_memory_titles=existing_titles,
                chunk_index=chunk.index,
                total_chunks=len(chunks),
            )

            try:
                raw_response = _call_llm(
                    system=EXTRACTION_SYSTEM_PROMPT,
                    user=user_prompt,
                    model=model,
                    provider=provider,
                    temperature=cfg.temperature,
                )
                api_calls += 1

                raw_items = _parse_llm_json(raw_response)

                for raw in raw_items:
                    validated = validate_draft(raw)
                    if not validated:
                        continue
                    # Resolve canonical_type from LLM output
                    canonical = resolve_canonical_type(validated.get("canonical_type", ""))
                    type_label = validated.get("type_label") or canonical
                    temporal = validated.get("temporal", {}) or {}

                    draft = ExtractedMemoryDraft(
                        canonical_type=canonical,
                        type_label=type_label,
                        title=validated["title"][:300],
                        content=validated["content"],
                        importance=int(validated["importance"]),
                        confidence=float(validated["confidence"]),
                        source_quote=validated.get("source_quote"),
                        source_message_ids=validated.get("source_message_ids", []),
                        entities=validated.get("entities", []),
                        links=validated.get("links", []),
                        tags=validated.get("tags", []),
                        related_files=validated.get("related_files", []),
                        privacy_level=validated.get("privacy_level", "internal"),
                        review_recommended=bool(validated.get("review_recommended", False)),
                        llm_reasoning=validated.get("llm_reasoning"),
                        valid_from=temporal.get("valid_from"),
                        valid_until=temporal.get("valid_until"),
                        supersedes_memory_id=temporal.get("supersedes_memory_id"),
                        extraction_backend="cloud_llm",
                    )
                    all_drafts.append(draft)

            except Exception as chunk_exc:
                logger.warning("Chunk %d extraction failed: %s", chunk.index, chunk_exc)
                continue

        # Step 4: Deduplication
        # 4a — across chunks
        dedup_result = deduplicate_drafts(all_drafts)
        # 4b — against existing DB memories
        dedup_vs_db = deduplicate_against_existing(
            dedup_result.kept,
            existing_titles,
            existing_contents,
        )

        final_drafts = dedup_vs_db.kept
        all_skipped = dedup_result.removed + dedup_vs_db.removed
        skipped_titles = dedup_result.removed_titles + dedup_vs_db.removed_titles

        # Dynamic type labels not in canonical map
        from .base import CANONICAL_TYPE_MAP
        dynamic_labels = [
            d.type_label for d in final_drafts
            if d.type_label and d.type_label not in CANONICAL_TYPE_MAP
        ]

        # Step 5: Write to DB
        created_ids = _write_drafts_to_db(
            final_drafts, session_id, session.project_id, db, "cloud_llm"
        )

        # Update session summary
        _update_session_summary(db, session, len(created_ids), model)

        return P2ExtractionResult(
            session_id=session_id,
            memories_created=len(created_ids),
            summary=(
                f"Cloud LLM ({model}) extracted {len(created_ids)} memories "
                f"from {len(chunks)} chunks. "
                f"Noise filtered: {gate_result.filtered_count} sentences. "
                f"Skipped duplicates: {len(all_skipped)}."
            ),
            memory_ids=created_ids,
            backend_used="cloud_llm",
            model_used=model,
            chunks_processed=len(chunks),
            noise_filtered_count=gate_result.filtered_count,
            llm_api_calls=api_calls,
            fallback_used=False,
            skipped_duplicates=len(all_skipped),
            duplicate_titles=skipped_titles,
            low_confidence_queued=sum(
                1 for d in final_drafts if d.confidence < 0.6
            ),
            dynamic_type_labels=list(set(dynamic_labels)),
        )


def _update_session_summary(db: Session, session, memory_count: int, model: str) -> None:
    """Update the session's summary field with extraction stats."""
    try:
        summary = (
            f"Extracted {memory_count} memories via {model}. "
            f"Session: {session.title}"
        )
        session.summary = summary[:500]
        db.commit()
    except Exception:
        pass


# Register singleton
_instance = CloudLLMExtractionBackend()
register_backend(_instance)

cloud_llm_backend: ExtractionBackend = _instance  # type: ignore[assignment]
