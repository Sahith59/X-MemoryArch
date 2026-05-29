"""
Pre-Phase 2 — LocalLLMExtractionBackend (Ollama).

Uses the same hybrid pipeline as cloud_llm:
  noise gate → chunking → Ollama LLM → dedup → write

Ollama communication is via HTTP to localhost (default: http://localhost:11434).
No streaming — uses /api/generate with format='json' for structured output.

Supported models: llama3.2, mistral, phi-4, gemma2, qwen2.5 (any Ollama model).
The model must be pulled locally before use.

Availability check: pings /api/tags to verify Ollama is running.
Falls back to rule_based if Ollama is unreachable.
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
from .cloud_llm import (
    _parse_llm_json,
    _write_drafts_to_db,
    _update_session_summary,
)
from .deduplicator import deduplicate_drafts, deduplicate_against_existing
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT_OLLAMA,
    build_extraction_user_prompt,
    build_project_summary,
    validate_draft,
)

_P1_ROOT = Path(__file__).resolve().parents[4] / "project-memory-core"
if str(_P1_ROOT) not in sys.path:
    sys.path.insert(0, str(_P1_ROOT))

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL = "llama3.2"


# ---------------------------------------------------------------------------
# Ollama HTTP client (uses httpx — no async needed for extraction)
# ---------------------------------------------------------------------------

def _ollama_generate(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    """
    Call Ollama /api/generate and return the raw response text.

    Uses format='json' to encourage structured output from Ollama models.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": 4096,
        },
    }

    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "[]")
    except httpx.TimeoutException:
        raise RuntimeError(f"Ollama request timed out after {timeout}s (model={model})")
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama HTTP error {exc.response.status_code}: {exc.response.text[:200]}")


def _ollama_is_running(base_url: str, timeout: float = 3.0) -> bool:
    """Ping Ollama /api/tags to verify the server is up."""
    try:
        import httpx
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _build_ollama_prompt(system: str, user: str) -> str:
    """
    Combine system + user into a single Ollama prompt string.

    Ollama's /api/generate doesn't natively support system/user separation
    for all models — we concatenate with clear delimiters.
    """
    return (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{user}\n<|end|>\n"
        f"<|assistant|>\n"
    )


# ---------------------------------------------------------------------------
# LocalLLMExtractionBackend
# ---------------------------------------------------------------------------

class LocalLLMExtractionBackend:
    """
    LLM extraction backend using Ollama (local, offline, no API key).

    Same hybrid pipeline as CloudLLMExtractionBackend:
      noise gate (0.15) → chunking → Ollama → dedup → write

    Falls back to rule_based if Ollama is unreachable.
    """

    @property
    def name(self) -> str:
        return "local_llm"

    def is_available(self) -> bool:
        """True if Ollama server is reachable."""
        base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)
        return _ollama_is_running(base_url)

    def extract(
        self,
        session_id: str,
        db: Session,
        config: ExtractionConfig | None = None,
    ) -> P2ExtractionResult:
        cfg = config or ExtractionConfig(backend="local_llm")
        base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)
        model = cfg.model or os.environ.get("LOCAL_LLM_DEFAULT_MODEL", _DEFAULT_OLLAMA_MODEL)

        if not _ollama_is_running(base_url):
            if cfg.fallback_to_rule_based:
                logger.info("Ollama not reachable — falling back to rule_based for session %s", session_id)
                from .rule_based import RuleBasedExtractionBackend
                result = RuleBasedExtractionBackend().extract(session_id, db, config)
                result.fallback_used = True
                result.backend_used = "local_llm→rule_based"
                return result
            raise RuntimeError(
                f"Ollama is not reachable at {base_url}. "
                "Start Ollama with: ollama serve"
            )

        try:
            return self._extract_with_ollama(session_id, db, cfg, model, base_url)
        except Exception as exc:
            logger.error("Local LLM extraction failed (%s): %s", model, exc)
            if cfg.fallback_to_rule_based:
                logger.info("Falling back to rule_based for session %s", session_id)
                from .rule_based import RuleBasedExtractionBackend
                result = RuleBasedExtractionBackend().extract(session_id, db, config)
                result.fallback_used = True
                result.backend_used = "local_llm→rule_based"
                return result
            raise

    def _extract_with_ollama(
        self,
        session_id: str,
        db: Session,
        cfg: ExtractionConfig,
        model: str,
        base_url: str,
    ) -> P2ExtractionResult:
        from app import crud, models

        session = crud.get_session(db, session_id)
        project = crud.get_project(db, session.project_id)

        project_summary = build_project_summary(project) if cfg.include_project_context else ""

        existing_memories = db.query(models.Memory).filter(
            models.Memory.project_id == session.project_id,
            models.Memory.status == "active",
        ).limit(cfg.max_existing_memories_context).all()
        existing_titles = [m.title for m in existing_memories]
        existing_contents = [m.content for m in existing_memories]

        # Step 1+2: Noise gate → chunks
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
                backend_used="local_llm",
                model_used=model,
                noise_filtered_count=gate_result.filtered_count,
            )

        # Step 3: Ollama extraction per chunk
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
            full_prompt = _build_ollama_prompt(EXTRACTION_SYSTEM_PROMPT_OLLAMA, user_prompt)

            try:
                raw_response = _ollama_generate(
                    prompt=full_prompt,
                    model=model,
                    base_url=base_url,
                    temperature=cfg.temperature,
                )
                api_calls += 1

                raw_items = _parse_llm_json(raw_response)

                for raw in raw_items:
                    validated = validate_draft(raw)
                    if not validated:
                        continue

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
                        extraction_backend="local_llm",
                    )
                    all_drafts.append(draft)

            except Exception as chunk_exc:
                logger.warning("Chunk %d Ollama extraction failed: %s", chunk.index, chunk_exc)
                continue

        # Step 4: Dedup
        dedup_result = deduplicate_drafts(all_drafts)
        dedup_vs_db = deduplicate_against_existing(
            dedup_result.kept, existing_titles, existing_contents,
        )
        final_drafts = dedup_vs_db.kept
        all_skipped = dedup_result.removed + dedup_vs_db.removed
        skipped_titles = dedup_result.removed_titles + dedup_vs_db.removed_titles

        from .base import CANONICAL_TYPE_MAP
        dynamic_labels = [
            d.type_label for d in final_drafts
            if d.type_label and d.type_label not in CANONICAL_TYPE_MAP
        ]

        # Step 5: Write to DB
        created_ids = _write_drafts_to_db(
            final_drafts, session_id, session.project_id, db, "local_llm"
        )

        _update_session_summary(db, session, len(created_ids), model)

        return P2ExtractionResult(
            session_id=session_id,
            memories_created=len(created_ids),
            summary=(
                f"Local LLM ({model}) extracted {len(created_ids)} memories "
                f"from {len(chunks)} chunks. "
                f"Noise filtered: {gate_result.filtered_count}. "
                f"Skipped duplicates: {len(all_skipped)}."
            ),
            memory_ids=created_ids,
            backend_used="local_llm",
            model_used=model,
            chunks_processed=len(chunks),
            noise_filtered_count=gate_result.filtered_count,
            llm_api_calls=api_calls,
            fallback_used=False,
            skipped_duplicates=len(all_skipped),
            duplicate_titles=skipped_titles,
            low_confidence_queued=sum(1 for d in final_drafts if d.confidence < 0.6),
            dynamic_type_labels=list(set(dynamic_labels)),
        )


# Register singleton
_instance = LocalLLMExtractionBackend()
register_backend(_instance)

local_llm_backend: ExtractionBackend = _instance  # type: ignore[assignment]
