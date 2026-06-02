"""
Memory extraction service — Sub-phase 1.20 Semantic Extraction Engine.

Pipeline (3 stages):
  1. Regex noise gate  — strip AI preamble / obvious filler fast, before model load.
  2. Semantic tech gate — all-MiniLM-L6-v2 cosine similarity to TECHNICAL_EXEMPLARS
                          centroid.  Filters casual conversation, greetings, filler.
  3. Semantic type classifier — cosine similarity to per-type exemplar centroids.
                          Returns (type, confidence).  Confidence becomes the memory's
                          confidence score — a real semantic signal, not a keyword count.

Regex is kept for two jobs where pattern matching is the right tool:
  - Structured field extraction  (_build_type_metadata)
  - File path extraction         (_extract_file_paths)

Public contract (unchanged):
    extract_memories_from_session(session_id: str, db: Session) -> ExtractionResult
"""
from __future__ import annotations
import difflib
import re
from sqlalchemy.orm import Session

from app import crud, schemas
from app.schemas import (
    ProblemMetadata, DecisionMetadata, StructureMetadata,
    HowToMetadata, OpenQuestionMetadata,
    ConstraintMetadata, WorkflowPatternMetadata, FailedApproachMetadata,
)
from app.services import semantic_classifier, entity_extractor


# ---------------------------------------------------------------------------
# Stage 1 — Fast regex noise gate (runs BEFORE model is loaded)
# Catches obvious AI preamble and filler without any ML cost.
# ---------------------------------------------------------------------------

_NOISE_RE = re.compile(
    r"^("
    r"thought for \d+|"
    r"thinking\.\.\.|"
    r"processing\.\.\.|"
    r"let me (think|check|look)|"
    r"sure[,!]?\s|"
    r"of course[,!]?\s|"
    r"great[,!]?\s|"
    r"okay[,!]?\s|"
    r"alright[,!]?\s|"
    r"certainly[,!]?\s|"
    r"absolutely[,!]?\s|"
    r"here('s| is) (what|how)|"
    r"i('ll| will| can| would)|"
    r"as (an|a) ai|"
    r"<[^>]+>|"
    r"hey[,!]?\s|"
    r"hi[,!]?\s|"
    r"hello[,!]?\s|"
    r"how('s| are| is)|"
    r"what('s| a) up|"
    r"glad to hear|"
    r"sounds (good|great|like)|"
    r"that('s| is) (great|good|interesting|cool|awesome|nice)|"
    r"thank(s| you)|"
    r"you're welcome|"
    r"no problem"
    r")",
    re.IGNORECASE,
)


def _is_obvious_noise(sentence: str) -> bool:
    return bool(_NOISE_RE.match(sentence.strip()))


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

_TYPE_WEIGHTS: dict[str, int] = {
    "decision": 4,
    "problem": 4,           # was: bug
    "structure": 4,         # was: architecture
    "failed_approach": 4,   # negative knowledge — equally critical for AI handoff
    "constraint": 3,
    "workflow_pattern": 3,
    "how_to": 2,            # was: setup_instruction
    "task": 2,
    "insight": 2,
    "open_question": 2,
    "reference_context": 2, # was: code_context
    "conversation_note": 1,
    "unclassified": 1,
}


def _position_score(idx: int, total: int) -> float:
    if total <= 1:
        return 1.0
    frac = idx / (total - 1)
    if frac < 0.33:
        return 1.0
    if frac < 0.66:
        return 0.9
    return 0.8


def _importance(mem_type: str, semantic_confidence: float, idx: int, total: int) -> int:
    """
    Three-factor importance score.
    semantic_confidence replaces the old keyword_strength — it's a real 0-1 signal
    from the model rather than a proxy keyword count.
    """
    tw = _TYPE_WEIGHTS.get(mem_type, 2)
    ps = _position_score(idx, total)
    # Scale confidence from [THRESHOLD, 1.0] to [0.70, 1.0] to use the full 1-5 range.
    threshold = semantic_classifier.TYPE_CONF_THRESHOLD
    scaled = 0.70 + max(0.0, semantic_confidence - threshold) / (1.0 - threshold) * 0.30
    return max(1, min(5, round(tw * scaled * ps)))


# ---------------------------------------------------------------------------
# File path extraction (regex is the right tool here)
# ---------------------------------------------------------------------------

_FILE_PATH_RE = re.compile(
    r"\b(?:(?:src|app|lib|test|tests|cmd|pkg|internal)/)"
    r"[\w\-]+(?:/[\w\-]+)*"
    r"\.(?:py|ts|js|tsx|jsx|rs|go|java|cpp|c|h|rb|sh|yaml|yml|json|md)\b"
    r"|"
    r"\./[\w\-]+(?:/[\w\-]+)*"
    r"\.(?:py|ts|js|tsx|jsx|rs|go|java|cpp|c|h|rb|sh|yaml|yml|json|md)\b"
)


def _extract_file_paths(text: str) -> list[str]:
    return list({m.group() for m in _FILE_PATH_RE.finditer(text)})


# ---------------------------------------------------------------------------
# Parsing helpers (unchanged from 1.13)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_HEADER_RE = re.compile(r"^#{1,4}\s+(.+)$")
_BULLET_RE = re.compile(r"^[-*•]\s+|^\d+\.\s+")
_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
_MIN_SENTENCE_LEN = 15

_TITLE_FLUFF_RE = re.compile(
    r"^(the |a |an |this |that |we |i |our |it |they |there |these |those )",
    re.IGNORECASE,
)


def _parse_raw(raw: str) -> list[tuple[str | None, str]]:
    # Strip triple-backtick code blocks — code lines are not memories
    raw = _CODE_BLOCK_RE.sub("\n", raw)
    items: list[tuple[str | None, str]] = []
    current_tag: str | None = None
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        m = _HEADER_RE.match(stripped)
        if m:
            current_tag = re.sub(r"[^\w\s]", "", m.group(1)).strip().lower()
            continue
        # Strip bullet/numbered-list prefix so "- Use JWT for auth" is treated
        # as a full sentence rather than a fragment starting with punctuation.
        stripped = _BULLET_RE.sub("", stripped).strip()
        if not stripped:
            continue
        for sent in _SENTENCE_RE.split(stripped):
            sent = sent.strip()
            if len(sent) >= _MIN_SENTENCE_LEN:
                items.append((current_tag, sent))
    return items


def _make_title(sentence: str, max_len: int = 80) -> str:
    t = sentence.strip().rstrip(".!?,;")
    t = _TITLE_FLUFF_RE.sub("", t).strip()
    if t:
        t = t[0].upper() + t[1:]
    return t[:max_len] + ("..." if len(t) > max_len else "")


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# Stage-3 structured field extraction (regex stays here — right tool for
# extracting known patterns like rationale clauses and error messages)
# ---------------------------------------------------------------------------

_RATIONALE_RE = re.compile(
    r"\b(because|since|due to|so that|in order to|as a result|given that|"
    r"therefore|thus|hence|the reason|we chose|we picked|we decided)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_RE = re.compile(
    r"\b(instead of|vs\.?|versus|over|compared to|alternative[s]?|"
    r"considered|evaluated|looked at|tried|rejected)\b",
    re.IGNORECASE,
)
_ERROR_MSG_RE = re.compile(
    r"(Error|Exception|Traceback|failed|crash|undefined|null|NullPointer|"
    r"AttributeError|TypeError|ValueError|KeyError|ImportError)[:\s][^\n.]*",
    re.IGNORECASE,
)
_FIX_RE = re.compile(
    r"\b(fix(ed)?|resolv(ed)?|solution|work ?around|patch(ed)?|"
    r"we changed|we updated|we added|by running|by adding)\b",
    re.IGNORECASE,
)
_COMMAND_RE = re.compile(
    r"(`[^`]+`|run\s+[\w\-]+|execute\s+[\w\-]+|\$\s*[\w\-\.]+)",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"(how|when|why|what|should|which)[^.?!]*\?", re.IGNORECASE)
_STEP_RE = re.compile(
    r"(\d+\.\s+[^.]+|first[,\s]|then[,\s]|finally[,\s]|next[,\s])",
    re.IGNORECASE,
)
_TOOL_RE = re.compile(
    r"\b(docker|kubernetes|k8s|nginx|redis|postgresql|postgres|sqlite|"
    r"fastapi|flask|django|react|next\.?js|typescript|python|alembic|"
    r"pytest|git|github|vercel|aws|gcp|azure|terraform)\b",
    re.IGNORECASE,
)

# Failed-approach extraction patterns
_FAILURE_RE = re.compile(
    r"\b(tried|attempted|experimented with|tested|evaluated|explored|"
    r"we had|used to use|switched away from|moved away from|abandoned|"
    r"dropped|replaced|gave up on|rolled back|reverted|"
    r"didn't work|does not work|failed|too slow|too complex|"
    r"ran into issues|caused problems|performance issues|"
    r"consistency issues|compatibility issues|security issues|"
    r"not suitable|couldn't scale|overwhelmed|unstable)\b",
    re.IGNORECASE,
)
_AVOID_RE = re.compile(
    r"\b(because|since|due to|as a result of|the reason|"
    r"it caused|which caused|leading to|resulted in)\b",
    re.IGNORECASE,
)
_SWITCHED_TO_RE = re.compile(
    r"\b(switched to|replaced with|moved to|migrated to|"
    r"now using|use .{1,30} instead|chose .{1,30} instead)\b",
    re.IGNORECASE,
)


def _extract_rationale(text: str) -> str | None:
    m = _RATIONALE_RE.search(text)
    if not m:
        return None
    snippet = text[m.start():m.start() + 150].split(".")[0].strip()
    return snippet if len(snippet) > 10 else None


def _extract_alternatives(text: str) -> list[str]:
    if not _ALTERNATIVE_RE.search(text):
        return []
    return list({m.group().lower() for m in _TOOL_RE.finditer(text)})[:4]


def _build_type_metadata(
    mem_type: schemas.MemoryType,
    sentence: str,
    context: str,
) -> dict | None:
    if mem_type == schemas.MemoryType.decision:
        rationale = _extract_rationale(context)
        alternatives = _extract_alternatives(context)
        if rationale or alternatives:
            return DecisionMetadata(
                rationale=rationale,
                alternatives_considered=alternatives,
                decision_status="accepted",
            ).model_dump(exclude_none=True)

    elif mem_type == schemas.MemoryType.problem:
        err_match = _ERROR_MSG_RE.search(context)
        error_msg = err_match.group(0)[:200] if err_match else None
        fix_match = _FIX_RE.search(context)
        fix_text = None
        if fix_match:
            fix_text = context[fix_match.start():fix_match.start() + 150].split(".")[0].strip()
        if error_msg or fix_text:
            return ProblemMetadata(
                error_message=error_msg,
                fix_applied=fix_text,
            ).model_dump(exclude_none=True)

    elif mem_type == schemas.MemoryType.structure:
        rationale = _extract_rationale(context)
        tools = list({m.group().lower() for m in _TOOL_RE.finditer(context)})[:4]
        if rationale or tools:
            return StructureMetadata(
                rationale=rationale,
                components_affected=tools,
            ).model_dump(exclude_none=True)

    elif mem_type == schemas.MemoryType.how_to:
        cmd_match = _COMMAND_RE.search(context)
        command = cmd_match.group(0).strip("`") if cmd_match else None
        tools = list({m.group().lower() for m in _TOOL_RE.finditer(context)})[:3]
        if command or tools:
            return HowToMetadata(
                command=command,
                prerequisites=tools,
            ).model_dump(exclude_none=True)

    elif mem_type == schemas.MemoryType.open_question:
        q_match = _QUESTION_RE.search(context)
        question_context = q_match.group(0)[:150] if q_match else None
        if question_context:
            return OpenQuestionMetadata(context=question_context).model_dump(exclude_none=True)

    elif mem_type == schemas.MemoryType.constraint:
        return ConstraintMetadata(
            source="extracted",
            impact=context[:100].split(".")[0] if context else None,
        ).model_dump(exclude_none=True)

    elif mem_type == schemas.MemoryType.workflow_pattern:
        tools = list({m.group().lower() for m in _TOOL_RE.finditer(context)})[:4]
        step_matches = _STEP_RE.findall(context)
        steps = [s[0].strip() if isinstance(s, tuple) else s.strip() for s in step_matches[:5]]
        if steps or tools:
            return WorkflowPatternMetadata(
                steps=steps,
                tools_used=tools,
            ).model_dump(exclude_none=True)

    elif mem_type == schemas.MemoryType.failed_approach:
        # Extract what was tried (the tool/approach noun near the failure verb)
        failure_match = _FAILURE_RE.search(context)
        approach_tried = None
        if failure_match:
            start = max(0, failure_match.start() - 60)
            snippet = context[start:failure_match.end() + 80]
            approach_tried = snippet.split(".")[0].strip()[:200]

        # Extract why it failed
        avoid_match = _AVOID_RE.search(context)
        reason_failed = None
        if avoid_match:
            reason_failed = context[avoid_match.start():avoid_match.start() + 150].split(".")[0].strip()

        # Extract what was chosen instead
        switch_match = _SWITCHED_TO_RE.search(context)
        alternative_chosen = None
        if switch_match:
            alternative_chosen = context[switch_match.start():switch_match.start() + 100].split(".")[0].strip()

        # Surface any observable symptoms (error messages, performance terms)
        tools_mentioned = list({m.group().lower() for m in _TOOL_RE.finditer(context)})[:4]

        return FailedApproachMetadata(
            approach_tried=approach_tried,
            reason_failed=reason_failed,
            symptoms=tools_mentioned,
            alternative_chosen=alternative_chosen,
            avoid_because=reason_failed[:100] if reason_failed else None,
        ).model_dump(exclude_none=True)

    return None


def _build_summary(raw: str, tool_name: str, keyword_counts: dict[str, int]) -> str:
    preview = raw.strip()[:300].replace("\n", " ")
    if len(raw) > 300:
        preview += "..."
    found = [(k, v) for k, v in keyword_counts.items() if v > 0]
    if found:
        extracted = ", ".join(
            f"{v} {k}{'s' if v > 1 else ''}" for k, v in found
        )
        return f"{preview}\n\n_Extracted: {extracted}_"
    return preview


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_memories_from_session(
    session_id: str, db: Session
) -> schemas.ExtractionResult:
    session = crud.get_session(db, session_id)
    raw = session.raw_content
    project_domain = (session.project.domain or "general") if session.project else "general"

    items = _parse_raw(raw)
    total = len(items)

    existing = crud.get_memories(db, session.project_id, limit=500)
    known_titles: list[str] = [m.title.lower() for m in existing]

    keyword_counts: dict[str, int] = {t.value: 0 for t in schemas.MemoryType}
    created: list[schemas.MemoryResponse] = []
    skipped_duplicates: list[str] = []
    seen_keys: set[str] = set()
    low_confidence_queued: int = 0

    # ── Stage 1 pre-filter (regex, no ML cost) ───────────────────────────────
    # Collect non-noise items so we can batch-encode them in one model call.
    surviving = [
        (idx, tag, sent)
        for idx, (tag, sent) in enumerate(items)
        if not _is_obvious_noise(sent)
    ]

    # ── Batch encode all surviving sentences in one shot ─────────────────────
    # Each sentence is encoded once and reused for gate + classifier.
    # Context window embeddings are encoded separately (different text).
    if surviving:
        sentences_only = [sent for _, _, sent in surviving]
        vecs = semantic_classifier.batch_encode(sentences_only)
        vec_map: dict[str, object] = {s: v for s, v in zip(sentences_only, vecs)}
    else:
        vec_map = {}

    for idx, section_tag, sentence in surviving:
        vec = vec_map[sentence]

        # ── Stage 2: semantic technical gate ────────────────────────────────
        if not semantic_classifier.is_technical_vec(vec):
            continue

        # ── Stage 3: semantic type classification ────────────────────────────
        result = semantic_classifier.classify_vec(vec, domain=project_domain)
        _suggested_type: str | None = None
        if result is None:
            # Gate passed but no type confident enough — queue for manual review
            best_type, best_score = semantic_classifier.classify_best_vec(
                vec, domain=project_domain
            )
            mem_type_str, semantic_confidence = "unclassified", best_score
            mem_type = schemas.MemoryType.unclassified
            _suggested_type = best_type
            low_confidence_queued += 1
        else:
            mem_type_str, semantic_confidence = result
            # Map string back to enum
            try:
                mem_type = schemas.MemoryType(mem_type_str)
            except ValueError:
                continue

        keyword_counts[mem_type.value] += 1

        extracted_title = _make_title(sentence)
        dedup_key = f"{mem_type.value}:{extracted_title}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        if any(_similarity(extracted_title, t) >= 0.8 for t in known_titles):
            skipped_duplicates.append(extracted_title)
            continue

        # Context window: sentence before + matched + sentence after
        window_parts: list[str] = []
        if idx > 0:
            window_parts.append(items[idx - 1][1])
        window_parts.append(sentence)
        if idx < total - 1:
            window_parts.append(items[idx + 1][1])
        context_content = " ".join(window_parts)

        importance = _importance(mem_type.value, semantic_confidence, idx, total)

        tags = ["auto-extracted"]
        if section_tag:
            tags.append(section_tag)

        file_paths = _extract_file_paths(context_content)
        if mem_type == schemas.MemoryType.unclassified:
            type_meta: dict | None = {
                "suggested_type": _suggested_type,
                "classifier_confidence": round(semantic_confidence, 4),
            }
        else:
            type_meta = _build_type_metadata(mem_type, sentence, context_content)

        review_status = (
            schemas.ReviewStatus.needs_review
            if mem_type == schemas.MemoryType.unclassified
            else schemas.ReviewStatus.auto_extracted
        )

        # Embed the context window — stored for Phase-2 vector retrieval
        embedding_bytes = semantic_classifier.embed_text(context_content)

        mem_data = schemas.MemoryCreate(
            type=mem_type,
            title=extracted_title,
            content=context_content,
            importance=importance,
            confidence=round(semantic_confidence, 4),   # real semantic signal
            tags=tags,
            related_tools=[session.tool_name],
            related_files=file_paths,
            status=schemas.MemoryStatus.active,
            source_session_id=session.id,
            file_path=file_paths[0] if file_paths else None,
            review_status=review_status,
            source_type=schemas.SourceType.ai_session,
            source_quote=sentence[:500],
            type_metadata=type_meta,
        )
        memory = crud.create_memory(db, session.project_id, mem_data,
                                    embedding=embedding_bytes)
        # Sub-phase 1.32: create parallel approved suggestion record (Option B)
        suggestion_data = schemas.SuggestionCreate(
            suggested_type=mem_type,
            title=extracted_title,
            content=context_content,
            confidence=round(semantic_confidence, 4),
            source_session_id=session.id,
        )
        crud.create_suggestion(
            db, session.project_id, suggestion_data,
            status="approved",
            created_by="rule_based",
            memory_id=memory.id,
        )
        # Entity extraction — runs on each extracted memory at write time (1.22)
        project_domain = session.project.domain if session.project else "general"
        entities = entity_extractor.extract_entities_for_memory(
            memory.title, memory.content, domain=project_domain or "general"
        )
        crud.create_memory_entities(db, memory.id, session.project_id, entities)
        # Auto-link by shared entities (1.29)
        crud.autolink_by_entities(db, session.project_id, memory.id)
        created.append(schemas.MemoryResponse.model_validate(memory))
        known_titles.append(extracted_title.lower())

    summary = _build_summary(raw, session.tool_name, keyword_counts)
    crud.update_session(db, session_id, schemas.SessionUpdate(summary=summary))

    # Sub-phase 1.27: recluster after bulk extraction so new memories get assignments
    if created:
        crud.recluster_project_memories(db, session.project_id)

    return schemas.ExtractionResult(
        session_id=session_id,
        memories_created=len(created),
        summary=summary,
        memories=created,
        skipped_duplicates=len(skipped_duplicates),
        duplicate_titles=skipped_duplicates,
        low_confidence_queued=low_confidence_queued,
    )
