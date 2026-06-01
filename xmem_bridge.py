"""
xmem_bridge.py — The only interface between Xmem CLI and the memory system.

Four public functions. Nothing else. The CLI never touches RetrievalEngine
internals, project-memory-core models, or the SQLite schema directly.

    write(content, project_name, session_id, provider, mode)
    retrieve_fast(query, project_name, top_k, mode)   → A7r  (28ms)
    retrieve_quality(query, project_name, top_k)      → A4mvr (200ms)
    build_handoff_packet(project_name, context_query) → str

Operational modes:
  "local"  — chunk embedding only, no API calls, A4r fallback for fast path
  "ollama" — Ollama fact extraction, A7r fast path
  "cloud"  — Claude Haiku fact extraction, both A7r + A4mvr available (default)
  "hybrid" — cloud with local fallback

All memories are stored in the shared SQLite at XMEM_DB_PATH.
Embeddings use GTE-large (1024-dim) — the same model as the retrieval engine's
champion approach A4mvr (0.822 R@5, beats Zep CE by 12%).

Memory tagging convention:
  source_type="xmem_chunk"  — paragraph splits, used by retrieve_quality (A4mvr)
  source_type="xmem_fact"   — Claude Haiku extracted facts, used by retrieve_fast (A7r)
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np

# ── Path bootstrap (same pattern as benchmark_4approaches.py) ─────────────────
# Merges project-memory-core's app package into RetrievalEngine's app namespace
# so SQLiteExactBackend, crud, models, etc. all resolve correctly.
_RE = Path(__file__).resolve().parent
_P1 = _RE.parent / "project-memory-core"

if str(_RE) not in sys.path:
    sys.path.insert(0, str(_RE))

import app as _p2
_p1a = str(_P1 / "app")
if _p1a not in list(_p2.__path__):
    _p2.__path__.append(_p1a)

import app.services as _p2s
_p1s = str(_P1 / "app" / "services")
if _p1s not in list(_p2s.__path__):
    _p2s.__path__.append(_p1s)

import app.models as _models
import app.crud as _crud
import app.schemas as _schemas
from app.database import Base

# ── Mode type ─────────────────────────────────────────────────────────────────
Mode = Literal["local", "ollama", "cloud", "hybrid"]

# ── Constants ─────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME  = "thenlper/gte-large"   # 1024-dim, matches A4mvr/A7r in benchmark
EMBED_DIM         = 1024
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_ALPHA        = 0.2    # learned in Phase 3.5: 80% Dense + 20% BM25 is optimal
RRF_K             = 60     # standard RRF constant
MAX_CANDIDATES    = 40     # candidates fed to cross-encoder

FACT_EXTRACT_PROMPT = """\
Extract {n} atomic memory facts from this AI conversation exchange.
Each fact must be:
- A complete standalone sentence (makes sense without any other context)
- Specific: include names, numbers, dates, file paths, decisions when present
- Worth remembering across sessions

Conversation:
{content}

Output exactly {n} facts, one per line, no numbering or bullets:"""

# ── Lazy singletons ───────────────────────────────────────────────────────────
_embed_model       = None
_embed_lock        = threading.Lock()
_reranker          = None
_reranker_lock     = threading.Lock()


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                from sentence_transformers import SentenceTransformer
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ── Database helpers ──────────────────────────────────────────────────────────

def _db_path() -> str:
    """Resolve the shared SQLite path from env or default."""
    raw = os.environ.get("XMEM_DB_PATH", str(Path.home() / ".xmem" / "memories.db"))
    p = Path(raw).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{p}"


def _make_engine():
    from sqlalchemy import create_engine
    return create_engine(_db_path(), connect_args={"check_same_thread": False})


def _make_session(engine=None):
    from sqlalchemy.orm import sessionmaker
    if engine is None:
        engine = _make_engine()
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _get_or_create_project(db, name: str) -> _models.Project:
    """Find project by name or create it. Returns the ORM object."""
    existing = db.query(_models.Project).filter(_models.Project.name == name).first()
    if existing:
        return existing
    return _crud.create_project(db, _schemas.ProjectCreate(
        name=name, description=f"Xmem project: {name}",
        tech_stack=[], goals=[], domain="general",
    ))


# ── Chunking (same logic as A4mvr in benchmark) ───────────────────────────────

def _chunk_content(content: str, min_chars: int = 50) -> list[str]:
    """Split exchange content into paragraph chunks (split on \\n\\n)."""
    raw = [c.strip() for c in content.split("\n\n") if c.strip()]
    if not raw:
        return [content.strip() or content]
    merged: list[str] = []
    for chunk in raw:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged if merged else [content]


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts with GTE-large. Returns [N, 1024] float32."""
    model = _get_embed_model()
    embs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embs, dtype=np.float32)


def _embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns [1024] float32."""
    return _embed_texts([query])[0]


# ── Claude Haiku fact extraction (cloud mode) ─────────────────────────────────

def _extract_facts_cloud(content: str, n_facts: int = 6) -> list[str]:
    """Call Claude Haiku to extract n atomic facts from content."""
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or _load_env_key("ANTHROPIC_API_KEY")
    )
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot use cloud mode")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": FACT_EXTRACT_PROMPT.format(
                    n=n_facts,
                    content=content[:4000],
                ),
            }],
        )
        raw = resp.content[0].text.strip()
        facts = [
            line.strip() for line in raw.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return facts[:n_facts]
    except Exception as e:
        raise RuntimeError(f"Claude Haiku fact extraction failed: {e}") from e


def _extract_facts_local(content: str) -> list[str]:
    """Local fallback: sentence-split the content as 'facts'."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", content.strip())
    merged: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if merged and len(merged[-1]) < 30:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return [s for s in merged if len(s) > 20][:8]


def _load_env_key(key: str) -> str | None:
    """Load key from RetrievalEngine/.env as fallback."""
    env_file = _RE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    return None


# ── Memory write helpers ───────────────────────────────────────────────────────

def _store_memories(
    db,
    project_id: str,
    session_id: str,
    texts: list[str],
    source_type: str,
    mem_type: str = "insight",
    title_prefix: str = "",
) -> None:
    """Embed and store a list of texts as Memory rows."""
    if not texts:
        return

    embs = _embed_texts(texts)

    for i, (text, emb) in enumerate(zip(texts, embs)):
        title = (title_prefix + " " + text[:60]).strip() if title_prefix else text[:80]
        mem = _models.Memory(
            id=str(uuid.uuid4()),
            project_id=project_id,
            source_session_id=session_id,
            type=mem_type,
            title=title,
            content=text,
            search_text=text,
            importance=3,
            confidence=0.9,
            privacy_level="internal",
            review_status="approved",
            status="active",
            source_type=source_type,         # "xmem_chunk" | "xmem_fact"
            embedding=emb.tobytes(),
            embedding_model=EMBED_MODEL_NAME,
            embedding_dim=EMBED_DIM,
        )
        db.add(mem)

    db.commit()


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def write(
    content: str,
    project_name: str,
    session_id: str | None = None,
    provider: str = "unknown",
    mode: Mode = "cloud",
    n_facts: int = 6,
) -> dict:
    """
    Extract and store memories from a single user/assistant exchange.

    Args:
        content:      Full exchange text, e.g. "User: ...\n\nAssistant: ..."
        project_name: Logical project name (e.g. "auth-fix", "inbox")
        session_id:   Optional existing session ID to attach memories to.
                      If None, a new session is created.
        provider:     AI tool name ("claude", "chatgpt", "gemini", ...)
        mode:         "local" | "ollama" | "cloud" | "hybrid"
        n_facts:      Number of atomic facts to extract (cloud/ollama modes)

    Returns:
        {"session_id": str, "chunks_stored": int, "facts_stored": int}

    Mode behaviour:
      local:  paragraph chunks only, GTE-large embedding, $0
      cloud:  paragraph chunks + Claude Haiku facts, both paths available
      hybrid: cloud with local fallback if API unavailable
    """
    db = _make_session()
    try:
        project = _get_or_create_project(db, project_name)

        # Create or reuse AISession
        if session_id is None:
            ai_session = _crud.create_session(db, project.id, _schemas.SessionCreate(
                tool_name=provider,
                title=f"{provider} session ({time.strftime('%Y-%m-%d %H:%M')})",
                raw_content=content,
                session_date=time.strftime("%Y-%m-%d"),
            ))
            session_id = ai_session.id
        else:
            # Append content to existing session if it exists
            try:
                ai_session = _crud.get_session(db, session_id)
                _crud.update_session(db, session_id, _schemas.SessionUpdate(
                    raw_content=ai_session.raw_content + "\n\n" + content
                ))
            except Exception:
                ai_session = _crud.create_session(db, project.id, _schemas.SessionCreate(
                    tool_name=provider,
                    title=f"{provider} session",
                    raw_content=content,
                    session_date=time.strftime("%Y-%m-%d"),
                ))
                session_id = ai_session.id

        # ── Paragraph chunks (always stored — powers retrieve_quality/A4mvr) ──
        chunks = _chunk_content(content)
        _store_memories(
            db, project.id, session_id,
            texts=chunks,
            source_type="xmem_chunk",
            mem_type="context",
            title_prefix="[chunk]",
        )

        # ── Atomic facts (stored in cloud/hybrid/ollama modes — powers retrieve_fast/A7r)
        facts_stored = 0
        if mode in ("cloud", "hybrid"):
            try:
                facts = _extract_facts_cloud(content, n_facts=n_facts)
            except RuntimeError:
                if mode == "hybrid":
                    facts = _extract_facts_local(content)
                else:
                    raise
            _store_memories(
                db, project.id, session_id,
                texts=facts,
                source_type="xmem_fact",
                mem_type="insight",
                title_prefix="[fact]",
            )
            facts_stored = len(facts)

        elif mode == "ollama":
            # Delegate to local sentence splitting for now (Ollama integration in v1.1)
            facts = _extract_facts_local(content)
            _store_memories(
                db, project.id, session_id,
                texts=facts,
                source_type="xmem_fact",
                mem_type="insight",
                title_prefix="[fact]",
            )
            facts_stored = len(facts)

        return {
            "session_id":    session_id,
            "project_id":    project.id,
            "chunks_stored": len(chunks),
            "facts_stored":  facts_stored,
        }

    finally:
        db.close()


def retrieve_fast(
    query: str,
    project_name: str,
    top_k: int = 5,
    mode: Mode = "cloud",
) -> list[dict]:
    """
    Fast retrieval for real-time context injection (before every user message).

    cloud/hybrid/ollama: A7r — BM25(α=0.2) + Dense(α=0.8) on extracted facts,
                         then ms-marco cross-encoder reranker. ~28ms p50.
    local:               A4r fallback — dense cosine on paragraph chunks. ~41ms p50.

    Args:
        query:        The user's current message / query.
        project_name: Which project's memories to search.
        top_k:        Number of unique sessions to return.
        mode:         Controls which memory type to search (facts vs chunks).

    Returns:
        list of dicts: [{
            "content": str,         # the matched fact or chunk
            "session_id": str,      # source session ID
            "title": str,           # memory title
            "score": float,         # cross-encoder score
            "source_type": str,     # "xmem_fact" or "xmem_chunk"
        }]
    """
    from rank_bm25 import BM25Okapi

    db = _make_session()
    try:
        project = db.query(_models.Project).filter(_models.Project.name == project_name).first()
        if not project:
            return []

        # Choose which memory type to search
        use_facts = mode in ("cloud", "hybrid", "ollama")
        target_source = "xmem_fact" if use_facts else "xmem_chunk"

        # Load memories of the target type for this project
        mems = (
            db.query(_models.Memory)
            .filter(
                _models.Memory.project_id == project.id,
                _models.Memory.source_type == target_source,
                _models.Memory.embedding.isnot(None),
                _models.Memory.status == "active",
            )
            .all()
        )

        if not mems:
            # Fallback: if no facts found (e.g. local mode or fresh project), use chunks
            mems = (
                db.query(_models.Memory)
                .filter(
                    _models.Memory.project_id == project.id,
                    _models.Memory.source_type == "xmem_chunk",
                    _models.Memory.embedding.isnot(None),
                    _models.Memory.status == "active",
                )
                .all()
            )
            if not mems:
                return []

        # Build numpy embedding matrix
        ids      = [m.id for m in mems]
        contents = [m.content for m in mems]
        sessions = [m.source_session_id or "" for m in mems]
        titles   = [m.title for m in mems]
        emb_mat  = np.array(
            [np.frombuffer(m.embedding, dtype=np.float32) for m in mems],
            dtype=np.float32,
        )  # [N, 1024]

        # Dense retrieval
        qvec        = _embed_query(query)
        dense_scores = emb_mat @ qvec  # [N]
        dense_top    = np.argsort(dense_scores)[::-1][:MAX_CANDIDATES]

        # BM25 retrieval (only when facts available — they're short and BM25-friendly)
        rrf: dict[int, float] = {}
        if use_facts:
            tokenized_corpus = [c.lower().split() for c in contents]
            bm25             = BM25Okapi(tokenized_corpus)
            bm25_scores      = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)
            bm25_top         = np.argsort(bm25_scores)[::-1][:MAX_CANDIDATES]

            for rank, idx in enumerate(bm25_top):
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + BM25_ALPHA / (RRF_K + rank)
            for rank, idx in enumerate(dense_top):
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + (1.0 - BM25_ALPHA) / (RRF_K + rank)
        else:
            # Local mode: dense only
            for rank, idx in enumerate(dense_top):
                rrf[int(idx)] = 1.0 / (RRF_K + rank)

        top_idxs = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:MAX_CANDIDATES]

        # Cross-encoder reranker
        try:
            reranker = _get_reranker()
            pairs    = [(query, contents[i]) for i in top_idxs]
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            top_idxs  = [idx for _, idx in sorted(zip(ce_scores, top_idxs), reverse=True)]
            score_map  = {idx: float(s) for idx, s in zip(top_idxs, sorted(ce_scores, reverse=True))}
        except Exception:
            score_map = {idx: float(rrf.get(idx, 0.0)) for idx in top_idxs}

        # Session-diversity MMR: max 2 memories per session, top_k unique sessions
        results: list[dict] = []
        sess_count: dict[str, int] = defaultdict(int)
        seen_sessions: set[str] = set()

        for idx in top_idxs:
            sid = sessions[idx]
            if sess_count[sid] < 2:
                results.append({
                    "content":     contents[idx],
                    "session_id":  sid,
                    "title":       titles[idx],
                    "score":       score_map.get(idx, 0.0),
                    "source_type": target_source,
                    "memory_id":   ids[idx],
                })
                sess_count[sid] += 1
                if sid not in seen_sessions:
                    seen_sessions.add(sid)
            if len(seen_sessions) >= top_k:
                break

        return results

    finally:
        db.close()


def retrieve_quality(
    query: str,
    project_name: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Quality retrieval using A4mvr (ColBERT-style chunk max-sim + cross-encoder).
    Works in ALL modes — chunk embeddings are always stored regardless of mode.
    Use for handoff packets and explicit --search commands.

    Pipeline (A4mvr, Phase 3.3 champion — 0.822 aggregate R@5):
      1. Load all chunk memories for the project (GTE-large 1024-dim embeddings)
      2. Group chunks by session
      3. For each session: compute max cosine similarity across its chunks (ColBERT)
      4. Rank sessions by max-sim → top-40 candidate sessions
      5. Cross-encoder reranker on (query, best_chunk) per session → top-k

    Args:
        query:        The query or context description.
        project_name: Which project's memories to search.
        top_k:        Number of unique sessions to return.

    Returns:
        list of dicts: [{
            "content": str,       # the best-matching chunk from this session
            "session_id": str,
            "title": str,
            "score": float,       # cross-encoder score
            "source_type": "xmem_chunk",
        }]
    """
    db = _make_session()
    try:
        project = db.query(_models.Project).filter(_models.Project.name == project_name).first()
        if not project:
            return []

        # Load all chunk memories
        mems = (
            db.query(_models.Memory)
            .filter(
                _models.Memory.project_id == project.id,
                _models.Memory.source_type == "xmem_chunk",
                _models.Memory.embedding.isnot(None),
                _models.Memory.status == "active",
            )
            .all()
        )

        if not mems:
            return []

        # Build numpy matrix
        ids      = [m.id for m in mems]
        contents = [m.content for m in mems]
        sessions = [m.source_session_id or m.id for m in mems]
        titles   = [m.title for m in mems]
        emb_mat  = np.array(
            [np.frombuffer(m.embedding, dtype=np.float32) for m in mems],
            dtype=np.float32,
        )  # [N, 1024]

        # Group chunk indices by session
        session_to_idxs: dict[str, list[int]] = defaultdict(list)
        for i, sid in enumerate(sessions):
            session_to_idxs[sid].append(i)

        # ColBERT max-sim: for each session, take max cosine similarity across chunks
        qvec         = _embed_query(query)
        all_sims     = emb_mat @ qvec  # [N]

        session_scores: list[tuple[float, str, int]] = []  # (max_sim, session_id, best_chunk_idx)
        for sid, idxs in session_to_idxs.items():
            chunk_sims  = all_sims[idxs]
            best_local  = int(np.argmax(chunk_sims))
            best_global = idxs[best_local]
            session_scores.append((float(chunk_sims[best_local]), sid, best_global))

        # Sort sessions by max-sim → top-40 candidates
        session_scores.sort(key=lambda x: x[0], reverse=True)
        top_sessions = session_scores[:MAX_CANDIDATES]

        # Cross-encoder reranker: (query, best_chunk_for_session)
        try:
            reranker = _get_reranker()
            pairs    = [(query, contents[best_idx]) for _, _, best_idx in top_sessions]
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            reranked  = sorted(
                zip(ce_scores, top_sessions),
                key=lambda x: x[0],
                reverse=True,
            )
        except Exception:
            # Fallback: use max-sim scores
            reranked = [(sim, entry) for sim, *entry_rest in [(s[0], s) for s in top_sessions]]
            reranked = sorted(reranked, key=lambda x: x[0], reverse=True)

        # Build results — one per unique session, top_k sessions
        results: list[dict] = []
        for ce_score, (_, sid, best_idx) in reranked[:top_k]:
            results.append({
                "content":     contents[best_idx],
                "session_id":  sid,
                "title":       titles[best_idx],
                "score":       float(ce_score),
                "source_type": "xmem_chunk",
                "memory_id":   ids[best_idx],
            })

        return results

    finally:
        db.close()


def build_handoff_packet(
    project_name: str,
    context_query: str | None = None,
    target_provider: str = "AI assistant",
    max_tokens: int = 500,
) -> str:
    """
    Build a formatted context string ready to prepend to any AI's system prompt.

    Uses retrieve_quality (A4mvr) internally — always works regardless of mode.
    The output is intentionally concise (<= max_tokens) to minimise context window cost.

    Args:
        project_name:    The Xmem project whose memories to summarise.
        context_query:   Optional query to guide which memories to retrieve.
                         If None, retrieves the most recent/relevant memories broadly.
        target_provider: The AI tool receiving this packet (for the header).
        max_tokens:      Approximate token cap on the output. (1 token ≈ 4 chars)

    Returns:
        A formatted string like:
            [Xmem Context — from Claude session, 14:30]
            • JWT refresh flow has a race condition in token validation
            • Backend fix applied: mutex lock in the refresh endpoint
            • Frontend breaking: API response shape changed
            [End Context — project "auth-fix", 3 memories]

        Returns empty string if no memories exist.
    """
    query = context_query or f"recent work and decisions in project {project_name}"
    memories = retrieve_quality(query, project_name, top_k=5)

    if not memories:
        return ""

    # Format bullets — keep each under 120 chars for readability
    bullets: list[str] = []
    char_budget = max_tokens * 4   # rough char estimate

    for mem in memories:
        content = mem["content"].strip()
        # Strip role labels from raw chunks (e.g. "User: ..." → "...")
        content = re.sub(r"^(User|Assistant|Human|AI):\s*", "", content, flags=re.IGNORECASE)
        # Take first sentence if too long
        sentences = re.split(r"(?<=[.!?])\s+", content)
        line = sentences[0] if sentences else content
        if len(line) > 120:
            line = line[:117] + "..."
        bullet = f"• {line}"
        if char_budget - len(bullet) < 50:
            break
        bullets.append(bullet)
        char_budget -= len(bullet) + 1

    if not bullets:
        return ""

    timestamp = time.strftime("%H:%M")
    header    = f"[Xmem Context — project \"{project_name}\", {timestamp}]"
    footer    = f"[End Context — {len(bullets)} memor{'y' if len(bullets)==1 else 'ies'} for {target_provider}]"
    body      = "\n".join(bullets)

    return f"{header}\n{body}\n{footer}"


# ── Convenience: list recent sessions ────────────────────────────────────────

def list_sessions(project_name: str, limit: int = 10) -> list[dict]:
    """
    Return recent AISession metadata for a project.
    Used by xmem --list.
    """
    db = _make_session()
    try:
        project = db.query(_models.Project).filter(_models.Project.name == project_name).first()
        if not project:
            return []
        sessions = _crud.get_sessions(db, project.id, limit=limit)
        return [
            {
                "session_id":  s.id,
                "tool_name":   s.tool_name,
                "title":       s.title,
                "date":        s.session_date,
                "created_at":  str(s.created_at),
            }
            for s in sessions
        ]
    finally:
        db.close()


def delete_session(session_id: str) -> bool:
    """Delete all memories and session record for a session_id. Used by xmem --forget."""
    db = _make_session()
    try:
        # Delete associated memories first
        db.query(_models.Memory).filter(
            _models.Memory.source_session_id == session_id
        ).delete()
        db.commit()
        # Delete session
        try:
            _crud.delete_session(db, session_id)
            return True
        except Exception:
            return False
    finally:
        db.close()


def search(query: str, project_name: str, top_k: int = 10) -> list[dict]:
    """
    Explicit memory search using retrieve_quality (best quality, A4mvr).
    Used by xmem --search "query string".
    """
    return retrieve_quality(query, project_name, top_k=top_k)


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test():
    """Quick smoke test — run with: python3 xmem_bridge.py"""
    import tempfile, os

    print("xmem_bridge self-test")
    print("=" * 50)

    # Use temp DB for test
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["XMEM_DB_PATH"] = str(Path(tmp) / "test.db")

        # 1. Write a fake exchange (local mode — no API needed)
        print("\n[1] write() — local mode, no API")
        result = write(
            content=(
                "User: We decided to use FastAPI over Flask for the backend.\n\n"
                "Assistant: Good choice. FastAPI gives you async support, automatic OpenAPI docs, "
                "and Pydantic validation out of the box. The main tradeoff is that it's newer "
                "and has a smaller ecosystem than Flask.\n\n"
                "User: Also, we're using PostgreSQL with SQLAlchemy ORM.\n\n"
                "Assistant: That's a solid stack. PostgreSQL handles complex queries well and "
                "SQLAlchemy gives you clean ORM abstractions."
            ),
            project_name="test-project",
            provider="claude",
            mode="local",
        )
        print(f"   stored: {result['chunks_stored']} chunks, {result['facts_stored']} facts")
        assert result["chunks_stored"] > 0

        # 2. Write another exchange
        result2 = write(
            content=(
                "User: What's the JWT token expiry we agreed on?\n\n"
                "Assistant: We set JWT access tokens to expire in 15 minutes and "
                "refresh tokens to expire in 7 days. The refresh token rotation is "
                "handled by the auth middleware."
            ),
            project_name="test-project",
            provider="claude",
            mode="local",
        )
        print(f"   stored: {result2['chunks_stored']} chunks")

        # 3. retrieve_quality
        print("\n[2] retrieve_quality() — A4mvr")
        hits = retrieve_quality("what database are we using?", "test-project", top_k=3)
        print(f"   returned {len(hits)} results")
        for h in hits:
            print(f"   • [{h['score']:.2f}] {h['content'][:80]}")
        assert len(hits) > 0

        # 4. retrieve_fast (local fallback — no facts in local mode)
        print("\n[3] retrieve_fast() — A7r local fallback")
        hits2 = retrieve_fast("FastAPI vs Flask decision", "test-project", mode="local")
        print(f"   returned {len(hits2)} results")
        for h in hits2:
            print(f"   • [{h['score']:.2f}] {h['content'][:80]}")

        # 5. build_handoff_packet
        print("\n[4] build_handoff_packet()")
        packet = build_handoff_packet(
            "test-project",
            context_query="tech stack decisions",
            target_provider="ChatGPT",
        )
        print(packet)
        assert "[Xmem Context" in packet

        print("\n✓ All tests passed")


if __name__ == "__main__":
    _self_test()
