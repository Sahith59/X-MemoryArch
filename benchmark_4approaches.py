"""
X-MemoryArch — Publishable 4-Approach × 3-Dataset Benchmark
============================================================
Datasets : SQuAD v1.1 · LoCoMo · LongMemEval (oracle)
Approaches:
  [1] Rule-based  — BM25 FTS5 + entity matching, zero ML
  [2] Local LLM   — Ollama nomic-embed-text + llama3.1 HyDE
  [3] Cloud LLM   — sentence-transformers + Claude Haiku contextual + HyDE
  [4] Hybrid Full — BM25+Dense+Entity+RRF+Graph+Rank+MMR + Cloud LLM

Usage:
  python3 benchmark_4approaches.py                  # 200 queries / dataset
  python3 benchmark_4approaches.py --n 500          # 500 queries / dataset
  python3 benchmark_4approaches.py --skip-ollama    # skip approach 2
  python3 benchmark_4approaches.py --skip-cloud     # skip approaches 3+4
"""
from __future__ import annotations

import argparse, json, math, os, random, struct, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import requests

# ── Path bootstrap ────────────────────────────────────────────────────────────
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

# ── DB factory ────────────────────────────────────────────────────────────────
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database import Base
import app.models as models
import app.p2_models          # noqa
import app.crud as crud
import app.schemas as schemas

def fresh_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                            poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        from app.search import setup_fts
        setup_fts(engine)
    except Exception:
        pass
    return engine, session

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200, help="queries per dataset (default 200)")
parser.add_argument("--ollama-n", type=int, default=100, help="max queries for Ollama (slow)")
parser.add_argument("--skip-ollama", action="store_true")
parser.add_argument("--skip-cloud", action="store_true")
parser.add_argument("--skip-a5", action="store_true", help="skip approach 5 (extracted facts)")
parser.add_argument("--embed-model", choices=["minilm", "bge", "gte"], default="gte",
                    help="dense embedding model: gte=thenlper/gte-small (default), minilm=all-MiniLM-L6-v2, bge=BAAI/bge-small-en-v1.5")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
random.seed(args.seed)

# ── Cache dir ─────────────────────────────────────────────────────────────────
CACHE = _RE / "benchmark_cache"
CACHE.mkdir(exist_ok=True)

# ── Embedding model config ─────────────────────────────────────────────────────
_EMBED_CONFIGS = {
    "minilm": {
        "hf_name":    "all-MiniLM-L6-v2",
        "cache_tag":  "minilm",
        "query_prefix": "",   # no instruction prefix needed
    },
    "bge": {
        "hf_name":    "BAAI/bge-small-en-v1.5",
        "cache_tag":  "bge_small",
        # BGE retrieval models benefit from an instruction prefix on queries only
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "gte": {
        "hf_name":    "thenlper/gte-small",
        "cache_tag":  "gte_small",
        # GTE uses symmetric embeddings — no special prefix for queries or documents
        "query_prefix": "",
    },
}
_ECFG = _EMBED_CONFIGS[args.embed_model]
EMBED_HF_NAME  = _ECFG["hf_name"]
EMBED_TAG      = _ECFG["cache_tag"]        # used in cache file names
EMBED_QPREFIX  = _ECFG["query_prefix"]    # prepended to query text only

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = _RE / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_EMBED = "nomic-embed-text:latest"

# ── Metric helpers ────────────────────────────────────────────────────────────
@dataclass
class RunMetrics:
    name: str
    recalls: dict[int, list[float]] = field(default_factory=dict)
    mrr: list[float] = field(default_factory=list)
    ndcg: list[float] = field(default_factory=list)
    lats: list[float] = field(default_factory=list)
    n_skipped: int = 0

    def add(self, retrieved: list[str], gold_ids: list[str], lat_ms: float, ks=(1,3,5,10)):
        gold_set = set(gold_ids)
        for k in ks:
            hit = int(any(r in gold_set for r in retrieved[:k]))
            self.recalls.setdefault(k, []).append(hit)
        # MRR
        mrr_val = 0.0
        for rank, mid in enumerate(retrieved[:10], 1):
            if mid in gold_set:
                mrr_val = 1.0 / rank
                break
        self.mrr.append(mrr_val)
        # NDCG (binary, single relevant)
        dcg = 0.0
        for rank, mid in enumerate(retrieved[:10], 1):
            if mid in gold_set:
                dcg = 1.0 / math.log2(rank + 1)
                break
        idcg = 1.0 / math.log2(2)
        self.ndcg.append(dcg / idcg if idcg else 0.0)
        self.lats.append(lat_ms)

    def r(self, k): a = self.recalls.get(k, []); return sum(a)/len(a) if a else 0
    def mrr_mean(self): return sum(self.mrr)/len(self.mrr) if self.mrr else 0
    def ndcg_mean(self): return sum(self.ndcg)/len(self.ndcg) if self.ndcg else 0
    def p(self, pct):
        if not self.lats: return 0
        s = sorted(self.lats); return s[min(int(len(s)*pct), len(s)-1)]

    def row(self, ks=(1,3,5,10)):
        parts = [f"{self.r(k):.3f}" for k in sorted(ks)]
        return " ".join(parts) + f"  {self.mrr_mean():.3f}  {self.ndcg_mean():.3f}  {self.p(0.5):.0f}ms  {self.p(0.95):.0f}ms"

# ── Dataset loaders ───────────────────────────────────────────────────────────
@dataclass
class MemEntry:
    mid: str
    title: str
    content: str
    search_text: str
    gold_key: str   # used as lookup key in queries

@dataclass
class QueryEntry:
    question: str
    gold_keys: list[str]   # list[gold_key] — all accepted correct answers

@dataclass
class BenchDataset:
    name: str
    memories: list[MemEntry]
    queries: list[QueryEntry]

def _cached(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text())
    result = fn()
    path.write_text(json.dumps(result, ensure_ascii=False))
    return result

def load_squad(n_queries: int) -> BenchDataset:
    print("  Loading SQuAD v1.1...")
    cache = CACHE / "squad.json"
    def _build():
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad", split="validation")
        ctx_map: dict[str, dict] = {}
        for ex in ds:
            ctx = ex["context"]
            if ctx not in ctx_map:
                ctx_map[ctx] = {"title": ex["title"], "questions": []}
            ctx_map[ctx]["questions"].append(ex["question"])
        mems, qs = [], []
        for i, (ctx, info) in enumerate(ctx_map.items()):
            key = f"ctx_{i}"
            mems.append({"mid": key, "title": info["title"], "content": ctx,
                         "search_text": info["title"] + " " + ctx, "gold_key": key})
            for q in info["questions"]:
                qs.append({"question": q, "gold_keys": [key]})
        return {"memories": mems, "queries": qs}
    raw = _cached(cache, _build)
    mems = [MemEntry(**m) for m in raw["memories"]]
    all_qs = [QueryEntry(**q) for q in raw["queries"]]
    sampled = random.sample(all_qs, min(n_queries, len(all_qs)))
    print(f"    {len(mems):,} memories  ·  {len(sampled):,} queries (of {len(all_qs):,})")
    return BenchDataset("SQuAD v1.1", mems, sampled)

def load_locomo(n_queries: int) -> BenchDataset:
    print("  Loading LoCoMo...")
    cache = CACHE / "locomo.json"
    def _build():
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        r = requests.get(url, timeout=60)
        data = r.json()
        mems, qs = [], []
        for conv_idx, conv in enumerate(data):
            conv_data = conv["conversation"]
            session_keys = sorted(
                [k for k in conv_data if k.startswith("session_") and "date" not in k]
            )

            # Build map: dia_id → session_key (within this conversation)
            # Needed to resolve QA evidence (turn-level) → session-level gold keys
            dia_to_session: dict[str, str] = {}

            for sk in session_keys:
                turns = conv_data[sk]
                lines = []
                for turn in turns:
                    dia_id = turn.get("dia_id", "")
                    speaker = turn.get("speaker", "?")
                    text = turn.get("text", "").strip()
                    if dia_id:
                        dia_to_session[dia_id] = sk
                    if text:
                        lines.append(f"{speaker}: {text}")

                if not lines:
                    continue

                # One memory per session — full context, capped at 1,200 chars
                session_content = "\n".join(lines)[:1200]
                session_id = f"c{conv_idx}_{sk}"
                date_val = conv_data.get(f"{sk}_date", "")
                title = f"Conv{conv_idx+1} {sk}" + (f" ({date_val})" if date_val else "")

                mems.append({
                    "mid": session_id, "title": title,
                    "content": session_content, "search_text": session_content,
                    "gold_key": session_id
                })

            for qa in conv["qa"]:
                evidence = qa.get("evidence", [])
                if not evidence:
                    continue
                if qa.get("category", 99) not in (1, 2, 3):
                    continue
                # Map evidence turn IDs → session IDs that contain those turns
                gold_sessions = list({
                    f"c{conv_idx}_{dia_to_session[e]}"
                    for e in evidence if e in dia_to_session
                })
                if not gold_sessions:
                    continue
                qs.append({"question": qa["question"], "gold_keys": gold_sessions})
        return {"memories": mems, "queries": qs}
    raw = _cached(cache, _build)
    mems = [MemEntry(**m) for m in raw["memories"]]
    all_qs = [QueryEntry(**q) for q in raw["queries"]]
    sampled = random.sample(all_qs, min(n_queries, len(all_qs)))
    print(f"    {len(mems):,} memories  ·  {len(sampled):,} queries (of {len(all_qs):,})")
    return BenchDataset("LoCoMo", mems, sampled)

def load_longmemeval(n_queries: int) -> BenchDataset:
    print("  Loading LongMemEval (oracle)...")
    cache = CACHE / "longmemeval.json"
    def _build():
        url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
        r = requests.get(url, timeout=60)
        data = r.json()
        # sessions: build unique session memories
        sess_map: dict[str, str] = {}  # session_id → content
        for item in data:
            ids = item.get("haystack_session_ids", [])
            sessions = item.get("haystack_sessions", [])
            for sid, turns in zip(ids, sessions):
                if sid in sess_map:
                    continue
                lines = []
                for turn in turns:
                    role = turn.get("role", "?")
                    content = turn.get("content", "")
                    lines.append(f"{role.capitalize()}: {content[:300]}")
                sess_map[sid] = "\n".join(lines[:20])  # cap at 20 turns
        mems = []
        for sid, content in sess_map.items():
            mems.append({"mid": sid, "title": f"Session {sid[:16]}",
                         "content": content, "search_text": content, "gold_key": sid})
        qs = []
        for item in data:
            ans_ids = item.get("answer_session_ids", [])
            if not ans_ids:
                continue
            qs.append({"question": item["question"], "gold_keys": ans_ids})
        return {"memories": mems, "queries": qs}
    raw = _cached(cache, _build)
    mems = [MemEntry(**m) for m in raw["memories"]]
    all_qs = [QueryEntry(**q) for q in raw["queries"]]
    sampled = random.sample(all_qs, min(n_queries, len(all_qs)))
    print(f"    {len(mems):,} memories  ·  {len(sampled):,} queries (of {len(all_qs):,})")
    return BenchDataset("LongMemEval", mems, sampled)

# ── LLM connectors ────────────────────────────────────────────────────────────
def make_ollama_llm(model: str = OLLAMA_LLM) -> Callable[[str], str] | None:
    def call(prompt: str) -> str:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120)
        return r.json().get("response", "")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        if model not in names:
            print(f"  [warn] Ollama model {model} not found. Available: {names}")
            return None
        call("hi")  # warm-up
        return call
    except Exception as e:
        print(f"  [warn] Ollama not reachable: {e}")
        return None

def ollama_embed(texts: list[str], model: str = OLLAMA_EMBED) -> np.ndarray | None:
    """Batch embed with Ollama nomic-embed-text. Returns (N, dim) float32 array or None."""
    try:
        vecs = []
        for text in texts:
            r = requests.post(f"{OLLAMA_URL}/api/embed",
                json={"model": model, "input": text}, timeout=30)
            data = r.json()
            emb = data.get("embeddings") or data.get("embedding")
            if emb is None:
                return None
            if isinstance(emb[0], list):
                emb = emb[0]
            vecs.append(np.array(emb, dtype=np.float32))
        mat = np.vstack(vecs)
        # Normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = np.where(norms > 0, mat / norms, mat)
        return mat
    except Exception as e:
        print(f"  [warn] Ollama embed error: {e}")
        return None

def make_claude_llm() -> Callable[[str], str] | None:
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        def call(prompt: str) -> str:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text
        return call
    except Exception as e:
        print(f"  [warn] Claude Haiku not available: {e}")
        return None

# ── Embedding helpers ─────────────────────────────────────────────────────────
_st_model = None
def get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBED_HF_NAME)
        print(f"    sentence-transformers {EMBED_HF_NAME} loaded")
    return _st_model

def st_embed_batch(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed a batch of texts. Set is_query=True to apply BGE query prefix."""
    model = get_st_model()
    if is_query and EMBED_QPREFIX:
        texts = [EMBED_QPREFIX + t for t in texts]
    batch = 256
    parts = []
    for i in range(0, len(texts), batch):
        parts.append(model.encode(texts[i:i+batch], convert_to_numpy=True, normalize_embeddings=True))
    return np.vstack(parts).astype(np.float32)

def st_embed_one(text: str, is_query: bool = True) -> list[float]:
    """Embed a single text. Defaults is_query=True — queries get the BGE prefix."""
    if is_query and EMBED_QPREFIX:
        text = EMBED_QPREFIX + text
    return get_st_model().encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()

# ── Contextual prefix generator (cached per dataset × LLM) ───────────────────
_FACT_PROMPT = """\
Extract 3-5 specific, self-contained facts from this conversation.
Each fact must stand alone without context. Capture specific events, decisions,
preferences, relationships, or information shared. One fact per line, no bullets or numbering.

Conversation:
{content}"""

def extract_session_facts(
    mems: list[MemEntry],
    cache_path: Path,
    max_workers: int = 8,
) -> dict[str, list[str]]:
    """Return {gold_key: [fact1, fact2, ...]}. Uses Claude Haiku directly; cached to disk."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    if not ANTHROPIC_API_KEY:
        return {}
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print(f"    Extracting facts from {len(mems):,} sessions via LLM...")
    results: dict[str, list[str]] = {}

    def _call(mem: MemEntry):
        content = mem.content[:800]
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{"role": "user", "content": _FACT_PROMPT.format(content=content)}],
            )
            raw = msg.content[0].text.strip()
            facts = [f.strip() for f in raw.split("\n") if f.strip()][:5]
        except Exception:
            facts = []
        return mem.gold_key, facts if facts else [mem.content[:200]]

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_call, m): m for m in mems}
        for f in as_completed(futs):
            key, facts = f.result()
            results[key] = facts
            done += 1
            if done % 200 == 0:
                print(f"      {done}/{len(mems)} sessions done", flush=True)
    cache_path.write_text(json.dumps(results, ensure_ascii=False))
    return results

_CTX_PROMPT = """\
Write a single-sentence context tag (max 20 words) for this memory snippet that describes
what it is about — who, what topic, or what event. Be specific and factual.
Memory: {content}
Context tag:"""

def generate_contextual_prefixes(
    mems: list[MemEntry],
    llm_fn: Callable,
    cache_path: Path,
    max_workers: int = 8,
) -> dict[str, str]:
    """Return {gold_key: prefix}. Cached to disk."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    print(f"    Generating contextual prefixes for {len(mems):,} memories via LLM...")
    results: dict[str, str] = {}
    def _call(mem: MemEntry):
        snippet = mem.content[:300]
        try:
            prefix = llm_fn(_CTX_PROMPT.format(content=snippet)).strip()[:120]
        except Exception:
            prefix = mem.title
        return mem.gold_key, prefix
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_call, m): m for m in mems}
        for f in as_completed(futs):
            key, prefix = f.result()
            results[key] = prefix
            done += 1
            if done % 200 == 0:
                print(f"      {done}/{len(mems)} prefixes done")
    cache_path.write_text(json.dumps(results, ensure_ascii=False))
    return results

# ── DB corpus builder ─────────────────────────────────────────────────────────
def build_corpus(
    mems: list[MemEntry],
    engine,
    db,
    embeddings: np.ndarray | None = None,
    embed_model: str | None = None,
) -> dict[str, str]:   # gold_key → memory.id (same here since mid = gold_key)
    proj = crud.create_project(db, schemas.ProjectCreate(
        name="benchmark", description="", tech_stack=[], goals=[], domain="software"))
    project_id = proj.id
    gid_to_dbid: dict[str, str] = {}
    import uuid
    batch = []
    for i, m in enumerate(mems):
        mem_id = str(uuid.uuid4())
        gid_to_dbid[m.gold_key] = mem_id
        mem = models.Memory(
            id=mem_id, project_id=project_id,
            type="insight", title=m.title[:200],
            content=m.content, search_text=m.search_text,
            importance=3, confidence=0.9,
            privacy_level="internal", review_status="approved", status="active",
        )
        if embeddings is not None:
            mem.embedding = embeddings[i].tobytes()
            mem.embedding_model = embed_model or "unknown"
        db.add(mem)
        if (i+1) % 500 == 0:
            db.commit()
    db.commit()
    return project_id, gid_to_dbid

# ── HyDE helper ───────────────────────────────────────────────────────────────
def hyde_augment(query: str, llm_fn: Callable, embed_fn: Callable) -> list[float]:
    """Generate HyDE-augmented query vector. Falls back to plain query vector."""
    q_vec = np.array(embed_fn(query), dtype=np.float32)
    n = np.linalg.norm(q_vec)
    if n > 0:
        q_vec /= n
    try:
        from app.services.retrieval.hyde import generate_hyde_text
        hyde_text = generate_hyde_text(query, llm_fn)
        h_vec = np.array(embed_fn(hyde_text), dtype=np.float32)
        hn = np.linalg.norm(h_vec)
        if hn > 0:
            h_vec /= hn
        combined = (q_vec + h_vec) / 2.0
        cn = np.linalg.norm(combined)
        return (combined / cn).tolist() if cn > 0 else q_vec.tolist()
    except Exception:
        return q_vec.tolist()

# ── Approach runners ──────────────────────────────────────────────────────────
def run_approach_1_rulebased(ds: BenchDataset) -> RunMetrics:
    """BM25 FTS5 + entity matching. Zero ML."""
    from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
    engine, db = fresh_db()
    project_id, gid_to_dbid = build_corpus(ds.memories, engine, db)
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    vb = SQLiteExactBackend(db)
    cfg = RetrievalConfig(top_k=10, embed_query=False,
                          enable_weighted_ranking=False, enable_reranker=False,
                          enable_mmr=False, enable_entity_boost=False,
                          enable_graph_expansion=False)
    m = RunMetrics("Rule-based (BM25+Entity)")
    # warm-up
    retrieve(db=db, project_id=project_id, query="warm", vector_backend=vb, config=cfg)
    for qe in ds.queries:
        t0 = time.monotonic()
        res = retrieve(db=db, project_id=project_id, query=qe.question,
                       vector_backend=vb, config=cfg)
        lat = (time.monotonic() - t0) * 1000
        retrieved = res.selected_memory_ids
        gold_db = [gid_to_dbid[g] for g in qe.gold_keys if g in gid_to_dbid]
        m.add(retrieved, gold_db, lat)
    db.close()
    return m

def run_approach_2_ollama(ds: BenchDataset, n_queries: int) -> RunMetrics | None:
    """Ollama: nomic-embed-text + llama3.1 HyDE."""
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    print("    Connecting to Ollama...")
    llm_fn = make_ollama_llm(OLLAMA_LLM)
    if llm_fn is None:
        print("    [skip] Ollama LLM not available")
        return None

    # Embed all memories with nomic-embed-text
    print(f"    Embedding {len(ds.memories):,} memories with nomic-embed-text...")
    cache_key = CACHE / f"embed_nomic_{ds.name.replace(' ','_')}.npy"
    if cache_key.exists():
        embs = np.load(str(cache_key))
        print(f"    Loaded embeddings from cache ({embs.shape})")
    else:
        texts = [m.content for m in ds.memories]
        embs = ollama_embed(texts, OLLAMA_EMBED)
        if embs is None:
            print("    [skip] nomic-embed-text embedding failed")
            return None
        np.save(str(cache_key), embs)
        print(f"    Embedded: {embs.shape}")

    engine, db = fresh_db()
    project_id, gid_to_dbid = build_corpus(ds.memories, engine, db, embs, "nomic-embed-text")
    vb = SQLiteExactBackend(db)

    import unittest.mock as mock

    def ollama_embed_one(text: str) -> list[float]:
        r = requests.post(f"{OLLAMA_URL}/api/embed",
            json={"model": OLLAMA_EMBED, "input": text}, timeout=30)
        data = r.json()
        emb = data.get("embeddings") or data.get("embedding")
        if isinstance(emb[0], list): emb = emb[0]
        v = np.array(emb, dtype=np.float32)
        n = np.linalg.norm(v)
        return (v / n).tolist() if n > 0 else v.tolist()

    m = RunMetrics("Local LLM (Ollama nomic-embed-text + llama3.1 HyDE)")
    sampled = random.sample(ds.queries, min(n_queries, len(ds.queries)))
    allowed, _ = __import__('app.services.retrieval.candidate_generators', fromlist=['apply_hard_filters']).apply_hard_filters(
        db=db, project_id=project_id, max_clearance="internal", include_superseded=False)
    # warm-up HyDE
    try: llm_fn("Say hi")
    except: pass

    for qe in sampled:
        t0 = time.monotonic()
        qvec = hyde_augment(qe.question, llm_fn, ollama_embed_one)
        results = vb.search(qvec, 10, project_id, allowed)
        lat = (time.monotonic() - t0) * 1000
        retrieved = [mid for mid, _ in results]
        gold_db = [gid_to_dbid[g] for g in qe.gold_keys if g in gid_to_dbid]
        m.add(retrieved, gold_db, lat)
    m.n_skipped = len(ds.queries) - len(sampled)
    db.close()
    return m

def run_approach_3_cloud(ds: BenchDataset, llm_fn: Callable) -> RunMetrics:
    """Cloud LLM: sentence-transformers + Claude Haiku contextual + HyDE."""
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    import unittest.mock as mock

    # Load or generate contextual prefixes (shared across embed models — text is text)
    ctx_cache = CACHE / f"ctx_claude_{ds.name.replace(' ','_')}.json"
    prefixes = generate_contextual_prefixes(ds.memories, llm_fn, ctx_cache, max_workers=10)

    # Build texts with contextual prefix prepended
    texts = []
    for m in ds.memories:
        pfx = prefixes.get(m.gold_key, "")
        texts.append(f"{pfx}\n\n{m.content}" if pfx else m.content)

    # Embed with sentence-transformers (cache is model-specific)
    print(f"    Embedding {len(texts):,} contextual memories ({EMBED_HF_NAME})...")
    cache_key = CACHE / f"embed_ctx_claude_{EMBED_TAG}_{ds.name.replace(' ','_')}.npy"
    if cache_key.exists():
        embs = np.load(str(cache_key))
        print(f"    Loaded from cache ({embs.shape})")
    else:
        embs = st_embed_batch(texts, is_query=False)
        np.save(str(cache_key), embs)

    engine, db = fresh_db()
    project_id, gid_to_dbid = build_corpus(ds.memories, engine, db, embs, EMBED_HF_NAME)
    vb = SQLiteExactBackend(db)
    allowed, _ = __import__('app.services.retrieval.candidate_generators', fromlist=['apply_hard_filters']).apply_hard_filters(
        db=db, project_id=project_id, max_clearance="internal", include_superseded=False)

    m = RunMetrics(f"Cloud LLM ({EMBED_HF_NAME} + Claude Haiku contextual+HyDE)")
    with mock.patch("app.services.semantic_classifier.embed_text",
                    side_effect=lambda t: st_embed_batch([t], is_query=False)[0].tobytes()):
        for qe in ds.queries:
            t0 = time.monotonic()
            qvec = hyde_augment(qe.question, llm_fn, st_embed_one)
            results = vb.search(qvec, 10, project_id, allowed)
            lat = (time.monotonic() - t0) * 1000
            retrieved = [mid for mid, _ in results]
            gold_db = [gid_to_dbid[g] for g in qe.gold_keys if g in gid_to_dbid]
            m.add(retrieved, gold_db, lat)

    db.close()
    return m

def run_approach_4_hybrid(ds: BenchDataset, llm_fn: Callable) -> RunMetrics:
    """Hybrid Full: BM25+Dense+Entity+RRF+Graph+Rank+MMR + Cloud contextual+HyDE."""
    from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    import unittest.mock as mock

    # Reuse contextual embeddings from approach 3 (same LLM, same embed model)
    ctx_cache = CACHE / f"ctx_claude_{ds.name.replace(' ','_')}.json"
    emb_cache = CACHE / f"embed_ctx_claude_{EMBED_TAG}_{ds.name.replace(' ','_')}.npy"

    if not ctx_cache.exists():
        print("    Generating contextual prefixes (reused from approach 3 if available)...")
        generate_contextual_prefixes(ds.memories, llm_fn, ctx_cache, max_workers=10)

    if emb_cache.exists():
        embs = np.load(str(emb_cache))
    else:
        prefixes = json.loads(ctx_cache.read_text()) if ctx_cache.exists() else {}
        texts = [f"{prefixes.get(m.gold_key,'')}\n\n{m.content}".strip() for m in ds.memories]
        embs = st_embed_batch(texts, is_query=False)
        np.save(str(emb_cache), embs)

    engine, db = fresh_db()
    project_id, gid_to_dbid = build_corpus(ds.memories, engine, db, embs, EMBED_HF_NAME)
    vb = SQLiteExactBackend(db)

    cfg = RetrievalConfig(
        top_k=10, embed_query=True,
        enable_weighted_ranking=True, enable_reranker=False,
        enable_mmr=True, mmr_lambda=0.70,
        enable_entity_boost=True, enable_graph_expansion=True, enable_2hop=False,
    )

    m = RunMetrics(f"Hybrid Full (BM25+Dense+RRF+Graph+MMR + {EMBED_HF_NAME} + Cloud LLM)")
    with mock.patch("app.services.semantic_classifier.embed_text",
                    side_effect=lambda t: st_embed_batch([t], is_query=False)[0].tobytes()):
        retrieve(db=db, project_id=project_id, query="warm", vector_backend=vb, config=cfg)  # warm-up
        for qe in ds.queries:
            t0 = time.monotonic()
            # HyDE-augmented query vector
            qvec = hyde_augment(qe.question, llm_fn, st_embed_one)
            # Override embed_query to use our HyDE vector
            # We patch the embed_text to return our pre-computed vector
            hyde_bytes = np.array(qvec, dtype=np.float32).tobytes()
            with mock.patch("app.services.semantic_classifier.embed_text",
                            return_value=hyde_bytes):
                res = retrieve(db=db, project_id=project_id, query=qe.question,
                               vector_backend=vb, config=cfg)
            lat = (time.monotonic() - t0) * 1000
            gold_db = [gid_to_dbid[g] for g in qe.gold_keys if g in gid_to_dbid]
            m.add(res.selected_memory_ids, gold_db, lat)

    db.close()
    return m

def run_approach_5_extracted_facts(ds: BenchDataset) -> RunMetrics:
    """A5: Dense cosine over LLM-extracted atomic facts. No HyDE — diagnostic baseline."""
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    from collections import defaultdict
    import uuid as _uuid

    facts_cache = CACHE / f"facts_claude_{ds.name.replace(' ', '_')}.json"
    emb_cache   = CACHE / f"embed_facts_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"

    # Step 1: load or extract facts
    session_facts = extract_session_facts(ds.memories, facts_cache, max_workers=8)
    if not session_facts:
        return RunMetrics("A5: Extracted Facts (unavailable)")

    # Step 2: flatten to one MemEntry per fact
    fact_mems: list[MemEntry] = []
    for mem in ds.memories:
        facts = session_facts.get(mem.gold_key) or [mem.content[:200]]
        for i, fact_text in enumerate(facts):
            fact_mems.append(MemEntry(
                mid=f"{mem.gold_key}__f{i}",
                title=f"{mem.title[:80]} [f{i+1}]",
                content=fact_text,
                search_text=fact_text,
                gold_key=mem.gold_key,  # tracks parent session
            ))
    print(f"    {len(fact_mems):,} facts from {len(ds.memories):,} sessions")

    # Step 3: embed facts (cached)
    if emb_cache.exists():
        fact_embs = np.load(str(emb_cache))
        print(f"    Loaded fact embeddings from cache ({fact_embs.shape})")
    else:
        print(f"    Embedding {len(fact_mems):,} facts ({EMBED_HF_NAME})...")
        fact_embs = st_embed_batch([m.content for m in fact_mems], is_query=False)
        np.save(str(emb_cache), fact_embs)

    # Step 4: build in-memory DB, track session_id → [fact_db_ids]
    engine, db = fresh_db()
    proj = crud.create_project(db, schemas.ProjectCreate(
        name="bench_a5", description="", tech_stack=[], goals=[], domain="software"))
    project_id = proj.id

    session_to_fact_ids: dict[str, list[str]] = defaultdict(list)
    for i, fm in enumerate(fact_mems):
        mem_id = str(_uuid.uuid4())
        session_to_fact_ids[fm.gold_key].append(mem_id)
        mem = models.Memory(
            id=mem_id, project_id=project_id,
            type="insight", title=fm.title[:200],
            content=fm.content, search_text=fm.search_text,
            importance=3, confidence=0.9,
            privacy_level="internal", review_status="approved", status="active",
            embedding=fact_embs[i].tobytes(),
            embedding_model=EMBED_HF_NAME,
        )
        db.add(mem)
        if (i + 1) % 500 == 0:
            db.commit()
    db.commit()

    vb = SQLiteExactBackend(db)
    allowed, _ = __import__(
        'app.services.retrieval.candidate_generators', fromlist=['apply_hard_filters']
    ).apply_hard_filters(db=db, project_id=project_id,
                         max_clearance="internal", include_superseded=False)

    # Step 5: retrieve via plain cosine — no HyDE, $0 diagnostic
    m = RunMetrics(f"A5: Extracted Facts ({EMBED_HF_NAME}, no HyDE)")
    for qe in ds.queries:
        t0 = time.monotonic()
        qvec = st_embed_one(qe.question, is_query=True)
        results = vb.search(qvec, 10, project_id, allowed)
        lat = (time.monotonic() - t0) * 1000
        retrieved_ids = [mid for mid, _ in results]
        gold_db: list[str] = []
        for gk in qe.gold_keys:
            gold_db.extend(session_to_fact_ids.get(gk, []))
        m.add(retrieved_ids, gold_db, lat)

    db.close()
    return m

# ── Evaluate one dataset across all approaches ────────────────────────────────
def evaluate_dataset(ds: BenchDataset, llm_fn: Callable | None) -> dict[str, RunMetrics]:
    results: dict[str, RunMetrics] = {}

    print(f"\n  [A1] Rule-based BM25+Entity...")
    results["rule_based"] = run_approach_1_rulebased(ds)
    r = results["rule_based"]
    print(f"    R@5={r.r(5):.3f}  MRR@10={r.mrr_mean():.3f}  p50={r.p(0.5):.0f}ms")

    if not args.skip_ollama:
        print(f"\n  [A2] Local LLM (Ollama)...")
        r2 = run_approach_2_ollama(ds, args.ollama_n)
        if r2 is not None:
            results["ollama"] = r2
            print(f"    R@5={r2.r(5):.3f}  MRR@10={r2.mrr_mean():.3f}  p50={r2.p(0.5):.0f}ms  (n={len(r2.mrr)}, skipped={r2.n_skipped})")
        else:
            print("    [skip] Ollama unavailable")
    else:
        print("  [A2] Ollama skipped (--skip-ollama)")

    if not args.skip_cloud and llm_fn is not None:
        print(f"\n  [A3] Cloud LLM (Claude Haiku contextual+HyDE)...")
        results["cloud"] = run_approach_3_cloud(ds, llm_fn)
        r3 = results["cloud"]
        print(f"    R@5={r3.r(5):.3f}  MRR@10={r3.mrr_mean():.3f}  p50={r3.p(0.5):.0f}ms")

        print(f"\n  [A4] Hybrid Full pipeline + Cloud LLM...")
        results["hybrid"] = run_approach_4_hybrid(ds, llm_fn)
        r4 = results["hybrid"]
        print(f"    R@5={r4.r(5):.3f}  MRR@10={r4.mrr_mean():.3f}  p50={r4.p(0.5):.0f}ms")
    elif args.skip_cloud:
        print("  [A3/A4] Cloud skipped (--skip-cloud)")
    else:
        print("  [A3/A4] Cloud skipped (no ANTHROPIC_API_KEY)")

    # A5 runs independently of --skip-cloud (A5 manages its own Anthropic calls,
    # uses only cheap fact extraction on first run, then cached dense retrieval)
    facts_cached = (CACHE / f"facts_claude_{ds.name.replace(' ', '_')}.json").exists()
    if not args.skip_a5 and ANTHROPIC_API_KEY:
        print(f"\n  [A5] Extracted Facts (LLM facts → dense cosine, no HyDE)...")
        results["extracted"] = run_approach_5_extracted_facts(ds)
        r5 = results["extracted"]
        if r5.mrr:
            print(f"    R@5={r5.r(5):.3f}  MRR@10={r5.mrr_mean():.3f}  p50={r5.p(0.5):.0f}ms")
        else:
            print("    [skip] A5 unavailable")
    elif args.skip_a5:
        print("  [A5] Extracted Facts skipped (--skip-a5)")
    elif not ANTHROPIC_API_KEY and not facts_cached:
        print("  [A5] Extracted Facts skipped (no API key + no cache)")

    return results

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("X-MemoryArch: 5-Approach × 3-Dataset Benchmark")
    print(f"Embed model: {EMBED_HF_NAME}")
    print("="*65)

    # Load LLM
    llm_fn = None
    if not args.skip_cloud:
        print("\nInitializing Claude Haiku (Cloud LLM)...")
        llm_fn = make_claude_llm()
        if llm_fn:
            print("  Claude Haiku ready")
        else:
            print("  [warn] Claude Haiku unavailable — cloud approaches will be skipped")

    # Load datasets
    print("\nLoading datasets...")
    datasets = [
        load_squad(args.n),
        load_locomo(args.n),
        load_longmemeval(args.n),
    ]

    # Run evaluations
    all_results: dict[str, dict[str, RunMetrics]] = {}
    for ds in datasets:
        print(f"\n{'='*65}")
        print(f"DATASET: {ds.name}  ({len(ds.memories):,} memories · {len(ds.queries):,} queries)")
        print("="*65)
        all_results[ds.name] = evaluate_dataset(ds, llm_fn)

    # ── Final report ──────────────────────────────────────────────────────────
    APPROACH_KEYS = [
        ("rule_based", "Rule-based (BM25+Entity)"),
        ("ollama",     "Local LLM  (Ollama + HyDE)"),
        ("cloud",      "Cloud LLM  (Claude Haiku ctx+HyDE)"),
        ("hybrid",     "Hybrid Full (BM25+Dense+LLM)"),
        ("extracted",  "A5: Extracted Facts (Dense, no HyDE)"),
    ]

    print("\n\n" + "="*90)
    print("FINAL RESULTS — X-MemoryArch 5 Approaches × 3 Datasets")
    print("="*90)
    print(f"{'Approach':<40} {'R@1':>5} {'R@3':>5} {'R@5':>5} {'R@10':>6} {'MRR@10':>7} {'NDCG@10':>8} {'p50':>6} {'p95':>6}")
    print("-"*90)

    for ds in datasets:
        print(f"\n  ── {ds.name}  ({len(ds.memories):,} memories · {len(ds.queries):,} queries) ──")
        dres = all_results[ds.name]
        for key, label in APPROACH_KEYS:
            m = dres.get(key)
            if m is None:
                print(f"  {'  '+label:<38} {'N/A':>5} {'':>5} {'':>5} {'':>6} {'':>7} {'':>8} {'':>6} {'':>6}")
                continue
            n_note = f"  (n={len(m.mrr)})" if m.n_skipped > 0 else ""
            print(f"  {'  '+label:<38} {m.r(1):>5.3f} {m.r(3):>5.3f} {m.r(5):>5.3f} {m.r(10):>6.3f} {m.mrr_mean():>7.3f} {m.ndcg_mean():>8.3f} {m.p(0.5):>5.0f}ms {m.p(0.95):>5.0f}ms{n_note}")

    print("\n" + "="*90)
    print("AGGREGATE — Mean across all datasets")
    print("="*90)
    print(f"{'Approach':<40} {'R@5':>5} {'MRR@10':>7} {'NDCG@10':>8}")
    print("-"*60)
    for key, label in APPROACH_KEYS:
        r5s, mrrs, ndcgs = [], [], []
        for ds in datasets:
            m = all_results[ds.name].get(key)
            if m and len(m.mrr) > 0:
                r5s.append(m.r(5)); mrrs.append(m.mrr_mean()); ndcgs.append(m.ndcg_mean())
        if not r5s:
            print(f"  {'  '+label:<38} {'N/A':>5}")
            continue
        print(f"  {'  '+label:<38} {sum(r5s)/len(r5s):>5.3f} {sum(mrrs)/len(mrrs):>7.3f} {sum(ndcgs)/len(ndcgs):>8.3f}")

    print("\n  Plan targets: Recall@5 ≥ 0.80  MRR@10 ≥ 0.78  NDCG@10 ≥ 0.55")
    print()

main()
