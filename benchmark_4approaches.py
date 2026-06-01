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
parser.add_argument("--embed-model",
                    choices=["minilm", "bge", "gte", "bgel", "gtel", "te3l", "xma_gte_v1"],
                    default="gtel",
                    help=(
                        "gte=thenlper/gte-small (default, local 384-dim), "
                        "gtel=thenlper/gte-large (local 1024-dim, best local), "
                        "xma_gte_v1=models/gte-large-xma-v1 (Phase 3.4 fine-tuned GTE-large), "
                        "te3l=text-embedding-3-large (OpenAI cloud 3072-dim, best overall), "
                        "bgel=BAAI/bge-large-en-v1.5 (local 1024-dim, passage-biased), "
                        "bge=BAAI/bge-small-en-v1.5, minilm=all-MiniLM-L6-v2"
                    ))
parser.add_argument("--multi-hyde", action="store_true",
                    help="2.9c: use 3-hypothesis Multi-HyDE for A3, A4, and A5r (3x API calls per query)")
parser.add_argument("--reranker-model",
                    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="Cross-encoder reranker model name or local path (default: ms-marco-MiniLM-L-6-v2)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
random.seed(args.seed)

# ── Cache dir ─────────────────────────────────────────────────────────────────
CACHE = _RE / "benchmark_cache"
CACHE.mkdir(exist_ok=True)

# ── Embedding model config ─────────────────────────────────────────────────────
_EMBED_CONFIGS = {
    "minilm": {
        "backend": "st",
        "hf_name": "all-MiniLM-L6-v2",
        "cache_tag": "minilm",
        "query_prefix": "",
    },
    "bge": {
        "backend": "st",
        "hf_name": "BAAI/bge-small-en-v1.5",
        "cache_tag": "bge_small",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "gte": {
        "backend": "st",
        "hf_name": "thenlper/gte-small",
        "cache_tag": "gte_small",
        "query_prefix": "",
    },
    "bgel": {
        "backend": "st",
        "hf_name": "BAAI/bge-large-en-v1.5",   # 1024-dim, passage-retrieval biased
        "cache_tag": "bge_large",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "gtel": {
        "backend": "st",
        "hf_name": "thenlper/gte-large",  # 1024-dim, same GTE family as gte-small, no custom code
        "cache_tag": "gte_large",
        "query_prefix": "",  # symmetric embeddings, no prefix needed
    },
    "xma_gte_v1": {
        "backend": "st",
        "hf_name": "models/gte-large-xma-v1",  # Phase 3.4: fine-tuned on LoCoMo+LME conversational facts
        "cache_tag": "xma_gte_v1",
        "query_prefix": "",
    },
    "te3l": {
        "backend": "openai",
        "openai_name": "text-embedding-3-large",  # 3072-dim, best overall
        "hf_name": None,
        "cache_tag": "te3l",
        "query_prefix": "",
    },
}
_ECFG         = _EMBED_CONFIGS[args.embed_model]
EMBED_BACKEND  = _ECFG["backend"]                         # "st" | "openai"
EMBED_HF_NAME  = _ECFG.get("hf_name") or _ECFG.get("openai_name", "unknown")
EMBED_TAG      = _ECFG["cache_tag"]
EMBED_QPREFIX  = _ECFG["query_prefix"]

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = _RE / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
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

# ── OpenAI embedding backend (te3l: text-embedding-3-large, 3072-dim) ────────
_openai_client = None
def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set — required for --embed-model te3l")
        try:
            import openai
            _openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            print(f"    OpenAI client ready ({_ECFG['openai_name']})")
        except ImportError:
            raise RuntimeError("openai package not installed: pip install openai")
    return _openai_client

def _openai_embed_batch(texts: list[str]) -> np.ndarray:
    client = _get_openai_client()
    model  = _ECFG["openai_name"]
    all_vecs: list[np.ndarray] = []
    # OpenAI limits: 2048 inputs OR 300K tokens per request.
    # Use dynamic batching: estimate tokens as chars/4, cap at 240K (80% buffer).
    TOKEN_LIMIT = 240_000
    CHARS_PER_TOK = 4
    idx = 0
    while idx < len(texts):
        batch: list[str] = []
        tok_est = 0
        while idx < len(texts) and len(batch) < 2048:
            t = max(1, len(texts[idx]) // CHARS_PER_TOK)
            if batch and tok_est + t > TOKEN_LIMIT:
                break
            batch.append(texts[idx])
            tok_est += t
            idx += 1
        resp = client.embeddings.create(model=model, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        all_vecs.extend(np.array(d.embedding, dtype=np.float32) for d in ordered)
    mat = np.vstack(all_vecs).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.where(norms > 0, mat / norms, mat)

def _openai_embed_one(text: str) -> list[float]:
    client = _get_openai_client()
    resp = client.embeddings.create(model=_ECFG["openai_name"], input=[text])
    vec  = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()

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
    """Embed a batch — dispatches to OpenAI or sentence-transformers based on EMBED_BACKEND."""
    if EMBED_BACKEND == "openai":
        return _openai_embed_batch(texts)
    model = get_st_model()
    if is_query and EMBED_QPREFIX:
        texts = [EMBED_QPREFIX + t for t in texts]
    batch = 256
    parts = []
    for i in range(0, len(texts), batch):
        parts.append(model.encode(texts[i:i+batch], convert_to_numpy=True, normalize_embeddings=True))
    return np.vstack(parts).astype(np.float32)

def st_embed_one(text: str, is_query: bool = True) -> list[float]:
    """Embed a single text — dispatches to OpenAI or sentence-transformers based on EMBED_BACKEND."""
    if EMBED_BACKEND == "openai":
        return _openai_embed_one(text)
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

# ── Multi-Query RRF helpers (Phase 3.1) ──────────────────────────────────────
_MQR_REPHRASE_PROMPT = """\
Generate 2 alternative phrasings of this question. Same meaning, different wording and structure.
Return exactly 2 lines, one phrasing per line. No bullets, no numbers, no explanation.

Question: {query}"""

def generate_query_rephrases(
    queries: list[str],
    cache_path: Path,
    max_workers: int = 8,
) -> dict[str, list[str]]:
    """Return {original_query: [rephrase1, rephrase2]}. Cached to disk.
    Cache key is dataset-specific — independent of embedding model (text is text).
    """
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    if not ANTHROPIC_API_KEY:
        return {}
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    print(f"    Generating rephrases for {len(queries):,} queries via Claude Haiku...")
    results: dict[str, list[str]] = {}

    def _call(query: str):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[{"role": "user", "content": _MQR_REPHRASE_PROMPT.format(query=query)}],
            )
            raw = msg.content[0].text.strip()
            rephrases = [r.strip() for r in raw.split("\n") if r.strip()][:2]
            while len(rephrases) < 2:
                rephrases.append(query)  # fallback: repeat original if haiku returns < 2 lines
        except Exception:
            rephrases = [query, query]
        return query, rephrases

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_call, q): q for q in queries}
        for f in as_completed(futs):
            query, rephrases = f.result()
            results[query] = rephrases
            done += 1
            if done % 100 == 0:
                print(f"      {done}/{len(queries)} rephrases done", flush=True)
    cache_path.write_text(json.dumps(results, ensure_ascii=False))
    return results

def rrf_merge(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of fact row indices.
    Returns [(fact_row_idx, rrf_score)] sorted descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

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

# ── HyDE helpers ─────────────────────────────────────────────────────────────
def hyde_augment(query: str, llm_fn: Callable, embed_fn: Callable) -> list[float]:
    """Single-HyDE: generate 1 hypothetical document, average with query vector."""
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

# Three prompts generate hypothetical documents from different angles —
# factual, personal/experiential, and completion-style — for richer coverage.
_MULTI_HYDE_SESSION_PROMPTS = [
    "Write a memory entry that directly answers this question: {q}",
    "Write a personal memory about an experience or preference relevant to: {q}",
    "Write a brief note containing the information needed to answer: {q}",
]
_MULTI_HYDE_FACT_PROMPTS = [
    "Write one short, self-contained fact that answers: {q}",
    "Write a specific memory fact about a person that answers: {q}",
    "State in one sentence what a memory would say to answer: {q}",
]

def multi_hyde_augment(query: str, llm_fn: Callable, embed_fn: Callable) -> list[float]:
    """Multi-HyDE (2.9c): 3 hypothetical session docs averaged with query vector."""
    q_vec = np.array(embed_fn(query, is_query=True), dtype=np.float32)
    qn = np.linalg.norm(q_vec)
    if qn > 0: q_vec /= qn
    vecs = [q_vec]
    for tmpl in _MULTI_HYDE_SESSION_PROMPTS:
        try:
            text = llm_fn(tmpl.format(q=query)).strip()
            h = np.array(embed_fn(text, is_query=False), dtype=np.float32)
            hn = np.linalg.norm(h)
            if hn > 0: vecs.append(h / hn)
        except Exception:
            pass
    avg = np.mean(vecs, axis=0)
    n = np.linalg.norm(avg)
    return (avg / n).tolist() if n > 0 else q_vec.tolist()

def multi_hyde_fact_augment(query: str, llm_fn: Callable) -> list[float]:
    """Multi-HyDE for fact corpus (2.9c): 3 hypothetical facts averaged with query vector."""
    q_vec = np.array(st_embed_one(query, is_query=True), dtype=np.float32)
    qn = np.linalg.norm(q_vec)
    if qn > 0: q_vec /= qn
    vecs = [q_vec]
    for tmpl in _MULTI_HYDE_FACT_PROMPTS:
        try:
            fact = llm_fn(tmpl.format(q=query)).strip()[:200]
            h = np.array(st_embed_one(fact, is_query=False), dtype=np.float32)
            hn = np.linalg.norm(h)
            if hn > 0: vecs.append(h / hn)
        except Exception:
            pass
    avg = np.mean(vecs, axis=0)
    n = np.linalg.norm(avg)
    return (avg / n).tolist() if n > 0 else q_vec.tolist()

# ── Chunking helper (A4mv) ───────────────────────────────────────────────────
def chunk_session(content: str, min_chars: int = 50) -> list[str]:
    """Split a session into paragraph-level chunks.
    Splits on double-newline boundaries; merges orphaned tiny fragments (< min_chars)
    into the preceding chunk. Sessions that don't split produce a single chunk,
    making A4mv identical to plain dense retrieval for short sessions.
    """
    raw = [c.strip() for c in content.split('\n\n') if c.strip()]
    if not raw:
        return [content.strip() or content]
    merged: list[str] = []
    for chunk in raw:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged if merged else [content]

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

    _hyde_tag = "Multi-HyDE" if args.multi_hyde else "HyDE"
    m = RunMetrics(f"Cloud LLM ({EMBED_HF_NAME} + Claude Haiku contextual+{_hyde_tag})")
    _augment_fn = multi_hyde_augment if args.multi_hyde else hyde_augment
    with mock.patch("app.services.semantic_classifier.embed_text",
                    side_effect=lambda t: st_embed_batch([t], is_query=False)[0].tobytes()):
        for qe in ds.queries:
            t0 = time.monotonic()
            qvec = _augment_fn(qe.question, llm_fn, st_embed_one)
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
        enable_weighted_ranking=True, enable_reranker=True,   # 2.9b: reranker on by default
        reranker_top_n=20,
        enable_mmr=True, mmr_lambda=0.70,
        enable_entity_boost=True, enable_graph_expansion=True, enable_2hop=False,
    )

    _hyde_tag = "Multi-HyDE" if args.multi_hyde else "HyDE"
    _augment_fn = multi_hyde_augment if args.multi_hyde else hyde_augment
    m = RunMetrics(f"Hybrid Full (BM25+Dense+RRF+Graph+Reranker+MMR + {EMBED_HF_NAME} + {_hyde_tag})")
    with mock.patch("app.services.semantic_classifier.embed_text",
                    side_effect=lambda t: st_embed_batch([t], is_query=False)[0].tobytes()):
        retrieve(db=db, project_id=project_id, query="warm", vector_backend=vb, config=cfg)  # warm-up
        for qe in ds.queries:
            t0 = time.monotonic()
            qvec = _augment_fn(qe.question, llm_fn, st_embed_one)
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

def run_approach_4r_dense_reranked(ds: BenchDataset) -> RunMetrics:
    """A4r: Full hybrid pipeline + cross-encoder reranker, plain GTE query (no HyDE).
    $0 diagnostic — isolates reranker effect without any cloud API calls."""
    from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    import unittest.mock as mock

    # Use plain GTE embeddings (no contextual prefix — this is the diagnostic, not the full A4)
    emb_cache = CACHE / f"embed_facts_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"
    plain_cache = CACHE / f"embed_plain_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"

    if plain_cache.exists():
        embs = np.load(str(plain_cache))
    else:
        print(f"    Embedding {len(ds.memories):,} memories ({EMBED_HF_NAME})...")
        embs = st_embed_batch([m.content for m in ds.memories], is_query=False)
        np.save(str(plain_cache), embs)

    engine, db = fresh_db()
    project_id, gid_to_dbid = build_corpus(ds.memories, engine, db, embs, EMBED_HF_NAME)
    vb = SQLiteExactBackend(db)

    cfg = RetrievalConfig(
        top_k=10, embed_query=True,
        enable_weighted_ranking=True, enable_reranker=True,
        reranker_top_n=20,
        enable_mmr=True, mmr_lambda=0.70,
        enable_entity_boost=True, enable_graph_expansion=True, enable_2hop=False,
    )

    m = RunMetrics(f"A4r: Hybrid+Reranker ({EMBED_HF_NAME}, no HyDE)")
    with mock.patch("app.services.semantic_classifier.embed_text",
                    side_effect=lambda t: st_embed_batch([t], is_query=False)[0].tobytes()):
        retrieve(db=db, project_id=project_id, query="warm", vector_backend=vb, config=cfg)  # warm-up
        for qe in ds.queries:
            t0 = time.monotonic()
            qvec = st_embed_one(qe.question, is_query=True)
            qbytes = np.array(qvec, dtype=np.float32).tobytes()
            with mock.patch("app.services.semantic_classifier.embed_text",
                            return_value=qbytes):
                res = retrieve(db=db, project_id=project_id, query=qe.question,
                               vector_backend=vb, config=cfg)
            lat = (time.monotonic() - t0) * 1000
            gold_db = [gid_to_dbid[g] for g in qe.gold_keys if g in gid_to_dbid]
            m.add(res.selected_memory_ids, gold_db, lat)

    db.close()
    return m

def run_approach_4mv_multivec(ds: BenchDataset) -> RunMetrics:
    """A4mv: ColBERT-style multi-vector retrieval (2.9d).
    Each session is split into paragraph chunks; each chunk is embedded independently.
    Retrieval scores every chunk against the query, then max-pools per session (late
    interaction). No reranker, no HyDE — $0 diagnostic.

    Unlike A5 (LLM-extracted facts), chunks are purely structural — no inference cost.
    This is the cheapest path to sub-session granularity.
    """
    chunk_emb_cache = CACHE / f"embed_chunks_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"

    # ── Step 1: build chunk list ──────────────────────────────────────────────
    all_chunks: list[str] = []
    all_session_keys: list[str] = []  # parallel: which session each chunk belongs to

    for mem in ds.memories:
        chunks = chunk_session(mem.content)
        all_chunks.extend(chunks)
        all_session_keys.extend([mem.gold_key] * len(chunks))

    n_chunks = len(all_chunks)
    avg_per_session = n_chunks / len(ds.memories) if ds.memories else 0
    print(f"    {n_chunks:,} chunks from {len(ds.memories):,} sessions "
          f"(avg {avg_per_session:.1f} chunks/session)")

    # ── Step 2: embed chunks (cached, separate from plain-session cache) ──────
    if chunk_emb_cache.exists():
        chunk_matrix = np.load(str(chunk_emb_cache))
        print(f"    Loaded chunk embeddings from cache ({chunk_matrix.shape})")
    else:
        print(f"    Embedding {n_chunks:,} chunks ({EMBED_HF_NAME})...")
        chunk_matrix = st_embed_batch(all_chunks, is_query=False)
        np.save(str(chunk_emb_cache), chunk_matrix)

    # ── Step 3: build session → chunk indices mapping ─────────────────────────
    # Preserve insertion order so results are deterministic
    unique_sessions: list[str] = list(dict.fromkeys(all_session_keys))
    session_to_idxs: dict[str, list[int]] = {}
    for idx, sk in enumerate(all_session_keys):
        session_to_idxs.setdefault(sk, []).append(idx)

    # ── Step 4: per-query max-similarity retrieval ────────────────────────────
    m = RunMetrics(f"A4mv: Multi-Vector ({EMBED_HF_NAME}, max-sim)")
    for qe in ds.queries:
        t0 = time.monotonic()
        qvec = np.array(st_embed_one(qe.question, is_query=True), dtype=np.float32)

        # Score every chunk in one matmul
        sims = chunk_matrix @ qvec  # shape: (N_chunks,)

        # Max-pool per session: each session's score = highest chunk score
        session_scores: list[tuple[float, str]] = [
            (float(sims[idxs].max()), sk)
            for sk, idxs in session_to_idxs.items()
        ]
        session_scores.sort(key=lambda x: x[0], reverse=True)
        retrieved = [sk for _, sk in session_scores[:10]]

        lat = (time.monotonic() - t0) * 1000
        # gold_keys and retrieved are both in gold_key namespace — no DB mapping needed
        m.add(retrieved, qe.gold_keys, lat)

    return m

def run_approach_4mvr_multivec_reranked(ds: BenchDataset) -> RunMetrics:
    """A4mvr: Multi-Vector max-sim + Cross-Encoder Reranker (Phase 3.3).

    A4mv finds the right session via chunk max-sim but loses rank-1 precision
    (MRR 0.617 vs A4r 0.650). Adding the cross-encoder on top-40 max-sim candidates
    fixes precision without any LLM extraction cost.

    Pipeline:
      1. max-sim over all chunks → ranked session list (same as A4mv)
      2. top-40 sessions → cross-encoder scores (query, full session content)
      3. return top-10 by reranker score
    """
    chunk_emb_cache = CACHE / f"embed_chunks_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"

    # ── Build chunk list (same as A4mv) ──────────────────────────────────────
    all_chunks: list[str] = []
    all_session_keys: list[str] = []
    for mem in ds.memories:
        chunks = chunk_session(mem.content)
        all_chunks.extend(chunks)
        all_session_keys.extend([mem.gold_key] * len(chunks))

    n_chunks = len(all_chunks)
    avg_per_session = n_chunks / len(ds.memories) if ds.memories else 0
    print(f"    {n_chunks:,} chunks from {len(ds.memories):,} sessions "
          f"(avg {avg_per_session:.1f} chunks/session)")

    if chunk_emb_cache.exists():
        chunk_matrix = np.load(str(chunk_emb_cache))
        print(f"    Loaded chunk embeddings from cache ({chunk_matrix.shape})")
    else:
        print(f"    Embedding {n_chunks:,} chunks ({EMBED_HF_NAME})...")
        chunk_matrix = st_embed_batch(all_chunks, is_query=False)
        np.save(str(chunk_emb_cache), chunk_matrix)

    session_to_idxs: dict[str, list[int]] = {}
    for idx, sk in enumerate(all_session_keys):
        session_to_idxs.setdefault(sk, []).append(idx)

    session_content: dict[str, str] = {mem.gold_key: mem.content for mem in ds.memories}

    from app.services.retrieval.reranker import get_default_reranker
    reranker = get_default_reranker()

    m = RunMetrics(f"A4mvr: Multi-Vector+Reranker ({EMBED_HF_NAME})")
    for qe in ds.queries:
        t0 = time.monotonic()
        qvec = np.array(st_embed_one(qe.question, is_query=True), dtype=np.float32)

        # Step 1: max-sim → top-40 candidate sessions
        sims = chunk_matrix @ qvec
        session_scores: list[tuple[float, str]] = [
            (float(sims[idxs].max()), sk)
            for sk, idxs in session_to_idxs.items()
        ]
        session_scores.sort(key=lambda x: x[0], reverse=True)
        top40 = [sk for _, sk in session_scores[:40]]

        # Step 2: cross-encoder reranks (query, full session content)
        pairs = [(qe.question, session_content[sk]) for sk in top40]
        scores = reranker._model.predict(pairs)
        reranked = [sk for _, sk in sorted(zip(scores, top40), reverse=True)]

        lat = (time.monotonic() - t0) * 1000
        m.add(reranked[:10], qe.gold_keys, lat)

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

def run_approach_5r_reranked(ds: BenchDataset, llm_fn: Callable | None = None) -> RunMetrics:
    """A5r: Extracted facts + cross-encoder reranker + session-diversity MMR.
    With --multi-hyde: query is augmented via 3 hypothetical facts before retrieval.
    Without: plain GTE cosine query ($0).
    """
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    from app.services.retrieval.reranker import get_default_reranker
    from collections import defaultdict
    import uuid as _uuid
    import app.models as _models

    facts_cache = CACHE / f"facts_claude_{ds.name.replace(' ', '_')}.json"
    emb_cache   = CACHE / f"embed_facts_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"

    # ── Step 1: load facts (reuse A5 cache) ──────────────────────────────────
    session_facts = extract_session_facts(ds.memories, facts_cache, max_workers=8)
    if not session_facts:
        return RunMetrics("A5r: Extracted Facts+Reranker (unavailable)")

    # ── Step 2: build flat fact list ─────────────────────────────────────────
    fact_mems: list[MemEntry] = []
    for mem in ds.memories:
        facts = session_facts.get(mem.gold_key) or [mem.content[:200]]
        for i, fact_text in enumerate(facts):
            fact_mems.append(MemEntry(
                mid=f"{mem.gold_key}__f{i}",
                title=f"{mem.title[:80]} [f{i+1}]",
                content=fact_text,
                search_text=fact_text,
                gold_key=mem.gold_key,
            ))
    print(f"    {len(fact_mems):,} facts from {len(ds.memories):,} sessions")

    # ── Step 3: load embeddings (reuse A5 cache) ─────────────────────────────
    if emb_cache.exists():
        fact_embs = np.load(str(emb_cache))
        print(f"    Loaded fact embeddings from cache ({fact_embs.shape})")
    else:
        print(f"    Embedding {len(fact_mems):,} facts ({EMBED_HF_NAME})...")
        fact_embs = st_embed_batch([m.content for m in fact_mems], is_query=False)
        np.save(str(emb_cache), fact_embs)

    # ── Step 4: build in-memory DB ───────────────────────────────────────────
    engine, db = fresh_db()
    proj = crud.create_project(db, schemas.ProjectCreate(
        name="bench_a5r", description="", tech_stack=[], goals=[], domain="software"))
    project_id = proj.id

    session_to_fact_ids: dict[str, list[str]] = defaultdict(list)
    fact_id_to_session: dict[str, str] = {}
    fact_id_to_content: dict[str, str] = {}
    fact_id_to_title:   dict[str, str] = {}

    for i, fm in enumerate(fact_mems):
        mem_id = str(_uuid.uuid4())
        session_to_fact_ids[fm.gold_key].append(mem_id)
        fact_id_to_session[mem_id]  = fm.gold_key
        fact_id_to_content[mem_id]  = fm.content
        fact_id_to_title[mem_id]    = fm.title
        mem_obj = _models.Memory(
            id=mem_id, project_id=project_id,
            type="insight", title=fm.title[:200],
            content=fm.content, search_text=fm.search_text,
            importance=3, confidence=0.9,
            privacy_level="internal", review_status="approved", status="active",
            embedding=fact_embs[i].tobytes(),
            embedding_model=EMBED_HF_NAME,
        )
        db.add(mem_obj)
        if (i + 1) % 500 == 0:
            db.commit()
    db.commit()

    vb = SQLiteExactBackend(db)
    allowed, _ = __import__(
        'app.services.retrieval.candidate_generators', fromlist=['apply_hard_filters']
    ).apply_hard_filters(db=db, project_id=project_id,
                         max_clearance="internal", include_superseded=False)

    # ── Step 5: load cross-encoder ───────────────────────────────────────────
    reranker = get_default_reranker()

    # ── Step 6: per-query retrieval + rerank + session-MMR ───────────────────
    _use_mhyde = args.multi_hyde and llm_fn is not None
    _hyde_tag = "+Multi-HyDE" if _use_mhyde else ""
    m = RunMetrics(f"A5r: Facts+Reranker+Session-MMR{_hyde_tag} ({EMBED_HF_NAME})")

    for qe in ds.queries:
        t0 = time.monotonic()

        # Dense retrieval: top-20 candidate facts (multi-HyDE or plain GTE)
        if _use_mhyde:
            qvec = multi_hyde_fact_augment(qe.question, llm_fn)
        else:
            qvec = st_embed_one(qe.question, is_query=True)
        top20 = vb.search(qvec, 20, project_id, allowed)
        candidate_ids = [mid for mid, _ in top20]

        # Cross-encoder rerank: build lightweight pseudo-Memory objects
        class _Mem:
            def __init__(self, id_, title_, content_):
                self.id = id_; self.title = title_; self.content = content_
        pseudo_mems = [_Mem(mid, fact_id_to_title[mid], fact_id_to_content[mid])
                       for mid in candidate_ids if mid in fact_id_to_content]
        reranked = reranker.rerank(qe.question, pseudo_mems, top_n=20)
        # reranked → [(fact_id, score)] sorted descending

        # Session-diversity MMR: max 2 facts per session, pick top-10
        # Cap=2 optimal: prevents session saturation; higher caps don't help
        selected: list[str] = []
        session_count: dict[str, int] = defaultdict(int)
        for fact_id, _ in reranked:
            if len(selected) >= 10:
                break
            sess = fact_id_to_session.get(fact_id, "")
            if session_count[sess] < 2:
                selected.append(fact_id)
                session_count[sess] += 1

        lat = (time.monotonic() - t0) * 1000
        gold_db: list[str] = []
        for gk in qe.gold_keys:
            gold_db.extend(session_to_fact_ids.get(gk, []))
        m.add(selected, gold_db, lat)

    db.close()
    return m

def run_approach_6_multiqueryRRF(ds: BenchDataset) -> RunMetrics | None:
    """A6: Multi-Query Retrieval + RRF Fusion (Phase 3.1).

    3 independent searches per query (original + 2 LLM rephrases) over the extracted
    fact corpus, merged via Reciprocal Rank Fusion (k=60), then session-diversity MMR
    (max 2 facts per session) → top-10 sessions.

    Contrast with Multi-HyDE (2.9c, -0.009 regression): this approach runs N independent
    searches and merges RESULTS — it never averages query vectors. Each search is sharp;
    RRF rewards facts that rank well across multiple phrasings.

    Reuses A5/A5r fact corpus (same cache files). Only new cost: query rephrasing (~$0.01
    per 200-query dataset, cached after first run → $0 on subsequent runs).
    """
    from collections import defaultdict

    facts_cache    = CACHE / f"facts_claude_{ds.name.replace(' ', '_')}.json"
    emb_cache      = CACHE / f"embed_facts_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"
    rephrase_cache = CACHE / f"mqr_rephrases_{ds.name.replace(' ', '_')}.json"

    # ── Step 1: load facts (reuse A5 cache) ──────────────────────────────────
    session_facts = extract_session_facts(ds.memories, facts_cache, max_workers=8)
    if not session_facts:
        return None

    # ── Step 2: build flat fact list ─────────────────────────────────────────
    fact_mems: list[MemEntry] = []
    for mem in ds.memories:
        facts = session_facts.get(mem.gold_key) or [mem.content[:200]]
        for i, fact_text in enumerate(facts):
            fact_mems.append(MemEntry(
                mid=f"{mem.gold_key}__f{i}",
                title=f"{mem.title[:80]} [f{i+1}]",
                content=fact_text,
                search_text=fact_text,
                gold_key=mem.gold_key,
            ))
    print(f"    {len(fact_mems):,} facts from {len(ds.memories):,} sessions")

    # ── Step 3: load fact embeddings (reuse A5 cache) ─────────────────────────
    if emb_cache.exists():
        fact_embs = np.load(str(emb_cache))
        print(f"    Loaded fact embeddings from cache ({fact_embs.shape})")
    else:
        print(f"    Embedding {len(fact_mems):,} facts ({EMBED_HF_NAME})...")
        fact_embs = st_embed_batch([m.content for m in fact_mems], is_query=False)
        np.save(str(emb_cache), fact_embs)

    # Parallel array: which session does each fact row belong to
    fact_session_keys = [fm.gold_key for fm in fact_mems]

    # ── Step 4: load / generate query rephrases ───────────────────────────────
    all_questions = [qe.question for qe in ds.queries]
    rephrases = generate_query_rephrases(all_questions, rephrase_cache, max_workers=8)
    if not rephrases:
        print("    [skip] No ANTHROPIC_API_KEY and no cached rephrases — A6 requires rephrases")
        return None
    cached_note = "(from cache)" if rephrase_cache.exists() else "(freshly generated)"
    print(f"    Query rephrases ready {cached_note}")

    # ── Step 5: per-query multi-search + RRF + session-MMR ───────────────────
    TOP_PER_SEARCH = 20

    m = RunMetrics(f"A6: Multi-Query RRF ({EMBED_HF_NAME}, 3 searches/query)")
    for qe in ds.queries:
        t0 = time.monotonic()

        # 3 query variants: original + 2 rephrases
        variants = [qe.question] + rephrases.get(qe.question, [qe.question, qe.question])
        variants = variants[:3]

        # Independent cosine search for each variant → ranked list of fact row indices
        ranked_lists: list[list[int]] = []
        for variant in variants:
            qvec = np.array(st_embed_one(variant, is_query=True), dtype=np.float32)
            sims  = fact_embs @ qvec                                          # (N_facts,)
            n     = min(TOP_PER_SEARCH, len(sims))
            top   = np.argpartition(sims, -n)[-n:]
            top   = top[np.argsort(sims[top])[::-1]]
            ranked_lists.append(top.tolist())

        # RRF merge → [(fact_row_idx, rrf_score)] descending
        merged = rrf_merge(ranked_lists, k=60)

        # Session-diversity MMR: max 2 facts per session, collect top-10 unique sessions
        selected_sessions: list[str] = []
        seen_sessions: set[str]      = set()
        session_count: dict[str, int] = defaultdict(int)
        for fact_idx, _ in merged:
            if len(selected_sessions) >= 10:
                break
            sess = fact_session_keys[fact_idx]
            if session_count[sess] < 2:
                session_count[sess] += 1
                if sess not in seen_sessions:
                    selected_sessions.append(sess)
                    seen_sessions.add(sess)

        lat = (time.monotonic() - t0) * 1000
        # Both selected_sessions and qe.gold_keys are in gold_key namespace — no DB mapping needed
        m.add(selected_sessions, qe.gold_keys, lat)

    return m

def run_approach_6r_multiqueryRRF_reranked(ds: BenchDataset) -> RunMetrics | None:
    """A6r: Multi-Query RRF + Cross-Encoder Reranker + Session-MMR (Phase 3.1).

    A6's multi-query RRF for recall + A5r's cross-encoder reranker for precision.
    Pipeline: 3 rephrased searches → RRF merge (top-40 pool) → cross-encoder rerank
    → session-diversity MMR (max 2/session) → top-10 sessions.

    Key diagnostic: does the ms-marco reranker help or hurt when given A6's wider
    multi-query candidate pool? A5r's reranker hurt LoCoMo standalone (0.595 vs A5 0.615).
    A6 hits 0.635 without reranker. A6r reveals whether more diverse candidates change
    the reranker's behaviour on conversational data.
    """
    from collections import defaultdict
    from app.services.retrieval.reranker import get_default_reranker

    facts_cache    = CACHE / f"facts_claude_{ds.name.replace(' ', '_')}.json"
    emb_cache      = CACHE / f"embed_facts_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"
    rephrase_cache = CACHE / f"mqr_rephrases_{ds.name.replace(' ', '_')}.json"

    # ── Steps 1-3: identical to A6 (all from cache) ───────────────────────────
    session_facts = extract_session_facts(ds.memories, facts_cache, max_workers=8)
    if not session_facts:
        return None

    fact_mems: list[MemEntry] = []
    for mem in ds.memories:
        facts = session_facts.get(mem.gold_key) or [mem.content[:200]]
        for i, fact_text in enumerate(facts):
            fact_mems.append(MemEntry(
                mid=f"{mem.gold_key}__f{i}",
                title=f"{mem.title[:80]} [f{i+1}]",
                content=fact_text,
                search_text=fact_text,
                gold_key=mem.gold_key,
            ))
    print(f"    {len(fact_mems):,} facts from {len(ds.memories):,} sessions")

    if emb_cache.exists():
        fact_embs = np.load(str(emb_cache))
        print(f"    Loaded fact embeddings from cache ({fact_embs.shape})")
    else:
        print(f"    Embedding {len(fact_mems):,} facts ({EMBED_HF_NAME})...")
        fact_embs = st_embed_batch([m.content for m in fact_mems], is_query=False)
        np.save(str(emb_cache), fact_embs)

    fact_session_keys = [fm.gold_key for fm in fact_mems]

    rephrases = generate_query_rephrases(
        [qe.question for qe in ds.queries], rephrase_cache, max_workers=8
    )
    if not rephrases:
        print("    [skip] No rephrases available — A6r requires query rephrases")
        return None
    print(f"    Query rephrases ready (from cache)")

    # ── Step 4: load cross-encoder ────────────────────────────────────────────
    reranker = get_default_reranker()

    # ── Step 5: per-query multi-search + RRF + rerank + session-MMR ──────────
    TOP_PER_SEARCH  = 20
    RRF_RERANK_POOL = 40   # wider pool than A5r (20) — 3 searches yield more diverse candidates

    class _Mem:
        def __init__(self, id_, title_, content_):
            self.id = id_; self.title = title_; self.content = content_

    m = RunMetrics(f"A6r: Multi-Query RRF + Reranker ({EMBED_HF_NAME})")
    for qe in ds.queries:
        t0 = time.monotonic()

        variants = [qe.question] + rephrases.get(qe.question, [qe.question, qe.question])
        variants = variants[:3]

        # 3 independent searches → per-variant ranked lists of fact row indices
        ranked_lists: list[list[int]] = []
        for variant in variants:
            qvec = np.array(st_embed_one(variant, is_query=True), dtype=np.float32)
            sims  = fact_embs @ qvec
            n     = min(TOP_PER_SEARCH, len(sims))
            top   = np.argpartition(sims, -n)[-n:]
            top   = top[np.argsort(sims[top])[::-1]]
            ranked_lists.append(top.tolist())

        # RRF merge → take top-40 as reranker input pool
        merged   = rrf_merge(ranked_lists, k=60)
        top_pool = [idx for idx, _ in merged[:RRF_RERANK_POOL]]

        # Cross-encoder rerank: score (query, fact_content) pairs
        pseudo_mems = [_Mem(str(idx), fact_mems[idx].title, fact_mems[idx].content)
                       for idx in top_pool]
        reranked = reranker.rerank(qe.question, pseudo_mems, top_n=RRF_RERANK_POOL)

        # Session-diversity MMR: max 2 facts per session, collect top-10 unique sessions
        selected_sessions: list[str] = []
        seen_sessions:     set[str]  = set()
        session_count: dict[str, int] = defaultdict(int)
        for str_idx, _ in reranked:
            if len(selected_sessions) >= 10:
                break
            sess = fact_session_keys[int(str_idx)]
            if session_count[sess] < 2:
                session_count[sess] += 1
                if sess not in seen_sessions:
                    selected_sessions.append(sess)
                    seen_sessions.add(sess)

        lat = (time.monotonic() - t0) * 1000
        m.add(selected_sessions, qe.gold_keys, lat)

    return m

# ── Evaluate one dataset across all approaches ────────────────────────────────
def run_approach_8r_graph_augmented(ds: BenchDataset) -> RunMetrics | None:
    """A8r: Entity knowledge graph walk + A4mvr max-sim + cross-encoder reranker (Phase 3.7).

    Pipeline:
      1. Build in-memory entity graph from corpus (one-time per dataset, O(N×E²))
      2. For each query:
         a. A4mvr: embed query → chunk max-sim per session → top-40 sessions (same as A4mvr)
         b. Graph walk: extract query entities → 2-hop BFS → candidate session IDs
         c. RRF merge: combine A4mvr scores + graph scores (k=60)
         d. Cross-encoder reranker on top-40 merged candidates
         e. Session-diversity cap → top-10

    The graph adds complementary candidates for entity-chain queries that chunk
    similarity alone misses (e.g. "what job does she have?" when job info is in
    a different session than the person's name).

    $0 cost: no API calls. Pure local NER + numpy + cross-encoder.
    """
    from collections import defaultdict
    from app.services.retrieval.reranker import get_default_reranker
    from app.services.knowledge_graph.entity_ner import extract_entities

    chunk_emb_cache = CACHE / f"embed_chunks_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"

    # ── Build chunk index (same as A4mvr) ────────────────────────────────────
    all_chunks:      list[str] = []
    all_session_keys: list[str] = []
    for mem in ds.memories:
        chunks = chunk_session(mem.content)
        all_chunks.extend(chunks)
        all_session_keys.extend([mem.gold_key] * len(chunks))

    if chunk_emb_cache.exists():
        chunk_matrix = np.load(str(chunk_emb_cache))
    else:
        print(f"    Embedding {len(all_chunks):,} chunks ({EMBED_HF_NAME})...")
        chunk_matrix = st_embed_batch(all_chunks, is_query=False)
        np.save(str(chunk_emb_cache), chunk_matrix)

    session_to_chunk_idxs: dict[str, list[int]] = defaultdict(list)
    for idx, sk in enumerate(all_session_keys):
        session_to_chunk_idxs[sk].append(idx)

    session_content = {mem.gold_key: mem.content for mem in ds.memories}

    # ── Build entity graph (in-memory, built once per dataset call) ───────────
    print(f"    Building entity graph for {len(ds.memories):,} sessions...", end="", flush=True)
    # entity_to_sessions: normalized entity label → set of session gold_keys
    entity_to_sessions: dict[str, set[str]] = defaultdict(set)
    # session_to_entities: gold_key → set of normalized entity labels
    session_to_entities: dict[str, set[str]] = defaultdict(set)

    for mem in ds.memories:
        text = mem.content or mem.search_text or ""
        entities = extract_entities(text)
        for ent in entities:
            entity_to_sessions[ent.normalized].add(mem.gold_key)
            session_to_entities[mem.gold_key].add(ent.normalized)

    total_entity_links = sum(len(v) for v in entity_to_sessions.values())
    print(f" {len(entity_to_sessions):,} entities · {total_entity_links:,} links")

    reranker = get_default_reranker()
    m = RunMetrics(f"A8r: Graph+MultiVec+Reranker ({EMBED_HF_NAME})")

    K_RRF  = 60
    TOP_N  = 40   # candidates fed to cross-encoder

    for qe in ds.queries:
        t0 = time.monotonic()

        # ── A4mvr leg: chunk max-sim per session ──────────────────────────────
        qvec = np.array(st_embed_one(qe.question, is_query=True), dtype=np.float32)
        sims  = chunk_matrix @ qvec
        session_maxsim: dict[str, float] = {}
        for sk, idxs in session_to_chunk_idxs.items():
            session_maxsim[sk] = float(sims[idxs].max())
        sorted_a4mvr = sorted(session_maxsim.items(), key=lambda x: x[1], reverse=True)

        # RRF score from A4mvr (dense)
        rrf_scores: dict[str, float] = {}
        for rank, (sk, _) in enumerate(sorted_a4mvr[:TOP_N]):
            rrf_scores[sk] = rrf_scores.get(sk, 0.0) + 1.0 / (K_RRF + rank)

        # ── Graph walk leg: entity-chain candidates ───────────────────────────
        query_entities = extract_entities(qe.question)
        graph_sessions: dict[str, float] = {}  # session_id → graph relevance score

        for ent in query_entities:
            norm = ent.normalized

            # Hop 0: sessions directly containing this entity
            direct_sessions = entity_to_sessions.get(norm, set())
            for sk in direct_sessions:
                graph_sessions[sk] = max(graph_sessions.get(sk, 0.0), 1.0)

            # Hop 1: sessions that share entities with hop-0 sessions
            for h0_sk in list(direct_sessions):
                for co_ent in session_to_entities.get(h0_sk, set()):
                    if co_ent == norm:
                        continue
                    for h1_sk in entity_to_sessions.get(co_ent, set()):
                        if h1_sk not in direct_sessions:
                            existing = graph_sessions.get(h1_sk, 0.0)
                            graph_sessions[h1_sk] = max(existing, 0.5)

        # Add graph-walk RRF scores
        sorted_graph = sorted(graph_sessions.items(), key=lambda x: x[1], reverse=True)
        for rank, (sk, _) in enumerate(sorted_graph[:TOP_N]):
            rrf_scores[sk] = rrf_scores.get(sk, 0.0) + 0.5 / (K_RRF + rank)
            # Weight graph leg at 0.5× dense to avoid over-relying on entity chains

        # ── Merge + cross-encoder reranker ────────────────────────────────────
        top_sessions = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        top_sks = [sk for sk, _ in top_sessions]

        pairs  = [(qe.question, session_content[sk]) for sk in top_sks]
        scores = reranker._model.predict(pairs)
        reranked = [sk for _, sk in sorted(zip(scores, top_sks), reverse=True)]

        lat = (time.monotonic() - t0) * 1000
        m.add(reranked[:10], qe.gold_keys, lat)

    return m


def run_approach_7r_bm25_dense_reranked(ds: BenchDataset) -> RunMetrics | None:
    """A7r: BM25+Dense weighted RRF + cross-encoder reranker (Phase 3.5).

    Weight learning: grid search α ∈ {0.0, 0.1, ..., 1.0} on 80% of eval queries.
    α=1.0 → pure BM25, α=0.0 → pure Dense, α=0.5 → equal blend.
    Full evaluation runs with optimal α + cross-encoder reranker.

    Uses same Claude-extracted fact corpus as A5r (must be cached).
    Bypasses SQLite; uses numpy for fast retrieval.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("    [skip] rank_bm25 not installed. Run: pip install rank-bm25")
        return None

    from collections import defaultdict
    from app.services.retrieval.reranker import get_default_reranker

    facts_cache = CACHE / f"facts_claude_{ds.name.replace(' ', '_')}.json"
    emb_cache   = CACHE / f"embed_facts_{EMBED_TAG}_{ds.name.replace(' ', '_')}.npy"

    if not facts_cache.exists():
        return RunMetrics("A7r: BM25+Dense+Reranker (no fact cache — run A5 first)")
    if not emb_cache.exists():
        return RunMetrics("A7r: BM25+Dense+Reranker (no embed cache — run A5 first)")

    # ── Load fact corpus ───────────────────────────────────────────────────────
    session_facts = json.loads(facts_cache.read_text())

    fact_texts:    list[str] = []
    fact_sessions: list[str] = []
    session_to_fact_idxs: dict[str, list[int]] = defaultdict(list)

    for mem in ds.memories:
        sid   = mem.gold_key
        facts = session_facts.get(sid) or [mem.content[:200]]
        for fact in facts:
            idx = len(fact_texts)
            fact_texts.append(fact)
            fact_sessions.append(sid)
            session_to_fact_idxs[sid].append(idx)

    fact_embs = np.load(str(emb_cache)).astype(np.float32)  # [N, D]
    print(f"    {len(fact_texts):,} facts · embeddings {fact_embs.shape}")

    # ── Build BM25 index ───────────────────────────────────────────────────────
    bm25 = BM25Okapi([t.lower().split() for t in fact_texts])

    reranker = get_default_reranker()

    # ── Pre-compute BM25 top-40 and dense top-40 for all eval queries ─────────
    # (avoids redundant embedding calls during grid search)
    all_bm25_top40:  list[np.ndarray] = []
    all_dense_top40: list[np.ndarray] = []

    print(f"    Pre-computing BM25 + dense scores for {len(ds.queries)} queries...", flush=True)
    all_qvecs = np.array(
        [st_embed_one(qe.question, is_query=True) for qe in ds.queries],
        dtype=np.float32,
    )  # [Q, D]
    dense_all = all_qvecs @ fact_embs.T  # [Q, N]

    for qi, qe in enumerate(ds.queries):
        bm25_scores = np.array(bm25.get_scores(qe.question.lower().split()), dtype=np.float32)
        all_bm25_top40.append(np.argsort(bm25_scores)[::-1][:40])
        all_dense_top40.append(np.argsort(dense_all[qi])[::-1][:40])

    # ── Weighted RRF helper ────────────────────────────────────────────────────
    def weighted_rrf_top40(qi: int, alpha: float) -> list[int]:
        k = 60
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(all_bm25_top40[qi]):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + alpha / (k + rank)
        for rank, idx in enumerate(all_dense_top40[qi]):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + (1.0 - alpha) / (k + rank)
        return sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:40]

    def top_sessions(ranked_idxs: list[int], n: int = 5) -> list[str]:
        sessions: list[str] = []
        seen: set[str] = set()
        for idx in ranked_idxs:
            s = fact_sessions[idx]
            if s not in seen:
                sessions.append(s)
                seen.add(s)
            if len(sessions) >= n:
                break
        return sessions

    # ── Grid search: find best α on 80% of queries (no reranker for speed) ────
    n_train = max(1, int(len(ds.queries) * 0.8))
    print(f"    Grid search α on {n_train} train queries:", end="", flush=True)

    best_alpha, best_r5 = 0.5, -1.0
    for alpha_10 in range(11):
        alpha = alpha_10 / 10.0
        hits  = 0
        for qi, qe in enumerate(ds.queries[:n_train]):
            top5 = top_sessions(weighted_rrf_top40(qi, alpha), n=5)
            if any(s in set(qe.gold_keys) for s in top5):
                hits += 1
        r5 = hits / n_train
        print(f" {alpha:.1f}→{r5:.3f}", end="", flush=True)
        if r5 > best_r5:
            best_r5, best_alpha = r5, alpha
    print(f"\n    Learned α={best_alpha:.1f}  (train R@5={best_r5:.3f})", flush=True)

    # ── Full evaluation: best α + cross-encoder reranker + session MMR ────────
    m = RunMetrics(f"A7r: BM25+Dense α={best_alpha:.1f}+Reranker (Phase 3.5)")

    for qi, qe in enumerate(ds.queries):
        t0 = time.monotonic()

        top40_idxs = weighted_rrf_top40(qi, best_alpha)

        # Cross-encoder reranker on (query, fact) pairs
        pairs  = [(qe.question, fact_texts[i]) for i in top40_idxs]
        scores = reranker._model.predict(pairs)
        reranked = [idx for _, idx in sorted(zip(scores, top40_idxs), reverse=True)]

        # Session-diversity MMR: max 2 facts/session → top-10 unique sessions
        selected: list[str] = []
        sess_count: dict[str, int] = defaultdict(int)
        for idx in reranked:
            s = fact_sessions[idx]
            if sess_count[s] < 2:
                if s not in selected:
                    selected.append(s)
                sess_count[s] += 1
            if len(selected) >= 10:
                break

        lat = (time.monotonic() - t0) * 1000
        m.add(selected, qe.gold_keys, lat)

    return m


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

    # A4r: full hybrid pipeline + cross-encoder reranker, no HyDE — $0 diagnostic
    print(f"\n  [A4r] Hybrid+Reranker (GTE dense, no HyDE)...")
    results["hybrid_reranked"] = run_approach_4r_dense_reranked(ds)
    r4r = results["hybrid_reranked"]
    print(f"    R@5={r4r.r(5):.3f}  MRR@10={r4r.mrr_mean():.3f}  p50={r4r.p(0.5):.0f}ms")

    # A4mv: ColBERT-style multi-vector, paragraph chunks, max-sim — $0
    print(f"\n  [A4mv] Multi-Vector chunks (ColBERT-style max-sim, no HyDE)...")
    results["multivec"] = run_approach_4mv_multivec(ds)
    r4mv = results["multivec"]
    print(f"    R@5={r4mv.r(5):.3f}  MRR@10={r4mv.mrr_mean():.3f}  p50={r4mv.p(0.5):.0f}ms")

    # A4mvr: multi-vector max-sim + cross-encoder reranker (Phase 3.3)
    print(f"\n  [A4mvr] Multi-Vector + Reranker (max-sim → cross-encoder, Phase 3.3)...")
    results["multivec_reranked"] = run_approach_4mvr_multivec_reranked(ds)
    r4mvr = results["multivec_reranked"]
    print(f"    R@5={r4mvr.r(5):.3f}  MRR@10={r4mvr.mrr_mean():.3f}  p50={r4mvr.p(0.5):.0f}ms")

    if not args.skip_cloud and llm_fn is not None:
        print(f"\n  [A3] Cloud LLM (Claude Haiku contextual+HyDE)...")
        results["cloud"] = run_approach_3_cloud(ds, llm_fn)
        r3 = results["cloud"]
        print(f"    R@5={r3.r(5):.3f}  MRR@10={r3.mrr_mean():.3f}  p50={r3.p(0.5):.0f}ms")

        print(f"\n  [A4] Hybrid Full pipeline + Cloud LLM + Reranker...")
        results["hybrid"] = run_approach_4_hybrid(ds, llm_fn)
        r4 = results["hybrid"]
        print(f"    R@5={r4.r(5):.3f}  MRR@10={r4.mrr_mean():.3f}  p50={r4.p(0.5):.0f}ms")
    elif args.skip_cloud:
        print("  [A3/A4] Cloud skipped (--skip-cloud)")
    else:
        print("  [A3/A4] Cloud skipped (no ANTHROPIC_API_KEY)")

    # A5 + A5r run independently of --skip-cloud (fact extraction is self-managed;
    # once facts are cached all subsequent runs are $0)
    facts_cached = (CACHE / f"facts_claude_{ds.name.replace(' ', '_')}.json").exists()
    if not args.skip_a5 and ANTHROPIC_API_KEY:
        print(f"\n  [A5] Extracted Facts (LLM facts → dense cosine, no HyDE)...")
        results["extracted"] = run_approach_5_extracted_facts(ds)
        r5 = results["extracted"]
        if r5.mrr:
            print(f"    R@5={r5.r(5):.3f}  MRR@10={r5.mrr_mean():.3f}  p50={r5.p(0.5):.0f}ms")
        else:
            print("    [skip] A5 unavailable")

        _a5r_tag = " + Multi-HyDE" if args.multi_hyde else ""
        print(f"\n  [A5r] Extracted Facts + Reranker + Session-MMR{_a5r_tag}...")
        results["extracted_reranked"] = run_approach_5r_reranked(ds, llm_fn=llm_fn)
        r5r = results["extracted_reranked"]
        if r5r.mrr:
            print(f"    R@5={r5r.r(5):.3f}  MRR@10={r5r.mrr_mean():.3f}  p50={r5r.p(0.5):.0f}ms")
        else:
            print("    [skip] A5r unavailable")
    elif args.skip_a5:
        print("  [A5/A5r] Extracted Facts skipped (--skip-a5)")
    elif not ANTHROPIC_API_KEY and not facts_cached:
        print("  [A5/A5r] Extracted Facts skipped (no API key + no cache)")

    # A6: Multi-Query RRF — runs whenever A5 can run (reuses fact cache, only adds rephrases)
    rephrase_cached = (CACHE / f"mqr_rephrases_{ds.name.replace(' ', '_')}.json").exists()
    if not args.skip_a5 and (ANTHROPIC_API_KEY or (facts_cached and rephrase_cached)):
        print(f"\n  [A6] Multi-Query RRF (3 searches/query + RRF merge)...")
        r6 = run_approach_6_multiqueryRRF(ds)
        if r6 is not None:
            results["multiqueryRRF"] = r6
            print(f"    R@5={r6.r(5):.3f}  MRR@10={r6.mrr_mean():.3f}  p50={r6.p(0.5):.0f}ms")
        else:
            print("    [skip] A6 unavailable")

    # A6r: Multi-Query RRF + Reranker — same gate as A6
    if not args.skip_a5 and (ANTHROPIC_API_KEY or (facts_cached and rephrase_cached)):
        print(f"\n  [A6r] Multi-Query RRF + Reranker (3 searches + cross-encoder)...")
        r6r = run_approach_6r_multiqueryRRF_reranked(ds)
        if r6r is not None:
            results["multiqueryRRF_reranked"] = r6r
            print(f"    R@5={r6r.r(5):.3f}  MRR@10={r6r.mrr_mean():.3f}  p50={r6r.p(0.5):.0f}ms")
        else:
            print("    [skip] A6r unavailable")

    # A7r: BM25+Dense weighted RRF + reranker (Phase 3.5) — requires fact cache
    if facts_cached:
        print(f"\n  [A7r] BM25+Dense RRF (learned weight) + Reranker (Phase 3.5)...")
        r7r = run_approach_7r_bm25_dense_reranked(ds)
        if r7r is not None:
            results["bm25_dense_reranked"] = r7r
            print(f"    R@5={r7r.r(5):.3f}  MRR@10={r7r.mrr_mean():.3f}  p50={r7r.p(0.5):.0f}ms")
        else:
            print("    [skip] A7r unavailable (install rank-bm25)")
    else:
        print("  [A7r] BM25+Dense+Reranker skipped (no fact cache — run A5 first)")

    # A8r: Graph-Augmented Multi-Vector + Reranker (Phase 3.7) — always runs ($0)
    print(f"\n  [A8r] Graph-Augmented A4mvr (entity graph walk + reranker, Phase 3.7)...")
    r8r = run_approach_8r_graph_augmented(ds)
    if r8r is not None:
        results["graph_augmented"] = r8r
        print(f"    R@5={r8r.r(5):.3f}  MRR@10={r8r.mrr_mean():.3f}  p50={r8r.p(0.5):.0f}ms")
    else:
        print("    [skip] A8r unavailable")

    return results

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Initialize reranker singleton with CLI model before any approach runs
    from app.services.retrieval.reranker import get_default_reranker as _init_reranker
    _init_reranker(args.reranker_model)

    print("\n" + "="*65)
    _mhyde_note = " + Multi-HyDE (2.9c)" if args.multi_hyde else ""
    _reranker_note = f" reranker={Path(args.reranker_model).name}" if args.reranker_model != "cross-encoder/ms-marco-MiniLM-L-6-v2" else ""
    print(f"X-MemoryArch: 11-Approach × 3-Dataset Benchmark{_mhyde_note}{_reranker_note}")
    _backend_note = f" [{EMBED_BACKEND.upper()}]" if EMBED_BACKEND != "st" else " [local]"
    print(f"Embed model: {EMBED_HF_NAME}{_backend_note}")
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
        ("rule_based",          "A1: Rule-based (BM25+Entity)"),
        ("ollama",              "A2: Local LLM  (Ollama + HyDE)"),
        ("cloud",               "A3: Cloud LLM  (Claude Haiku ctx+HyDE)"),
        ("hybrid",              "A4: Hybrid Full (BM25+Dense+Reranker+LLM)"),
        ("hybrid_reranked",     "A4r: Hybrid+Reranker (GTE, no HyDE)"),
        ("multivec",            "A4mv: Multi-Vector (ColBERT max-sim)"),
        ("multivec_reranked",   "A4mvr: Multi-Vector+Reranker (Phase 3.3)"),
        ("extracted",           "A5: Extracted Facts (GTE, no HyDE)"),
        ("extracted_reranked",  "A5r: Facts+Reranker+Session-MMR (GTE)"),
        ("multiqueryRRF",           "A6: Multi-Query RRF (3 searches, Phase 3.1)"),
        ("multiqueryRRF_reranked",  "A6r: Multi-Query RRF + Reranker (Phase 3.1)"),
        ("bm25_dense_reranked",     "A7r: BM25+Dense RRF+Reranker (Phase 3.5)"),
        ("graph_augmented",         "A8r: Graph+MultiVec+Reranker (Phase 3.7)"),
    ]

    print("\n\n" + "="*90)
    print("FINAL RESULTS — X-MemoryArch 10 Approaches × 3 Datasets")
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
