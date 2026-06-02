"""
Real-corpus benchmark — SQuAD v1.1 validation set.

2,067 unique passage memories · sampled 1,000 queries (or pass --all for full 10,570)

Measures 4 retrieval approaches:
  [1] Rule-based   — BM25 (FTS5) + entity leg only, no ML
  [2] Dense        — pure vector search (sentence-transformers all-MiniLM-L6-v2)
  [3] Hybrid RRF   — BM25 + Dense fused with RRF (no extra pipeline stages)
  [4] Full pipeline— Hybrid + Graph expansion + Weighted ranking + MMR

Usage:
  cd RetrievalEngine
  python3 benchmark_real_corpus.py              # 1,000 queries
  python3 benchmark_real_corpus.py --all        # all 10,570 queries (slow)
  python3 benchmark_real_corpus.py --k 1 3 5 10 # custom Recall@k values
"""
from __future__ import annotations

import argparse
import math
import random
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Path bootstrap (mirrors conftest.py) ─────────────────────────────────────
_RE_ROOT = Path(__file__).resolve().parent
_P1_ROOT = _RE_ROOT.parent / "project-memory-core"
if str(_RE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RE_ROOT))
import app as _p2_app
_p1_dir = str(_P1_ROOT / "app")
if _p1_dir not in list(_p2_app.__path__):
    _p2_app.__path__.append(_p1_dir)
import app.services as _p2_svc
_p1_svc = str(_P1_ROOT / "app" / "services")
if _p1_svc not in list(_p2_svc.__path__):
    _p2_svc.__path__.append(_p1_svc)

# ── DB ────────────────────────────────────────────────────────────────────────
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database import Base
import app.models as models
import app.p2_models          # noqa — registers RetrievalRun
import app.crud as crud
import app.schemas as schemas

engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
Base.metadata.create_all(bind=engine)
DB = sessionmaker(bind=engine)

# ── Metric helpers ────────────────────────────────────────────────────────────
def recall_at_k(retrieved: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in retrieved[:k] else 0.0

def mrr_at_k(retrieved: list[str], gold_id: str, k: int) -> float:
    for rank, mid in enumerate(retrieved[:k], start=1):
        if mid == gold_id:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list[str], gold_id: str, k: int) -> float:
    for rank, mid in enumerate(retrieved[:k], start=1):
        if mid == gold_id:
            return 1.0 / math.log2(rank + 1)
    ideal = 1.0 / math.log2(2)   # rank-1 hit
    return 0.0 / ideal if rank > k else 0.0  # not reached

def ndcg_at_k_correct(retrieved: list[str], gold_id: str, k: int) -> float:
    dcg = 0.0
    for rank, mid in enumerate(retrieved[:k], start=1):
        if mid == gold_id:
            dcg = 1.0 / math.log2(rank + 1)
            break
    idcg = 1.0 / math.log2(2)   # ideal: hit at rank 1
    return dcg / idcg

@dataclass
class RunMetrics:
    name: str
    recall: dict[int, list[float]] = field(default_factory=dict)
    mrr: list[float] = field(default_factory=list)
    ndcg: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    def add(self, retrieved, gold_id, latency_ms, k_vals):
        for k in k_vals:
            self.recall.setdefault(k, []).append(recall_at_k(retrieved, gold_id, k))
        self.mrr.append(mrr_at_k(retrieved, gold_id, 10))
        self.ndcg.append(ndcg_at_k_correct(retrieved, gold_id, 10))
        self.latencies.append(latency_ms)

    def report(self, k_vals):
        lines = [f"\n{'─'*60}", f"  {self.name}", f"{'─'*60}"]
        for k in sorted(k_vals):
            vals = self.recall.get(k, [])
            avg = sum(vals) / len(vals) if vals else 0
            lines.append(f"  Recall@{k:<3} = {avg:.4f}  ({avg*100:.1f}%)")
        mrr  = sum(self.mrr)  / len(self.mrr)  if self.mrr  else 0
        ndcg = sum(self.ndcg) / len(self.ndcg) if self.ndcg else 0
        lines.append(f"  MRR@10   = {mrr:.4f}  ({mrr*100:.1f}%)")
        lines.append(f"  NDCG@10  = {ndcg:.4f}  ({ndcg*100:.1f}%)")
        lats = sorted(self.latencies)
        p50 = lats[len(lats)//2]
        p95 = lats[min(int(len(lats)*0.95), len(lats)-1)]
        p99 = lats[min(int(len(lats)*0.99), len(lats)-1)]
        lines.append(f"  Latency  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--all", action="store_true")
parser.add_argument("--n-queries", type=int, default=1000)
parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
K_VALS = args.k
random.seed(args.seed)

# ── Load SQuAD ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOADING SQuAD v1.1 validation set...")
print("="*60)
from datasets import load_dataset
squad = load_dataset("rajpurkar/squad", split="validation")

# Build context → questions mapping
ctx_map: dict[str, dict] = {}   # context_text → {title, questions: [{q, a}]}
for ex in squad:
    ctx = ex["context"]
    if ctx not in ctx_map:
        ctx_map[ctx] = {"title": ex["title"], "questions": []}
    ctx_map[ctx]["questions"].append({
        "q": ex["question"],
        "a": ex["answers"]["text"][0] if ex["answers"]["text"] else "",
    })

contexts = list(ctx_map.items())
print(f"  Unique memories (contexts): {len(contexts):,}")
print(f"  Total questions available:  {len(squad):,}")

# Sample questions
all_qa_pairs = []   # (question_text, context_text)
for ctx, info in contexts:
    for qa in info["questions"]:
        all_qa_pairs.append((qa["q"], ctx))

if args.all:
    sampled_qa = all_qa_pairs
else:
    n = min(args.n_queries, len(all_qa_pairs))
    sampled_qa = random.sample(all_qa_pairs, n)

print(f"  Queries to evaluate:       {len(sampled_qa):,}")

# ── Create DB project + memories ─────────────────────────────────────────────
print("\n" + "="*60)
print("INGESTING MEMORIES INTO DB...")
print("="*60)
db = DB()

proj = crud.create_project(db, schemas.ProjectCreate(
    name="SQuAD Benchmark",
    description="SQuAD v1.1 passage corpus for retrieval benchmarking",
    tech_stack=[],
    goals=["benchmark"],
    domain="general",
))
project_id = proj.id

# Set up FTS before inserting memories
try:
    from app.search import setup_fts
    setup_fts(engine)
    print("  FTS5 BM25 index ready")
except Exception as e:
    print(f"  [warn] FTS setup failed: {e}")

# Create memories (batch insert)
ctx_to_mid: dict[str, str] = {}   # context_text → memory_id
t_ingest = time.monotonic()
for i, (ctx_text, info) in enumerate(contexts):
    title = info["title"].replace("_", " ")
    mem = models.Memory(
        id=str(__import__("uuid").uuid4()),
        project_id=project_id,
        type="insight",
        title=f"{title} ({i+1})",
        content=ctx_text,
        search_text=f"{title} {ctx_text}",
        importance=3,
        confidence=0.9,
        privacy_level="internal",
        review_status="approved",
        source_quote=ctx_text[:200],
        status="active",
    )
    db.add(mem)
    ctx_to_mid[ctx_text] = mem.id
    if (i+1) % 500 == 0:
        db.commit()
        print(f"  Inserted {i+1:,}/{len(contexts):,} memories...")

db.commit()
ingest_ms = (time.monotonic() - t_ingest) * 1000
print(f"  Done: {len(contexts):,} memories in {ingest_ms:.0f}ms")

# ── Embed memories with sentence-transformers ─────────────────────────────────
print("\n" + "="*60)
print("EMBEDDING MEMORIES (all-MiniLM-L6-v2)...")
print("="*60)
from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"  Model loaded. Embedding {len(contexts):,} passages...")

t_embed = time.monotonic()
texts = [ctx for ctx, _ in contexts]
batch_size = 256
all_embeddings = []
for start in range(0, len(texts), batch_size):
    batch = texts[start:start+batch_size]
    embs = st_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
    all_embeddings.append(embs)
    if (start + batch_size) % 500 < batch_size:
        pct = min(start + batch_size, len(texts))
        print(f"  Embedded {pct:,}/{len(texts):,}...")

all_embeddings = np.vstack(all_embeddings).astype(np.float32)
embed_ms = (time.monotonic() - t_embed) * 1000
print(f"  Done: {len(all_embeddings):,} embeddings in {embed_ms:.0f}ms  ({embed_ms/len(texts):.1f}ms/mem)")

# Store embeddings in DB memories
mids = [ctx_to_mid[ctx] for ctx, _ in contexts]
mem_rows = db.query(models.Memory).filter(models.Memory.project_id == project_id).all()
mid_to_obj = {m.id: m for m in mem_rows}
for mid, emb in zip(mids, all_embeddings):
    if mid in mid_to_obj:
        mid_to_obj[mid].embedding = emb.tobytes()
        mid_to_obj[mid].embedding_model = "all-MiniLM-L6-v2"
db.commit()
print(f"  Embeddings stored in DB")

# Build SQLiteExactBackend index
from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
vb = SQLiteExactBackend(db)
allowed_ids = list(ctx_to_mid.values())

# ── Helper: embed a query ─────────────────────────────────────────────────────
def embed_query_text(query: str) -> list[float]:
    vec = st_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    return vec.tolist()

# ── APPROACH 1: Rule-based (BM25 + Entity only) ───────────────────────────────
print("\n" + "="*60)
print("APPROACH 1: Rule-based (BM25 FTS5 + Entity leg)")
print("="*60)

from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve

m1 = RunMetrics("Rule-based: BM25 + Entity (no ML)")

# Warm-up
retrieve(db=db, project_id=project_id, query="test", vector_backend=vb,
         config=RetrievalConfig(top_k=10, embed_query=False,
                                enable_weighted_ranking=False,
                                enable_mmr=False, enable_entity_boost=False,
                                enable_graph_expansion=False))

cfg_bm25 = RetrievalConfig(
    top_k=10, embed_query=False,
    enable_weighted_ranking=False,
    enable_reranker=False,
    enable_mmr=False,
    enable_entity_boost=False,
    enable_graph_expansion=False,
)

for i, (question, ctx_text) in enumerate(sampled_qa):
    gold_id = ctx_to_mid[ctx_text]
    t0 = time.monotonic()
    result = retrieve(db=db, project_id=project_id, query=question, vector_backend=vb, config=cfg_bm25)
    lat = (time.monotonic() - t0) * 1000
    m1.add(result.selected_memory_ids, gold_id, lat, K_VALS)
    if (i+1) % 200 == 0:
        r5 = sum(m1.recall[5]) / len(m1.recall[5])
        print(f"  [{i+1}/{len(sampled_qa)}] running Recall@5={r5:.3f}")

print(m1.report(K_VALS))

# ── APPROACH 1b: BM25 with OR matching (fairer standalone BM25) ──────────────
print("\n" + "="*60)
print("APPROACH 1b: Rule-based BM25 with OR matching (term-level scoring)")
print("="*60)
print("  (FTS5 default is AND — this variant joins query terms with OR for recall)")

from app.services.retrieval.candidate_generators import (
    apply_hard_filters, generate_bm25_candidates, generate_dense_candidates,
    generate_entity_candidates,
)

def bm25_or_candidates(db, project_id, query: str, allowed_ids: list, top_k=50):
    """BM25 with OR-joined terms for better recall on semantic queries."""
    import re
    _FTS_SPECIAL = re.compile(r'[^\w\s]')
    cleaned = _FTS_SPECIAL.sub(' ', query).strip()
    terms = [t for t in cleaned.split() if len(t) > 2]   # skip stopword-length terms
    if not terms:
        return []
    fts_q = " OR ".join(terms)
    try:
        from sqlalchemy import text
        rows = db.execute(text("""
            SELECT memories_fts.memory_id, bm25(memories_fts) AS bm25_score
            FROM memories_fts
            JOIN memories m ON m.id = memories_fts.memory_id
            WHERE memories_fts MATCH :q
              AND m.project_id = :pid
            LIMIT 200
        """), {"q": fts_q, "pid": project_id}).fetchall()
    except Exception:
        return []
    allowed_set = set(allowed_ids)
    scored = [(r[0], -r[1]) for r in rows if r[0] in allowed_set]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

m1b = RunMetrics("Rule-based: BM25 OR-terms (no ML)")

allowed_bm25, _ = apply_hard_filters(db=db, project_id=project_id,
                                      max_clearance="internal", include_superseded=False)
# Warm-up
bm25_or_candidates(db, project_id, "test warm", allowed_bm25)

for i, (question, ctx_text) in enumerate(sampled_qa):
    gold_id = ctx_to_mid[ctx_text]
    t0 = time.monotonic()
    candidates = bm25_or_candidates(db, project_id, question, allowed_bm25, top_k=10)
    lat = (time.monotonic() - t0) * 1000
    retrieved = [mid for mid, _ in candidates]
    m1b.add(retrieved, gold_id, lat, K_VALS)
    if (i+1) % 200 == 0:
        r5 = sum(m1b.recall[5]) / len(m1b.recall[5])
        print(f"  [{i+1}/{len(sampled_qa)}] running Recall@5={r5:.3f}")

print(m1b.report(K_VALS))

# ── APPROACH 2: Dense-only (pure vector search) ───────────────────────────────
print("\n" + "="*60)
print("APPROACH 2: Dense-only (all-MiniLM-L6-v2 cosine similarity)")
print("="*60)

m2 = RunMetrics("Dense-only: all-MiniLM-L6-v2")

# Warm-up  — correct param order: (query_vector, top_k, project_id, allowed_ids)
_ = vb.search(embed_query_text("test"), 10, project_id, allowed_ids)

for i, (question, ctx_text) in enumerate(sampled_qa):
    gold_id = ctx_to_mid[ctx_text]
    t0 = time.monotonic()
    qvec = embed_query_text(question)
    results = vb.search(qvec, 10, project_id, allowed_ids)   # fixed order
    lat = (time.monotonic() - t0) * 1000
    retrieved = [mid for mid, _ in results]
    m2.add(retrieved, gold_id, lat, K_VALS)
    if (i+1) % 200 == 0:
        r5 = sum(m2.recall[5]) / len(m2.recall[5])
        print(f"  [{i+1}/{len(sampled_qa)}] running Recall@5={r5:.3f}")

print(m2.report(K_VALS))

# ── APPROACH 3: Hybrid RRF (BM25 + Dense, RRF fusion only) ───────────────────
print("\n" + "="*60)
print("APPROACH 3: Hybrid RRF (BM25 + Dense, no downstream pipeline)")
print("="*60)

from app.services.retrieval.fusion import rrf_fuse

m3 = RunMetrics("Hybrid RRF: BM25 + Dense (k=60)")

for i, (question, ctx_text) in enumerate(sampled_qa):
    gold_id = ctx_to_mid[ctx_text]
    t0 = time.monotonic()

    # Hard filters
    allowed, _ = apply_hard_filters(db=db, project_id=project_id,
                                     max_clearance="internal", include_superseded=False)

    # Embed query
    qvec = embed_query_text(question)

    # BM25 candidates
    bm25 = generate_bm25_candidates(db=db, project_id=project_id, query=question,
                                     allowed_ids=allowed, top_k=50)
    # Dense candidates
    dense = generate_dense_candidates(vector_backend=vb, query_vector=qvec,
                                       allowed_ids=allowed, top_k=50, project_id=project_id)
    # RRF fusion
    fused = rrf_fuse(
        ranked_lists=[[mid for mid, _ in bm25], [mid for mid, _ in dense]],
        k=60,
    )
    retrieved = [mid for mid, _ in fused[:10]]
    lat = (time.monotonic() - t0) * 1000
    m3.add(retrieved, gold_id, lat, K_VALS)
    if (i+1) % 200 == 0:
        r5 = sum(m3.recall[5]) / len(m3.recall[5])
        print(f"  [{i+1}/{len(sampled_qa)}] running Recall@5={r5:.3f}")

print(m3.report(K_VALS))

# ── APPROACH 4: Full pipeline (BM25 + Dense + Entity + RRF + Ranking + MMR) ──
print("\n" + "="*60)
print("APPROACH 4: Full pipeline (all phases: 2.1→2.5)")
print("="*60)

m4 = RunMetrics("Full pipeline: BM25+Dense+Entity+RRF+Graph+Ranking+MMR")

# Monkey-patch embed_text to use our already-loaded sentence-transformers model
import app.services.semantic_classifier as sc_module
_orig_embed = None
def _st_embed(text: str) -> bytes:
    vec = st_model.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return vec.tobytes()

# Patch the embed_text used inside retrieval_service.py
import unittest.mock as mock
cfg_full = RetrievalConfig(
    top_k=10,
    embed_query=True,
    enable_weighted_ranking=True,
    enable_reranker=False,
    enable_mmr=True,
    mmr_lambda=0.70,
    enable_entity_boost=True,
    enable_graph_expansion=True,
    enable_2hop=False,   # sparse corpus — 2-hop adds noise
)

with mock.patch("app.services.semantic_classifier.embed_text", side_effect=_st_embed):
    # Warm-up
    retrieve(db=db, project_id=project_id, query="test", vector_backend=vb, config=cfg_full)

    for i, (question, ctx_text) in enumerate(sampled_qa):
        gold_id = ctx_to_mid[ctx_text]
        t0 = time.monotonic()
        result = retrieve(db=db, project_id=project_id, query=question,
                          vector_backend=vb, config=cfg_full)
        lat = (time.monotonic() - t0) * 1000
        m4.add(result.selected_memory_ids, gold_id, lat, K_VALS)
        if (i+1) % 200 == 0:
            r5 = sum(m4.recall[5]) / len(m4.recall[5])
            print(f"  [{i+1}/{len(sampled_qa)}] running Recall@5={r5:.3f}")

print(m4.report(K_VALS))

# ── FINAL COMPARISON TABLE ────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK RESULTS — REAL CORPUS")
print(f"Dataset: SQuAD v1.1 · {len(contexts):,} memories · {len(sampled_qa):,} queries")
print("="*60)

def avg(lst): return sum(lst) / len(lst) if lst else 0

rows = [m1, m1b, m2, m3, m4]
k_cols = sorted(K_VALS)

# Header
hdr = f"  {'Approach':<42}"
for k in k_cols:
    hdr += f" R@{k:<2}"
hdr += "  MRR@10  NDCG@10  p50ms  p95ms"
print(hdr)
print("  " + "-"*105)

for m in rows:
    line = f"  {m.name:<42}"
    for k in k_cols:
        r = avg(m.recall.get(k, [0]))
        line += f" {r:.3f}"
    mrr  = avg(m.mrr)
    ndcg = avg(m.ndcg)
    lats = sorted(m.latencies)
    p50  = lats[len(lats)//2]
    p95  = lats[min(int(len(lats)*0.95), len(lats)-1)]
    line += f"  {mrr:.3f}   {ndcg:.3f}    {p50:5.1f}  {p95:5.1f}"
    print(line)

print("\n  Targets:")
print("  Recall@5 ≥ 0.80   MRR@10 ≥ 0.78   NDCG@10 ≥ 0.55   p50 ≤ 150ms   p95 ≤ 400ms")
print()

# Uplift table
print("="*60)
print("INCREMENTAL UPLIFT (each stage over previous)")
print("="*60)
k5_bm25_and = avg(m1.recall.get(5, [0]))
k5_bm25_or  = avg(m1b.recall.get(5, [0]))
k5_dense    = avg(m2.recall.get(5, [0]))
k5_hybrid   = avg(m3.recall.get(5, [0]))
k5_full     = avg(m4.recall.get(5, [0]))

print(f"  BM25 AND (strict keyword)      : Recall@5 = {k5_bm25_and:.4f}")
print(f"  BM25 OR  (liberal keyword)     : Recall@5 = {k5_bm25_or:.4f}  ({(k5_bm25_or-k5_bm25_and)*100:+.1f}pp vs AND)")
print(f"  Dense only (semantic)          : Recall@5 = {k5_dense:.4f}  ({(k5_dense-k5_bm25_or)*100:+.1f}pp vs BM25-OR)")
print(f"  Hybrid RRF (BM25+Dense)        : Recall@5 = {k5_hybrid:.4f}  ({(k5_hybrid-k5_dense)*100:+.1f}pp vs Dense-only)")
print(f"  Full pipeline (+Graph+MMR+Rank): Recall@5 = {k5_full:.4f}  ({(k5_full-k5_hybrid)*100:+.1f}pp vs Hybrid)")

db.close()
