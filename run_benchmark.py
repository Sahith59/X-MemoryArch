"""
Standalone benchmark runner — produces real scores against plan targets.

Run from RetrievalEngine/:
  python run_benchmark.py
  python run_benchmark.py --dense   # include dense leg (requires sentence_transformers)
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

# ── sys.path setup (mirrors conftest.py) ────────────────────────────────────
_RE_ROOT = Path(__file__).resolve().parent
_P1_ROOT = _RE_ROOT.parent / "project-memory-core"

if str(_RE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RE_ROOT))

import app as _p2_app
_p1_app_dir = str(_P1_ROOT / "app")
if _p1_app_dir not in list(_p2_app.__path__):
    _p2_app.__path__.append(_p1_app_dir)

import app.services as _p2_services
_p1_services_dir = str(_P1_ROOT / "app" / "services")
if _p1_services_dir not in list(_p2_services.__path__):
    _p2_services.__path__.append(_p1_services_dir)

# ── DB setup ─────────────────────────────────────────────────────────────────
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
import app.models  # noqa — registers Phase 1 tables (projects, memories, etc.)
import app.p2_models  # noqa — registers RetrievalRun with metadata

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
db = Session()

# ── FTS setup ─────────────────────────────────────────────────────────────────
try:
    from app.search import setup_fts
    setup_fts(engine)
except Exception as e:
    print(f"[warn] FTS setup failed (BM25 leg disabled): {e}")

# ── Build synthetic corpus ────────────────────────────────────────────────────
from app.services.benchmark.corpus import build_synthetic_corpus
from app.services.benchmark.runner import run_benchmark, run_privacy_suite
from app.services.benchmark.eval_metrics import BenchmarkSuite, EvalReport

import app.crud as crud
import app.schemas as schemas

proj = crud.create_project(db, schemas.ProjectCreate(
    name="Benchmark Project",
    description="Evaluation corpus for X-MemoryArch Phase 2",
    tech_stack=["Python", "FastAPI", "PostgreSQL", "Redis"],
    goals=["Measure retrieval quality"],
    domain="software",
))

print("Building synthetic corpus...")
corpus = build_synthetic_corpus(db, proj.id)
print(f"  {len(corpus.test_cases)} test cases, {len(corpus.memory_map)} memories\n")

# ── Run benchmark (BM25 + Entity legs, no dense) ─────────────────────────────
print("=" * 65)
print("RUN 1: BM25 + Entity legs (embed_query=False, fully offline)")
print("=" * 65)
t0 = time.monotonic()
suite = run_benchmark(db, corpus, top_k=10)
elapsed = time.monotonic() - t0
print(suite.full_report())
print(f"Total benchmark time: {elapsed*1000:.0f}ms\n")

# Per-scenario detail
print("PER-SCENARIO BREAKDOWN")
print("-" * 65)
for rep in suite.reports:
    status = []
    if not rep.meets_recall_target: status.append(f"Recall@5={rep.recall_at_5:.3f} < {rep.TARGET_RECALL_5} MISS")
    if not rep.meets_mrr_target:    status.append(f"MRR@10={rep.mrr_at_10:.3f} < {rep.TARGET_MRR_10} MISS")
    if not rep.meets_ndcg_target:   status.append(f"NDCG@10={rep.ndcg_at_10:.3f} < {rep.TARGET_NDCG_10} MISS")
    if not rep.meets_latency_p50:   status.append(f"p50={rep.latency_p50_ms:.0f}ms MISS")
    if not rep.meets_latency_p95:   status.append(f"p95={rep.latency_p95_ms:.0f}ms MISS")
    if not rep.privacy_clean:       status.append(f"LEAKAGE={rep.total_privacy_violations}")
    flag = "  PASS" if not status else f"  FAIL: {', '.join(status)}"
    print(f"  {rep.scenario_name:<28} R@5={rep.recall_at_5:.3f}  MRR@10={rep.mrr_at_10:.3f}  NDCG@10={rep.ndcg_at_10:.3f}  p50={rep.latency_p50_ms:.0f}ms{flag}")

# ── Privacy suite ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PRIVACY HARDENING SUITE")
print("=" * 65)
violations = run_privacy_suite(db, corpus, top_k=20)
all_clean = all(v == 0 for v in violations.values())
for level, count in violations.items():
    status = "PASS (0 leaks)" if count == 0 else f"FAIL ({count} leaks)"
    print(f"  {level:<12} {status}")
print(f"\n  Overall privacy: {'CLEAN ✓' if all_clean else 'LEAKAGE DETECTED ✗'}")

# ── Dense leg (optional) ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dense", action="store_true", help="Run dense leg too")
args, _ = parser.parse_known_args()

if args.dense:
    print("\n" + "=" * 65)
    print("RUN 2: BM25 + Entity + Dense legs (embed_query=True)")
    print("=" * 65)

    # Re-build corpus with embed_query=True cases and real embeddings
    from app import models
    import struct, numpy as np
    from app.services.semantic_classifier import embed_text

    print("  Embedding corpus memories...")
    memories = db.query(models.Memory).filter(models.Memory.project_id == proj.id).all()
    for m in memories:
        raw = embed_text(m.content or "")
        m.embedding = raw
        m.embedding_model = "all-MiniLM-L6-v2"
    db.commit()
    print(f"  Embedded {len(memories)} memories")

    from app.services.retrieval.retrieval_service import RetrievalConfig, retrieve
    from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
    from app.services.benchmark.eval_metrics import evaluate, EvalReport

    vb = SQLiteExactBackend(db)
    dense_report = EvalReport(scenario_name="dense_full_pipeline")

    for tc in corpus.test_cases:
        cfg = RetrievalConfig(
            top_k=10,
            max_clearance=tc.max_clearance,
            embed_query=True,
        )
        t_s = time.monotonic()
        result = retrieve(db=db, project_id=proj.id, query=tc.query, vector_backend=vb, config=cfg)
        lat = int((time.monotonic() - t_s) * 1000)
        expected = corpus.resolve_expected_ids(tc)
        excluded = corpus.resolve_excluded_ids(tc)
        violations_q = sum(1 for eid in excluded if eid in result.selected_memory_ids)
        qr = evaluate(tc.query, result.selected_memory_ids, expected, lat, violations_q)
        dense_report.query_results.append(qr)

    print(dense_report.summary())

# ── Summary vs competitors ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SCORE vs TARGETS & COMPETITORS")
print("=" * 65)
print(f"{'Metric':<20} {'Ours':>8} {'Target':>8} {'Mem0':>8} {'Zep':>8} {'Status':>8}")
print("-" * 65)

our_r5   = suite.overall_recall_at_5
our_mrr  = suite.overall_mrr_at_10
our_ndcg = suite.overall_ndcg_at_10

# Competitor numbers from deep-research-report (Mem0 v1.1, Zep CE 0.14)
MEM0_R5   = 0.72
MEM0_MRR  = 0.68
MEM0_NDCG = 0.51
ZEP_R5    = 0.69
ZEP_MRR   = 0.64
ZEP_NDCG  = 0.48

TARGET_R5   = 0.80
TARGET_MRR  = 0.78   # (plan says 0.75, using 0.78 from MRR@10 column)
TARGET_NDCG = 0.55

def row(name, ours, target, m0, zep):
    status = "✓ PASS" if ours >= target else "✗ MISS"
    beat_m0 = "↑" if ours > m0 else "↓"
    beat_zep = "↑" if ours > zep else "↓"
    return f"  {name:<18} {ours:>8.3f} {target:>8.3f} {m0:>6.3f}{beat_m0} {zep:>6.3f}{beat_zep} {status:>8}"

print(row("Recall@5",   our_r5,   TARGET_R5,   MEM0_R5,   ZEP_R5))
print(row("MRR@10",     our_mrr,  TARGET_MRR,  MEM0_MRR,  ZEP_MRR))
print(row("NDCG@10",    our_ndcg, TARGET_NDCG, MEM0_NDCG, ZEP_NDCG))

total_violations = suite.total_privacy_violations
print(f"\n  {'Privacy leakage':<18} {total_violations:>8}    0       0       0   {'✓ PASS' if total_violations == 0 else '✗ FAIL':>8}")

all_p50 = sorted(r.latency_ms for rep in suite.reports for r in rep.query_results)
p50 = all_p50[len(all_p50)//2] if all_p50 else 0
p95 = all_p50[min(int(len(all_p50)*0.95), len(all_p50)-1)] if all_p50 else 0
print(f"\n  {'Latency p50':<18} {p50:>7}ms  150ms     --      --   {'✓ PASS' if p50 <= 150 else '✗ MISS':>8}")
print(f"  {'Latency p95':<18} {p95:>7}ms  400ms     --      --   {'✓ PASS' if p95 <= 400 else '✗ MISS':>8}")

db.close()
