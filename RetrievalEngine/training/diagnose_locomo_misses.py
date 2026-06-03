#!/usr/bin/env python3
"""
Phase 5.6 — Failure analysis for LoCoMo R@10 misses.

For every query where the gold session is NOT in our top-10, dump:
  - the query
  - the gold session's extracted memories (did we even capture the topic?)
  - what we retrieved instead (top-5)
  - cosine similarity of query to the BEST gold memory (embedding gap signal)

Then categorize each miss:
  COVERAGE   — gold session has no memory matching the query topic (extraction miss)
  EMBEDDING  — gold memory exists & is on-topic but cosine sim is low (embedding gap)
  RANKING    — gold memory has high cosine but lost in pool/rerank (ranking miss)

This tells us the real bottleneck instead of guessing.

Usage:
    python3 training/diagnose_locomo_misses.py --model-tag sonnet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", default="sonnet")
parser.add_argument("--dataset", choices=["locomo", "lme"], default="locomo")
parser.add_argument("--top-k", type=int, default=10)
parser.add_argument("--max-show", type=int, default=100, help="max misses to print in detail")
args = parser.parse_args()

# Dataset → (bench cache file, rich-memory file label)
_DS = {
    "locomo": ("locomo.json",      "LoCoMo"),
    "lme":    ("longmemeval.json", "LongMemEval"),
}[args.dataset]
_BENCH_FILE, _DS_LABEL = _DS

BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE     = BENCH_DIR / "benchmark_cache"
sys.path.insert(0, str(BENCH_DIR))

# Load .env
_env = BENCH_DIR / ".env"
if _env.exists():
    import os
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from sentence_transformers import SentenceTransformer
from app.services.retrieval.reranker import get_default_reranker
from app.services.retrieval.multi_signal_retrieval import MultiSignalRetriever

TAG = args.model_tag

# ── Load data ───────────────────────────────────────────────────────────────
bench = json.loads((CACHE / _BENCH_FILE).read_text())
memories_meta = bench["memories"]   # list of {mid, content, gold_key}
queries = bench["queries"]          # list of {question, gold_keys}

all_memories = json.loads((CACHE / f"rich_memories_{TAG}_{_DS_LABEL}.json").read_text())
entity_store = json.loads((CACHE / f"entity_store_{TAG}_{_DS_LABEL}.json").read_text())
rich_embs = np.load(str(CACHE / f"embed_rich_gtel_{TAG}_{_DS_LABEL}.npy"))

# Build flat arrays in gold_key order (same as benchmark)
mem_texts, mem_session_keys, mem_positions, mem_ids = [], [], [], []
sid_to_records = all_memories
for m in memories_meta:
    sid = m["gold_key"]
    recs = sid_to_records.get(sid, [])
    if recs:
        for rec in recs:
            mem_texts.append(rec["memory"])
            mem_session_keys.append(sid)
            mem_positions.append(rec.get("session_position", 1))
            mem_ids.append(rec.get("memory_id", ""))
    else:
        mem_texts.append(m["content"][:200])
        mem_session_keys.append(sid)
        mem_positions.append(1)
        mem_ids.append("")

assert rich_embs.shape[0] == len(mem_texts), f"{rich_embs.shape[0]} vs {len(mem_texts)}"

# session_id → list of (memory_text, row_index)
session_to_rows: dict[str, list[int]] = {}
for i, sid in enumerate(mem_session_keys):
    session_to_rows.setdefault(sid, []).append(i)

# ── Rephrases ───────────────────────────────────────────────────────────────
rephrase_cache = CACHE / f"mqr_rephrases_{_DS_LABEL}.json"
rephrases_raw = json.loads(rephrase_cache.read_text()) if rephrase_cache.exists() else {}

# ── Embedder ────────────────────────────────────────────────────────────────
print("Loading GTE-large...")
embedder = SentenceTransformer("thenlper/gte-large")

def embed_one(q: str) -> np.ndarray:
    return embedder.encode(q, convert_to_numpy=True, normalize_embeddings=True)

reranker = get_default_reranker()

retriever = MultiSignalRetriever(
    mem_texts=mem_texts,
    mem_embs=rich_embs,
    mem_session_keys=mem_session_keys,
    mem_positions=mem_positions,
    entity_store=entity_store,
    embed_fn=embed_one,
    reranker=reranker,
    mem_ids=mem_ids if any(mem_ids) else None,
)

# Run on ALL queries — removes sampling noise, gives true failure distribution
sampled = queries[:]

# ── Run + categorize ────────────────────────────────────────────────────────
def best_cosine_to_session(qvec: np.ndarray, sid: str) -> tuple[float, str]:
    """Highest cosine sim between query and any memory in the gold session."""
    rows = session_to_rows.get(sid, [])
    if not rows:
        return -1.0, "(no memories extracted for this session)"
    sims = rich_embs[rows] @ qvec
    best_idx = int(np.argmax(sims))
    return float(sims[best_idx]), mem_texts[rows[best_idx]]

misses = []
hits = 0
for qe in sampled:
    q = qe["question"]
    gold = set(qe["gold_keys"])
    reps = rephrases_raw.get(q, [q, q])
    retrieved = retriever.retrieve(q, rephrases=reps, top_k=args.top_k)
    if gold & set(retrieved):
        hits += 1
        continue

    qvec = embed_one(q)
    # For each gold session, what's the best cosine?
    gold_info = []
    for gsid in gold:
        sim, mem = best_cosine_to_session(qvec, gsid)
        gold_info.append((gsid, sim, mem))
    gold_info.sort(key=lambda x: -x[1])

    misses.append({
        "query": q,
        "gold_keys": list(gold),
        "best_gold_sim": gold_info[0][1],
        "best_gold_mem": gold_info[0][2],
        "retrieved": retrieved[:5],
    })

# ── Categorize ──────────────────────────────────────────────────────────────
# Heuristic thresholds on best cosine sim of query→gold memory:
#   sim < 0.0          → COVERAGE (no memory / no on-topic memory at all)
#   0.0 <= sim < 0.78  → EMBEDDING (on-topic memory exists but embeds far from query)
#   sim >= 0.78        → RANKING (close memory exists but lost in pool/rerank)
COVERAGE, EMBEDDING, RANKING = [], [], []
for ms in misses:
    s = ms["best_gold_sim"]
    if s < 0.0:
        COVERAGE.append(ms)
    elif s < 0.78:
        EMBEDDING.append(ms)
    else:
        RANKING.append(ms)

n = len(sampled)
print("\n" + "=" * 70)
print(f"{_DS_LABEL} R@{args.top_k} FAILURE ANALYSIS  ({n} queries)")
print("=" * 70)
print(f"  Hits (gold in top-{args.top_k}): {hits}/{n}  = R@{args.top_k} {hits/n:.3f}")
print(f"  Misses:                    {len(misses)}/{n}")
print()
print(f"  Failure breakdown:")
print(f"    COVERAGE  (no on-topic memory):       {len(COVERAGE):3d}  ({len(COVERAGE)/max(1,len(misses))*100:.0f}% of misses)")
print(f"    EMBEDDING (on-topic but sim<0.78):    {len(EMBEDDING):3d}  ({len(EMBEDDING)/max(1,len(misses))*100:.0f}% of misses)")
print(f"    RANKING   (close mem sim>=0.78 lost): {len(RANKING):3d}  ({len(RANKING)/max(1,len(misses))*100:.0f}% of misses)")
print("=" * 70)

def dump(label, items):
    if not items:
        return
    print(f"\n{'─'*70}\n{label}  ({len(items)} cases)\n{'─'*70}")
    for ms in items[:args.max_show]:
        print(f"\nQ: {ms['query']}")
        print(f"  gold: {ms['gold_keys']}   best_gold_sim={ms['best_gold_sim']:.3f}")
        print(f"  best gold memory: {ms['best_gold_mem'][:150]}")
        print(f"  we retrieved:     {ms['retrieved']}")

dump("COVERAGE MISSES — extraction never captured the query topic", COVERAGE)
dump("EMBEDDING MISSES — on-topic memory exists but embeds far from query", EMBEDDING)
dump("RANKING MISSES — close memory exists but lost in pool/rerank", RANKING)

# Save full dump
out = CACHE / f"{_DS_LABEL.lower()}_miss_analysis.json"
out.write_text(json.dumps({
    "r_at_k": hits / n, "k": args.top_k, "n": n,
    "coverage": COVERAGE, "embedding": EMBEDDING, "ranking": RANKING,
}, indent=2))
print(f"\nSaved full analysis: {out}")
