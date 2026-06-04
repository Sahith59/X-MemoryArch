#!/usr/bin/env python3
"""
Per-conversation-scoped LoCoMo eval — the STANDARD Mem0 protocol.

Mem0 ingests one conversation at a time (~27 sessions) and retrieves within it.
Our default benchmark pools all 272 sessions globally. This script measures the
scoped protocol to confirm our 0.923 global number is apples-to-apples (or
conservative) vs Mem0.

For each query, retrieval is scoped to ONLY the sessions of that query's own
conversation (c0..c9), then R@5/R@10/MRR is computed.

Usage: python3 training/scoped_locomo_eval.py --model-tag sonnet
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model-tag", default="sonnet")
args = ap.parse_args()
TAG = args.model_tag

BENCH = Path(__file__).resolve().parent.parent
CACHE = BENCH / "benchmark_cache"
sys.path.insert(0, str(BENCH))
_env = BENCH / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())

from sentence_transformers import SentenceTransformer
from app.services.retrieval.reranker import get_default_reranker
from app.services.retrieval.multi_signal_retrieval import MultiSignalRetriever

bench = json.loads((CACHE / "locomo.json").read_text())
queries = bench["queries"]
all_mem = json.loads((CACHE / f"rich_memories_{TAG}_LoCoMo.json").read_text())
embs = np.load(str(CACHE / f"embed_rich_gtel_{TAG}_LoCoMo.npy"))
entity_store = json.loads((CACHE / f"entity_store_{TAG}_LoCoMo.json").read_text())
rephrases = json.loads((CACHE / "mqr_rephrases_LoCoMo.json").read_text()) if (CACHE / "mqr_rephrases_LoCoMo.json").exists() else {}

# flat arrays aligned to embs (same insertion order as build: bench memories order)
mem_texts, mem_sids = [], []
for m in bench["memories"]:
    sid = m["gold_key"]
    for r in all_mem.get(sid, []):
        mem_texts.append(r["memory"]); mem_sids.append(sid)
assert len(mem_texts) == embs.shape[0], f"{len(mem_texts)} vs {embs.shape[0]}"

# conversation prefix (c0..c9) → row indices
def conv_of(sid): return sid.split("_session_")[0]
conv_rows: dict[str, list[int]] = {}
for i, sid in enumerate(mem_sids):
    conv_rows.setdefault(conv_of(sid), []).append(i)

embedder = SentenceTransformer("thenlper/gte-large")
reranker = get_default_reranker()
emb1 = lambda x: embedder.encode(x, convert_to_numpy=True, normalize_embeddings=True)

# entity store filtered per conversation (entities reference memory_ids; simplest: pass full store,
# scoped retriever only sees scoped rows so entity boost over out-of-scope ids is a no-op)
r1=r3=r5=r10=mrr=0.0; n=0
for q in queries:
    gold = set(q["gold_keys"])
    conv = conv_of(next(iter(gold)))
    rows = conv_rows.get(conv, [])
    if not rows: continue
    sub_e = embs[rows]; sub_t=[mem_texts[i] for i in rows]; sub_s=[mem_sids[i] for i in rows]
    retr = MultiSignalRetriever(mem_texts=sub_t, mem_embs=sub_e, mem_session_keys=sub_s,
        mem_positions=[1]*len(rows), entity_store=[], embed_fn=emb1, reranker=reranker, mem_ids=None)
    reps = rephrases.get(q["question"], [q["question"], q["question"]])
    got = retr.retrieve(q["question"], rephrases=reps, top_k=10)
    def h(k): return 1.0 if gold & set(got[:k]) else 0.0
    r1+=h(1); r3+=h(3); r5+=h(5); r10+=h(10)
    rr=0.0
    for rank,sid in enumerate(got[:10],1):
        if sid in gold: rr=1.0/rank; break
    mrr+=rr; n+=1

print("\n"+"="*60)
print(f"LoCoMo PER-CONVERSATION SCOPED (standard Mem0 protocol) — {TAG}, {n} queries")
print("="*60)
print(f"  R@1={r1/n:.3f}  R@3={r3/n:.3f}  R@5={r5/n:.3f}  R@10={r10/n:.3f}  MRR@10={mrr/n:.3f}")
print(f"  Global-pool baseline (our default): R@5=0.923")
print(f"  Mem0 NEW LoCoMo: 0.916")
print("="*60)
