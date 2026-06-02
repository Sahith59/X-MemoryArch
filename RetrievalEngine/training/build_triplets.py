#!/usr/bin/env python3
"""
Phase 3.4 — Build (anchor, positive, negative) triplets for GTE embedding fine-tuning.

Training data:
  LoCoMo: 490 training queries (200 benchmark eval held out)
  LME:    300 training queries (200 benchmark eval held out)
  SQuAD:  NOT included — kept as out-of-distribution validation

For each (query, gold_session) pair:
  positive = each Claude-extracted fact from the gold session
  negative = hard negative fact from a non-gold session (top cosine-similar non-gold)

Multiple triplets are created per gold fact by rotating through top hard negatives.

Output:
  training/triplets_train.jsonl  — 80% split
  training/triplets_val.jsonl    — 20% split
  Format: {"anchor": query, "positive": gold_fact, "negative": hard_neg_fact}

Usage:
  cd RetrievalEngine/
  python3 training/build_triplets.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

BENCH_DIR    = Path(__file__).resolve().parent.parent
CACHE        = BENCH_DIR / "benchmark_cache"
TRAINING     = Path(__file__).resolve().parent
OUTPUT_TRAIN = TRAINING / "triplets_train.jsonl"
OUTPUT_VAL   = TRAINING / "triplets_val.jsonl"

# Hard negatives per gold fact — rotate through top hits
N_HARD_NEG = 3
VAL_FRAC   = 0.20
SEED       = 13


def get_benchmark_eval_questions() -> tuple[set[str], set[str]]:
    """Return (locomo_eval, lme_eval) holdout sets — exact same sampling as benchmark."""
    rng = random.Random(42)
    squad_qs  = json.loads((CACHE / "squad.json").read_text())["queries"]
    rng.sample(squad_qs, min(200, len(squad_qs)))   # advance rng past SQuAD
    locomo_qs = json.loads((CACHE / "locomo.json").read_text())["queries"]
    locomo_eval = {q["question"] for q in rng.sample(locomo_qs, min(200, len(locomo_qs)))}
    lme_qs    = json.loads((CACHE / "longmemeval.json").read_text())["queries"]
    lme_eval  = {q["question"] for q in rng.sample(lme_qs, min(200, len(lme_qs)))}
    return locomo_eval, lme_eval


DATASET_CONFIGS = [
    {
        "name":       "LoCoMo",
        "cache_file": "locomo.json",
        "facts_file": "facts_claude_LoCoMo.json",
        "emb_file":   "embed_facts_gte_large_LoCoMo.npy",
        "holdout":    "locomo",
    },
    {
        "name":       "LongMemEval",
        "cache_file": "longmemeval.json",
        "facts_file": "facts_claude_LongMemEval.json",
        "emb_file":   "embed_facts_gte_large_LongMemEval.npy",
        "holdout":    "lme",
    },
]


def build_fact_rows(memories: list[dict], facts_by_session: dict[str, list[str]]) -> list[dict]:
    rows: list[dict] = []
    for mem in memories:
        sid   = mem["gold_key"]
        facts = facts_by_session.get(sid) or [mem["content"][:400]]
        for fact_text in facts:
            rows.append({"idx": len(rows), "fact": fact_text, "session_id": sid})
    return rows


def mine_dataset_triplets(
    ds_cfg:       dict,
    model,
    rng:          random.Random,
    holdout_sets: dict[str, set[str]],
) -> list[dict]:
    ds_name    = ds_cfg["name"]
    cache_path = CACHE / ds_cfg["cache_file"]
    facts_path = CACHE / ds_cfg["facts_file"]
    emb_path   = CACHE / ds_cfg["emb_file"]

    if not cache_path.exists():
        print(f"  ERROR: {cache_path} not found.")
        sys.exit(1)
    if not emb_path.exists():
        print(f"  ERROR: {emb_path} not found — run benchmark once to build fact embeddings.")
        sys.exit(1)

    raw      = json.loads(cache_path.read_text())
    memories = raw["memories"]
    all_qs   = raw["queries"]

    holdout_key = ds_cfg.get("holdout")
    if holdout_key and holdout_key in holdout_sets:
        holdout_qs = holdout_sets[holdout_key]
        all_qs = [q for q in all_qs if q["question"] not in holdout_qs]
        print(f"  Holdout: {len(holdout_qs)} eval queries removed → {len(all_qs)} training queries remain")

    facts_by_session: dict[str, list[str]] = json.loads(facts_path.read_text())
    fact_rows  = build_fact_rows(memories, facts_by_session)
    fact_embs  = np.load(str(emb_path)).astype(np.float32)

    assert fact_embs.shape[0] == len(fact_rows), (
        f"Embedding count {fact_embs.shape[0]} != fact count {len(fact_rows)}"
    )
    print(f"  {len(memories):,} sessions · {len(all_qs):,} queries · {len(fact_rows):,} facts")

    session_to_idxs: dict[str, list[int]] = {}
    for row in fact_rows:
        session_to_idxs.setdefault(row["session_id"], []).append(row["idx"])

    all_idxs = list(range(len(fact_rows)))

    # Embed all training queries
    query_texts = [q["question"] for q in all_qs]
    print(f"  Embedding {len(query_texts):,} queries...")
    query_embs = model.encode(
        query_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    triplets: list[dict] = []

    for qi, qe in enumerate(all_qs):
        gold_sessions = set(qe["gold_keys"])

        gold_idxs: set[int] = set()
        for gs in gold_sessions:
            gold_idxs.update(session_to_idxs.get(gs, []))

        if not gold_idxs:
            continue

        nongold_idxs = [i for i in all_idxs if i not in gold_idxs]
        if not nongold_idxs:
            continue

        # Find top hard negatives for this query
        qvec         = query_embs[qi]
        nongold_embs = fact_embs[nongold_idxs]
        sims         = nongold_embs @ qvec
        top_k        = min(N_HARD_NEG * 3, len(nongold_idxs))  # pool to sample from
        top_k_idxs   = np.argsort(sims)[::-1][:top_k].tolist()
        hard_neg_pool = [nongold_idxs[j] for j in top_k_idxs]

        # Create N_HARD_NEG triplets per gold fact (rotating negatives)
        gold_idx_list = list(gold_idxs)
        neg_cursor    = 0

        for gidx in gold_idx_list:
            for _ in range(N_HARD_NEG):
                neg_idx = hard_neg_pool[neg_cursor % len(hard_neg_pool)]
                neg_cursor += 1
                triplets.append({
                    "anchor":   qe["question"],
                    "positive": fact_rows[gidx]["fact"],
                    "negative": fact_rows[neg_idx]["fact"],
                    "dataset":  ds_name,
                })

    print(f"  → {len(triplets):,} triplets")
    return triplets


def main():
    rng = random.Random(SEED)

    print("Computing benchmark eval holdout sets...")
    locomo_eval, lme_eval = get_benchmark_eval_questions()
    print(f"  LoCoMo holdout: {len(locomo_eval)} · LME holdout: {len(lme_eval)}")

    holdout_sets = {"locomo": locomo_eval, "lme": lme_eval}

    from sentence_transformers import SentenceTransformer
    print("\nLoading GTE-large for query embedding...")
    model = SentenceTransformer("thenlper/gte-large")

    all_triplets: list[dict] = []
    for ds_cfg in DATASET_CONFIGS:
        print(f"\n── {ds_cfg['name']} ──")
        triplets = mine_dataset_triplets(ds_cfg, model, rng, holdout_sets)
        all_triplets.extend(triplets)

    # Global stats
    by_ds: dict[str, int] = {}
    for t in all_triplets:
        by_ds[t["dataset"]] = by_ds.get(t["dataset"], 0) + 1
    print(f"\nTotal: {len(all_triplets):,} triplets")
    for ds, cnt in sorted(by_ds.items(), key=lambda x: -x[1]):
        print(f"  {ds}: {cnt:,}  ({100*cnt/len(all_triplets):.1f}%)")

    # Shuffle and split
    rng.shuffle(all_triplets)
    val_size    = int(len(all_triplets) * VAL_FRAC)
    val_data    = all_triplets[:val_size]
    train_data  = all_triplets[val_size:]

    print(f"\nSplit: {len(train_data):,} train · {len(val_data):,} val")

    OUTPUT_TRAIN.write_text("\n".join(json.dumps(t) for t in train_data))
    OUTPUT_VAL.write_text("\n".join(json.dumps(t) for t in val_data))
    print(f"Saved: {OUTPUT_TRAIN}")
    print(f"Saved: {OUTPUT_VAL}")


if __name__ == "__main__":
    main()
