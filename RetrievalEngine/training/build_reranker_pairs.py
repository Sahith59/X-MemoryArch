#!/usr/bin/env python3
"""
Phase 3.2 (v2) — Build (query, fact, label) training pairs for xma-reranker-v2.

Key changes from v1:
  - SQuAD: uses TRAIN split (87,599 questions, 18,891 passages) instead of val split.
    Zero overlap with benchmark data (benchmark uses val split).
  - LoCoMo / LME: benchmark eval queries (seed=42, 200 each) are held out from
    training. Clean train/eval separation — no leakage.
  - LoCoMo / LME N_HARD_NEG raised to 20 to compensate for fewer training queries.

Dataset configs (after holdout):
  SQuAD train  — 5,000 of 87,599 queries  · N_HARD_NEG=5  · N_EASY_NEG=2
  LoCoMo       — 490 of 690 queries        · N_HARD_NEG=20 · N_EASY_NEG=5
  LongMemEval  — 300 of 500 queries        · N_HARD_NEG=20 · N_EASY_NEG=5

Target pair ratio: ~65% SQuAD / ~21% LoCoMo / ~14% LME  (same as v1)

Output:
  training/reranker_train.jsonl   -- 80% split
  training/reranker_val.jsonl     -- 20% split
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
OUTPUT_TRAIN = TRAINING / "reranker_train.jsonl"
OUTPUT_VAL   = TRAINING / "reranker_val.jsonl"

# ── Reproduce benchmark eval query sets (seed=42, SQuAD first → LoCoMo → LME) ──
# These queries are held out from LoCoMo and LME training to prevent leakage.
# SQuAD uses train split (separate from val split used in benchmark) — no holdout needed.
def get_benchmark_eval_questions() -> tuple[set[str], set[str]]:
    """Return (locomo_eval_questions, lme_eval_questions) as sets of question strings."""
    rng = random.Random(42)

    squad_qs = json.loads((CACHE / "squad.json").read_text())["queries"]
    rng.sample(squad_qs, min(200, len(squad_qs)))           # advance rng past SQuAD

    locomo_qs = json.loads((CACHE / "locomo.json").read_text())["queries"]
    locomo_eval = {q["question"] for q in rng.sample(locomo_qs, min(200, len(locomo_qs)))}

    lme_qs = json.loads((CACHE / "longmemeval.json").read_text())["queries"]
    lme_eval = {q["question"] for q in rng.sample(lme_qs, min(200, len(lme_qs)))}

    return locomo_eval, lme_eval


DATASET_CONFIGS = [
    {
        "name":        "SQuAD train",
        "cache_file":  "squad_train.json",
        "facts_file":  "facts_squad_train.json",
        "emb_file":    "embed_facts_gte_large_SQuAD_train.npy",
        "max_queries": 5000,
        "n_hard_neg":  5,
        "n_easy_neg":  2,
        "holdout":     None,    # train split — no holdout needed
    },
    {
        "name":        "LoCoMo",
        "cache_file":  "locomo.json",
        "facts_file":  "facts_claude_LoCoMo.json",
        "emb_file":    "embed_facts_gte_large_LoCoMo.npy",
        "max_queries": None,    # use all non-holdout queries
        "n_hard_neg":  20,
        "n_easy_neg":  5,
        "holdout":     "locomo",
    },
    {
        "name":        "LongMemEval",
        "cache_file":  "longmemeval.json",
        "facts_file":  "facts_claude_LongMemEval.json",
        "emb_file":    "embed_facts_gte_large_LongMemEval.npy",
        "max_queries": None,    # use all non-holdout queries
        "n_hard_neg":  20,
        "n_easy_neg":  5,
        "holdout":     "lme",
    },
]

VAL_FRAC = 0.20
SEED     = 7


def load_gte_large():
    from sentence_transformers import SentenceTransformer
    print("Loading GTE-large for query embedding...")
    model = SentenceTransformer("thenlper/gte-large")
    print("  GTE-large ready")
    return model


def build_fact_rows(memories: list[dict], facts_by_session: dict[str, list[str]]) -> list[dict]:
    rows: list[dict] = []
    for mem in memories:
        sid   = mem["gold_key"]
        facts = facts_by_session.get(sid) or [mem["content"][:400]]
        for fact_text in facts:
            rows.append({"idx": len(rows), "fact": fact_text, "session_id": sid})
    return rows


def mine_dataset_pairs(
    ds_cfg:       dict,
    model,
    rng:          random.Random,
    holdout_sets: dict[str, set[str]],
) -> list[dict]:
    ds_name    = ds_cfg["name"]
    cache_path = CACHE / ds_cfg["cache_file"]
    facts_path = CACHE / ds_cfg["facts_file"]
    emb_path   = CACHE / ds_cfg["emb_file"]
    n_hard     = ds_cfg["n_hard_neg"]
    n_easy     = ds_cfg["n_easy_neg"]

    print(f"\n── {ds_name} ──")

    if not cache_path.exists():
        print(f"  ERROR: {cache_path} not found.")
        print(f"  For SQuAD train: run python3 training/build_squad_train_cache.py first.")
        sys.exit(1)

    raw       = json.loads(cache_path.read_text())
    memories  = raw["memories"]
    all_qs    = raw["queries"]

    # Apply holdout: remove benchmark eval queries
    holdout_key = ds_cfg.get("holdout")
    if holdout_key and holdout_key in holdout_sets:
        holdout_qs = holdout_sets[holdout_key]
        all_qs = [q for q in all_qs if q["question"] not in holdout_qs]
        print(f"  Holdout: {len(holdout_qs)} eval queries removed → {len(all_qs)} training queries remain")

    # Cap query count
    max_q = ds_cfg.get("max_queries")
    if max_q and len(all_qs) > max_q:
        all_qs = rng.sample(all_qs, max_q)
    print(f"  {len(memories):,} sessions · {len(all_qs):,} queries")

    facts_by_session: dict[str, list[str]] = json.loads(facts_path.read_text())
    fact_rows  = build_fact_rows(memories, facts_by_session)
    fact_embs  = np.load(str(emb_path)).astype(np.float32)

    assert fact_embs.shape[0] == len(fact_rows), (
        f"Embedding count {fact_embs.shape[0]} != fact count {len(fact_rows)}"
    )
    print(f"  {len(fact_rows):,} facts · embeddings {fact_embs.shape}")

    session_to_idxs: dict[str, list[int]] = {}
    for row in fact_rows:
        session_to_idxs.setdefault(row["session_id"], []).append(row["idx"])

    query_texts = [q["question"] for q in all_qs]
    print(f"  Embedding {len(query_texts):,} queries...")
    query_embs = model.encode(
        query_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    all_idxs = list(range(len(fact_rows)))
    pairs: list[dict] = []

    for qi, qe in enumerate(all_qs):
        gold_sessions = set(qe["gold_keys"])

        gold_idxs: set[int] = set()
        for gs in gold_sessions:
            gold_idxs.update(session_to_idxs.get(gs, []))

        if not gold_idxs:
            continue

        nongold_idxs = [i for i in all_idxs if i not in gold_idxs]

        # ── Positives ──────────────────────────────────────────────────────────
        for idx in gold_idxs:
            pairs.append({
                "query":   qe["question"],
                "text":    fact_rows[idx]["fact"],
                "label":   1.0,
                "dataset": ds_name,
                "type":    "positive",
            })

        # ── Hard negatives: highest cosine non-gold ────────────────────────────
        if nongold_idxs:
            qvec         = query_embs[qi]
            nongold_embs = fact_embs[nongold_idxs]
            sims         = nongold_embs @ qvec
            top40        = np.argsort(sims)[::-1][:40].tolist()
            n_sample     = min(n_hard, len(top40))
            hard_chosen  = rng.sample(top40[:min(20, len(top40))], n_sample)
            for j in hard_chosen:
                pairs.append({
                    "query":   qe["question"],
                    "text":    fact_rows[nongold_idxs[j]]["fact"],
                    "label":   0.0,
                    "dataset": ds_name,
                    "type":    "hard_neg",
                })

        # ── Easy negatives: random distant ────────────────────────────────────
        if len(nongold_idxs) > 60:
            easy_chosen = rng.sample(nongold_idxs, min(n_easy, len(nongold_idxs)))
            for idx in easy_chosen:
                pairs.append({
                    "query":   qe["question"],
                    "text":    fact_rows[idx]["fact"],
                    "label":   0.0,
                    "dataset": ds_name,
                    "type":    "easy_neg",
                })

    pos  = sum(1 for p in pairs if p["type"] == "positive")
    hard = sum(1 for p in pairs if p["type"] == "hard_neg")
    easy = sum(1 for p in pairs if p["type"] == "easy_neg")
    print(f"  Pairs: {len(pairs):,} total  ({pos:,} pos · {hard:,} hard_neg · {easy:,} easy_neg)")
    return pairs


def main():
    rng = random.Random(SEED)

    print("Computing benchmark eval holdout sets...")
    locomo_eval, lme_eval = get_benchmark_eval_questions()
    print(f"  LoCoMo eval holdout: {len(locomo_eval)} queries")
    print(f"  LME eval holdout:    {len(lme_eval)} queries")

    holdout_sets = {"locomo": locomo_eval, "lme": lme_eval}

    model = load_gte_large()

    all_pairs: list[dict] = []
    for ds_cfg in DATASET_CONFIGS:
        pairs = mine_dataset_pairs(ds_cfg, model, rng, holdout_sets)
        all_pairs.extend(pairs)

    # ── Global stats ────────────────────────────────────────────────────────────
    pos  = sum(1 for p in all_pairs if p["type"] == "positive")
    hard = sum(1 for p in all_pairs if p["type"] == "hard_neg")
    easy = sum(1 for p in all_pairs if p["type"] == "easy_neg")
    print(f"\nTotal: {len(all_pairs):,} pairs")
    print(f"  Positive:  {pos:,}  ({100*pos/len(all_pairs):.1f}%)")
    print(f"  Hard neg:  {hard:,}  ({100*hard/len(all_pairs):.1f}%)")
    print(f"  Easy neg:  {easy:,}  ({100*easy/len(all_pairs):.1f}%)")

    by_ds: dict[str, int] = {}
    for p in all_pairs:
        by_ds[p["dataset"]] = by_ds.get(p["dataset"], 0) + 1
    print("\nBy dataset:")
    for ds, cnt in sorted(by_ds.items(), key=lambda x: -x[1]):
        print(f"  {ds}: {cnt:,}  ({100*cnt/len(all_pairs):.1f}%)")

    # ── Train/val split ──────────────────────────────────────────────────────────
    rng.shuffle(all_pairs)
    val_size    = int(len(all_pairs) * VAL_FRAC)
    val_pairs   = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    print(f"\nSplit: {len(train_pairs):,} train · {len(val_pairs):,} val")

    OUTPUT_TRAIN.write_text("\n".join(json.dumps(p) for p in train_pairs))
    OUTPUT_VAL.write_text("\n".join(json.dumps(p) for p in val_pairs))
    print(f"Saved: {OUTPUT_TRAIN}")
    print(f"Saved: {OUTPUT_VAL}")


if __name__ == "__main__":
    main()
