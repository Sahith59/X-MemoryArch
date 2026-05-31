#!/usr/bin/env python3
"""
Phase 3.4 — Fine-tune GTE-large bi-encoder on conversational fact retrieval.

Base model : thenlper/gte-large (1024-dim, symmetric embeddings)
Loss       : Custom MNRL (6 lines) — avoids sentence-transformers v5 MNRL MPS bug
             (library MNRL uses torch.eye masking + hardness penalty ops that fail on MPS backward)
Device     : MPS (Apple Silicon) — ~12 min/epoch, 3 epochs ≈ 37 min total
Output     : models/gte-large-xma-v1/

Custom MNRL: for each (anchor, positive, negative) triplet batch:
  - docs = concat([positives, negatives])       [2B, D]
  - scores = anchor @ docs.T * scale            [B, 2B]
  - loss = cross_entropy(scores, arange(B))     diagonal is the target
  Effective negatives per step = 2 * batch_size - 1 (all other positives + all negatives)

Usage:
  cd RetrievalEngine/
  python3 training/build_triplets.py   # first
  python3 training/finetune_gte.py
  python3 training/finetune_gte.py --epochs 5 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-model",  default="thenlper/gte-large")
parser.add_argument("--epochs",      type=int,   default=3)
parser.add_argument("--batch-size",  type=int,   default=16)
parser.add_argument("--lr",          type=float, default=2e-5)
parser.add_argument("--warmup-frac", type=float, default=0.1)
parser.add_argument("--scale",       type=float, default=20.0,
                    help="MNRL temperature scale (default 20.0)")
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--output",      default=None)
parser.add_argument("--seed",        type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)

TRAINING   = Path(__file__).resolve().parent
BENCH_DIR  = TRAINING.parent
MODELS     = BENCH_DIR / "models"
TRAIN_PATH = TRAINING / "triplets_train.jsonl"
VAL_PATH   = TRAINING / "triplets_val.jsonl"
OUTPUT_DIR = Path(args.output) if args.output else MODELS / "gte-large-xma-v1"

if not TRAIN_PATH.exists():
    print(f"ERROR: {TRAIN_PATH} not found. Run training/build_triplets.py first.")
    raise SystemExit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def batches(data: list, size: int, shuffle: bool = True, rng: random.Random | None = None):
    indices = list(range(len(data)))
    if shuffle:
        (rng or random).shuffle(indices)
    for start in range(0, len(indices), size):
        yield [data[indices[i]] for i in range(start, min(start + size, len(indices)))]


def mnrl_loss(emb_a, emb_p, emb_n, scale: float):
    """
    Custom MultipleNegativesRankingLoss — MPS-safe, no library internals.
    docs = [positives; negatives] → 2B candidates for each anchor.
    Diagonal of [B × 2B] score matrix is the target (anchor_i matches positive_i).
    All other B-1 positives + B negatives = in-batch negatives.
    """
    import torch
    import torch.nn.functional as F
    docs   = torch.cat([emb_p, emb_n], dim=0)         # [2B, D]
    scores = torch.mm(emb_a, docs.T) * scale           # [B, 2B]
    labels = torch.arange(emb_a.size(0), device=emb_a.device)
    return F.cross_entropy(scores, labels)


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import get_linear_schedule_with_warmup

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_raw = load_jsonl(TRAIN_PATH)
    val_raw   = load_jsonl(VAL_PATH) if VAL_PATH.exists() else []
    random.shuffle(train_raw)

    for split_name, raw in [("train", train_raw), ("val", val_raw)]:
        by_ds: dict[str, int] = {}
        for t in raw:
            by_ds[t["dataset"]] = by_ds.get(t["dataset"], 0) + 1
        breakdown = "  ".join(f"{k}={v}" for k, v in sorted(by_ds.items()))
        print(f"{split_name}: {len(raw):,} triplets  |  {breakdown}")

    steps_per_epoch = math.ceil(len(train_raw) / args.batch_size)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(total_steps * args.warmup_frac)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model, device=str(device))
    model = model.float()   # force float32 weights for stable gradient updates

    print(f"Epochs     : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}  |  Scale: {args.scale}")
    print(f"Steps      : {total_steps:,} total  |  {steps_per_epoch:,}/epoch  |  warmup: {warmup_steps:,}")
    print(f"Output     : {OUTPUT_DIR}\n")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Tokenize helper ───────────────────────────────────────────────────────
    def tok(texts: list[str]) -> dict:
        return {k: v.to(device) for k, v in model.tokenize(texts).items() if hasattr(v, "to")}

    # ── Training loop ─────────────────────────────────────────────────────────
    t0 = time.monotonic()
    best_val_loss = float("inf")
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in batches(train_raw, args.batch_size, shuffle=True, rng=rng):
            feat_a = tok([t["anchor"]   for t in batch])
            feat_p = tok([t["positive"] for t in batch])
            feat_n = tok([t["negative"] for t in batch])

            optimizer.zero_grad()
            emb_a = model(feat_a)["sentence_embedding"]
            emb_p = model(feat_p)["sentence_embedding"]
            emb_n = model(feat_n)["sentence_embedding"]

            loss = mnrl_loss(emb_a, emb_p, emb_n, args.scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

            if n_batches % max(1, steps_per_epoch // 5) == 0:
                elapsed = (time.monotonic() - t0) / 60
                lr_now  = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}  step {n_batches}/{steps_per_epoch}"
                      f"  loss={epoch_loss/n_batches:.4f}  lr={lr_now:.2e}  [{elapsed:.1f}min]",
                      flush=True)

        avg_loss = epoch_loss / n_batches

        # Validation loss
        val_loss = None
        if val_raw:
            model.eval()
            v_loss, v_n = 0.0, 0
            with torch.no_grad():
                for batch in batches(val_raw[:2000], args.batch_size, shuffle=False):
                    feat_a = tok([t["anchor"]   for t in batch])
                    feat_p = tok([t["positive"] for t in batch])
                    feat_n = tok([t["negative"] for t in batch])
                    emb_a = model(feat_a)["sentence_embedding"]
                    emb_p = model(feat_p)["sentence_embedding"]
                    emb_n = model(feat_n)["sentence_embedding"]
                    v_loss += mnrl_loss(emb_a, emb_p, emb_n, args.scale).item()
                    v_n    += 1
            val_loss = v_loss / v_n
            model.train()

        elapsed = (time.monotonic() - t0) / 60
        print(f"\nEpoch {epoch+1}/{args.epochs}  train={avg_loss:.4f}"
              + (f"  val={val_loss:.4f}" if val_loss else "")
              + f"  [{elapsed:.1f}min]", flush=True)

        if val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss or avg_loss
            model.save(str(OUTPUT_DIR))
            print(f"  Saved best → {OUTPUT_DIR}", flush=True)

    elapsed = (time.monotonic() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min")

    model.save(str(OUTPUT_DIR / "final"))
    print(f"Final model: {OUTPUT_DIR / 'final'}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    model.eval()
    embs = model.encode([
        "When did Alice start working at the hospital?",
        "Alice began her nursing career at City Hospital in March 2019.",
        "Bob enjoys hiking on weekends.",
    ], normalize_embeddings=True, convert_to_tensor=False)
    import numpy as np
    embs = np.array(embs)
    print(f"\nSanity check:")
    print(f"  (query, relevant):   {float(embs[0] @ embs[1]):.4f}  ← should be high")
    print(f"  (query, irrelevant): {float(embs[0] @ embs[2]):.4f}  ← should be lower")
    print(f"\nTo benchmark:")
    print(f"  python3 benchmark_4approaches.py --skip-ollama --skip-cloud --embed-model xma_gte_v1")


if __name__ == "__main__":
    main()
