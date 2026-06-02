#!/usr/bin/env python3
"""
Build SQuAD train split cache files for xma-reranker-v2 training.

The SQuAD val split (squad.json, 10,570 questions) is used exclusively for
benchmarking. This script builds from the train split (87,599 questions,
18,891 unique passages) — zero overlap with benchmark data.

Outputs (in benchmark_cache/):
  squad_train.json                        — memories + queries (same format as squad.json)
  facts_squad_train.json                  — one fact per passage = the passage itself
  embed_facts_gte_large_SQuAD_train.npy  — GTE-large embeddings of all passages

Usage:
  cd RetrievalEngine/
  python3 training/build_squad_train_cache.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE = BENCH_DIR / "benchmark_cache"

OUT_CACHE = CACHE / "squad_train.json"
OUT_FACTS = CACHE / "facts_squad_train.json"
OUT_EMB   = CACHE / "embed_facts_gte_large_SQuAD_train.npy"


def main():
    if OUT_CACHE.exists() and OUT_FACTS.exists() and OUT_EMB.exists():
        print("SQuAD train cache already exists — skipping build.")
        data = json.loads(OUT_CACHE.read_text())
        print(f"  {len(data['memories']):,} passages  ·  {len(data['queries']):,} questions")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print("Downloading SQuAD v1.1 train split from HuggingFace...")
    ds = load_dataset("rajpurkar/squad", split="train")
    print(f"  {len(ds):,} questions loaded")

    # Deduplicate passages → one memory per unique context
    print("Deduplicating passages...")
    ctx_to_mid: dict[str, str] = {}
    memories: list[dict] = []

    for row in ds:
        ctx = row["context"]
        if ctx not in ctx_to_mid:
            mid = f"train_ctx_{len(memories)}"
            ctx_to_mid[ctx] = mid
            memories.append({
                "mid":         mid,
                "title":       row["title"].replace(" ", "_"),
                "content":     ctx,
                "search_text": ctx,
                "gold_key":    mid,
            })

    print(f"  {len(memories):,} unique passages")

    # Build queries (one per question)
    queries: list[dict] = []
    for row in ds:
        mid = ctx_to_mid[row["context"]]
        queries.append({"question": row["question"], "gold_keys": [mid]})

    print(f"  {len(queries):,} questions")

    OUT_CACHE.write_text(json.dumps({"memories": memories, "queries": queries}))
    print(f"Saved: {OUT_CACHE}")

    # Facts: split each passage into sentences (avg 2-4 per passage, 20-40 words each).
    # This matches the length distribution of LoCoMo/LME atomic facts (12-17 words),
    # so the reranker trains on consistently short texts across all 3 datasets.
    import re

    def split_sentences(text: str) -> list[str]:
        # Split on . ! ? followed by space+capital or end-of-string
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        # Merge very short fragments (< 30 chars) with next sentence
        merged: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if merged and len(merged[-1]) < 30:
                merged[-1] = merged[-1] + " " + part
            else:
                merged.append(part)
        return merged if merged else [text[:300]]

    facts_by_session: dict[str, list[str]] = {}
    total_facts = 0
    for mem in memories:
        sentences = split_sentences(mem["content"])
        facts_by_session[mem["mid"]] = sentences
        total_facts += len(sentences)

    avg_facts = total_facts / len(memories)
    avg_len   = sum(len(s.split()) for sents in facts_by_session.values() for s in sents) / total_facts
    print(f"  Sentence split: {total_facts:,} facts across {len(memories):,} passages "
          f"(avg {avg_facts:.1f} facts/passage, avg {avg_len:.1f} words/fact)")

    OUT_FACTS.write_text(json.dumps(facts_by_session))
    print(f"Saved: {OUT_FACTS}")

    # Embed all sentence-facts with GTE-large (one embedding per sentence, not per passage).
    # Order must match build_fact_rows: iterate memories in order, flatten sentences per passage.
    from sentence_transformers import SentenceTransformer
    all_sentences: list[str] = []
    for mem in memories:
        all_sentences.extend(facts_by_session[mem["mid"]])

    print(f"\nEmbedding {len(all_sentences):,} sentence-facts with GTE-large...")
    model = SentenceTransformer("thenlper/gte-large")
    embs = model.encode(
        all_sentences,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"  Shape: {embs.shape}")

    np.save(str(OUT_EMB), embs)
    print(f"Saved: {OUT_EMB}")
    print("\nSQuAD train cache ready.")


if __name__ == "__main__":
    main()
