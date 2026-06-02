#!/usr/bin/env python3
"""
Phase 4.4 — Build rich memory caches for both model tiers.

Extracts 15-80 word temporally-grounded memories using:
  --model sonnet  →  Claude Sonnet 4.6 (internal ceiling benchmark)
  --model haiku   →  Claude Haiku 4.5  (public product benchmark)

Run TWICE — once per model tier — to produce dual-benchmark cache:
  python3 training/build_rich_memories.py --model sonnet --dataset all
  python3 training/build_rich_memories.py --model haiku  --dataset all

Output files (in benchmark_cache/):
  rich_memories_{model}_{ds}.json            ← flat list of memory records
  entity_store_{model}_{ds}.json             ← entity → memory links
  embed_rich_gtel_{model}_{ds}.npy           ← GTE-large embeddings (float32)
  embed_rich_gtel_{model}_{ds}.session_ids.json  ← matching session ID per row
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["sonnet", "haiku"],
    default="sonnet",
    help="Extraction model tier. sonnet=Sonnet 4.6 (internal), haiku=Haiku 4.5 (product)",
)
parser.add_argument(
    "--dataset",
    choices=["locomo", "lme", "all"],
    default="all",
    help="Dataset(s) to process. Default: all (LoCoMo + LME)",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Re-extract even if cache exists",
)
args = parser.parse_args()

MODEL_IDS = {
    "sonnet": "claude-sonnet-4-6",
    "haiku":  "claude-haiku-4-5-20251001",
}
MODEL_TAG = args.model          # "sonnet" or "haiku" — used in filenames
MODEL_ID  = MODEL_IDS[MODEL_TAG]

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE     = BENCH_DIR / "benchmark_cache"
sys.path.insert(0, str(BENCH_DIR))

# ── Load API key ───────────────────────────────────────────────────────────────
_env = BENCH_DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set in RetrievalEngine/.env")
    sys.exit(1)

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "locomo": {
        "cache_file":  "locomo.json",
        "ds_label":    "LoCoMo",
        "display_name": "LoCoMo",
    },
    "lme": {
        "cache_file":  "longmemeval.json",
        "ds_label":    "LongMemEval",
        "display_name": "LongMemEval",
    },
}


def _output_paths(ds_label: str) -> tuple[Path, Path, Path, Path]:
    """Return (memories_path, entity_store_path, emb_path, ids_path) for this model+dataset."""
    tag = f"{MODEL_TAG}_{ds_label}"
    return (
        CACHE / f"rich_memories_{tag}.json",
        CACHE / f"entity_store_{tag}.json",
        CACHE / f"embed_rich_gtel_{tag}.npy",
        CACHE / f"embed_rich_gtel_{tag}.session_ids.json",
    )


def build_dataset(cfg: dict) -> None:
    from app.services.extraction.rich_memory_extractor import (
        RichMemoryExtractor,
        build_entity_store,
        check_memory_quality,
        session_position_from_id,
    )
    import numpy as np
    from sentence_transformers import SentenceTransformer

    ds_label = cfg["ds_label"]
    mem_path, ent_path, emb_path, ids_path = _output_paths(ds_label)

    # ── Load raw dataset ────────────────────────────────────────────────────────
    raw_path = CACHE / cfg["cache_file"]
    if not raw_path.exists():
        print(f"  [skip] {ds_label}: cache file not found at {raw_path}")
        return

    ds_data  = json.loads(raw_path.read_text())
    memories = ds_data["memories"]
    print(f"\n{ds_label}: {len(memories)} sessions → model={MODEL_ID}")

    # ── Resume from existing cache ──────────────────────────────────────────────
    existing: dict[str, list[dict]] = {}
    if mem_path.exists() and not args.force:
        existing = json.loads(mem_path.read_text())
        cached_n = sum(1 for m in memories if m["gold_key"] in existing)
        print(f"  Existing cache: {cached_n}/{len(memories)} sessions")
    else:
        print(f"  Starting fresh extraction.")

    needs_extraction = [m for m in memories if m["gold_key"] not in existing]
    print(f"  To extract: {len(needs_extraction)} sessions")

    if needs_extraction:
        extractor = RichMemoryExtractor(api_key=ANTHROPIC_KEY, model=MODEL_ID)

        sessions_list = [
            {
                "session_id":       m["gold_key"],
                "content":          m.get("content") or m.get("search_text", ""),
                "session_position": session_position_from_id(m["gold_key"]),
                "dataset":          ds_label,
            }
            for m in needs_extraction
        ]

        ckpt = CACHE / f"_ckpt_rich_{MODEL_TAG}_{ds_label}.json"
        print(f"  Extracting with {MODEL_ID} (checkpoint every 20 sessions)...")
        t0 = time.monotonic()

        new_results = extractor.extract_batch(
            sessions_list,
            verbose=True,
            checkpoint_path=ckpt,
            checkpoint_every=20,
        )

        if ckpt.exists():
            ckpt.unlink()

        elapsed = time.monotonic() - t0
        print(f"  Extraction done in {elapsed / 60:.1f} min")

        all_results = {**existing, **new_results}
        mem_path.write_text(json.dumps(all_results, indent=2))
        print(f"  Saved memories: {mem_path}")
    else:
        all_results = existing
        print("  All sessions cached — skipping extraction.")

    # ── Quality audit ───────────────────────────────────────────────────────────
    all_records_flat = [r for records in all_results.values() for r in records]
    quality = check_memory_quality(all_records_flat)
    total_mems = quality.get("total_memories", 0)
    avg_w      = quality.get("avg_words", 0)
    temp_pct   = quality.get("temporal_grounded_pct", 0)
    pron_pct   = quality.get("pronoun_pct", 0)
    ent_cov    = quality.get("entity_coverage_pct", 0)
    short_n    = quality.get("short_count", 0)
    long_n     = quality.get("long_count", 0)

    avg_per_sess = total_mems / max(1, len(all_results))
    print(f"\n  Quality audit ({total_mems:,} memories, {avg_per_sess:.1f}/session):")
    print(f"    avg words/memory:       {avg_w:.1f}  (target 30-60)")
    print(f"    temporal grounded:      {temp_pct:.1%}  (target >90%)")
    print(f"    pronoun leakage:        {pron_pct:.1%}  (target <5%)")
    print(f"    entity coverage:        {ent_cov:.1%}  (memories with ≥1 entity)")
    print(f"    too-short (<15w):       {short_n}  (discarded by validator)")
    print(f"    too-long (>80w):        {long_n}  (truncated by validator)")

    # ── Entity store ────────────────────────────────────────────────────────────
    print(f"\n  Building entity store...")
    entity_store = build_entity_store(all_records_flat)
    ent_path.write_text(json.dumps(entity_store, indent=2))
    top_entities = entity_store[:5]
    print(f"  Entity store: {len(entity_store):,} entities")
    print(f"  Top entities by memory count:")
    for ent in top_entities:
        print(f"    {ent['canonical_name']!r}: {ent['memory_count']} memories")
    print(f"  Saved entity store: {ent_path}")

    # ── GTE-large embeddings ────────────────────────────────────────────────────
    # Build flat (session_id, memory_text) rows in session order
    rows: list[tuple[str, str]] = []
    for mem in memories:
        sid = mem["gold_key"]
        records = all_results.get(sid, [])
        if records:
            for rec in records:
                rows.append((sid, rec["memory"]))
        else:
            # No memories extracted — embed raw content truncated as fallback
            rows.append((sid, (mem.get("content") or mem.get("search_text", ""))[:200]))

    texts      = [r[1] for r in rows]
    session_ids = [r[0] for r in rows]

    print(f"\n  Embedding {len(texts):,} memory texts with GTE-large...")
    st_model = SentenceTransformer("thenlper/gte-large")
    embs = st_model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    np.save(str(emb_path), embs)
    ids_path.write_text(json.dumps(session_ids))
    print(f"  Saved embeddings: {emb_path}  shape={embs.shape}")
    print(f"  Saved session ID map: {ids_path}")


def main() -> None:
    datasets = ["locomo", "lme"] if args.dataset == "all" else [args.dataset]

    print("=" * 65)
    print(f"Phase 4.4 — Rich Memory Extraction")
    print(f"Model:    {MODEL_ID}  (tag: {MODEL_TAG})")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Force:    {args.force}")
    print("=" * 65)

    for ds_key in datasets:
        build_dataset(DATASET_CONFIGS[ds_key])

    print("\n" + "=" * 65)
    print("Done! Run Phase 4.4 benchmark:")
    print(f"  python3 benchmark_4approaches.py --skip-ollama --skip-cloud --only-a11s --model-tag {MODEL_TAG}")
    print("=" * 65)


if __name__ == "__main__":
    main()
