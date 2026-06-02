#!/usr/bin/env python3
"""
Phase 3.7 — Build comprehensive fact caches for benchmark datasets.

Fixes the root cause of LoCoMo 66% retrieval:
  1. Topic bias: extracts from BOTH first and second half of each session
  2. Temporal vagueness: preserves all dates/months/years verbatim
  3. Incomplete coverage: extracts facts for ALL people and events in the session
  4. Pronoun resolution: uses progressively-built entity roster across sessions

Datasets processed:
  LoCoMo:      272 sessions → facts_comprehensive_LoCoMo.json
  LongMemEval: 940 sessions → facts_comprehensive_LongMemEval.json

Embeddings are re-created for the new fact caches using GTE-large.

Usage:
  cd RetrievalEngine/
  python3 training/build_comprehensive_facts.py
  python3 training/build_comprehensive_facts.py --dataset locomo
  python3 training/build_comprehensive_facts.py --dataset lme --n-facts 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["locomo", "lme", "all"], default="all")
parser.add_argument("--n-facts", type=int, default=5,
                    help="Facts to extract per half-chunk (total ≈ 2× this). Default: 5 → ~10 per session")
parser.add_argument("--force", action="store_true", help="Re-extract even if cache exists")
args = parser.parse_args()

BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE     = BENCH_DIR / "benchmark_cache"
RE_DIR    = BENCH_DIR

# ── Path bootstrap (same as benchmark_4approaches.py) ─────────────────────────
sys.path.insert(0, str(RE_DIR))
import app as _p2
_P1 = RE_DIR.parent / "project-memory-core"
_p1a = str(_P1 / "app")
if _p1a not in list(_p2.__path__):
    _p2.__path__.append(_p1a)
import app.services as _p2s
_p1s = str(_P1 / "app" / "services")
if _p1s not in list(_p2s.__path__):
    _p2s.__path__.append(_p1s)

# ── Load API key ───────────────────────────────────────────────────────────────
_env = RE_DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set in RetrievalEngine/.env")
    sys.exit(1)

DATASET_CONFIGS = {
    "locomo": {
        "cache_file":    "locomo.json",
        "output_facts":  "facts_comprehensive_LoCoMo.json",
        "output_emb":    "embed_facts_gte_large_LoCoMo_comprehensive.npy",
        "display_name":  "LoCoMo",
    },
    "lme": {
        "cache_file":    "longmemeval.json",
        "output_facts":  "facts_comprehensive_LongMemEval.json",
        "output_emb":    "embed_facts_gte_large_LongMemEval_comprehensive.npy",
        "display_name":  "LongMemEval",
    },
}


def build_dataset(cfg: dict):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from app.services.extraction.comprehensive_extractor import (
        ComprehensiveFactExtractor, check_fact_quality
    )
    from app.services.knowledge_graph.entity_ner import extract_entities

    facts_path = CACHE / cfg["output_facts"]
    emb_path   = CACHE / cfg["output_emb"]
    ds_name    = cfg["display_name"]

    # Load dataset
    ds_data = json.loads((CACHE / cfg["cache_file"]).read_text())
    memories = ds_data["memories"]
    print(f"\n{ds_name}: {len(memories)} sessions")

    # Load existing cache
    existing: dict[str, list[str]] = {}
    if facts_path.exists() and not args.force:
        existing = json.loads(facts_path.read_text())
        cached_n = sum(1 for m in memories if m["gold_key"] in existing)
        print(f"  Existing cache: {cached_n}/{len(memories)} sessions")

    needs_extraction = [m for m in memories if m["gold_key"] not in existing]
    print(f"  To extract: {len(needs_extraction)} sessions × ~{args.n_facts * 2} facts each")

    if not needs_extraction:
        print("  All sessions cached — skipping extraction.")
        all_facts = existing
    else:
        extractor = ComprehensiveFactExtractor(api_key=ANTHROPIC_KEY)

        # Build sessions list with content
        sessions_list = [
            {"session_id": m["gold_key"], "content": m["content"] or m.get("search_text", "")}
            for m in needs_extraction
        ]

        # Process sequentially with progressive entity roster
        # Checkpoint file: saves every 20 sessions → safe to kill/resume
        checkpoint_file = CACHE / f"_checkpoint_{cfg['output_facts']}"
        print(f"  Extracting facts via Claude Haiku (checkpoint every 20 sessions)...")
        t0 = time.monotonic()
        new_facts = extractor.extract_batch(
            sessions_list,
            n_facts=args.n_facts,
            verbose=True,
            checkpoint_path=checkpoint_file,
            checkpoint_every=20,
        )
        # Remove checkpoint file after successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        elapsed = time.monotonic() - t0
        print(f"  Extraction done in {elapsed/60:.1f} min")

        # Merge with existing
        all_facts = {**existing, **new_facts}
        facts_path.write_text(json.dumps(all_facts, indent=2))
        print(f"  Saved: {facts_path}")

    # ── Quality audit ───────────────────────────────────────────────────────────
    all_fact_list = [f for facts in all_facts.values() for f in facts]
    quality = check_fact_quality(all_fact_list)
    print(f"\n  Quality audit ({len(all_fact_list):,} facts):")
    print(f"    quality_score:  {quality['quality_score']:.3f}  (1.0 = perfect)")
    print(f"    pronoun_subject:{quality['issues']['pronoun_subject']} facts start with pronoun")
    print(f"    vague_temporal: {quality['issues']['vague_temporal']} facts use vague time")
    print(f"    no_named_entity:{quality['issues']['no_named_entity']} facts lack a named subject")
    if quality.get("examples", {}).get("vague_temporal"):
        print(f"    examples of vague temporal:")
        for ex in quality["examples"]["vague_temporal"][:2]:
            print(f"      > {ex[:90]}")

    # ── Average facts per session ───────────────────────────────────────────────
    avg_facts = len(all_fact_list) / len(all_facts)
    print(f"    avg facts/session: {avg_facts:.1f} (old approach: 6.0)")

    # ── Build flat fact list for embedding ─────────────────────────────────────
    fact_rows: list[dict] = []
    for mem in memories:
        sid   = mem["gold_key"]
        facts = all_facts.get(sid, [mem["content"][:200]])
        for f in facts:
            fact_rows.append({"fact": f, "session_id": sid})

    print(f"\n  Embedding {len(fact_rows):,} facts with GTE-large...")
    model = SentenceTransformer("thenlper/gte-large")
    texts = [r["fact"] for r in fact_rows]
    embs  = model.encode(texts, batch_size=64, normalize_embeddings=True,
                         show_progress_bar=True).astype("float32")
    import numpy as np
    np.save(str(emb_path), embs)
    print(f"  Saved embeddings: {emb_path}  shape={embs.shape}")

    return all_facts, fact_rows, embs


def main():
    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    print("="*60)
    print("Comprehensive Fact Extraction — Phase 3.7")
    print("="*60)
    print(f"n_facts per chunk: {args.n_facts}  (total per session ≈ {args.n_facts * 2})")

    for ds_key in datasets:
        cfg = DATASET_CONFIGS[ds_key]
        build_dataset(cfg)

    print("\n" + "="*60)
    print("Done! Run benchmark with:")
    print("  python3 benchmark_4approaches.py --skip-ollama --skip-cloud --comprehensive-facts")
    print("="*60)


if __name__ == "__main__":
    main()
