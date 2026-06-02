#!/usr/bin/env python3
"""
Phase 4.1 — Build triple fact caches for benchmark datasets.

Extracts structured entity-relationship triples instead of atomic sentences.
This is the Phase 4.1 architectural pivot: subject-first, relation-explicit
fact representation for better cosine similarity with entity-state queries.

Output per session:
  [{"subject": "Alice", "relation": "WORKS_AT", "object": "City Hospital",
    "detail": "as a nurse", "text": "Alice works at City Hospital as a nurse",
    "temporal": "current"}, ...]

Files created (in benchmark_cache/):
  triple_facts_LoCoMo.json
  triple_facts_LongMemEval.json
  triple_facts_SQuAD_v1_1.json       (optional, SQuAD already >90%)
  embed_triples_gtel_LoCoMo.npy
  embed_triples_gtel_LongMemEval.npy
  embed_triples_gtel_SQuAD_v1_1.npy  (optional)

Usage:
  cd RetrievalEngine/
  python3 training/build_triple_facts.py                          # LoCoMo + LME
  python3 training/build_triple_facts.py --dataset locomo
  python3 training/build_triple_facts.py --dataset all            # includes SQuAD
  python3 training/build_triple_facts.py --force                  # re-extract all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["locomo", "lme", "squad", "all", "conv"],
    default="conv",
    help=(
        "conv = LoCoMo + LME (default, the gap datasets), "
        "all = conv + SQuAD, "
        "locomo/lme/squad = single dataset"
    ),
)
parser.add_argument("--n-per-chunk", type=int, default=6,
                    help="Target triples per half-session (total ≈ 2×). Default: 6 → ~12/session")
parser.add_argument("--force", action="store_true",
                    help="Re-extract even if cache exists")
args = parser.parse_args()

BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE     = BENCH_DIR / "benchmark_cache"
RE_DIR    = BENCH_DIR

# ── Path bootstrap ─────────────────────────────────────────────────────────────
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

# ── Dataset config ─────────────────────────────────────────────────────────────
# cache_file: raw dataset JSON in benchmark_cache/
# output_facts: where we write extracted triples JSON
# output_emb: GTE-large embeddings over triple .text fields
DATASET_CONFIGS = {
    "locomo": {
        "cache_file":   "locomo.json",
        "output_facts": "triple_facts_LoCoMo.json",
        "output_emb":   "embed_triples_gtel_LoCoMo.npy",
        "display_name": "LoCoMo",
    },
    "lme": {
        "cache_file":   "longmemeval.json",
        "output_facts": "triple_facts_LongMemEval.json",
        "output_emb":   "embed_triples_gtel_LongMemEval.npy",
        "display_name": "LongMemEval",
    },
    "squad": {
        "cache_file":   "squad.json",
        "output_facts": "triple_facts_SQuAD_v1_1.json",
        "output_emb":   "embed_triples_gtel_SQuAD_v1_1.npy",
        "display_name": "SQuAD v1.1",
    },
}


def build_dataset(cfg: dict):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from app.services.extraction.triple_extractor import (
        TripleExtractor, check_triple_quality,
    )

    facts_path = CACHE / cfg["output_facts"]
    emb_path   = CACHE / cfg["output_emb"]
    ds_name    = cfg["display_name"]

    # ── Load raw dataset ────────────────────────────────────────────────────────
    raw_path = CACHE / cfg["cache_file"]
    if not raw_path.exists():
        print(f"  [skip] {ds_name}: cache file not found at {raw_path}")
        return

    ds_data  = json.loads(raw_path.read_text())
    memories = ds_data["memories"]
    print(f"\n{ds_name}: {len(memories)} sessions")

    # ── Load existing cache (checkpoint resume) ─────────────────────────────────
    existing: dict[str, list[dict]] = {}
    if facts_path.exists() and not args.force:
        existing = json.loads(facts_path.read_text())
        cached_n = sum(1 for m in memories if m["gold_key"] in existing)
        print(f"  Existing cache: {cached_n}/{len(memories)} sessions")
    else:
        print(f"  Starting fresh extraction.")

    needs_extraction = [m for m in memories if m["gold_key"] not in existing]
    print(f"  To extract: {len(needs_extraction)} sessions × ~{args.n_per_chunk * 2} triples each")

    if not needs_extraction:
        print("  All sessions cached — skipping extraction.")
        all_triples = existing
    else:
        extractor = TripleExtractor(api_key=ANTHROPIC_KEY)

        sessions_list = [
            {
                "session_id": m["gold_key"],
                "content": m.get("content") or m.get("search_text", ""),
            }
            for m in needs_extraction
        ]

        checkpoint_file = CACHE / f"_checkpoint_{cfg['output_facts']}"
        print(f"  Extracting triples via Claude Haiku (checkpoint every 20 sessions)...")
        t0 = time.monotonic()

        new_triples = extractor.extract_batch(
            sessions_list,
            n_per_chunk=args.n_per_chunk,
            verbose=True,
            checkpoint_path=checkpoint_file,
            checkpoint_every=20,
        )

        if checkpoint_file.exists():
            checkpoint_file.unlink()

        elapsed = time.monotonic() - t0
        print(f"  Extraction done in {elapsed / 60:.1f} min")

        all_triples = {**existing, **new_triples}
        facts_path.write_text(json.dumps(all_triples, indent=2))
        print(f"  Saved triples: {facts_path}")

    # ── Quality audit ────────────────────────────────────────────────────────────
    all_triple_list = [t for triples in all_triples.values() for t in triples]
    quality = check_triple_quality(all_triple_list)
    print(f"\n  Quality audit ({len(all_triple_list):,} triples):")
    print(f"    quality_score:    {quality['quality_score']:.3f}  (1.0 = perfect)")
    print(f"    fallback_pct:     {quality['fallback_pct']:.1%}  (HAS_ATTRIBUTE fallback usage)")
    print(f"    pronoun_subject:  {quality['issues']['pronoun_subject']} triples have pronoun subject")
    print(f"    empty_object:     {quality['issues']['empty_object']} triples missing object")
    if quality["examples"].get("fallback_relation"):
        print(f"    fallback examples: {quality['examples']['fallback_relation'][:2]}")

    avg_triples = len(all_triple_list) / max(1, len(all_triples))
    print(f"    avg triples/session: {avg_triples:.1f}")

    # ── Relation distribution ────────────────────────────────────────────────────
    rel_counts: dict[str, int] = {}
    for t in all_triple_list:
        rel = t.get("relation", "HAS_ATTRIBUTE")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    top_rels = sorted(rel_counts.items(), key=lambda x: -x[1])[:8]
    print(f"  Top relations: " + " | ".join(f"{r}={n}" for r, n in top_rels))

    # ── Build flat list for embedding ────────────────────────────────────────────
    # text field = natural-language rendering: "Alice works at City Hospital as a nurse"
    triple_rows: list[dict] = []
    for mem in memories:
        sid     = mem["gold_key"]
        triples = all_triples.get(sid)
        if not triples:
            # Fallback: use raw content truncated
            triple_rows.append({"text": mem.get("content", "")[:200], "session_id": sid})
            continue
        for t in triples:
            triple_rows.append({"text": t.get("text", ""), "session_id": sid})

    print(f"\n  Embedding {len(triple_rows):,} triple texts with GTE-large...")
    model = SentenceTransformer("thenlper/gte-large")
    texts = [r["text"] for r in triple_rows]
    embs  = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    import numpy as np
    np.save(str(emb_path), embs)
    print(f"  Saved embeddings: {emb_path}  shape={embs.shape}")

    # Save session_id mapping alongside embeddings (needed by A10)
    id_map_path = emb_path.with_suffix(".session_ids.json")
    id_map_path.write_text(json.dumps([r["session_id"] for r in triple_rows]))
    print(f"  Saved session ID map: {id_map_path}")

    return all_triples, triple_rows, embs


def main():
    if args.dataset == "conv":
        datasets = ["locomo", "lme"]
    elif args.dataset == "all":
        datasets = ["locomo", "lme", "squad"]
    else:
        datasets = [args.dataset]

    print("=" * 60)
    print("Triple Fact Extraction — Phase 4.1")
    print("=" * 60)
    print(f"n_per_chunk: {args.n_per_chunk}  (total per session ≈ {args.n_per_chunk * 2})")
    print(f"Datasets:    {', '.join(datasets)}")
    print()

    for ds_key in datasets:
        cfg = DATASET_CONFIGS[ds_key]
        build_dataset(cfg)

    print("\n" + "=" * 60)
    print("Done! Run A10 benchmark with:")
    print("  python3 benchmark_4approaches.py --skip-ollama --skip-cloud --only-a10")
    print("  # or run full suite:")
    print("  python3 benchmark_4approaches.py --skip-ollama --skip-cloud --comprehensive-facts")
    print("=" * 60)


if __name__ == "__main__":
    main()
