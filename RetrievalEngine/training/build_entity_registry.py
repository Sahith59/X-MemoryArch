#!/usr/bin/env python3
"""
Phase 4.2 — Build entity registry caches for benchmark datasets.

Processes Phase 4.1 triple_facts JSON files → builds EntityRegistry →
saves to benchmark_cache/. Zero API cost (purely local processing).

Output per dataset:
  entity_registry_LoCoMo.json
  entity_registry_LongMemEval.json

The registry maps entity names → session_ids, enabling Phase 4.3
entity graph walk retrieval: "Alice" → [session_3, session_7, ...]

Usage:
  cd RetrievalEngine/
  python3 training/build_entity_registry.py                   # LoCoMo + LME
  python3 training/build_entity_registry.py --dataset locomo
  python3 training/build_entity_registry.py --dataset all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["locomo", "lme", "all"],
    default="all",
    help="Which datasets to build registry for (default: all available triple caches)",
)
parser.add_argument("--force", action="store_true", help="Rebuild even if registry exists")
args = parser.parse_args()

BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE     = BENCH_DIR / "benchmark_cache"
RE_DIR    = BENCH_DIR

sys.path.insert(0, str(RE_DIR))

# ── Path config ───────────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    "locomo": {
        "triple_facts": "triple_facts_LoCoMo.json",
        "output":       "entity_registry_LoCoMo.json",
        "display_name": "LoCoMo",
    },
    "lme": {
        "triple_facts": "triple_facts_LongMemEval.json",
        "output":       "entity_registry_LongMemEval.json",
        "display_name": "LongMemEval",
    },
}


def build_registry(cfg: dict):
    from app.services.knowledge_graph.entity_registry import EntityRegistry

    triple_path = CACHE / cfg["triple_facts"]
    output_path = CACHE / cfg["output"]
    ds_name     = cfg["display_name"]

    print(f"\n{ds_name}")
    print("-" * 40)

    if not triple_path.exists():
        print(f"  [skip] Triple facts not found: {triple_path}")
        print(f"         Run: python3 training/build_triple_facts.py --dataset {ds_name.lower()}")
        return

    if output_path.exists() and not args.force:
        print(f"  Registry already exists: {output_path}")
        # Load and print stats
        reg = EntityRegistry.load(output_path)
        st  = reg.stats()
        print(f"  Entities: {st['total_entities']:,}  |  Aliases: {st['total_aliases']:,}  "
              f"|  Sessions covered: {st['sessions_covered']:,}")
        return

    t0 = time.monotonic()
    triple_facts = json.loads(triple_path.read_text())
    n_sessions   = len(triple_facts)
    n_triples    = sum(len(v) for v in triple_facts.values())
    print(f"  Input: {n_sessions:,} sessions, {n_triples:,} triples")

    print(f"  Building entity registry (zero API cost)...")
    registry = EntityRegistry.build_from_triples(triple_facts, dataset_name=ds_name)

    elapsed = time.monotonic() - t0
    st = registry.stats()

    print(f"  Done in {elapsed:.2f}s")
    print(f"  Entities:         {st['total_entities']:,}")
    print(f"  Aliases:          {st['total_aliases']:,}")
    print(f"  Sessions covered: {st['sessions_covered']:,} / {n_sessions:,}")
    print(f"  By type:          {st['by_type']}")

    # ── Sample: show top-10 entities by session count ─────────────────────────
    entities_by_sessions = sorted(
        registry._entities.values(),
        key=lambda e: len(e.session_ids),
        reverse=True,
    )
    print(f"\n  Top entities by session count:")
    for ent in entities_by_sessions[:10]:
        aliases_str = ", ".join(ent.aliases[:3])
        if len(ent.aliases) > 3:
            aliases_str += f" +{len(ent.aliases) - 3} more"
        print(f"    [{ent.entity_type}] {ent.canonical:25s} → {len(ent.session_ids):3d} sessions  "
              f"(aliases: {aliases_str})")

    # ── Sample: show a few registry lookups to validate ───────────────────────
    print(f"\n  Registry lookup validation:")
    sample_subjects = list(triple_facts.keys())[:3]
    for sid in sample_subjects:
        triples = triple_facts[sid]
        if triples:
            first_subj = triples[0].get("subject", "")
            if first_subj:
                sessions = registry.sessions_for_entity(first_subj)
                print(f"    sessions_for_entity({first_subj!r:20s}) → {len(sessions)} sessions")

    registry.save(output_path)
    print(f"\n  Saved: {output_path}")


def main():
    if args.dataset == "all":
        datasets = list(DATASET_CONFIGS.keys())
    else:
        datasets = [args.dataset]

    print("=" * 60)
    print("Entity Registry Build — Phase 4.2")
    print("=" * 60)
    print("Zero API cost — purely local processing of triple_facts cache")

    for ds_key in datasets:
        if ds_key in DATASET_CONFIGS:
            build_registry(DATASET_CONFIGS[ds_key])
        else:
            print(f"\n[warn] Unknown dataset: {ds_key}")

    print("\n" + "=" * 60)
    print("Done. Entity registries saved to benchmark_cache/")
    print("Next: Phase 4.3 — Entity graph walk retrieval")
    print("  Approach A10 will use these registries for direct entity lookup")
    print("=" * 60)


if __name__ == "__main__":
    main()
