#!/usr/bin/env python3
"""
Phase 4.2 — Entity Registry Benchmark.

Validates EntityRegistry + QueryEntityExtractor quality before Phase 4.3.

The key question: if Phase 4.3 does a direct entity graph walk, what is the
theoretical upper bound on recall? This benchmark answers it:

  Entity lookup recall = P(gold session in entity sessions | entity found in query)

If this is 80%, Phase 4.3's graph walk can at best reach 80% for entity queries.
If this is 40%, there's a coverage gap we need to fix first.

Metrics reported:
  1. Entity extraction rate  — % of queries where ≥1 entity found
  2. Entity lookup recall    — % of entity-found queries where gold session is covered
  3. Recall@5 upper bound    — assuming entity sessions are perfectly ranked top-5
  4. Intent distribution     — ENTITY_STATE / EPISODE / SEMANTIC breakdown
  5. Intent × recall         — entity lookup recall per intent type
  6. Coverage gap analysis   — queries where entity found but gold not covered

Usage:
  cd RetrievalEngine/
  python3 training/benchmark_entity_registry.py                # LoCoMo (all queries)
  python3 training/benchmark_entity_registry.py --dataset lme
  python3 training/benchmark_entity_registry.py --n 100       # sample 100 queries
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=["locomo", "lme", "both"], default="locomo",
    help="Dataset to benchmark (default: locomo)",
)
parser.add_argument(
    "--n", type=int, default=0,
    help="Max queries to evaluate (0 = all)",
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE     = BENCH_DIR / "benchmark_cache"
RE_DIR    = BENCH_DIR
sys.path.insert(0, str(RE_DIR))

random.seed(args.seed)

DATASET_CONFIGS = {
    "locomo": {
        "raw_cache":      "locomo.json",
        "registry_cache": "entity_registry_LoCoMo.json",
        "display_name":   "LoCoMo",
    },
    "lme": {
        "raw_cache":      "longmemeval.json",
        "registry_cache": "entity_registry_LongMemEval.json",
        "display_name":   "LongMemEval",
    },
}


def load_queries(cfg: dict) -> tuple[list[dict], dict[str, str]]:
    """
    Returns (queries, session_content_map).
    queries: [{"question": str, "gold_keys": list[str]}]
    session_content_map: gold_key → content (for context-based pronoun resolution)
    """
    raw_path = CACHE / cfg["raw_cache"]
    if not raw_path.exists():
        print(f"  [skip] Dataset not found: {raw_path}")
        return [], {}

    raw = json.loads(raw_path.read_text())
    queries   = raw.get("queries", [])
    memories  = raw.get("memories", [])
    session_map = {m["gold_key"]: m.get("content", "") for m in memories}
    return queries, session_map


def run_benchmark(cfg: dict):
    from app.services.knowledge_graph.entity_registry import EntityRegistry
    from app.services.knowledge_graph.query_entity_extractor import QueryEntityExtractor

    ds_name     = cfg["display_name"]
    reg_path    = CACHE / cfg["registry_cache"]

    print(f"\n{'='*60}")
    print(f"Entity Registry Benchmark — {ds_name}")
    print(f"{'='*60}")

    if not reg_path.exists():
        print(f"  [skip] Registry not found: {reg_path}")
        print(f"         Run: python3 training/build_entity_registry.py")
        return

    registry  = EntityRegistry.load(reg_path)
    extractor = QueryEntityExtractor(registry)
    st = registry.stats()
    print(f"  Registry: {st['total_entities']:,} entities, {st['total_aliases']:,} aliases, "
          f"{st['sessions_covered']:,} sessions covered")

    queries, session_map = load_queries(cfg)
    if not queries:
        return

    if args.n and args.n < len(queries):
        queries = random.sample(queries, args.n)
    print(f"  Evaluating {len(queries):,} queries\n")

    # ── Per-query evaluation ──────────────────────────────────────────────────
    total               = len(queries)
    entity_found        = 0    # queries where ≥1 entity extracted
    gold_covered        = 0    # entity found + gold session in entity sessions
    entity_state_total  = 0
    entity_state_covered = 0
    episode_total       = 0
    episode_covered     = 0
    semantic_total      = 0
    semantic_covered    = 0

    # Detailed breakdown
    coverage_by_intent: dict[str, dict] = {
        "ENTITY_STATE": {"found": 0, "covered": 0, "not_found": 0},
        "EPISODE":      {"found": 0, "covered": 0, "not_found": 0},
        "SEMANTIC":     {"found": 0, "covered": 0, "not_found": 0},
    }

    # Sample failure cases for qualitative analysis
    failures: list[dict] = []   # entity found but gold not covered
    no_entity: list[dict] = []  # no entity found at all

    for qe in queries:
        question  = qe["question"]
        gold_keys = qe.get("gold_keys", [])
        if not gold_keys:
            continue

        result = extractor.extract(question)
        intent = result.intent
        coverage_by_intent[intent]["found" if result.entity_ids else "not_found"] \
            = coverage_by_intent[intent].get("found" if result.entity_ids else "not_found", 0) + 1

        if not result.entity_ids:
            no_entity.append({"question": question, "intent": intent, "gold": gold_keys[0]})
            coverage_by_intent[intent]["not_found"] += 1
            continue

        entity_found += 1

        # Collect all session_ids from matched entities
        entity_sessions: set[str] = set()
        for eid in result.entity_ids:
            entity_sessions.update(registry.sessions_for_entity(
                registry.get_entity(eid).canonical if registry.get_entity(eid) else ""
            ))

        # Check if any gold key is covered
        is_covered = any(gk in entity_sessions for gk in gold_keys)

        if is_covered:
            gold_covered += 1
            coverage_by_intent[intent]["covered"] = coverage_by_intent[intent].get("covered", 0) + 1
        else:
            failures.append({
                "question":       question,
                "intent":         intent,
                "entities_found": result.entities,
                "gold":           gold_keys[0],
                "entity_sessions_count": len(entity_sessions),
            })

    # ── Results ───────────────────────────────────────────────────────────────
    entity_extraction_rate = entity_found / max(1, total)
    entity_lookup_recall   = gold_covered / max(1, entity_found)
    overall_coverage       = gold_covered / max(1, total)

    print(f"  ENTITY EXTRACTION RATE")
    print(f"    Queries with ≥1 entity found:  {entity_found:4d} / {total:4d}  "
          f"({entity_extraction_rate:.1%})")
    print(f"    Queries with no entity found:  {total - entity_found:4d} / {total:4d}  "
          f"({(total - entity_found) / total:.1%})")
    print()

    print(f"  ENTITY LOOKUP RECALL (given entity found)")
    print(f"    Gold session in entity sessions: {gold_covered:4d} / {entity_found:4d}  "
          f"({entity_lookup_recall:.1%})")
    print(f"    Gold NOT covered (gap):          "
          f"{entity_found - gold_covered:4d} / {entity_found:4d}  "
          f"({(entity_found - gold_covered) / max(1, entity_found):.1%})")
    print()

    print(f"  OVERALL COVERAGE (lookup recall × extraction rate)")
    print(f"    Gold covered / all queries:      {gold_covered:4d} / {total:4d}  "
          f"({overall_coverage:.1%})")
    print(f"    → Phase 4.3 upper bound (entity queries only): {entity_lookup_recall:.1%}")
    print(f"    → Must be supplemented with semantic fallback for remaining "
          f"{100 * (1 - overall_coverage):.0f}%")
    print()

    print(f"  INTENT DISTRIBUTION + RECALL")
    print(f"    {'Intent':<14} {'Total':>6} {'Entity%':>8} {'Covered%':>10}")
    print(f"    {'-'*42}")
    for intent in ("ENTITY_STATE", "EPISODE", "SEMANTIC"):
        d = coverage_by_intent[intent]
        total_intent = d.get("found", 0) + d.get("not_found", 0)
        found_pct    = d.get("found", 0) / max(1, total_intent)
        covered_pct  = d.get("covered", 0) / max(1, d.get("found", 0))
        print(f"    {intent:<14} {total_intent:>6d}  {found_pct:>7.1%}  {covered_pct:>9.1%}")
    print()

    # ── Failure analysis ─────────────────────────────────────────────────────
    if failures:
        print(f"  SAMPLE FAILURES — entity found but gold session not in entity sessions")
        print(f"  (first 5 of {len(failures)} failures)")
        for f in failures[:5]:
            print(f"    Q: {f['question'][:70]}")
            print(f"       intent={f['intent']}  entities={f['entities_found']}  "
                  f"entity_sessions={f['entity_sessions_count']}  gold={f['gold']}")
        print()

    if no_entity:
        print(f"  SAMPLE NO-ENTITY QUERIES — registry couldn't help at all")
        print(f"  (first 5 of {len(no_entity)} queries)")
        for q in no_entity[:5]:
            print(f"    Q: {q['question'][:70]}")
            print(f"       intent={q['intent']}  gold={q['gold']}")
        print()

    # ── Phase 4.3 projection ─────────────────────────────────────────────────
    print(f"  PHASE 4.3 PROJECTION")
    print(f"    Current A10 (pure embedding):  LoCoMo 0.685")
    print(f"    Entity lookup ceiling:         {entity_lookup_recall:.3f}  "
          f"(for queries with entity match)")
    print(f"    If Phase 4.3 achieves ceiling: overall recall would be "
          f"≥ {min(1.0, overall_coverage + 0.2):.3f}  (ceiling × rate + semantic)")
    print(f"    Note: Phase 4.3 adds relation-filtered walk (WORKS_AT, LIVES_AT...)")
    print(f"          which is MORE precise than raw entity sessions — actual gain")
    print(f"          depends on how many gold sessions have the right relation type.")


def main():
    datasets = ["locomo", "lme"] if args.dataset == "both" else [args.dataset]

    print("=" * 60)
    print("Phase 4.2 — Entity Registry Benchmark")
    print("=" * 60)
    print("Measures: entity extraction rate + lookup recall + intent coverage")
    if args.n:
        print(f"Queries:  {args.n} per dataset (sampled, seed={args.seed})")
    else:
        print("Queries:  all available")

    for ds_key in datasets:
        if ds_key in DATASET_CONFIGS:
            run_benchmark(DATASET_CONFIGS[ds_key])


if __name__ == "__main__":
    main()
