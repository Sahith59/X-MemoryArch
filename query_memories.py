#!/usr/bin/env python3
"""
Phase 3.9 — Interactive memory query + human judgment for real-world validation.

Loads the ingested fact index, runs BM25+Dense+reranker retrieval for each query,
asks for human judgment (R/P/N), and computes precision@5 at the end.

Usage:
  cd X-MemoryArch/
  python3 query_memories.py              # full interactive mode
  python3 query_memories.py --auto       # run preset queries, skip judgment prompts
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

parser = argparse.ArgumentParser()
parser.add_argument("--cache-dir", default="evaluation_cache")
parser.add_argument("--auto",      action="store_true",
                    help="Run preset queries without judgment (for pipeline testing)")
parser.add_argument("--top-k",     type=int, default=5)
args = parser.parse_args()

ROOT      = Path(__file__).resolve().parent
CACHE_DIR = ROOT / args.cache_dir
RE_DIR    = ROOT / "RetrievalEngine"

# Add RetrievalEngine to path so we can load the reranker
sys.path.insert(0, str(RE_DIR))

# ── Pre-defined test queries covering all 3 scenarios ────────────────────────
# These are answerable from the conversations and test different query types:
# entity recall, decision recall, temporal, relationship, numeric.

PRESET_QUERIES = [
    # ── FlowDesk (Scenario 1) ─────────────────────────────────────────────────
    ("FlowDesk", "What database did the FlowDesk team decide to use and why?"),
    ("FlowDesk", "Who is Ravi and what problem is he having with FlowDesk?"),
    ("FlowDesk", "Why did FlowDesk pivot away from targeting solo lawyers?"),
    ("FlowDesk", "Who is Daniel Chen and what did he say about FlowDesk's direction?"),
    ("FlowDesk", "How much runway does FlowDesk have left and who invested?"),
    ("FlowDesk", "What is the tension between Alex and Maya about at FlowDesk?"),
    ("FlowDesk", "What is Priya's role at FlowDesk and what is she waiting on?"),
    ("FlowDesk", "What auth system did FlowDesk consider and what did they decide?"),
    # ── PhD Student (Scenario 2) ──────────────────────────────────────────────
    ("PhD",      "What is the PhD student's research topic and which university?"),
    ("PhD",      "Who is Jonas and why is he a problem in the lab?"),
    ("PhD",      "What issue did they discover with the STRING DB dataset?"),
    ("PhD",      "When is the NeurIPS deadline and what is the submission status?"),
    ("PhD",      "What is the PhD student's annual stipend?"),
    ("PhD",      "Who is Prof. Anita Sharma and how available is she?"),
    ("PhD",      "Who is Wei and why does the PhD student need to deal with him?"),
    ("PhD",      "What is the situation with Alex moving to Seattle?"),
    ("PhD",      "Should the student apply to DeepMind or stay in academia?"),
    # ── Meridian Pay PM (Scenario 3) ──────────────────────────────────────────
    ("MeridianPay", "Why is there a 6-week gap between the PM's promise and engineering's estimate?"),
    ("MeridianPay", "Who is Marcus and why does he keep blocking the Meridian Pay feature?"),
    ("MeridianPay", "What are the current and target activation rates for Meridian Pay?"),
    ("MeridianPay", "Which competitor launched a similar feature and created urgency?"),
    ("MeridianPay", "What vendor is Meridian Pay considering for document verification?"),
    ("MeridianPay", "Who is Sarah and what is her role at Meridian Pay?"),
    ("MeridianPay", "What is the PM's promotion situation and what is at stake?"),
    ("MeridianPay", "What did the user research session reveal about the account opening flow?"),
    ("MeridianPay", "Who is Tom and how does he respond to timeline pressure?"),
]


# ── Load index ────────────────────────────────────────────────────────────────

def load_index():
    for f, name in [
        (CACHE_DIR / "sessions.json",   "sessions.json"),
        (CACHE_DIR / "facts.json",      "facts.json"),
        (CACHE_DIR / "fact_index.json", "fact_index.json"),
        (CACHE_DIR / "embed_facts.npy", "embed_facts.npy"),
    ]:
        if not f.exists():
            print(f"ERROR: {name} not found in {CACHE_DIR}")
            print("Run: python3 ingest_chatgpt_export.py")
            sys.exit(1)

    sessions   = json.loads((CACHE_DIR / "sessions.json").read_text())
    fact_index = json.loads((CACHE_DIR / "fact_index.json").read_text())
    embs       = np.load(str(CACHE_DIR / "embed_facts.npy")).astype(np.float32)

    assert embs.shape[0] == len(fact_index), (
        f"Embedding count {embs.shape[0]} != fact count {len(fact_index)}"
    )
    return sessions, fact_index, embs


# ── Retrieval ─────────────────────────────────────────────────────────────────

def build_bm25(fact_index):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("ERROR: rank_bm25 not installed. Run: pip install rank-bm25")
        sys.exit(1)
    tokenized = [row["fact"].lower().split() for row in fact_index]
    return BM25Okapi(tokenized)


def load_reranker():
    try:
        # Load CrossEncoder directly — avoids lazy-init issue with the app service wrapper
        from sentence_transformers.cross_encoder import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return model
    except Exception as e:
        print(f"  [warn] Reranker load failed: {e} — running BM25+Dense only")
        return None


def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("thenlper/gte-large")


def retrieve(
    query: str,
    fact_index: list[dict],
    embs: np.ndarray,
    bm25,
    embed_model,
    reranker,
    top_k: int = 5,
) -> list[dict]:
    """
    BM25 + Dense RRF (α=0.2) → cross-encoder reranker → top_k unique sessions.
    Returns list of dicts: {session_id, title, scenario, date_range, fact, score}
    """
    alpha = 0.2  # learned optimal from Phase 3.5 (slight BM25 weight)
    k_rrf = 60
    top_n_candidates = 40

    # BM25
    bm25_scores = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)
    bm25_top    = np.argsort(bm25_scores)[::-1][:top_n_candidates]

    # Dense (normalize query vector to match unit-norm fact embeddings)
    qvec = embed_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    norm = np.linalg.norm(qvec)
    if norm > 0:
        qvec = qvec / norm
    dense_scores = embs @ qvec
    dense_top    = np.argsort(dense_scores)[::-1][:top_n_candidates]

    # Weighted RRF
    rrf: dict[int, float] = {}
    for rank, idx in enumerate(bm25_top):
        rrf[int(idx)] = rrf.get(int(idx), 0.0) + alpha / (k_rrf + rank)
    for rank, idx in enumerate(dense_top):
        rrf[int(idx)] = rrf.get(int(idx), 0.0) + (1.0 - alpha) / (k_rrf + rank)

    top_idxs = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:top_n_candidates]

    # Cross-encoder reranker
    if reranker is not None:
        pairs  = [(query, fact_index[i]["fact"]) for i in top_idxs]
        scores = reranker.predict(pairs)
        top_idxs = [idx for _, idx in sorted(zip(scores, top_idxs), reverse=True)]

    # Session dedup: pick top_k unique sessions (max 2 facts/session for MMR)
    results: list[dict] = []
    sess_count: dict[str, int] = defaultdict(int)
    seen_sessions: set[str] = set()

    for idx in top_idxs:
        row  = fact_index[idx]
        sid  = row["session_id"]
        if sess_count[sid] < 2:
            result = {**row, "rank_idx": idx}
            results.append(result)
            sess_count[sid] += 1
            if len(set(r["session_id"] for r in results)) >= top_k:
                break

    # Return one result per unique session (best-ranked fact per session)
    seen: set[str] = set()
    unique_results: list[dict] = []
    for r in results:
        if r["session_id"] not in seen:
            unique_results.append(r)
            seen.add(r["session_id"])
        if len(unique_results) >= top_k:
            break

    return unique_results


# ── Display ───────────────────────────────────────────────────────────────────

def display_results(query: str, results: list[dict], query_num: int):
    print(f"\n{'─'*65}")
    print(f"Query {query_num}: {query}")
    print(f"{'─'*65}")
    for i, r in enumerate(results, 1):
        dates = " → ".join(r.get("date_range", []))
        print(f"\n  [{i}] {r['title']}")
        print(f"      {r['scenario'].replace('_', ' ')}  |  {dates}")
        print(f"      Fact: {r['fact']}")


# ── Judgment ──────────────────────────────────────────────────────────────────

JUDGMENT_LEGEND = "R=Relevant  P=Partial  N=Not relevant  S=Skip"

def get_judgment(query: str, results: list[dict]) -> list[str]:
    """Ask user to judge each result. Returns list of R/P/N/S per result."""
    print(f"\n{JUDGMENT_LEGEND}")
    judgments: list[str] = []
    for i, r in enumerate(results, 1):
        while True:
            raw = input(f"  Result [{i}] '{r['fact'][:60]}...' → ").strip().upper()
            if raw in ("R", "P", "N", "S", ""):
                judgments.append(raw or "N")
                break
            print("    Type R, P, N, or S")
    return judgments


# ── Precision calculation ─────────────────────────────────────────────────────

def compute_metrics(all_judgments: list[list[str]]) -> dict:
    """
    precision@5: fraction of top-5 results that are R or P.
    strict_p@5: fraction that are R only.
    """
    total_results = 0
    relevant      = 0
    strict_rel    = 0
    queries_with_hit = 0

    for judgments in all_judgments:
        top5 = judgments[:5]
        hits     = sum(1 for j in top5 if j in ("R", "P"))
        str_hits = sum(1 for j in top5 if j == "R")
        relevant      += hits
        strict_rel    += str_hits
        total_results += len(top5)
        if hits > 0:
            queries_with_hit += 1

    n = len(all_judgments)
    return {
        "queries":          n,
        "precision_at_5":   relevant / total_results if total_results else 0,
        "strict_p_at_5":    strict_rel / total_results if total_results else 0,
        "query_hit_rate":   queries_with_hit / n if n else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("X-MemoryArch — Phase 3.9 Real-World Validation")
    print("="*65)

    print("\nLoading index...")
    sessions, fact_index, embs = load_index()
    print(f"  {len(sessions)} sessions · {len(fact_index):,} facts · {embs.shape[1]}-dim embeddings")

    print("\nBuilding BM25 index...")
    bm25 = build_bm25(fact_index)

    print("Loading embedding model (GTE-large)...")
    embed_model = load_embed_model()

    print("Loading cross-encoder reranker...")
    reranker = load_reranker()
    if reranker is None:
        print("  [warn] Reranker unavailable — running BM25+Dense only")
    else:
        print("  ms-marco reranker ready")

    print(f"\n{'='*65}")
    print(f"{'='*65}")
    if args.auto:
        print("Mode: AUTO (running preset queries, no judgment prompts)")
    else:
        print("Mode: INTERACTIVE — judge each result (R/P/N/S)")
        print(f"  {JUDGMENT_LEGEND}")
    print(f"{'='*65}")

    all_judgments: list[list[str]] = []
    query_log:     list[dict]      = []

    # ── Run preset queries ─────────────────────────────────────────────────────
    for qi, (scenario, query) in enumerate(PRESET_QUERIES, 1):
        results = retrieve(query, fact_index, embs, bm25, embed_model, reranker, args.top_k)
        display_results(query, results, qi)

        if args.auto:
            judgments = ["?"] * len(results)
        else:
            judgments = get_judgment(query, results)

        all_judgments.append(judgments)
        query_log.append({
            "query_num": qi, "scenario": scenario, "query": query,
            "results":   [{"title": r["title"], "fact": r["fact"]} for r in results],
            "judgments": judgments,
        })

    # ── Custom queries ─────────────────────────────────────────────────────────
    if not args.auto:
        print(f"\n{'='*65}")
        print("Custom queries (press Enter with empty query to finish):")
        print(f"{'='*65}")
        cq_num = len(PRESET_QUERIES) + 1
        while True:
            query = input(f"\nQuery {cq_num}: ").strip()
            if not query:
                break
            results = retrieve(query, fact_index, embs, bm25, embed_model, reranker, args.top_k)
            display_results(query, results, cq_num)
            judgments = get_judgment(query, results)
            all_judgments.append(judgments)
            query_log.append({
                "query_num": cq_num, "scenario": "custom", "query": query,
                "results":   [{"title": r["title"], "fact": r["fact"]} for r in results],
                "judgments": judgments,
            })
            cq_num += 1

    # ── Precision@5 ───────────────────────────────────────────────────────────
    if not args.auto and all_judgments[0][0] != "?":
        metrics = compute_metrics(all_judgments)
        print(f"\n{'='*65}")
        print("PHASE 3.9 RESULTS")
        print(f"{'='*65}")
        print(f"  Queries evaluated:  {metrics['queries']}")
        print(f"  Precision@5 (R+P):  {metrics['precision_at_5']:.3f}  "
              f"({'✓ PASS' if metrics['precision_at_5'] >= 0.60 else '✗ needs work'}, target ≥ 0.60)")
        print(f"  Strict P@5 (R only): {metrics['strict_p_at_5']:.3f}")
        print(f"  Query hit rate:     {metrics['query_hit_rate']:.3f}  "
              f"(fraction of queries with ≥1 hit in top-5)")

        verdict = "PASS — system works on real-world data → ready for MVP" \
                  if metrics["precision_at_5"] >= 0.60 else \
                  "NEEDS WORK — investigate retrieval failures before MVP"
        print(f"\n  Verdict: {verdict}")

        # Save results log
        log_path = CACHE_DIR / "evaluation_log.json"
        log_path.write_text(json.dumps({
            "metrics": metrics, "query_log": query_log
        }, indent=2))
        print(f"\n  Full log saved: {log_path}")
    else:
        print(f"\n{'='*65}")
        print("Auto mode — no judgment recorded.")
        print("Re-run without --auto to score retrieval quality.")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
