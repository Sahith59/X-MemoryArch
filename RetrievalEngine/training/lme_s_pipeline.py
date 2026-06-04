#!/usr/bin/env python3
"""
LongMemEval-S (standard protocol) — extraction + per-question-scoped evaluation.

Unlike our oracle setup (global pool over 940 sessions), this follows the STANDARD
LongMemEval protocol: each question retrieves ONLY within its own haystack (~48
sessions), exactly how Mem0 and others report their 94.8%. This is the apples-to-apples
comparison.

Two modes:
  --extract   Extract rich memories from the unique haystack sessions (checkpointed).
  --eval      For each question, build a retriever scoped to its haystack memories,
              measure R@5 / R@10 / MRR@10.

Scope with --n-questions (default 100 for a ~$90 Sonnet subset; 500 for the full run).

Usage:
  python3 training/lme_s_pipeline.py --extract --model sonnet --n-questions 100
  python3 training/lme_s_pipeline.py --eval    --model sonnet --n-questions 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--extract", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--model", choices=["sonnet", "haiku", "gpt4omini"], default="sonnet")
parser.add_argument("--n-questions", type=int, default=100)
parser.add_argument("--content-limit", type=int, default=6000)
parser.add_argument("--workers", type=int, default=8, help="concurrent extraction workers")
args = parser.parse_args()
WORKERS = args.workers

MODEL_IDS = {
    "sonnet":    "claude-sonnet-4-6",
    "haiku":     "claude-haiku-4-5-20251001",
    "gpt4omini": "gpt-4o-mini",
}
MODEL_ID = MODEL_IDS[args.model]
TAG = args.model

BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE = BENCH_DIR / "benchmark_cache"
sys.path.insert(0, str(BENCH_DIR))

_env = BENCH_DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

RAW = CACHE / "longmemeval_s_cleaned.json"
MEM_PATH = CACHE / f"lme_s_memories_{TAG}_n{args.n_questions}.json"
EMB_PATH = CACHE / f"lme_s_embed_{TAG}_n{args.n_questions}.npy"
IDS_PATH = CACHE / f"lme_s_embed_{TAG}_n{args.n_questions}.ids.json"
CKPT = CACHE / f"_ckpt_lme_s_{TAG}_n{args.n_questions}.json"


def _format_session(turns: list) -> str:
    lines = []
    for t in turns:
        role = t.get("role", "?").capitalize()
        content = t.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)[: args.content_limit]


def load_questions() -> list[dict]:
    data = json.loads(RAW.read_text())
    return data[: args.n_questions]


def collect_unique_sessions(questions: list[dict]) -> dict[str, str]:
    """session_id -> formatted content, for every haystack session in scope."""
    sessions: dict[str, str] = {}
    for q in questions:
        ids = q.get("haystack_session_ids", [])
        hs = q.get("haystack_sessions", [])
        for sid, turns in zip(ids, hs):
            if sid not in sessions:
                sessions[sid] = _format_session(turns)
    return sessions


# ── EXTRACT ──────────────────────────────────────────────────────────────────
def run_extract() -> None:
    from app.services.extraction.rich_memory_extractor import (
        RichMemoryExtractor, build_entity_store, check_memory_quality,
    )
    from sentence_transformers import SentenceTransformer

    questions = load_questions()
    sessions = collect_unique_sessions(questions)
    print(f"LME-S extract: {len(questions)} questions → {len(sessions):,} unique sessions")
    print(f"Model: {MODEL_ID}")

    results: dict[str, list[dict]] = {}
    if CKPT.exists():
        results = json.loads(CKPT.read_text())
        print(f"  Resumed: {len(results)}/{len(sessions)} sessions done")

    # api_key=None → extractor auto-loads the right key by provider (OPENAI for gpt*, ANTHROPIC else)
    extractor = RichMemoryExtractor(api_key=None, model=MODEL_ID)
    extractor._get_client()  # pre-warm client before concurrent use (thread-safe after init)
    todo = [(sid, c) for sid, c in sessions.items() if sid not in results]
    print(f"  To extract: {len(todo)}  (concurrent, {WORKERS} workers)")
    t0 = time.monotonic()

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = threading.Lock()

    def _do(sid_content):
        sid, content = sid_content
        recs = extractor.extract(content, session_id=sid, session_position=1,
                                 session_date=None, dataset="LongMemEval-S")
        return sid, recs

    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_do, sc) for sc in todo]
        for fut in as_completed(futures):
            sid, recs = fut.result()
            with lock:
                results[sid] = recs
                completed += 1
                if completed % 100 == 0:
                    n_mem = sum(len(v) for v in results.values())
                    print(f"  {len(results)}/{len(sessions)} sessions  ({n_mem:,} memories)  "
                          f"{(time.monotonic()-t0)/60:.1f} min", flush=True)
                    CKPT.write_text(json.dumps(results))
    CKPT.write_text(json.dumps(results))
    MEM_PATH.write_text(json.dumps(results))

    all_recs = [r for recs in results.values() for r in recs]
    q = check_memory_quality(all_recs)
    print(f"\n  Quality: {q['total_memories']:,} memories, "
          f"{q['total_memories']/max(1,len(sessions)):.1f}/session, "
          f"pronoun {q.get('pronoun_pct',0)*100:.1f}%, "
          f"avg_words {q.get('avg_words',0):.1f}")

    # Embed
    print("  Embedding with GTE-large...")
    texts, ids = [], []
    for sid, recs in results.items():
        for r in recs:
            texts.append(r["memory"]); ids.append(sid)
    model = SentenceTransformer("thenlper/gte-large")
    embs = []
    for j in range(0, len(texts), 64):
        embs.append(model.encode(texts[j:j+64], convert_to_numpy=True, normalize_embeddings=True))
    embs = np.vstack(embs).astype(np.float32)
    np.save(str(EMB_PATH), embs)
    IDS_PATH.write_text(json.dumps(ids))
    print(f"  Saved {embs.shape} embeddings → {EMB_PATH.name}")
    print(f"\nDone. Eval: python3 training/lme_s_pipeline.py --eval --model {TAG} --n-questions {args.n_questions}")


# ── EVAL (per-question scoped) ───────────────────────────────────────────────
def run_eval() -> None:
    from sentence_transformers import SentenceTransformer
    from app.services.retrieval.reranker import get_default_reranker
    from app.services.retrieval.multi_signal_retrieval import MultiSignalRetriever

    questions = load_questions()
    all_mem = json.loads(MEM_PATH.read_text())
    embs = np.load(str(EMB_PATH))
    ids = json.loads(IDS_PATH.read_text())

    # session_id -> list of (row_index)
    sid_rows: dict[str, list[int]] = {}
    for i, sid in enumerate(ids):
        sid_rows.setdefault(sid, []).append(i)
    # flat text array aligned to embs
    flat_texts, flat_sids = [], []
    for sid, recs in all_mem.items():
        for r in recs:
            flat_texts.append(r["memory"]); flat_sids.append(sid)
    assert len(flat_texts) == embs.shape[0], f"{len(flat_texts)} vs {embs.shape[0]}"

    embedder = SentenceTransformer("thenlper/gte-large")
    reranker = get_default_reranker()

    def hit_at(retrieved, gold, k):
        return 1.0 if set(gold) & set(retrieved[:k]) else 0.0

    r1 = r3 = r5 = r10 = mrr = 0.0
    n = 0
    for q in questions:
        gold = set(q.get("answer_session_ids", []))
        hay = q.get("haystack_session_ids", [])
        if not gold or not hay:
            continue
        # Build scoped arrays for this question's haystack only
        rows = [i for sid in hay for i in sid_rows.get(sid, [])]
        if not rows:
            continue
        sub_embs = embs[rows]
        sub_texts = [flat_texts[i] for i in rows]
        sub_sids = [flat_sids[i] for i in rows]
        retr = MultiSignalRetriever(
            mem_texts=sub_texts, mem_embs=sub_embs, mem_session_keys=sub_sids,
            mem_positions=[1]*len(rows), entity_store=[],
            embed_fn=lambda x: embedder.encode(x, convert_to_numpy=True, normalize_embeddings=True),
            reranker=reranker, mem_ids=None,
        )
        got = retr.retrieve(q["question"], rephrases=[q["question"], q["question"]], top_k=10)
        r1 += hit_at(got, gold, 1); r3 += hit_at(got, gold, 3)
        r5 += hit_at(got, gold, 5); r10 += hit_at(got, gold, 10)
        # MRR
        rr = 0.0
        for rank, sid in enumerate(got[:10], 1):
            if sid in gold:
                rr = 1.0 / rank; break
        mrr += rr
        n += 1

    print("\n" + "="*60)
    print(f"LongMemEval-S (STANDARD per-question scoped)  —  {TAG}, {n} questions")
    print("="*60)
    print(f"  R@1={r1/n:.3f}  R@3={r3/n:.3f}  R@5={r5/n:.3f}  R@10={r10/n:.3f}  MRR@10={mrr/n:.3f}")
    print(f"  vs Mem0 NEW LME = 0.948")
    print("="*60)


if __name__ == "__main__":
    if args.extract:
        run_extract()
    elif args.eval:
        run_eval()
    else:
        print("Specify --extract or --eval")
