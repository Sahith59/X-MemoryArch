#!/usr/bin/env python3
"""
QA-accuracy evaluation — the metric Mem0/Zep/Honcho actually publish.

Pipeline per question (standard LongMemEval / LoCoMo methodology):
  1. RETRIEVE top-k sessions with our engine (scoped to the standard protocol)
  2. GENERATE an answer from the retrieved memories (GPT-4o-mini generator)
  3. JUDGE the answer vs the gold answer (GPT-4o judge → correct/incorrect)
  QA accuracy = % judged correct  →  directly comparable to Mem0's 91.6 / 94.8

This is the like-for-like number. Retrieval recall (R@5) is a different, easier
metric and is NOT comparable to competitors' QA-accuracy scores.

Usage:
  python3 training/qa_accuracy_eval.py --dataset lme_s   --model-tag gpt4omini --n 500
  python3 training/qa_accuracy_eval.py --dataset locomo  --model-tag sonnet    --n 690
"""
from __future__ import annotations
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["lme_s", "locomo"], required=True)
ap.add_argument("--model-tag", default="gpt4omini")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--top-k", type=int, default=10, help="retrieved sessions fed to the generator")
ap.add_argument("--gen-model", default="gpt-4o-mini")
ap.add_argument("--judge-model", default="gpt-4o")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

BENCH = Path(__file__).resolve().parent.parent
CACHE = BENCH / "benchmark_cache"
sys.path.insert(0, str(BENCH))
_env = BENCH / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())

import openai
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
from sentence_transformers import SentenceTransformer
from app.services.retrieval.reranker import get_default_reranker
from app.services.retrieval.multi_signal_retrieval import MultiSignalRetriever

embedder = SentenceTransformer("thenlper/gte-large")
reranker = get_default_reranker()
emb1 = lambda x: embedder.encode(x, convert_to_numpy=True, normalize_embeddings=True)

GEN_PROMPT = """You answer a question using ONLY the memories provided. Be concise — answer in as few words as possible (a name, date, number, or short phrase). If the memories do not contain the answer, reply "I don't know".

Memories:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are grading whether a predicted answer to a question is correct, given the gold answer. Be lenient about phrasing, formatting, and extra words — judge only whether the predicted answer conveys the same factual information as the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer correct? Reply with exactly one word: CORRECT or INCORRECT."""


def gen_answer(context: str, question: str) -> str:
    r = client.chat.completions.create(
        model=args.gen_model, max_tokens=80, temperature=0,
        messages=[{"role": "user", "content": GEN_PROMPT.format(context=context[:8000], question=question)}],
    )
    return (r.choices[0].message.content or "").strip()


def judge(question: str, gold: str, pred: str) -> bool:
    r = client.chat.completions.create(
        model=args.judge_model, max_tokens=5, temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)}],
    )
    return "CORRECT" in (r.choices[0].message.content or "").upper()


# ── Load dataset + memories + per-question scope ─────────────────────────────
def load_lme_s():
    MEMN = 500  # memories were extracted for the full 500-question set
    data = json.loads((CACHE / "longmemeval_s_cleaned.json").read_text())[: args.n]
    mem = json.loads((CACHE / f"lme_s_memories_{args.model_tag}_n{MEMN}.json").read_text())
    embs = np.load(str(CACHE / f"lme_s_embed_{args.model_tag}_n{MEMN}.npy"))
    ids = json.loads((CACHE / f"lme_s_embed_{args.model_tag}_n{MEMN}.ids.json").read_text())
    flat_texts, flat_sids = [], []
    for sid, recs in mem.items():
        for r in recs:
            flat_texts.append(r["memory"]); flat_sids.append(sid)
    sid_rows = {}
    for i, sid in enumerate(ids):
        sid_rows.setdefault(sid, []).append(i)
    items = []
    for q in data:
        gold = q.get("answer_session_ids", [])
        hay = q.get("haystack_session_ids", [])
        if not q.get("answer") or not hay:
            continue
        items.append({"question": q["question"], "gold_answer": str(q["answer"]),
                      "scope_sids": hay})
    return items, embs, flat_texts, flat_sids, sid_rows


def load_locomo():
    bench = json.loads((CACHE / "locomo.json").read_text())
    raw = json.loads((CACHE / "locomo10_raw.json").read_text())
    mem = json.loads((CACHE / f"rich_memories_{args.model_tag}_LoCoMo.json").read_text())
    embs = np.load(str(CACHE / f"embed_rich_gtel_{args.model_tag}_LoCoMo.npy"))
    flat_texts, flat_sids = [], []
    for m in bench["memories"]:
        sid = m["gold_key"]
        for r in mem.get(sid, []):
            flat_texts.append(r["memory"]); flat_sids.append(sid)
    sid_rows = {}
    for i, sid in enumerate(flat_sids):
        sid_rows.setdefault(sid, []).append(i)
    # build question→gold answer + conversation scope from raw qa
    CONV = {f"c{i}": i for i in range(10)}
    items = []
    # map gold_keys (from bench queries) by question text
    q2gold = {q["question"]: q["gold_keys"] for q in bench["queries"]}
    for ci in range(10):
        for qa in raw[ci]["qa"]:
            ques = qa.get("question", "")
            if qa.get("category", 99) not in (1, 2, 3) or "answer" not in qa:
                continue
            if ques not in q2gold:
                continue
            conv = q2gold[ques][0].split("_session_")[0]
            scope = [s for s in sid_rows if s.split("_session_")[0] == conv]
            items.append({"question": ques, "gold_answer": str(qa["answer"]), "scope_sids": scope})
    return items[: args.n], embs, flat_texts, flat_sids, sid_rows


items, embs, flat_texts, flat_sids, sid_rows = (load_lme_s() if args.dataset == "lme_s" else load_locomo())
print(f"QA-accuracy eval: {args.dataset}, {len(items)} questions, gen={args.gen_model}, judge={args.judge_model}")


def retrieve_context(scope_sids, question):
    rows = [i for sid in scope_sids for i in sid_rows.get(sid, [])]
    if not rows:
        return ""
    sub_e = embs[rows]; sub_t = [flat_texts[i] for i in rows]; sub_s = [flat_sids[i] for i in rows]
    retr = MultiSignalRetriever(mem_texts=sub_t, mem_embs=sub_e, mem_session_keys=sub_s,
        mem_positions=[1]*len(rows), entity_store=[], embed_fn=emb1, reranker=reranker, mem_ids=None)
    got = retr.retrieve(question, rephrases=[question, question], top_k=args.top_k)
    # gather memory texts from the retrieved sessions
    ctx = []
    for sid in got:
        for i in sid_rows.get(sid, []):
            ctx.append(flat_texts[i])
    return "\n".join(ctx[:40])


def run_one(it):
    ctx = retrieve_context(it["scope_sids"], it["question"])
    pred = gen_answer(ctx, it["question"]) if ctx else "I don't know"
    ok = judge(it["question"], it["gold_answer"], pred)
    return ok


# Pre-warm lazy loaders (NLTK WordNet lemmatizer used by BM25) in the main thread —
# concurrent first-load across workers races and crashes otherwise.
try:
    from nltk.stem import WordNetLemmatizer
    WordNetLemmatizer().lemmatize("warming")
except Exception:
    pass
if items:
    retrieve_context(items[0]["scope_sids"], items[0]["question"])  # warm retriever path

lock = threading.Lock()
correct = 0; done = 0
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futs = [pool.submit(run_one, it) for it in items]
    for f in as_completed(futs):
        ok = f.result()
        with lock:
            correct += int(ok); done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(items)}  running QA acc={correct/done:.3f}", flush=True)

print("\n" + "=" * 60)
print(f"QA ACCURACY — {args.dataset} ({args.model_tag} extraction, {len(items)} questions)")
print("=" * 60)
print(f"  QA accuracy = {correct/len(items):.3f}  ({correct}/{len(items)} correct)")
print(f"  Generator: {args.gen_model} | Judge: {args.judge_model} | top-{args.top_k} retrieved")
print(f"  Comparable to: Mem0 LoCoMo 91.6-92.5 / LME 94.4-94.8 (QA accuracy)")
print("=" * 60)
