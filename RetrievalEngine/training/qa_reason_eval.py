#!/usr/bin/env python3
"""
QA-accuracy eval WITH a reasoning layer — the system that turns retrieved memory
into a correct final answer.

Pipeline per question:
  1. RETRIEVE top-k sessions with our memory engine (scoped to the standard protocol)
  2. Build context from the RAW text of those retrieved sessions (full detail)
  3. ANSWER with a reasoning prompt: anchored to the question's reference date,
     chain-of-thought, explicit temporal arithmetic + multi-hop aggregation +
     calibrated abstention, ending in a single short ANSWER line
  4. JUDGE the answer vs gold (GPT-4o)

This is the layer the diagnostic said was missing. Baseline (no reasoning) was ~0.33.

Usage:
  python3 training/qa_reason_eval.py --dataset lme_s --model-tag gpt4omini --answerer gpt-4o-mini --n 100
  python3 training/qa_reason_eval.py --dataset lme_s --model-tag gpt4omini --answerer gpt-4o      --n 500
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["lme_s", "locomo"], default="lme_s")
ap.add_argument("--model-tag", default="gpt4omini", help="extraction tag for the memory index")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--top-k", type=int, default=5, help="retrieved sessions fed to the answerer")
ap.add_argument("--answerer", default="gpt-4o-mini")
ap.add_argument("--judge-model", default="gpt-4o")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--no-reason", action="store_true", help="ablation: simple answerer, no reasoning")
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

REASON_PROMPT = """You answer a question about a user's own past conversations with an AI assistant. You are given excerpts from those conversations, each tagged with its date. Today's date is {today}.

Conversation excerpts:
{context}

Question: {question}

Reason carefully before answering:
- Find the specific fact(s) in the excerpts that bear on the question.
- DATES / DURATIONS ("how many days ago", "how long since", "between X and Y", "most recent"): locate the exact date(s) in the excerpts and compute the difference using today ({today}) as the reference. Count the days carefully and inclusively as the question implies.
- COUNTS / TOTALS ("how many", "total", "in total"): find EVERY relevant item across all excerpts and add them up. Do not stop at the first one.
- UPDATES ("currently", "most recently", "switch"): prefer the latest-dated fact over earlier ones.
- If the excerpts genuinely do not contain the information, do not guess — answer that it was not mentioned.

Think step by step, then end with exactly one final line:
ANSWER: <the shortest possible answer: a name, number, date, or short phrase; or "not mentioned">"""

SIMPLE_PROMPT = """Answer the question using ONLY these conversation excerpts. Be concise. If not present, say "not mentioned".

{context}

Question: {question}
ANSWER:"""

JUDGE_PROMPT = """Grade whether the predicted answer is correct given the gold answer. Be lenient about phrasing, formatting and extra words — judge only whether it conveys the same factual information. For "not mentioned"/abstention gold answers, the prediction is correct only if it also declines.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Reply with exactly one word: CORRECT or INCORRECT."""


import time as _time
def _chat(model, prompt, max_tokens, _tries=8):
    for attempt in range(_tries):
        try:
            r = client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=0,
                                               messages=[{"role": "user", "content": prompt}])
            return (r.choices[0].message.content or "").strip()
        except openai.RateLimitError:
            _time.sleep(min(2 ** attempt, 30))  # exponential backoff for TPM/RPM limits
        except Exception:
            _time.sleep(2)
    return ""  # gave up after retries — counts as a miss, not a crash


_CTX_CAP = 28000  # chars of retrieved raw session text fed to the answerer (~7k tokens)
def answer(context, question, today):
    if args.no_reason:
        out = _chat(args.answerer, SIMPLE_PROMPT.format(context=context[:_CTX_CAP], question=question), 100)
    else:
        out = _chat(args.answerer, REASON_PROMPT.format(today=today, context=context[:_CTX_CAP], question=question), 700)
    m = re.search(r"ANSWER:\s*(.+)", out, re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip() if m else out).split("\n")[0].strip()


def judge(question, gold, pred):
    out = _chat(args.judge_model, JUDGE_PROMPT.format(question=question, gold=gold, pred=pred), 5).upper()
    if "INCORRECT" in out:
        return False
    return "CORRECT" in out


# ── Load memory index + per-question scope + raw session text + question date ──
def build_index(mem_path, emb_path, ids_path):
    mem = json.loads(mem_path.read_text())
    embs = np.load(str(emb_path))
    ids = json.loads(ids_path.read_text())
    flat_texts, flat_sids = [], []
    for sid, recs in mem.items():
        for r in recs:
            flat_texts.append(r["memory"]); flat_sids.append(sid)
    assert embs.shape[0] == len(flat_texts), f"{embs.shape[0]} vs {len(flat_texts)}"
    sid_rows = {}
    for i, sid in enumerate(flat_sids):
        sid_rows.setdefault(sid, []).append(i)
    return embs, flat_texts, flat_sids, sid_rows


def load_lme_s():
    MEMN = 500
    data = json.loads((CACHE / "longmemeval_s_cleaned.json").read_text())[: args.n]
    embs, ft, fs, sr = build_index(
        CACHE / f"lme_s_memories_{args.model_tag}_n{MEMN}.json",
        CACHE / f"lme_s_embed_{args.model_tag}_n{MEMN}.npy",
        CACHE / f"lme_s_embed_{args.model_tag}_n{MEMN}.ids.json")
    items = []
    for q in data:
        if not q.get("answer") or not q.get("haystack_session_ids"):
            continue
        raw = {sid: "\n".join(f"{t.get('role')}: {t.get('content','')}" for t in turns)
               for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"])}
        items.append({"question": q["question"], "gold": str(q["answer"]),
                      "scope": q["haystack_session_ids"], "raw": raw,
                      "today": q.get("question_date", "unknown")})
    return items, embs, ft, fs, sr


items, embs, ft, fs, sr = load_lme_s()
print(f"Reasoning QA: {args.dataset}, {len(items)} q, answerer={args.answerer}, reason={not args.no_reason}, top-{args.top_k}")

# pre-warm NLTK (BM25 lemmatizer) before threads
try:
    from nltk.stem import WordNetLemmatizer; WordNetLemmatizer().lemmatize("x")
except Exception:
    pass


def retrieve_context(it):
    rows = [i for sid in it["scope"] for i in sr.get(sid, [])]
    if not rows:
        return ""
    R = MultiSignalRetriever(mem_texts=[ft[i] for i in rows], mem_embs=embs[rows],
        mem_session_keys=[fs[i] for i in rows], mem_positions=[1]*len(rows),
        entity_store=[], embed_fn=emb1, reranker=reranker, mem_ids=None)
    got = R.retrieve(it["question"], rephrases=[it["question"]]*2, top_k=args.top_k)
    return "\n\n".join(it["raw"].get(sid, "") for sid in got)


if items:
    retrieve_context(items[0])  # warm

def run_one(it):
    ctx = retrieve_context(it)
    pred = answer(ctx, it["question"], it["today"]) if ctx else "not mentioned"
    return judge(it["question"], it["gold"], pred)

lock = threading.Lock(); correct = 0; done = 0
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    for f in as_completed([pool.submit(run_one, it) for it in items]):
        ok = f.result()
        with lock:
            correct += int(ok); done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(items)}  QA acc={correct/done:.3f}", flush=True)

print("\n" + "=" * 60)
print(f"QA ACCURACY (reasoning={not args.no_reason}) — {args.dataset}")
print(f"  answerer={args.answerer}  judge={args.judge_model}  top-{args.top_k}  n={len(items)}")
print(f"  QA accuracy = {correct/len(items):.3f}  ({correct}/{len(items)})")
print("=" * 60)
