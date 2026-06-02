#!/usr/bin/env python3
"""
Phase 3.9 — Ingest JSONL conversation files for real-world validation.

Parses synthetic ChatGPT-style JSONL conversations into sessions, extracts
atomic facts via Claude Haiku, embeds with GTE-large, and caches everything
for the query_memories.py interactive evaluation.

Usage:
  cd X-MemoryArch/
  python3 ingest_chatgpt_export.py
  python3 ingest_chatgpt_export.py --input-dir Chats_for_evaluation --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir",  default="Chats_for_evaluation",
                    help="Directory containing *.jsonl conversation files")
parser.add_argument("--cache-dir",  default="evaluation_cache",
                    help="Output directory for sessions, facts, embeddings")
parser.add_argument("--force",      action="store_true",
                    help="Re-extract facts even if cached")
parser.add_argument("--max-facts",  type=int, default=6,
                    help="Max atomic facts per session (default 6)")
args = parser.parse_args()

ROOT       = Path(__file__).resolve().parent
INPUT_DIR  = ROOT / args.input_dir
CACHE_DIR  = ROOT / args.cache_dir
RE_DIR     = ROOT / "RetrievalEngine"

CACHE_DIR.mkdir(exist_ok=True)

SESSIONS_FILE  = CACHE_DIR / "sessions.json"
FACTS_FILE     = CACHE_DIR / "facts.json"
EMB_FILE       = CACHE_DIR / "embed_facts.npy"
FACT_IDX_FILE  = CACHE_DIR / "fact_index.json"

# ── Load API key from RetrievalEngine/.env ────────────────────────────────────
_env = RE_DIR / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not found in RetrievalEngine/.env")
    sys.exit(1)


# ── Parse JSONL files into sessions ──────────────────────────────────────────

def parse_jsonl_files(input_dir: Path) -> list[dict]:
    """Parse all *.jsonl files → list of session dicts."""
    sessions: list[dict] = []
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        print(f"ERROR: No *.jsonl files found in {input_dir}")
        sys.exit(1)

    for f in files:
        convs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        print(f"  {f.name}: {len(convs)} conversations")
        for conv in convs:
            # Build readable session content from exchanges
            lines: list[str] = []
            for ex in conv.get("exchanges", []):
                date = ex.get("date", "")
                user_text = ex.get("user", "").strip()
                asst_text = ex.get("assistant", "").strip()
                if user_text:
                    lines.append(f"[{date}] User: {user_text}")
                if asst_text:
                    lines.append(f"[{date}] Assistant: {asst_text}")
            content = "\n\n".join(lines)

            sessions.append({
                "session_id":  conv["conversation_id"],
                "title":       conv["title"],
                "scenario":    conv.get("scenario", "unknown"),
                "project":     conv.get("project", ""),
                "date_range":  conv.get("date_range", []),
                "n_exchanges": len(conv.get("exchanges", [])),
                "content":     content,
            })

    print(f"\nTotal: {len(sessions)} sessions across {len(files)} file(s)")
    return sessions


# ── Fact extraction via Claude Haiku ─────────────────────────────────────────

FACT_PROMPT = """\
You are extracting memory facts from a conversation. Extract {n} atomic facts \
that a person would want to remember later — specific decisions, names, numbers, \
deadlines, preferences, and outcomes. Skip pleasantries and filler.

Each fact must be:
- A complete, standalone sentence (makes sense without the conversation)
- Specific (include names, numbers, dates when present)
- Not duplicated

Conversation (excerpt):
{content}

Output exactly {n} facts, one per line, no numbering:"""


def extract_facts(session: dict, client, existing: dict) -> list[str]:
    """Extract atomic facts from a session using Claude Haiku. Cached."""
    sid = session["session_id"]
    if not args.force and sid in existing:
        return existing[sid]

    # Use a 1200-char excerpt from the middle of the session for diversity
    content = session["content"]
    if len(content) > 3000:
        # Take beginning + middle + end
        third = len(content) // 3
        excerpt = content[:800] + "\n...\n" + content[third:third+800] + "\n...\n" + content[-800:]
    else:
        excerpt = content

    prompt = FACT_PROMPT.format(n=args.max_facts, content=excerpt[:4000])
    n_facts = args.max_facts

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        facts = [line.strip() for line in raw.splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        return facts[:n_facts] if facts else [session["title"]]
    except Exception as e:
        print(f"    [warn] Haiku error on {sid}: {e}")
        return [session["title"]]


# ── Embedding with GTE-large ─────────────────────────────────────────────────

def embed_facts(fact_texts: list[str]) -> "np.ndarray":
    """Embed all facts with GTE-large. Returns [N, 1024] float32."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    print(f"\nLoading GTE-large for embedding {len(fact_texts):,} facts...")
    model = SentenceTransformer("thenlper/gte-large")
    embs = model.encode(
        fact_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    print(f"  Embeddings shape: {embs.shape}")
    return embs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import numpy as np

    print("\n" + "="*60)
    print("X-MemoryArch — Phase 3.9 Ingestion")
    print("="*60)

    # ── Parse sessions ────────────────────────────────────────────────────────
    print(f"\nParsing JSONL files from: {INPUT_DIR}")
    sessions = parse_jsonl_files(INPUT_DIR)

    # ── Load existing fact cache ───────────────────────────────────────────────
    existing_facts: dict[str, list[str]] = {}
    if FACTS_FILE.exists() and not args.force:
        existing_facts = json.loads(FACTS_FILE.read_text())
        cached_n = sum(1 for s in sessions if s["session_id"] in existing_facts)
        print(f"\nFact cache: {cached_n}/{len(sessions)} sessions already cached")

    # ── Extract facts ─────────────────────────────────────────────────────────
    needs_extraction = [s for s in sessions if s["session_id"] not in existing_facts]

    if needs_extraction:
        print(f"\nExtracting facts for {len(needs_extraction)} sessions via Claude Haiku...")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            print("ERROR: anthropic package not installed. Run: pip install anthropic")
            sys.exit(1)

        for i, sess in enumerate(needs_extraction):
            print(f"  [{i+1}/{len(needs_extraction)}] {sess['session_id'][:50]}...", end="", flush=True)
            t0 = time.monotonic()
            facts = extract_facts(sess, client, existing_facts)
            existing_facts[sess["session_id"]] = facts
            elapsed = time.monotonic() - t0
            print(f" → {len(facts)} facts  [{elapsed:.1f}s]")
            time.sleep(0.1)  # light rate limiting

        FACTS_FILE.write_text(json.dumps(existing_facts, indent=2))
        print(f"\nSaved facts: {FACTS_FILE}")
    else:
        print("All facts cached — skipping extraction.")

    # ── Build flat fact index ─────────────────────────────────────────────────
    fact_index: list[dict] = []  # [{fact, session_id, scenario, title, date_range}]
    for sess in sessions:
        facts = existing_facts.get(sess["session_id"], [])
        for fact in facts:
            fact_index.append({
                "fact":       fact,
                "session_id": sess["session_id"],
                "scenario":   sess["scenario"],
                "title":      sess["title"],
                "date_range": sess["date_range"],
                "project":    sess["project"],
            })

    print(f"\nTotal facts in index: {len(fact_index):,}")
    print(f"  Per session avg: {len(fact_index)/len(sessions):.1f}")

    # ── Embed facts ───────────────────────────────────────────────────────────
    if EMB_FILE.exists() and not args.force:
        existing_emb = np.load(str(EMB_FILE))
        if existing_emb.shape[0] == len(fact_index):
            print(f"\nEmbedding cache hit: {existing_emb.shape} — skipping embedding.")
            embs = existing_emb
        else:
            print(f"\nEmbedding count mismatch ({existing_emb.shape[0]} vs {len(fact_index)}) — re-embedding.")
            embs = embed_facts([r["fact"] for r in fact_index])
            np.save(str(EMB_FILE), embs)
    else:
        embs = embed_facts([r["fact"] for r in fact_index])
        np.save(str(EMB_FILE), embs)
        print(f"Saved embeddings: {EMB_FILE}")

    # ── Save sessions and fact index ──────────────────────────────────────────
    # Strip content from sessions (large, already in facts)
    sessions_meta = [{k: v for k, v in s.items() if k != "content"} for s in sessions]
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2))
    FACT_IDX_FILE.write_text(json.dumps(fact_index, indent=2))

    print(f"\nSaved sessions metadata: {SESSIONS_FILE}")
    print(f"Saved fact index:        {FACT_IDX_FILE}")
    print(f"Saved embeddings:        {EMB_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    scenario_counts = Counter(s["scenario"] for s in sessions)
    print("\n" + "="*60)
    print("Ingestion complete!")
    print("="*60)
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count} sessions")
    print(f"  Total facts: {len(fact_index):,}")
    print(f"\nRun next: python3 query_memories.py")


if __name__ == "__main__":
    main()
