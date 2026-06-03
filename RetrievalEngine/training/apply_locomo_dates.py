#!/usr/bin/env python3
"""
Phase 5.5 — Apply actual LoCoMo session dates to already-extracted memories.

The extraction ran with session-position markers ("As of session N").
This script replaces them with actual ISO dates from the raw LoCoMo dataset.

Cost: $0 (string replacement + local re-embedding, no API calls).
Run after build_rich_memories.py --dataset locomo completes.

Usage:
    python3 training/apply_locomo_dates.py
    python3 training/apply_locomo_dates.py --dry-run   # preview without saving
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Preview replacements without saving")
parser.add_argument("--model-tag", default="sonnet")
args = parser.parse_args()

BENCH_DIR = Path(__file__).resolve().parent.parent
CACHE     = BENCH_DIR / "benchmark_cache"
sys.path.insert(0, str(BENCH_DIR))

# ── Benchmark conversation → raw conversation index mapping ─────────────────
# Determined by matching speaker names between locomo.json and locomo10_raw.json
CONV_MAP = {
    "c0": 0,   # Caroline/Melanie → conv-26
    "c1": 1,   # Gina/Jon        → conv-30
    "c2": 2,   # Maria/John      → conv-41
    "c3": 3,   # Nate/Joanna     → conv-42
    "c4": 4,   # John/Tim        → conv-43
    "c5": 5,   # Audrey/Andrew   → conv-44
    "c6": 6,   # John/James      → conv-47
    "c7": 7,   # Deborah/Jolene  → conv-48
    "c8": 8,   # Sam/Evan        → conv-49
    "c9": 9,   # Calvin/Dave     → conv-50
}


def _parse_locomo_date(raw_str: str) -> str:
    """Parse '1:56 pm on 8 May, 2023' → 'May 08, 2023'."""
    raw_str = raw_str.strip()
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
        try:
            dt = datetime.strptime(raw_str, fmt)
            return dt.strftime("%B %d, %Y")
        except ValueError:
            continue
    # fallback: return as-is if unparseable
    return raw_str


def build_date_map() -> dict[str, str]:
    """Build {session_id → formatted_date} for all LoCoMo sessions."""
    raw_path = CACHE / "locomo10_raw.json"
    if not raw_path.exists():
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        print(f"Downloading raw LoCoMo dataset...")
        urllib.request.urlretrieve(url, raw_path)

    raw_data = json.loads(raw_path.read_text())
    date_map: dict[str, str] = {}

    for conv_key, raw_idx in CONV_MAP.items():
        conv = raw_data[raw_idx]
        convo = conv["conversation"]
        for key, val in convo.items():
            if key.endswith("_date_time") and isinstance(val, str):
                # key: "session_1_date_time" → sess_num = 1
                parts = key.split("_")  # ["session", "1", "date", "time"]
                try:
                    sess_num = int(parts[1])
                    session_id = f"{conv_key}_session_{sess_num}"
                    date_map[session_id] = _parse_locomo_date(val)
                except (ValueError, IndexError):
                    pass

    return date_map


def apply_dates_to_memory(memory_text: str, session_id: str, date: str) -> str:
    """Replace session-position temporal markers with actual date in memory text."""
    sess_pos = session_id.split("_session_")[-1]  # "1", "10", etc.

    replacements = [
        # State memories
        (f"As of session {sess_pos},", f"As of {date},"),
        (f"as of session {sess_pos},", f"as of {date},"),
        # Episodic memories
        (f"During session {sess_pos},", f"On {date},"),
        (f"during session {sess_pos},", f"on {date},"),
        # Plan memories
        (f"mentioned in session {sess_pos})", f"mentioned on {date})"),
        (f"session {sess_pos})", f"{date})"),
        # Generic session reference
        (f"session {sess_pos}", f"{date}"),
    ]

    result = memory_text
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def main() -> None:
    mem_path = CACHE / f"rich_memories_{args.model_tag}_LoCoMo.json"
    if not mem_path.exists():
        print(f"ERROR: {mem_path} not found. Run build_rich_memories.py first.")
        sys.exit(1)

    print("Building date map from raw LoCoMo dataset...")
    date_map = build_date_map()
    print(f"Loaded {len(date_map)} session dates")
    print("Sample:", list(date_map.items())[:3])

    all_memories: dict[str, list[dict]] = json.loads(mem_path.read_text())

    total_replaced = 0
    total_memories = 0
    sessions_with_dates = 0

    updated: dict[str, list[dict]] = {}
    for session_id, records in all_memories.items():
        date = date_map.get(session_id)
        new_records = []
        for rec in records:
            total_memories += 1
            old_text = rec["memory"]
            if date:
                new_text = apply_dates_to_memory(old_text, session_id, date)
                if new_text != old_text:
                    total_replaced += 1
            else:
                new_text = old_text
            new_records.append({**rec, "memory": new_text, "session_date": date})
        updated[session_id] = new_records
        if date:
            sessions_with_dates += 1

    print(f"\nResults:")
    print(f"  Sessions with actual dates: {sessions_with_dates}/{len(all_memories)}")
    print(f"  Memories updated:           {total_replaced}/{total_memories}")

    # Show a few examples
    print("\nExamples of replacements:")
    shown = 0
    for session_id, records in updated.items():
        if shown >= 3:
            break
        date = date_map.get(session_id)
        if not date:
            continue
        old_recs = all_memories[session_id]
        for old_rec, new_rec in zip(old_recs, records):
            if old_rec["memory"] != new_rec["memory"]:
                print(f"\n  [{session_id}] date={date}")
                print(f"  BEFORE: {old_rec['memory'][:120]}")
                print(f"  AFTER:  {new_rec['memory'][:120]}")
                shown += 1
                break

    if args.dry_run:
        print("\n[DRY RUN] No files saved.")
        return

    # Save updated memories
    mem_path.write_text(json.dumps(updated))
    print(f"\nSaved updated memories to: {mem_path}")
    print("Next: re-embed with  python3 training/build_rich_memories.py --model sonnet --dataset locomo")
    print("      (will skip extraction, only re-embed since memories are updated)")


if __name__ == "__main__":
    main()
