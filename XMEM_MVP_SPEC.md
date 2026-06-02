# Xmem MVP — Complete Build Specification

## What You Are Building

**Xmem** is a passive memory layer that wraps any AI CLI terminal. The user keeps using Claude, ChatGPT, and Gemini exactly as they always have — Xmem sits in between, captures what the user chooses to save, and silently injects relevant context when they switch tools.

```
Normal usage without Xmem:    user → claude  (isolated, no memory, no handoff)
Normal usage with Xmem:        user → xmem --claude → claude  (identical UX + memory)
```

**The killer feature — cross-AI handoff:**
User opens `xmem --chatgpt` in a new terminal and types "let's fix the frontend issue".
Xmem retrieves relevant memories from their Claude session and silently prepends them to ChatGPT's prompt.
ChatGPT responds as if it was already in the conversation.
The user typed one line. It felt like magic.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Xmem CLI  (~/Desktop/Xmem/)   ← NEW REPO, this session     │
│                                                              │
│  xmem --claude  →  spawns `claude` subprocess               │
│                     pipes all stdin/stdout transparently     │
│                     detects exchange boundaries              │
│                     asks consent after each response         │
│                     on save: calls xmem_bridge.write()       │
│                                                              │
│  xmem --chatgpt →  spawns `chatgpt` subprocess              │
│                     before forwarding user prompt:           │
│                     calls xmem_bridge.retrieve_fast()        │
│                     silently prepends context packet         │
│                     user sees only ChatGPT's response        │
└──────────────────────────────────────────────────────────────┘
         ↕ imports xmem_bridge.py (single clean interface)
┌──────────────────────────────────────────────────────────────┐
│  xmem_bridge.py  (lives in RetrievalEngine/)                │
│  THE ONLY INTERFACE XMEM TOUCHES — 4 functions              │
│                                                              │
│  write(content, project_id, mode)  → extract + store        │
│  retrieve_fast(query, project_id)  → A7r, 28ms              │
│  retrieve_quality(query, project_id) → A4mvr, 200ms         │
│  build_handoff_packet(project_id, query) → context string   │
└──────────────────────────────────────────────────────────────┘
         ↕ delegates internally to:
┌─────────────────────────────────┐  ┌──────────────────────────┐
│  project-memory-core            │  │  RetrievalEngine          │
│  (write path)                   │  │  (read path)              │
│                                 │  │                           │
│  CloudLLMExtractionBackend      │  │  A4mvr: 0.822 R@5         │
│  LocalChunkBackend              │  │  A7r:   0.800 R@5 / 28ms  │
│  OllamaExtractionBackend        │  │  ms-marco reranker        │
│  GTE-large embedder             │  │  GTE-large embedder       │
│  SQLite schema (Phase 1)        │  │  SQLiteExactBackend       │
│  1058 tests passing             │  │  BM25 (rank_bm25)         │
└─────────────────────────────────┘  └──────────────────────────┘
         ↕ shared file                        ↕ reads from
              ~/.xmem/memories.db  ←──────────────┘
```

**Critical rule for the CLI session:**
Xmem ONLY touches `xmem_bridge.py`. It never imports from `benchmark_4approaches.py`,
`app/services/retrieval/`, or any other internal module. The bridge is the contract.
When 3.7 (knowledge graph) and 3.8 (agent loops) land, the bridge internals improve —
the CLI changes nothing.

---

## The Bridge: xmem_bridge.py

This file lives at:
```
~/Desktop/X-MemoryArch/RetrievalEngine/xmem_bridge.py
```

It is the **only interface between Xmem CLI and the memory system.** Four functions. Nothing else.

### Function 1 — write()

```python
def write(
    content: str,         # "User: ...\n\nAssistant: ..."
    project_id: str,      # e.g. "auth-fix" or "claude_20260601_143022"
    mode: str = "cloud",  # "local" | "ollama" | "cloud" (see Modes section)
) -> None:
    """
    Extract facts from an exchange and store in memory.

    local:  split content into paragraph chunks → embed with GTE-large → store
    ollama: Ollama LLM extracts facts → embed → store
    cloud:  Claude Haiku extracts 6 atomic facts → embed → store (best quality)

    All modes write to ~/.xmem/memories.db via project-memory-core.
    All modes produce GTE-large embeddings for retrieval.
    Cloud mode additionally produces a fact index used by retrieve_fast().
    """
```

### Function 2 — retrieve_fast()

```python
def retrieve_fast(
    query: str,
    project_id: str,
    top_k: int = 5,
    mode: str = "cloud",
) -> list[dict]:
    """
    Fast retrieval for real-time query injection (before every user message).
    Target latency: < 50ms.

    cloud/ollama mode:  A7r — BM25 + Dense RRF on extracted facts, 28ms, R@5=0.800
    local mode:         A4r — Dense cosine only on chunks, 41ms, R@5=0.748

    Returns: [{"fact": str, "session_id": str, "title": str, "score": float}, ...]

    Use this path for: pre-query context injection on every user turn.
    Do NOT use A6r here — it needs Claude Haiku per query, too expensive for real-time.
    """
```

### Function 3 — retrieve_quality()

```python
def retrieve_quality(
    query: str,
    project_id: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Quality retrieval for handoff packets and explicit search.
    Target latency: 200ms (one-time call, user is not waiting in a flow).

    ALL modes:  A4mvr — chunk max-sim + ms-marco reranker, 200ms, R@5=0.822

    A4mvr uses RAW SESSION CHUNKS (not extracted facts) — works in every mode
    including fully local with zero API calls ever.

    Returns: [{"chunk": str, "session_id": str, "title": str, "score": float}, ...]

    Use this path for: explicit handoff, xmem --search, session summaries.
    """
```

### Function 4 — build_handoff_packet()

```python
def build_handoff_packet(
    project_id: str,
    context_query: str = None,  # if None, retrieves recent memories broadly
    max_tokens: int = 500,
) -> str:
    """
    Build a formatted context string ready to prepend to any AI's system prompt.
    Uses retrieve_quality() internally (A4mvr).

    Output format:
    [Xmem Context — from Claude session, 14:30]
    • JWT refresh flow has a race condition in token validation
    • Backend fix: added mutex lock in refresh endpoint
    • Frontend breaking: token format changed, API response shape updated
    [End Context — 3 memories from project "auth-fix"]

    Capped at max_tokens. Never dumps raw conversation — only extracted facts/chunks.
    """
```

---

## The Four Operational Modes

This is the key architectural concept. Different users have different API budgets and
privacy requirements. Xmem supports all four modes — the same CLI command, different
`--mode` flag (or config setting).

### Mode 1 — Fully Local (`--mode local`)

```
Zero API calls. Works offline. No API key required.

Write path:
  content → split into paragraph chunks (on \n\n)
           → embed each chunk with GTE-large (local, MPS/CPU)
           → store chunks + embeddings in ~/.xmem/memories.db

Retrieve fast:   A4r  — dense cosine similarity on chunks  — 41ms,  R@5=0.748
Retrieve quality: A4mvr — chunk max-sim + ms-marco reranker — 200ms, R@5=0.822

Cost:    $0 forever
Privacy: data never leaves device
Tradeoff: retrieval uses whole paragraphs, not atomic facts
          (slightly less precise recall on short specific queries)
```

### Mode 2 — Local + Ollama (`--mode ollama`)

```
Uses local Ollama for fact extraction. Zero cloud cost.
Requires Ollama running locally with a capable model (llama3, mistral, etc.)

Write path:
  content → Ollama extracts facts (quality ≈ model quality)
           → embed facts with GTE-large (local)
           → store facts + embeddings in ~/.xmem/memories.db

Retrieve fast:    A7r — BM25+Dense on Ollama-extracted facts — 28ms, R@5≈0.78*
Retrieve quality: A4mvr — chunk max-sim + reranker — 200ms, R@5=0.822

Cost:    $0 (compute cost only)
Privacy: data never leaves device
Tradeoff: fact quality depends on Ollama model; smaller models miss nuance
*estimated; Ollama fact quality varies vs Claude Haiku
```

### Mode 3 — Cloud (`--mode cloud`) ← Recommended default

```
Claude Haiku extracts facts (best quality). Everything else is local.
Requires: ANTHROPIC_API_KEY in ~/.xmem/.env

Write path:
  content → Claude Haiku extracts 6 atomic facts (one-time per saved exchange)
           → embed facts with GTE-large (local)
           → store facts + chunks + embeddings in ~/.xmem/memories.db

Retrieve fast:    A7r  — BM25+Dense on Claude-extracted facts — 28ms,  R@5=0.800
Retrieve quality: A4mvr — chunk max-sim + ms-marco reranker  — 200ms, R@5=0.822

Cost:    ~$0.001 per saved exchange (Claude Haiku fact extraction, one-time)
         $0 for ALL retrieval (no API calls at query time)
Privacy: exchange content sent to Anthropic at write time only
Tradeoff: tiny cost per save; Claude-extracted facts are highest quality
```

### Mode 4 — Hybrid (`--mode hybrid`)

```
Same as Cloud mode but with auto-fallback to local if API unavailable.
Attempts Claude Haiku extraction; falls back to local chunking on failure.

Cost:    Same as cloud when available, $0 when offline
Use for: laptop users who switch between on/offline
```

### Why A6r is NOT exposed in the bridge

A6r (Multi-Query RRF + Reranker, R@5=0.808) needs Claude Haiku to generate 3 query
rephrases PER QUERY at retrieval time. In a CLI context:
- retrieve_fast() is called before every single user message
- That would be 3 Claude Haiku API calls per message the user types
- Cost: ~$0.003/message → unacceptable for an interactive tool
- Latency: 62ms base + API round-trip → 200-500ms → noticeable terminal lag

A6r is the right approach for offline batch search (xmem --search "JWT issue" where
latency and cost are acceptable). It is NOT right for real-time injection.
The bridge does not expose it. CLI session should not implement it.

---

## SQLite: The Two Layers (Common Confusion — Read This)

There are two completely different SQLite uses in the architecture. They happen to
share the same file but serve different purposes:

```
~/.xmem/memories.db
├── [Relational layer — Phase 1 / project-memory-core]
│   Tables: projects, memories, entities, memory_links, retrieval_runs
│   Used for: metadata, text search (FTS5 BM25), write path, schema
│   Retrieval quality with this layer alone: A1 = 0.025 R@5 (pure keyword match)
│   This is NOT the layer you retrieve from in production
│
└── [Vector layer — Phase 2 / RetrievalEngine / SQLiteExactBackend]
    Stored in: memories.embedding field (1024-dim float32 binary blobs)
    Used for: cosine similarity search, A4mvr, A5r, A7r
    Retrieval quality: A4mvr = 0.822 R@5, A7r = 0.800 R@5
    This IS the layer powering quality retrieval
```

**A4mvr's 0.822 R@5 comes from the vector layer, not the relational layer.**
The relational layer (A1 at 0.025) is only used directly by the rule-based BM25 approach,
which is a diagnostic baseline, not a production path.

**Why SQLite is sufficient for MVP (not Qdrant/FAISS/Chroma):**
- SQLiteExactBackend does exact cosine similarity over all embeddings — correct and fast for <100K memories
- Zero dependencies (no Docker, no server, no cloud account)
- VectorBackend interface is already abstracted — swap to Qdrant for Phase 4.3 (multi-tenant cloud) with one config line
- Personal CLI use case: a developer accumulates ~10-50K memories over years of use — SQLite handles this trivially

---

## How the Two Repos Communicate

**Python imports + one shared SQLite file. No network, no HTTP, no IPC for MVP.**

```bash
# In Xmem's virtualenv — install both as editable packages:
pip install -e ~/Desktop/X-MemoryArch/project-memory-core
pip install -e ~/Desktop/X-MemoryArch/RetrievalEngine

# Set shared DB path before any imports:
export XMEM_DB_PATH=~/.xmem/memories.db
```

```python
# xmem_bridge.py only — no direct imports elsewhere in Xmem

# Bridge imports write path:
from app.services.extraction.cloud_llm import CloudLLMExtractionBackend
from app.services.extraction.local import LocalChunkBackend
from app.crud import create_project_if_not_exists

# Bridge imports read path (internal to bridge, hidden from Xmem CLI):
import sys; sys.path.insert(0, str(RE_DIR))
from app.services.vector_backends.sqlite_exact import SQLiteExactBackend
from app.services.retrieval.reranker import get_default_reranker
from rank_bm25 import BM25Okapi
```

The Xmem CLI (`wrapper.py`, `handoff.py`, `consent.py`) imports ONLY:
```python
from xmem_bridge import write, retrieve_fast, retrieve_quality, build_handoff_packet
```

Nothing else from either repo. Ever.

---

## Repository Structure

```
~/Desktop/Xmem/                    ← NEW repo (CLI session builds this)
├── xmem/
│   ├── __init__.py
│   ├── cli.py                     ← entry point (Click or Typer)
│   ├── session.py                 ← session ID, state, project detection
│   ├── wrapper.py                 ← pty subprocess + stdin/stdout pipe
│   ├── boundary.py                ← exchange boundary detection per provider
│   ├── consent.py                 ← [Y/n] prompt after each exchange
│   ├── injector.py                ← calls retrieve_fast(), prepends context
│   ├── handoff.py                 ← calls build_handoff_packet(), launches new provider
│   └── providers.py               ← provider registry
├── tests/
│   ├── test_boundary.py           ← test boundary detection for each provider
│   ├── test_consent.py
│   └── test_injection.py
├── requirements.txt
├── setup.py
└── .env.example                   ← ANTHROPIC_API_KEY, XMEM_DB_PATH, XMEM_MODE

~/Desktop/X-MemoryArch/
└── RetrievalEngine/
    └── xmem_bridge.py             ← THE BRIDGE (built in this repo, not Xmem)

~/.xmem/                           ← runtime data, not in any repo
├── memories.db
├── .env                           ← user's API keys
└── config.json                    ← mode, consent_default, project_default
```

---

## Session Identity

Claude/ChatGPT/Gemini CLIs do not expose session IDs in terminal output. Xmem generates its own:

```python
session_id = f"{provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
# e.g.: claude_20260601_143022_12345
```

User can optionally name the project:
```bash
xmem --claude --project "backend-auth-fix"
# All memories from this session tagged under project "backend-auth-fix"
# xmem --chatgpt --project "backend-auth-fix" retrieves from the same project
```

If no `--project` given:
- Xmem uses a default `"inbox"` project
- After first 3 exchanges, auto-detects a 3-word topic name via Claude Haiku (cloud mode)
  or keyword extraction (local mode)
- User can rename: `xmem --rename "auth-fix"` at any time

**How handoff finds the right memories:**
- All memories have `project_id` in the Phase 1 schema (already built)
- `retrieve_fast()` and `retrieve_quality()` both filter by `project_id`
- Handoff retrieves from the same `project_id` regardless of which AI wrote the memories

---

## Core Technical Problem: Exchange Boundary Detection

**Solve this first. Everything else depends on it.**

Xmem needs to know when the AI CLI has finished its response before showing the consent
prompt. Each CLI has different output patterns.

**Strategy: sentinel pattern matching on stdout stream**

```python
# boundary.py
PROVIDER_BOUNDARIES = {
    "claude": {
        "sentinels": [r"^> $", r"^Human:"],
        "idle_timeout_ms": 500,   # if no output for 500ms, assume done
    },
    "chatgpt": {
        "sentinels": [r"^You: $", r"^\[ChatGPT\]"],
        "idle_timeout_ms": 500,
    },
    "gemini": {
        "sentinels": [r"^> $", r"^User: $", r"^\[Gemini\]"],
        "idle_timeout_ms": 500,
    },
}
# Fallback: if idle_timeout_ms passes with no new output, boundary reached.
# Must be calibrated empirically — run each CLI and watch output.
```

**Implementation approach:**
```python
import pty, os, select, re

master_fd, slave_fd = pty.openpty()
proc = subprocess.Popen([cmd], stdin=slave_fd, stdout=slave_fd, stderr=slave_fd)

buffer = ""
last_output_time = time.time()

while proc.poll() is None:
    r, _, _ = select.select([master_fd], [], [], 0.05)
    if r:
        chunk = os.read(master_fd, 1024).decode(errors="replace")
        os.write(sys.stdout.fileno(), chunk.encode())  # pass through to user
        buffer += chunk
        last_output_time = time.time()
    elif time.time() - last_output_time > IDLE_TIMEOUT:
        if buffer.strip():  # non-empty exchange
            on_exchange_complete(buffer)
            buffer = ""
```

---

## The Full Memory Pipeline

### 1. Capture + Boundary Detection (wrapper.py)

Spawn AI subprocess via pty. Stream stdout through to user's terminal unchanged.
Accumulate buffer. Detect exchange end via sentinel or idle timeout.

### 2. Consent (consent.py)

```
[Xmem] Save this exchange? [Y/n]:
```
- Default: N (skip) — user must opt in. Privacy first.
- User presses Y → triggers write pipeline
- User presses Enter or N → continues without saving
- Optional: `xmem --consent always` to save everything automatically

### 3. Write (injector.py → xmem_bridge.write())

```python
content = f"User: {user_turn}\n\nAssistant: {assistant_turn}"
write(content, project_id=session.project_id, mode=config.mode)
# Bridge handles: noise gate → chunking → extraction → embedding → SQLite
```

### 4. Inject (injector.py → xmem_bridge.retrieve_fast())

```python
# Called BEFORE forwarding every user message to the AI subprocess
def inject_context(user_input: str, session) -> str:
    memories = retrieve_fast(user_input, session.project_id, mode=config.mode)
    if not memories:
        return user_input
    context = format_context(memories)  # < 300 tokens
    return f"[Context]\n{context}\n[/Context]\n\n{user_input}"
    # User never sees this — it's injected into the subprocess stdin
```

### 5. Handoff (handoff.py → xmem_bridge.build_handoff_packet())

```python
def execute_handoff(target_provider: str, session):
    packet = build_handoff_packet(session.project_id)
    # Launch new provider with packet pre-loaded as first system message
    new_session = Session(provider=target_provider, project_id=session.project_id)
    new_session.prepend_system(packet)
    spawn_provider(target_provider, new_session)
```

---

## Handoff Flow (End-to-End Example)

```bash
# Terminal 1:
xmem --claude --project "auth-fix"

User: explain the JWT race condition we discussed
Claude: The refresh token flow fails because two requests can simultaneously...
[Xmem] Save? [Y/n]: Y  ← user saves

User: what fix did we apply?
Claude: We added a mutex lock in the refresh endpoint...
[Xmem] Save? [Y/n]: Y  ← user saves

# Something breaks in frontend. User opens Terminal 2:
xmem --chatgpt --project "auth-fix"

# ChatGPT launches. User types:
User: let's fix the frontend issue

# What Xmem does (invisible):
# 1. retrieve_fast("let's fix the frontend issue", "auth-fix") → 3 memories
# 2. Builds: "[Context]\n• JWT race condition fixed with mutex lock...\n[/Context]\n\nlet's fix the frontend issue"
# 3. Sends this to chatgpt subprocess stdin

# What user sees:
ChatGPT: Sure — given the mutex lock you added to the JWT refresh endpoint,
         the frontend is likely seeing a timing issue where the token hasn't
         refreshed before the next API call. Here's how to fix it...

# User never saw the context injection. It felt like ChatGPT already knew.
```

---

## Provider Registry

```python
# providers.py
PROVIDERS = {
    "claude": {
        "command": "claude",          # CLI command to spawn
        "boundary_patterns": [r"^> $", r"^Human:"],
        "idle_timeout_ms": 500,
        "inject_method": "stdin",     # prepend to stdin before forwarding
    },
    "chatgpt": {
        "command": "chatgpt",
        "boundary_patterns": [r"^You: $"],
        "idle_timeout_ms": 500,
        "inject_method": "stdin",
    },
    "gemini": {
        "command": "gemini",
        "boundary_patterns": [r"^> $", r"^User: $"],
        "idle_timeout_ms": 500,
        "inject_method": "stdin",
    },
}
# Adding new AI: add one dict entry. Zero code change to wrapper or bridge.
```

---

## CLI Commands

```bash
# Start sessions:
xmem --claude                              # default project "inbox"
xmem --claude --project "auth-fix"         # named project
xmem --chatgpt --project "auth-fix"        # same project, different AI
xmem --claude --mode local                 # fully local, no API keys

# Handoff:
xmem --handoff chatgpt                     # build packet, launch chatgpt same project
xmem --handoff gemini --project "auth-fix" # explicit project

# Memory management:
xmem --list                                # recent sessions + project names
xmem --search "JWT race condition"         # search memories (uses retrieve_quality)
xmem --forget <session-id>                # delete session memories

# Config:
xmem --config set mode=cloud               # cloud | local | ollama | hybrid
xmem --config set consent_default=ask     # ask | always | never
```

---

## What NOT to Build in MVP

- No browser extension — next layer
- No IDE integration — next layer
- No Qdrant/Chroma — SQLite sufficient; backend abstracted, swap in Phase 4.3
- No cloud sync — Phase 4.3
- No multi-user — Phase 4.3
- No A6r in real-time injection — 3 API calls per message, wrong tradeoff
- No UI beyond terminal — terminal IS the UI

---

## Environment Setup

```bash
# 1. Create Xmem repo
mkdir ~/Desktop/Xmem && cd ~/Desktop/Xmem
git init
python3 -m venv .venv && source .venv/bin/activate

# 2. Install both memory repos as editable packages (no code duplication)
pip install -e ~/Desktop/X-MemoryArch/project-memory-core
pip install -e ~/Desktop/X-MemoryArch/RetrievalEngine

# 3. Install Xmem-specific dependencies
pip install click rich pyte rank-bm25 anthropic

# 4. Runtime config
mkdir -p ~/.xmem
cat > ~/.xmem/.env << 'EOF'
ANTHROPIC_API_KEY=your_key_here
XMEM_DB_PATH=~/.xmem/memories.db
XMEM_MODE=cloud
EOF
```

---

## Files to Read Before Writing Any Code

```
~/Desktop/X-MemoryArch/
├── XMEM_MVP_SPEC.md                             ← this file
├── phase-3-plan.md                              ← full research context + Phase 4 plan
├── RetrievalEngine/
│   ├── APPROACHES.md                            ← all benchmark results with numbers
│   ├── PHASE3_NOTES.md                          ← what worked, what failed, lessons
│   ├── xmem_bridge.py                           ← THE BRIDGE (build this first)
│   ├── benchmark_4approaches.py                 ← A4mvr + A7r source of truth
│   └── app/services/retrieval/reranker.py       ← ms-marco cross-encoder
└── project-memory-core/
    ├── app/services/extraction/                 ← CloudLLMExtractionBackend, LocalChunk
    ├── app/crud.py                              ← DB write operations
    └── app/models.py                            ← Memory, Project SQLAlchemy models
```

---

## Benchmark Context

| Approach | R@5 | MRR | p50 | API at retrieve | Use in bridge |
|---|---|---|---|---|---|
| A4mvr | **0.822** | 0.717 | 200ms | None | retrieve_quality() — ALL modes |
| A6r | 0.808 | 0.700 | 62ms | Claude Haiku/query | NOT exposed — too expensive per query |
| A7r | 0.800 | 0.694 | **28ms** | None | retrieve_fast() — cloud/ollama modes |
| A5r | 0.792 | 0.693 | 48ms | None | superseded by A7r |
| A4r | 0.748 | 0.650 | 41ms | None | retrieve_fast() fallback — local mode |
| A1 | 0.025 | 0.038 | 4ms | None | NOT used — diagnostic baseline only |

**Phase 3.9 real-world validation:**
- Precision@5 = 0.759 on 75 sessions (target ≥ 0.60 — PASS)
- Query hit rate = 1.000 (every query got ≥1 relevant result)
- Competitor position: beats Zep CE on LongMemEval by 12.3%

---

## Build Order for the New Session

1. **`xmem_bridge.py`** — build this in the X-MemoryArch repo first. Define the 4 functions. Wire them to A4mvr and A7r. Test them standalone with the 3.9 evaluation data.

2. **`boundary.py`** — empirically test each CLI's output pattern. Get boundary detection working for Claude Code first.

3. **`wrapper.py`** — pty spawn + stdin/stdout pipe. No memory logic yet. Just prove the transparency.

4. **`consent.py`** + **`injector.py`** — wire the consent prompt and retrieval injection.

5. **`handoff.py`** — build the handoff packet and cross-AI launch.

6. **`cli.py`** — Click entrypoint wiring everything together.

7. End-to-end test: `xmem --claude --project test` → have a conversation → save → `xmem --chatgpt --project test` → verify context injection works.

---

## Open Questions for the New Session

1. **Boundary detection calibration** — each CLI must be run and its output observed to find the right sentinel patterns. The patterns in this spec are educated guesses. Calibrate empirically first.

2. **stdin injection compatibility** — does prepending context to stdin work without breaking Claude Code's input parser? Test with a short prefix first. Some CLIs may reject inputs >N characters.

3. **Consent UX timing** — `[Y/n]` blocks the terminal. Consider a 3-second auto-skip: if no key pressed, assume N and continue. Feels less intrusive.

4. **Context packet length** — target <300 tokens for fast path injection. A4mvr returns 5 chunk results; format as 3-5 bullet points. Never dump raw exchanges.

5. **Auto-project naming** — if no `--project` given, auto-detect from first 3 exchanges. Use Claude Haiku (cloud mode) or a simple TF-IDF keyword extractor (local mode) to generate a 3-word project name.
