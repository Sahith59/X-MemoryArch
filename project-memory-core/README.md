# X-MemoryArch — Phase 1: Memory Core

> A local-first memory layer that captures what you build, decide, and learn across AI conversations — and hands that context seamlessly to any AI tool, any time.

**1058 tests passing · No cloud API calls · No API key required · Fully offline after first run**

> **Model note:** The embedding model (`all-MiniLM-L6-v2`, 22 MB) is downloaded once from HuggingFace on first extraction and cached locally by `sentence-transformers`. After that, everything runs on your CPU with no internet connection. No data ever leaves your machine.

---

## The Problem This Solves

Every time you start a new conversation with Claude, ChatGPT, Cursor, or Gemini — you start from zero. You re-explain your stack, re-describe your decisions, re-paste your errors. Context is constantly lost between sessions, between tools, and between team members.

**X-MemoryArch fixes this.** It captures everything meaningful from your AI conversations — decisions, bugs, architecture notes, failed approaches — and stores them as structured, queryable memories. When you need context again, you export a single Markdown file and paste it into any AI tool. Instant context, zero re-explaining.

---

## How Phase 1 Works — The Complete Flow

```
YOUR AI CONVERSATION
(Claude, ChatGPT, Cursor, Gemini)
          │
          │  You paste the raw conversation text
          ▼
┌─────────────────────────┐
│       PROJECT           │
│                         │
│  A named workspace that │
│  groups everything      │
│  (name, tech stack,     │
│   goals, repo path,     │
│   domain)               │
└────────────┬────────────┘
             │
             │  You add the conversation as a Session
             ▼
┌─────────────────────────┐
│        SESSION          │
│                         │
│  Raw conversation text  │
│  stored as-is.          │
│  Tool: Claude / GPT /   │
│  Cursor / Gemini        │
│  Date, Title            │
│  Messages (individual   │
│  turns, role-tagged)    │
└────────────┬────────────┘
             │
             │  You trigger: Extract Memories
             ▼
┌────────────────────────────────────────────────────────┐
│                  EXTRACTION PIPELINE                   │
│              (the brain of Phase 1)                    │
│                                                        │
│  Step 1 — Noise Gate                                   │
│    Removes filler text, AI preamble, greetings.        │
│    Only substantive sentences move forward.            │
│                                                        │
│  Step 2 — Substantive Gate (Domain-Aware)             │
│    An AI embedding model (all-MiniLM-L6-v2, 22MB,     │
│    runs on your CPU, no internet needed) converts      │
│    each sentence into a 384-number fingerprint.        │
│    Max cosine similarity against 38 exemplars          │
│    (across software, design, research, business,       │
│    marketing, and personal domains) determines         │
│    whether the sentence is real work content.          │
│    Threshold: 0.30. Casual/social text is discarded.  │
│                                                        │
│  Step 3 — Type Classifier (Domain-Weighted)           │
│    The same model compares each sentence against       │
│    13 memory type templates and assigns a type.        │
│    For projects with a known domain, centroids blend   │
│    70% domain-specific + 30% global for precision.    │
│    Confidence threshold: 0.23.                         │
│                                                        │
│  Step 3a — Low-Confidence Queue                       │
│    Sentences passing the gate but below the type       │
│    confidence floor are not discarded. They are        │
│    stored as type=unclassified with review_status=     │
│    needs_review and a suggested_type hint in           │
│    type_metadata. A developer can later assign the     │
│    correct type via POST /memories/{id}/classify.      │
│                                                        │
│  Step 4 — Structured Field Extraction                 │
│    Pattern matching pulls type-specific details:       │
│    For a problem → error message, root cause, fix      │
│    For a decision → rationale, alternatives            │
│    For how_to → the command, prerequisites             │
│                                                        │
│  Step 5 — Embedding Storage                           │
│    The 384-number fingerprint of each memory           │
│    is stored in the database. Phase 2 will use         │
│    this for semantic (meaning-based) search.           │
│                                                        │
│  Shadow Write (Option B)                              │
│    Every extracted memory also creates a parallel      │
│    MemorySuggestion record (status=approved,           │
│    created_by=rule_based). Manual suggestions          │
│    enter as status=pending for human review.           │
└────────────────────┬───────────────────────────────────┘
                     │
                     │  Each extracted sentence becomes a Memory
                     ▼
┌────────────────────────────────────────────────────────┐
│                      MEMORY                            │
│                                                        │
│  A single structured unit of knowledge.               │
│                                                        │
│  Core fields:                                          │
│    type         → what kind of memory it is           │
│    title        → short description                   │
│    content      → full text                           │
│    importance   → 1–5 (auto-scored or user-set)       │
│    confidence   → 0–1 (how certain the extraction is) │
│    tags         → labels for filtering                │
│    status       → active / resolved / archived        │
│    review_status → auto_extracted / needs_review /    │
│                    verified / rejected / draft         │
│                                                        │
│  Signals computed automatically at write time:         │
│    quality_score   → composite health signal          │
│    decay_score     → freshness (fades over 90 days)   │
│    retrieval_hint  → one-line TL;DR for fast reading  │
│    tier            → working (hot) or archival (cold) │
│    embedding       → 384-dim vector for Phase 2       │
└───────┬────────────────────────┬───────────────────────┘
        │                        │
        │ Entity Extraction       │ Relationships
        ▼                        ▼
┌───────────────┐    ┌───────────────────────────────────┐
│   ENTITIES    │    │            MEMORY LINKS            │
│               │    │                                   │
│ spaCy NER     │    │ If two memories mention the same  │
│ + 60+ tech    │    │ technology or organization,       │
│ keywords      │    │ a "related_to" link is created    │
│               │    │ between them automatically.       │
│ Extracts:     │    │                                   │
│  TECH         │    │ You can also create links         │
│  ORG          │    │ manually with typed relationships:│
│  PRODUCT      │    │  related_to / supersedes /        │
│  PERSON       │    │  conflicts_with / resolves /      │
│               │    │  blocks / influenced_by           │
│ Stored per    │    │                                   │
│ memory and    │    │ This creates a lightweight        │
│ per project   │    │ knowledge graph inside SQLite —   │
└───────────────┘    │ no graph database needed.         │
                     └───────────────────────────────────┘
        │
        │  After enough memories are added
        ▼
┌────────────────────────────────────────────────────────┐
│                   CLUSTERING                           │
│                                                        │
│  The DBSCAN algorithm groups memories that have        │
│  similar meaning (based on their 384-dim embeddings)   │
│  into clusters automatically.                          │
│                                                        │
│  Each cluster gets a label — the top words from        │
│  its members' titles (e.g. "redis / caching / ttl").  │
│                                                        │
│  This reveals hidden topic groups in your memories     │
│  without you having to tag or organize anything.       │
└────────────────────────┬───────────────────────────────┘
                         │
                         │  When you need context in a new AI session
                         ▼
┌────────────────────────────────────────────────────────┐
│               CANONICAL EXPORT LAYER                   │
│              (the read surface of Phase 1)             │
│                                                        │
│  Three complementary export formats:                   │
│                                                        │
│  1. context.md — Full Markdown context packet          │
│     Project overview, entities, clusters,              │
│     all memory types with freshness bars,              │
│     retrieval hints, and relationship links.           │
│     Includes a Review Queue section for any            │
│     unclassified memories awaiting manual review.      │
│                                                        │
│  2. memories/{type}.md — Per-type focused export       │
│     One document per memory type (decision,            │
│     problem, structure, etc.) with type-specific       │
│     metadata rendered in the most readable format.     │
│     Filterable by status, importance, privacy.         │
│                                                        │
│  3. memory.yaml — Canonical machine-readable export    │
│     All memories + sessions + entity index +           │
│     cluster assignments in a single structured file.   │
│     Schema-versioned, insertion-order preserved,       │
│     stats always consistent with the memories list.    │
│     This is Phase 2's primary input format.            │
│                                                        │
│  All three respect privacy_level gating               │
│  (public → internal → sensitive → secret).            │
└────────────────────────────────────────────────────────┘
        │
        │  Handoff tracking
        ▼
┌────────────────────────────────────────────────────────┐
│             CONTEXT PACKETS & HANDOFFS                 │
│                                                        │
│  ContextPacket — a bundled handoff payload             │
│    Select specific memories + sessions → system        │
│    assembles a Markdown document with token_estimate.  │
│    Useful for targeted handoffs (not full exports).    │
│                                                        │
│  HandoffEvent — an audit trail record                  │
│    Logs every Claude→ChatGPT / ChatGPT→Cursor          │
│    transition. Links to the ContextPacket used.        │
│    Tracks status: pending / completed / failed.        │
│    handoff_at is separate from created_at to allow     │
│    retroactive recording.                              │
└────────────────────────────────────────────────────────┘
```

---

## The 13 Memory Types

| Type | What it captures |
|------|-----------------|
| `decision` | An architectural or product choice — with rationale and alternatives considered |
| `problem` | A known defect or issue — with error message, root cause, and fix applied |
| `task` | A TODO or action item |
| `structure` | A structural design note — pattern, components, reasoning |
| `failed_approach` | Something tried and rejected — prevents repeating the same mistake |
| `constraint` | A hard requirement or limitation |
| `workflow_pattern` | A repeatable process or procedure |
| `how_to` | Install/configure/run steps — with command and prerequisites |
| `open_question` | An unresolved decision or question |
| `insight` | A realization or lesson learned |
| `reference_context` | Notes about a specific piece of code, design, or resource |
| `conversation_note` | General notes from a session |
| `unclassified` | Passed the substantive gate but below type-confidence threshold — queued for manual review via `POST /memories/{id}/classify` |

> **Universal extraction:** all 13 types work across every domain — software, design, research, marketing, business, and personal projects.

---

## Supported Project Domains

16 domains map to 6 canonical groups used by the semantic classifier:

| Group | Domains |
|-------|---------|
| software | software, data, security |
| design | design, creative |
| research | research, education |
| business | business, product, sales, finance, legal, hr |
| marketing | marketing |
| general | general (default — no domain bias) |

Setting a project's `domain` improves extraction precision. The classifier blends 70% domain-specific type centroids with 30% global centroids for known domains.

---

## How the Components Connect

```
Project
  ├── Sessions (raw conversations)
  │     ├── Messages (individual turns, role-tagged)
  │     └── Memories (extracted knowledge units)
  │           ├── type_metadata (structured fields per type)
  │           ├── Entities (TECH / ORG / PRODUCT tags)
  │           ├── Links → other Memories (knowledge graph)
  │           ├── Cluster (semantic group it belongs to)
  │           └── Signals: quality_score, decay_score,
  │                        retrieval_hint, tier, embedding
  ├── MemorySuggestions (staging table for human review)
  │     status: pending → approved / rejected / edited
  ├── ContextPackets (bundled handoff payloads)
  │     Selected memories + sessions → assembled Markdown
  └── HandoffEvents (tool transition audit trail)
        Claude→ChatGPT→Cursor, links to ContextPacket
```

---

## What Gets Stored Per Memory (Full Schema)

| Column | Description |
|--------|-------------|
| `type` | One of 13 memory types |
| `title` | Short label |
| `content` | Full text |
| `importance` | 1–5 |
| `confidence` | 0–1 extraction certainty |
| `status` | active / resolved / stale / archived / superseded |
| `review_status` | auto_extracted / needs_review / verified / rejected / draft |
| `tags` | Freeform labels |
| `related_files` | Code file paths |
| `related_tools` | AI tools or frameworks involved |
| `type_metadata` | Type-specific structured fields (rationale, error_message, command, suggested_type, etc.) |
| `privacy_level` | public / internal / sensitive / secret |
| `tier` | working (hot) or archival (cold) |
| `file_path` | Code anchor |
| `commit_sha` | Git anchor |
| `superseded_by` | Points to replacement memory |
| `valid_until` | When this memory stopped being true |
| `quality_score` | Composite 0–1 health signal |
| `decay_score` | Ebbinghaus freshness — fades over 90 days |
| `retrieval_hint` | One-line TL;DR (max 300 chars) |
| `embedding` | 384-dim float32 vector (all-MiniLM-L6-v2) |
| `cluster_id` | DBSCAN cluster assignment |
| `access_count` | How many times this memory has been retrieved |
| `last_accessed_at` | Last retrieval timestamp |

---

## Full API Surface

```
# Projects
POST   /projects
GET    /projects
GET    /projects/{id}
PUT    /projects/{id}
DELETE /projects/{id}

# Sessions
POST   /projects/{id}/sessions
GET    /projects/{id}/sessions
GET    /sessions/{id}
PUT    /sessions/{id}
DELETE /sessions/{id}
POST   /sessions/{id}/extract-memories       ← triggers the extraction pipeline

# Session Messages (individual conversation turns)
POST   /sessions/{id}/messages
GET    /sessions/{id}/messages               ?role=user|assistant|system
GET    /messages/{id}
PUT    /messages/{id}
DELETE /messages/{id}

# Memories — CRUD
POST   /projects/{id}/memories               ?check_dedup=true     ← block creation if near-duplicate exists (similarity ≥ 0.97)
POST   /projects/{id}/memories               ?auto_supersede=true  ← auto-supersede existing same-type memories with similar title (≥ 0.85)
GET    /projects/{id}/memories               ?type= &status= &importance_min= &tag= &tier= &max_privacy_level=
GET    /memories/{id}                        ← increments access_count
PUT    /memories/{id}
DELETE /memories/{id}

# Memories — Intelligence
GET    /projects/{id}/memories/search        ?q=                   ← BM25 hybrid search
GET    /projects/{id}/memories/stale                               ← memories not updated in 30+ days
GET    /projects/{id}/memories/most-accessed                       ← ranked by access_count
POST   /projects/{id}/memories/find-duplicates                     ← preview near-duplicates without creating (non-blocking)
POST   /projects/{id}/memories/autolink                            ← create entity-based links
POST   /projects/{id}/memories/recluster                           ← run DBSCAN clustering
POST   /projects/{id}/memories/retier                              ← recompute working/archival tiers
POST   /projects/{id}/memories/embed-all                           ← backfill embeddings for memories missing one
GET    /projects/{id}/clusters                                     ← cluster overview
GET    /projects/{id}/entities               ?label=               ← entity index

# Memory History & Links
POST   /memories/{id}/supersede                                    ← version a memory (superseded_at recorded on the link)
GET    /memories/{id}/history                                      ← full audit changelog
GET    /memories/{id}/links
POST   /memories/{id}/links
DELETE /memory-links/{id}

# Memory Review — Low-Confidence Queue
POST   /memories/{id}/classify               {"type": "decision"}  ← manually assign type, sets review_status=verified

# Memory Suggestions (staging / review workflow)
POST   /projects/{id}/suggestions                                  ← create a pending suggestion
GET    /projects/{id}/suggestions            ?status= &created_by=
GET    /suggestions/{id}
PATCH  /suggestions/{id}
POST   /suggestions/{id}/approve                                   ← approve → creates a Memory
DELETE /suggestions/{id}

# Context Packets (bundled handoff payloads)
POST   /projects/{id}/context-packets                              ← assemble from memory + session IDs
GET    /projects/{id}/context-packets
GET    /context-packets/{id}
DELETE /context-packets/{id}

# Handoff Events (tool transition audit trail)
POST   /projects/{id}/handoff-events                               ← record a Claude→GPT→Cursor handoff
GET    /projects/{id}/handoff-events         ?status= &source_tool= &target_tool=
GET    /handoff-events/{id}
PATCH  /handoff-events/{id}                                        ← update status / note only
DELETE /handoff-events/{id}

# Export — Canonical Layer
GET    /projects/{id}/export/context.md          ?max_privacy_level=                              ← full Markdown context packet
GET    /projects/{id}/export/memories/{type}.md  ?status= &min_importance= &max_privacy_level=   ← per-type Markdown
GET    /projects/{id}/export/memory.yaml         ?status= &min_importance= &max_privacy_level=   ← canonical YAML
```

---

## The Canonical Export Layer

The three export endpoints are the primary interface between Phase 1 (write layer) and Phase 2 (retrieval engine). Each serves a different consumer:

| Export | Format | Consumer | What makes it useful |
|--------|--------|----------|---------------------|
| `context.md` | Markdown | Human / AI prompt | Paste directly into Claude, ChatGPT, or Cursor |
| `memories/{type}.md` | Markdown | Focused review | One document per memory type with type-specific fields rendered readably |
| `memory.yaml` | YAML | Machines / Phase 2 | All signals, links, entities, clusters in one schema-versioned file |

**Why the YAML export is Phase 2's foundation:**
- Insertion order preserved (`_OrderedDumper`) — no alphabetical key scrambling
- Stats recomputed from the filtered memories list — always internally consistent
- 8 sections: `schema_version`, `exported_at`, `project`, `stats`, `memories`, `sessions`, `entity_index`, `clusters`
- Every memory includes: `entities` (extracted NER tags), `links` (typed outgoing relationships), all computed signals
- Entity index grouped by normalized text, sorted by `memory_count desc` — ready for frequency-based ranking
- Cluster assignments included — Phase 2 can do cluster-scoped retrieval without recomputing DBSCAN

**Why the per-type Markdown is useful for targeted handoffs:**
- One URL per concern: `/export/memories/decision.md` gives you every architectural choice in one readable document
- Type-specific renderers: a `failed_approach` shows `approach_tried`, `reason_failed`, `avoid_because`; a `problem` shows `error_message`, `root_cause`, `fix_applied`
- Unknown type → 422. Empty type → friendly message (not an error). Sensible behavior at both edges.

---

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model (for entity extraction)
python -m spacy download en_core_web_sm

# 4. Copy config
cp .env.example .env

# 5. Run database migrations  ← required, the server will not start without this
alembic upgrade head

# 6. Start the server
uvicorn app.main:app --reload
# The embedding model (~22 MB) downloads automatically on the first extraction call.
# All subsequent runs are fully offline.
```

API running at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

```bash
# Run all tests (use the venv python directly)
.venv/bin/python3 -m pytest tests/ -v
# 1058 tests across 38 files — all should pass
```

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11+ |
| Framework | FastAPI |
| Database | SQLite (local, zero config) |
| ORM | SQLAlchemy (sync) |
| Migrations | Alembic (22 migration scripts) |
| Embeddings | all-MiniLM-L6-v2 via sentence-transformers (22 MB, CPU, offline) |
| NER | spaCy en_core_web_sm + TECH keyword list |
| Clustering | scikit-learn DBSCAN (cosine distance) |
| Full-text search | SQLite FTS5 (BM25) |
| YAML export | PyYAML with `_OrderedDumper` for key-order preservation |
| Tests | pytest + httpx |

---

## Project Structure

```
app/
  main.py                          # App entry point, router registration
  database.py                      # SQLite engine and session factory
  models.py                        # All 9 database tables (ORM)
  schemas.py                       # Request/response types (Pydantic)
  crud.py                          # All database operations
  search.py                        # BM25 hybrid search (FTS5)
  services/
    memory_service.py              # Extraction pipeline (5 stages + low-confidence queue + Option B shadow write)
    semantic_classifier.py         # Embedding model + domain-aware type classifier + classify_best()
    exemplars.py                   # 38 substantive gate exemplars + type exemplars across 6 domain groups
    entity_extractor.py            # spaCy NER + TECH keyword supplement
    clustering.py                  # DBSCAN cluster computation
    retrieval_hint.py              # TL;DR generation per memory type
    context_export_service.py      # Markdown context packet generator (includes Review Queue section)
    per_type_export_service.py     # Per-type Markdown export (13 type renderers)
    yaml_export_service.py         # Canonical YAML export (8 sections, ordered)
    session_service.py             # Session helpers
  routers/
    projects.py / sessions.py / memories.py
    memory_links.py / exports.py / conflicts.py
    suggestions.py                 # MemorySuggestion approval workflow
    context_packets.py             # Context packet assembly
    handoff_events.py              # Tool handoff audit trail
    messages.py                    # Session message management
  utils/
    datetime_utils.py
alembic/                           # 22 migration scripts (full schema history)
tests/                             # 1058 tests, 38 files
static/
  index.html                       # Developer testing UI (single-file, vanilla JS)
```

---

## Database Tables

| Table | Sub-phase | Purpose |
|-------|-----------|---------|
| `projects` | 1.1 | Top-level workspace |
| `ai_sessions` | 1.2 | Raw conversation storage |
| `memories` | 1.3–1.41 | Structured knowledge units |
| `memory_entities` | 1.22 | NER tags per memory |
| `memory_links` | 1.11 | Typed relationships between memories |
| `memory_changelog` | 1.14 | Full audit history |
| `session_messages` | 1.31 | Individual conversation turns |
| `memory_suggestions` | 1.32 | Suggestion staging / review workflow |
| `context_packets` | 1.33 | Bundled handoff payloads |
| `handoff_events` | 1.34 | Tool transition audit trail |

---

## Phase 1 Enhancements

Post-foundation features built on top of the core extraction and storage layer.

### Sub-phase 1.43 — Supersedes Link Auto-Creation
`POST /memories/{id}/supersede` and the conflict resolver both automatically create a `supersedes` link (old→new) so the replacement chain is queryable from both ends, not just via the `superseded_by` field.

### Sub-phase 1.44 — Near-Duplicate Detection
Two opt-in surfaces to catch duplicate memories before they accumulate:

- `POST /projects/{id}/memories?check_dedup=true` — blocks creation with `409` if cosine similarity ≥ 0.97 against any existing memory. Default off — extraction pipeline is unaffected.
- `POST /projects/{id}/memories/find-duplicates` — preview duplicates for any proposed content without creating anything. Returns `is_duplicate`, `threshold`, and the list of near-matches with similarity scores.

### Sub-phase 1.45 — Temporal Supersession Tracking
Two additions to make the supersession timeline precise and auditable:

**`superseded_at` on `MemoryLink`**  
Every `relationship_type="supersedes"` link now records the exact UTC timestamp when the old memory was replaced. The field is `null` for all other link types. Exposed in `MemoryLinkResponse`.

**`auto_supersede` at creation**  
`POST /projects/{id}/memories?auto_supersede=true` scans existing active memories of the same type and auto-supersedes any whose title is ≥ 85% similar (Levenshtein ratio via `difflib`). Only applies to semantic types where supersession makes sense: `decision`, `constraint`, `structure`, `how_to`. When supersession fires, the endpoint returns `AutoSupersedeResult` instead of `MemoryResponse`:

```json
{
  "memory": { ...new memory... },
  "superseded_count": 1,
  "superseded_memories": [ { ...old memory with status=superseded... } ]
}
```

The old memory gets `status=superseded`, `valid_until=now`, `superseded_by=<new_id>`, a changelog entry for each changed field, and a `supersedes` link with `superseded_at` set.

---

## What's Next — Phase 2: Retrieval Engine

Phase 1 is the **write layer** — it captures, structures, and stores everything.

Phase 2 is the **read layer** — it retrieves the right memories at the right time, automatically.

**Phase 1 is ready for Phase 2.** Every signal Phase 2 needs is already computed and stored:

| Phase 2 need | Where it lives in Phase 1 |
|-------------|--------------------------|
| Vector search corpus | `memories.embedding` — 384-dim float32, every memory |
| Re-ranking signals | `decay_score`, `quality_score`, `importance`, `access_count` |
| Entity-scoped retrieval | `memory_entities` table, entity index in YAML export |
| Cluster-scoped retrieval | `memories.cluster_id`, clusters section in YAML export |
| Structured metadata | `type_metadata` JSON + 13 dedicated renderers |
| Fast TL;DR generation | `retrieval_hint` — pre-computed at write time |
| Machine-readable snapshot | `memory.yaml` — the canonical Phase 2 input format |
| Handoff context | `context_packets` — pre-assembled, token-estimated |

What Phase 2 will add on top of Phase 1:

- **Vector search** — Qdrant ANN index over the 384-dim embeddings. Semantic queries like *"find everything related to our auth decisions"* without exact keyword matches.
- **LLM-backed extraction** — Replace the rule-based pipeline with an LLM call for higher accuracy on long, complex sessions. The extraction interface stays identical — only the internals change.
- **Hybrid retrieval** — Merge BM25 (Phase 1 FTS5) + ANN results, re-rank using Phase 1's pre-computed signals.
- **Context budgeter** — Given a query and a token limit, automatically select and order the most relevant memories to fit.
- **Capture system** — Browser extension / IDE plugin / CLI hook that captures conversations automatically and routes them to the right project.
