# X-MemoryArch

A research project building a production-grade long-term memory **retrieval engine** for AI assistants. It extracts structured, temporally-grounded memories from conversations and retrieves the right ones with high accuracy — fully transparent, reproducible, and open.

> ### ⚠️ A note on metrics — read this before the numbers
> The scores below are **retrieval recall** (R@5 / R@10): *did the relevant session land in the top-K retrieved results?* This is **not the same metric** as the headline scores Mem0, Zep, and Honcho publish, which are **end-to-end QA accuracy** (retrieve → generate an answer → an LLM judge scores answer correctness). Retrieval recall typically runs **20–30 points higher** than QA accuracy, so **our retrieval-recall numbers are NOT directly comparable to their QA-accuracy numbers, and we make no "beat Mem0" claim.** A like-for-like QA-accuracy evaluation (using the same retrieve→generate→judge harness) is in progress; until it lands, treat the competitor numbers as *context for a different metric*, not a head-to-head.

---

## What This Is

Most LLM memory systems store raw conversation history and retrieve it by brute-force embedding search. X-MemoryArch extracts structured, temporally-grounded memories from each session, indexes them across multiple retrieval signals, and ranks results with a cross-encoder reranker. Every architectural decision is documented, benchmarked, and justified.

It started as a rule-based keyword engine (Phase 1) and evolved into a multi-signal retrieval system. The headline engineering result: on the full 690-query LoCoMo set, **retrieval recall R@5 rose from 0.738 → 0.923** after a failure diagnostic revealed that session content was being truncated to 42% before extraction — feeding full sessions fixed it.

---

## Benchmarks

Three standard datasets. **Metric reported here: retrieval recall (R@5 / R@10)** — whether the gold session is in the top-K retrieved. (See the metrics note above: this differs from the QA-accuracy scores competitors publish.)

| Dataset | Description | Memories | Queries (full) |
|---|---|---|---|
| **SQuAD v1.1** | Factual QA over Wikipedia paragraphs | 2,067 | 10,570 |
| **LoCoMo** | Long-term conversational memory, 10 conversations | 272 | 690 |
| **LongMemEval** | Multi-session episodic memory recall | 940 | 500 |

### Our Retrieval-Recall Results (Phase 5.6, full query sets)

**A11-sonnet** (Claude Sonnet 4.6 extraction, internal/ceiling tier):

| Dataset | Protocol | R@5 (recall) | R@10 | MRR@10 |
|---|---|---|---|---|
| LoCoMo | per-conversation scoped (standard) | **0.929** | 0.971 | 0.825 |
| LoCoMo | global pool (harder) | 0.923 | 0.965 | 0.819 |
| LongMemEval-oracle | global pool | 0.908 | — | 0.760 |

**Same engine, cheaper extraction models** (retrieval recall, full sets):

| Extraction model | Dataset | R@5 (recall) | Notes |
|---|---|---|---|
| Claude Haiku 4.5 | LoCoMo | 0.900 | ~10× cheaper than Sonnet |
| **GPT-4o-mini** | **LongMemEval-S (standard protocol)** | **0.920** | same model Mem0 uses; ~$15 for 19k sessions |

The GPT-4o-mini result matters for one reason: it isolates the **retrieval architecture** as the variable (same extraction model as Mem0). But again — this is recall, not QA accuracy.

### Competitor Context (different metric — NOT a head-to-head)

These are the scores competitors publish. **They are QA accuracy (end-to-end answer correctness), not retrieval recall.** Listed for context only; a comparable QA-accuracy eval of our system is pending.

| System | Dataset | Their published score | Metric |
|---|---|---|---|
| Mem0 NEW | LoCoMo | 91.6–92.5% | QA accuracy |
| Mem0 NEW | LongMemEval | 94.4–94.8% | QA accuracy |
| Honcho | LoCoMo | 89.9% | QA accuracy |
| Honcho | LongMemEval | 90.4% | QA accuracy |
| Letta | LoCoMo | 74.0% | QA accuracy |
| Zep | LongMemEval | 71.2% | QA accuracy |

Our retrieval recall (≈0.92) being a higher number than some of these does **not** mean we score higher on their metric — recall is the easier measurement. We will only claim a comparison once we run the same QA-accuracy harness.

---

## Architecture

### Core Design Principles

**ADD-ONLY Memory Storage.** No deduplication, no supersede, no versioning at ingestion. This is not a simplification — it is the proven optimal architecture. Mem0's old algorithm used ADD/UPDATE/DELETE/Noop at ingestion and scored 71.4% LoCoMo. They dropped everything except ADD and reached 91.6%. Conflict resolution at ingestion requires predicting future relevance without future context. That is impossible. The retrieval layer handles staleness via temporal grounding baked into every memory text ("As of session 8, Alice works at Startup X...") combined with a recency bias signal.

**Extraction-First Quality.** Retrieval algorithm improvements hit a hard ceiling defined by extraction quality. A session whose topics were never extracted cannot be retrieved — ever. Every algorithmic improvement in Phase 2-3 was ultimately bounded by what the extractor wrote. Phase 4 pivot: make the extracted memories as semantically rich and topic-diverse as possible.

**Temporal Grounding in Memory Text.** Every extracted memory encodes its session position explicitly: `"As of session {N}, [Person] [fact]."` or `"During session {N}, [Person] [event]."` This makes staleness resolvable at retrieval time without a separate temporal reasoning step.

### Two-Stage Retrieval Pipeline (A11)

```
Query
  │
  ├─► Variant 1: raw query         ─┐
  ├─► Variant 2: rephrased query   ─┼─► top-25 each ─► RRF merge ─► top-80 pool
  └─► Variant 3: rephrased query   ─┘                      │
                                                            ▼
                                             Entity boost (weight=5.0, spread-attenuated)
                                           + Recency bias (for "now/current/still" queries)
                                                            │
                                                            ▼
                                              Top-40 → ms-marco cross-encoder
                                                            │
                                                            ▼
                                              Session diversity filter → top-5
```

---

## Project Structure

```
X-MemoryArch/
├── RetrievalEngine/              # Core retrieval system (Phase 2+)
│   ├── app/
│   │   ├── services/
│   │   │   ├── extraction/       # Memory extractors (rich, comprehensive, triple)
│   │   │   ├── retrieval/        # MultiSignalRetriever, HyDE, BM25
│   │   │   ├── knowledge_graph/  # Entity graph, bi-temporal edges
│   │   │   ├── memory_loops/     # Autonomous loop system (Phase 3.8)
│   │   │   ├── vector_backends/  # GTE-large, text-embedding-3-small adapters
│   │   │   └── benchmark/        # Per-approach benchmark runners
│   │   └── routers/              # FastAPI endpoints
│   ├── training/
│   │   ├── build_rich_memories.py        # Phase 4.4+ extraction (main script)
│   │   ├── build_comprehensive_facts.py  # Phase 3.7 comprehensive fact extraction
│   │   ├── build_triple_facts.py         # Phase 4.1 triple extraction
│   │   ├── build_reranker_pairs.py       # Fine-tuning data generation
│   │   ├── finetune_reranker.py          # xma-reranker-v1 training script
│   │   └── grid_search_phase51.py        # Phase 5.1 hyperparameter search
│   ├── benchmark_4approaches.py          # Main benchmark runner
│   ├── APPROACHES.md                     # All approaches with benchmark results
│   └── requirements.txt
├── project-memory-core/          # Phase 1: SQLite-backed memory store
│   ├── app/                      # FastAPI CRUD API
│   ├── alembic/                  # DB migrations
│   └── tests/
├── Chats_for_evaluation/         # Synthetic evaluation datasets
├── evaluation_cache/             # Cached evaluation results
├── ingest_chatgpt_export.py      # ChatGPT export → memory ingest
├── query_memories.py             # CLI query interface
└── XMEM_MVP_SPEC.md              # Original MVP specification
```

---

## Phase History

### Phase 1 — Rule-Based Memory Store

**Goal:** Build a working memory store with basic retrieval.

**What was built:**
- SQLite-backed memory store with FastAPI CRUD API (`project-memory-core/`)
- BM25 keyword search via SQLite FTS5
- Entity extraction and boosting (hand-crafted entity types: person, place, job, date)
- `ingest_chatgpt_export.py` — parse ChatGPT export JSON → structured memories
- `query_memories.py` — CLI interface for querying stored memories

**Benchmark result (A1):**

| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.025 | 0.025 |
| LoCoMo | 0.060 | 0.055 |
| LongMemEval | 0.035 | 0.035 |
| **Aggregate** | **0.040** | **0.038** |

**Why so low:** Conversational memories don't share keywords with questions. "Loves hiking" ≠ "outdoor activities." BM25 requires lexical overlap; natural language queries don't provide it.

---

### Phase 2 — Dense Retrieval + Hybrid Stack

**Goal:** Add semantic (embedding) search and build a full hybrid retrieval pipeline.

**What was built:**
- GTE-large (1024-dim, local) and text-embedding-3-large (OpenAI, 3072-dim) embedding backends
- Hybrid retrieval: BM25 + dense cosine + entity matching → RRF fusion
- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- MMR session diversity (λ=0.70)
- HyDE query expansion (A3, cloud path): Claude Haiku generates a hypothetical memory, its embedding is averaged with the raw query embedding
- Atomic fact decomposition (A5): Claude Haiku extracts 3-5 standalone facts per session; facts embed more precisely than full session blobs
- Comprehensive extraction study: fact quality >> retrieval algorithm complexity

**Key finding:** `A5 (fact decomposition) → 0.768 aggregate R@5` beats `A4 (full hybrid pipeline) → 0.715` despite A4 having graph expansion, intent weighting, MMR, and HyDE. Representation quality dominates algorithm complexity.

**Key negative result:** Multi-HyDE (average 3 hypothetical embeddings) → -0.009 aggregate. Averaging vectors produces a blurry centroid that weakens the query signal. The right solution for ambiguous queries is multi-query retrieval over ranked lists (RRF), not multi-vector averaging.

**Best Phase 2 result (A5r — facts + reranker + session-MMR):**

| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.955 | 0.880 |
| LoCoMo | 0.590 | 0.519 |
| LongMemEval | 0.815 | 0.680 |
| **Aggregate** | **0.792** | **0.693** |

---

### Phase 3 — Multi-Query RRF, Fine-Tuned Reranker, Knowledge Graph

**Goal:** Cross the R@5 ≥ 0.80 plan target across all three datasets.

**Major components built:**

**A6r — Multi-Query RRF + Reranker (Phase 3.1):** First time crossing 0.80. Generate 2 query rephrases via Claude Haiku (cached, ~$0.01 one-time), run 3 independent searches, merge via RRF (k=60), rerank top-40 with ms-marco. Key insight: RRF over multiple ranked lists preserves signal sharpness (unlike Multi-HyDE which blurs embeddings).

| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.970 | 0.882 |
| LoCoMo | 0.630 | 0.532 |
| LongMemEval | 0.825 | 0.688 |
| **Aggregate** | **0.808** ✓ | **0.700** |

**xma-reranker-v1 — Fine-Tuned Cross-Encoder (Phase 3.2):** ms-marco reranker is trained on 100-500 word Wikipedia passages. Our atomic facts are 12-17 words — outside its distribution. Fine-tuned on 47,517 labeled (query, fact) pairs from all 3 datasets (50% positive, 36% hard negative, 14% easy negative). Trained on Apple MPS in 10.9 minutes. Result: +0.045 LoCoMo, +0.020 LME. SQuAD regressed -0.030 (mild LoCoMo bias introduced by fine-tuning).

**A4mvr — ColBERT-Style Multi-Vector + Reranker (Phase 3.3):** Split sessions into paragraph chunks, embed each independently, max-pool per session → cross-encoder reranker. No LLM facts needed. Aggregate champion at 0.822 R@5, +0.100 MRR vs A6r. Works on raw session text with no extraction step.

| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.970 | — |
| LoCoMo | 0.660 | — |
| LongMemEval | 0.835 | — |
| **Aggregate** | **0.822** | **0.717** |

**A8r — Entity Knowledge Graph + Multi-Vector (Phase 3.7):** Builds an entity co-occurrence graph with bi-temporal edges (valid_from/valid_until). Graph walk adds 0 R@5 over dense retrieval alone — entity graph finds the exact sessions dense retrieval already retrieves, because LoCoMo uses pronouns heavily (NER misses entities without coreference resolution). Graph implementation retained for production value: point-in-time entity state queries work correctly.

**Comprehensive Fact Extraction (Phase 3.7):** Split each session into first-half and second-half, extract independently with mandatory "at least one fact per person, per event" rules. Increased facts from 6→10/session. LoCoMo: +0.080 R@5 (0.630→0.710 on A6r, new record). LME mild regression (sessions already well-covered at 6 facts).

**Autonomous Memory Loops (Phase 3.8):** 5 production-grade background loops with 50 tests passing: ingestion loop, embedding refresh loop, entity refresh loop, feedback loop (learns RRF weights), promotion loop. `FeedbackLoop` exports learned RRF weights.

**A9r — Intent-Routed Retrieval (Phase 3.8.6):** Intent classifier routes queries to ENTITY, TEMPORAL, FACT, or MERGE legs. Aggregate 0.813 — below A4mvr (0.822). Rule-based routing regressed on all three datasets; MERGE fallback loses FACT path precision. Scrapped in Phase 4.

**Phase 3 LoCoMo ceiling: 0.710.** Every algorithmic improvement exhausted. Moving to better memory representation.

---

### Phase 4 — Mem0 Architecture: Rich Memories + ADD-ONLY

**Goal:** Replicate Mem0's new algorithm. Abandon atomic facts. Switch to 15-80 word temporally-grounded memories, ADD-ONLY storage, multi-signal retrieval with entity boost.

**Why Phase 4 exists:** Mem0 published a new algorithm (April 2026) jumping from 71.4% → 91.6% LoCoMo. Their key changes: drop conflict resolution at ingestion entirely (ADD-ONLY), switch from 3-5 word triples to 15-80 word rich memories, add explicit temporal grounding in every memory. Phase 3's ceiling (0.710 LoCoMo) matched Mem0's OLD level. Phase 4 replicates their NEW architecture.

**A10 — Triple-Fact RRF (Phase 4.1):** Test structured entity-relationship triples ("Caroline has role counselor"). LoCoMo: 0.683. Better than A6r standard facts (+0.025) but the embedding gap between a 4-word triple and a natural language question remains too large.

**A11s — Rich Memory Extraction (Phase 4.4):** The core pivot. Replace 4-word triples with 15-80 word memories extracted by Claude Sonnet 4.6 (internal tier) or Haiku 4.5 (product tier). Format:

```
Phase 4.1: "Caroline has role counselor"
Phase 4.4: "As of session 4, Caroline has made a firm decision to pursue counseling,
            inspired by years of supporting friends through personal mental health challenges."
```

Cosine similarity between "career path decided to pursue" and the 35-word memory: ~0.82 vs ~0.35 for the triple. The lexical richness bridges the embedding gap directly.

Result: LME gap to Mem0 closed from **22 pts → 5 pts in one phase.** LoCoMo improved from 0.710 (Phase 3 record) to 0.716 → 0.738 after tuning.

| Tier | LoCoMo R@5 | LME R@5 |
|---|---|---|
| A11s-sonnet (Phase 4.4) | 0.716 | **0.898** |
| A11s-haiku (Phase 4.4) | 0.719 | 0.892 |
| Sonnet vs Haiku delta | ≤ 0.003 | ≤ 0.006 |

**A11 — Multi-Signal Retrieval + Phase 4.5r Extraction (Phase 4.5):**

Infrastructure added:
- Multi-query RRF with entity boost (weight=5.0, spread-attenuated by entity memory count)
- Recency bias for "now/currently/still" queries
- BM25 tested and disabled (vocabulary matching promotes irrelevant conversational candidates)

Extraction improvements (Phase 4.5r):
- Pronoun leakage reduced: 36.1% → 1.6% (all pronouns replaced with full person names)
- Session coverage improved: 5.2 → 8.7 memories/session
- Temporal grounding: 100% (was ~85%)

Phase 5.1 hyperparameter grid search:
- Entity weight: tested 0.0, 0.3, 1.0, 2.0, 5.0, 10.0 → 5.0 optimal
- Pool size: top-25 per variant → +0.002 LME vs top-20
- 94 tests passing

**Final A11-sonnet results:**

| Metric | LoCoMo | LongMemEval |
|---|---|---|
| R@5 | **0.738** | **0.900** |
| R@10 | **0.806** | ~0.940 |
| MRR@10 | **0.613** | ~0.750 |
| Latency p50 | 66ms | 72ms |

**Phase 4.6 — Haiku Temporal Reranker (FAILED, dropped):** Added a second reranker pass using Claude Haiku to reason over top-5 candidates and reorder by temporal relevance. Result: -0.006 LoCoMo, -0.010 LME, 23× latency increase. Haiku reasoning over sparse 50-word snippets performs worse than ms-marco's implicit temporal calibration from training. Permanently removed.

---

### Phase 5 — Raising the Retrieval-Recall Ceiling

Phase 5 spent weeks tuning extraction prompts (topic-first, quantity+diversity, 4 granularity levels, 15-20 memories/session), swapping embedding models (text-embedding-3-small), and adding actual date grounding. **None moved LoCoMo R@5 off ~0.73.** Every hypothesis was a guess.

**Phase 5.6 — The Breakthrough (full-session extraction):**

Instead of guessing again, we wrote a failure diagnostic (`training/diagnose_locomo_misses.py`) that categorizes every R@10 miss as COVERAGE / EMBEDDING / RANKING. It revealed that **96% of misses had the gold session indexed but the specific queried fact missing.** Tracing those facts to the raw data exposed the real bug:

> `load_locomo` truncated every session to its first **1,200 characters** before extraction — discarding **58%** of every conversation (mean raw session = 2,842 chars). Facts in the back half of every session were physically removed from the extractor's input. No prompt, embedding, or reranker could recover text that was never seen.

Concrete example — query *"When did Caroline go to the adoption meeting?"*: the raw session contained *"Last Friday I went to a council meeting for adoption"* at character ~1,400, past the cut. Extraction wrote the vague *"Caroline had a busy week, specific details unclear."* After the fix, it captured the actual event and retrieved correctly.

**The fix:** content cap 1,200 → 6,000 chars; extractor `_CONTENT_LIMIT` 4,000 → 6,000; a completeness prompt ("extract EVERY concrete fact from BOTH speakers, never vague-summarize, scan the entire conversation"); memory target 8-10 → 10-16.

**Result** — retrieval recall on the full 690-query LoCoMo set (pool widened to 50 in the same phase):

| Metric (retrieval recall) | Phase 4 | Phase 5.6 |
|---|---|---|
| LoCoMo R@5 | 0.738 | **0.923** |
| LoCoMo R@10 | 0.806 | **0.965** |
| LoCoMo MRR@10 | 0.613 | **0.819** |

R@10 jumped 0.828 → 0.965 (119 → 24 misses out of 690) — a **+18.5-point gain in retrieval recall** from fixing one truncation bug. (Mem0's published 0.916 LoCoMo is QA accuracy, a different metric — see the metrics note at the top; no head-to-head is claimed here.)

**Next:** LongMemEval. A failure diagnostic confirmed LME's misses are genuine *ranking* misses (facts extracted but ranked low), not coverage gaps — so the fix is ranking-side (pool widening lifted recall 0.900 → 0.908), not re-extraction. On the standard LongMemEval-S protocol with GPT-4o-mini extraction, retrieval recall R@5 = 0.920. The remaining work is multi-hop / "how many" aggregation queries, plus the pending **QA-accuracy evaluation** that will make competitor comparisons valid.

---

## All Retrieval Approaches — Benchmark Summary

Evolution from baseline to current best:

```
A1   (BM25 keyword)                                     → 0.040 aggregate R@5
A3   (Cloud LLM + HyDE)                                 → 0.710
A4r  (Full hybrid + reranker, no HyDE)                  → 0.748
A4mv (Multi-vector chunks + max-sim)                    → 0.777   $0, 16ms p50
A5   (Fact decomposition, GTE-large)                    → 0.768   representation > algorithm
A5r  (Facts + reranker + session-MMR)                   → 0.792   LME 0.825 ✓ beats Zep (71.2%)
A6   (Multi-Query RRF, 3 searches)                      → 0.783   LoCoMo 0.635 record
A6r  (Multi-Query RRF + reranker)                       → 0.808 ✓ first crossing R@5 0.80 target
A4mvr (Multi-Vector + Reranker)                         → 0.822 ✓ aggregate champion; +0.100 MRR
A8r  (Entity graph + A4mvr)                             → 0.822   graph adds 0 over dense
A6r  (comprehensive facts, Phase 3.7)                   → LoCoMo 0.710 Phase 3 record
──── Phase 4: Mem0 architecture pivot ────
A10  (Triple-fact RRF + reranker)                       → LoCoMo 0.683 / LME 0.728
A11s (Rich memories, Phase 4.4)                         → LoCoMo 0.716 / LME 0.898 ← LME +0.170!
A11  (Multi-signal, Phase 4.5 + 5.1 tuning)            → LoCoMo 0.738 / LME 0.900
──── Phase 5.6: full-session extraction (truncation bug fixed) + wider pool ────
A11  (full content + completeness prompt)               → LoCoMo 0.923 / LME 0.908  retrieval recall R@5
```

Per-dataset retrieval recall (R@5, full query sets):
- **SQuAD:** A4mvr / A8r — 0.970
- **LoCoMo:** A11-sonnet (Phase 5.6) — **0.929** (scoped) / 0.923 (global)
- **LongMemEval:** A11-sonnet — **0.908** (oracle) · GPT-4o-mini LME-S standard — **0.920**

All figures above are **retrieval recall**, not QA accuracy — see the metrics note at the top of this README.

---

## Setup

### Prerequisites

- Python 3.10+
- macOS (tested on Apple Silicon; MPS used for local model inference)
- `ANTHROPIC_API_KEY` in `.env` (required for extraction and query rephrasing)
- `OPENAI_API_KEY` in `.env` (optional — only for text-embedding-3-small / text-embedding-3-large tests)

### Installation

```bash
# Phase 1 — memory store
cd project-memory-core
pip install -r requirements.txt

# Phase 2+ — retrieval engine
cd ../RetrievalEngine
pip install -r requirements.txt
```

GTE-large and the ms-marco cross-encoder download automatically on first use from HuggingFace. Models are cached locally under `RetrievalEngine/models/` (gitignored — ~3GB total).

### Environment

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY
# Fill in OPENAI_API_KEY (optional)
```

---

## Running the Benchmark

```bash
cd RetrievalEngine

# Fast diagnostic — A1, A4r, A5, A5r using local GTE-large, no API calls, no Ollama
python3 -u benchmark_4approaches.py --skip-ollama --skip-cloud

# Current best (A11-sonnet) — requires pre-extracted rich memories cache
python3 benchmark_4approaches.py --skip-ollama --skip-cloud --only-a11 --model-tag sonnet

# Full smoke test — all approaches including Claude HyDE (~$2-3 API cost)
python3 -u benchmark_4approaches.py --skip-ollama

# Cloud embedding path — OpenAI text-embedding-3-large (~$0.05 per run)
python3 -u benchmark_4approaches.py --skip-ollama --skip-cloud --embed-model te3l

# Full publishable run — all 690 LoCoMo + 500 LME queries (~$12)
python3 -u benchmark_4approaches.py --skip-ollama --n 10000
```

### Rebuilding Extraction Caches

```bash
# Rich memory extraction (Phase 4.4+ — current architecture)
python3 training/build_rich_memories.py --model sonnet --dataset locomo
python3 training/build_rich_memories.py --model sonnet --dataset lme

# Comprehensive fact extraction (Phase 3.7)
python3 training/build_comprehensive_facts.py

# Fine-tune the cross-encoder reranker (Phase 3.2)
python3 training/build_reranker_pairs.py
python3 training/finetune_reranker.py
# Outputs: models/xma-reranker-v1/final/ (~10 min on Apple MPS)
```

---

## Key Findings

**1. Representation dominates algorithm.** A5 (plain fact decomposition, no reranker) at 0.768 aggregate R@5 beats the full A4 hybrid pipeline (BM25 + dense + entity graph + intent routing + HyDE + reranker + MMR) at 0.715. What you extract matters more than how you rank it.

**2. ADD-ONLY is the correct memory architecture.** Conflict resolution at ingestion (ADD/UPDATE/DELETE/Noop) requires knowing which fact will become stale — impossible without future context. ADD-ONLY + temporal grounding in memory text handles staleness correctly and outperforms conflict-resolving approaches by ~20 points on LoCoMo.

**3. Multi-HyDE fails; multi-query RRF works.** Averaging N hypothetical embeddings produces a blurry centroid (-0.009 vs single HyDE). Running N independent searches and merging ranked lists via RRF preserves each search's signal sharpness (+0.025 over single-search reranker).

**4. Extraction diversity is harder than extraction quantity.** 8.7 memories/session is worthless if all 8 are about the same topic. The model grabs the most emotionally salient topic and generates 8 phrasings of it. Enforcing topic diversity requires explicit anti-examples in the prompt; quantity and diversity must be decoupled.

**5. BM25 hurts conversational memory retrieval.** Unlike passage retrieval, conversational memories rarely share exact vocabulary with queries. BM25 over 15-80 word memories promotes vocabulary-matching irrelevant sessions ahead of semantically relevant ones. Disabled in production.

**6. Sonnet vs Haiku: ≤ 0.003 delta.** For memory extraction, Haiku 4.5 (product tier) produces results within 0.003 R@5 of Sonnet 4.6 (internal tier). Haiku is the correct production default — same recall quality at 5-10× lower cost.

**7. Fine-tuned rerankers are domain-specific.** ms-marco cross-encoder trained on 100-500 word Wikipedia passages performs poorly on 12-17 word conversational atomic facts. Fine-tuning on 47,517 actual (query, fact) pairs gained +0.045 LoCoMo but introduced mild SQuAD regression (-0.030). Domain-specific fine-tuning is only worth it if the deployment domain differs significantly from ms-marco's training distribution.

---

## Test Suite

```bash
cd RetrievalEngine
pytest tests/ -v
# 94 tests passing
```

Tests cover: extraction pipeline, entity store, multi-signal retrieval, RRF fusion, entity boost, recency bias, BM25 diagnostic, memory loop orchestration.

---

## Gitignore Policy

What is **excluded** from the repository:
- `benchmark_cache/` (~1.2 GB — rebuilt by training scripts)
- `models/` (~3 GB — xma-reranker-v1, GTE-large local copy)
- `checkpoints/` — training checkpoints
- `evaluation_cache/embed_facts.npy` — large embedding arrays
- `*.db` — SQLite memory stores
- `.env` — API keys
- Phase notes docs (`PHASE3_NOTES.md`, `PHASE4_NOTES.md`, `PHASE5_NOTES.md`) — internal running logs
- Analysis and planning docs — stay local

What is **included**:
- All source code
- Training scripts (for reproducibility)
- `APPROACHES.md` — complete benchmark reference with results for all 22+ approaches
- Test suite

---

## Roadmap

**The next milestone is the QA-accuracy evaluation.** All current numbers are retrieval recall. To make any valid comparison to Mem0/Zep/Honcho, the system must be run through the same end-to-end harness they use: retrieve → generate an answer → LLM-judge answer correctness. This is the single most important next step for a trustworthy claim, and it's cheap (~$5–10).

**LoCoMo retrieval: strong.** Retrieval recall R@5 = 0.923 (global) / 0.929 (scoped), R@10 = 0.965 — up from 0.738 after the truncation fix (Phase 5.5 date grounding, Phase 5.6 full-session extraction, pool widening all shipped).

**LongMemEval retrieval:** R@5 = 0.908 (oracle) / 0.920 (LME-S standard, GPT-4o-mini). Remaining misses are genuine *ranking* misses, not coverage gaps (diagnostic-confirmed). Dominated by multi-hop / "how many" aggregation queries (e.g. *"How many museums did I visit in December?"*) — the fix is an aggregation retrieval path inside the single pipeline.

**Remaining LoCoMo frontier:** 24 genuine RANKING misses (gold memory exists with high cosine but loses in pool/rerank). Candidate fixes: reranker fine-tuning, entity-store cleanup (month names currently leak in as entities from date text).

**Horizon:** Multi-agent memory consolidation, production deployment with FastAPI streaming, generalization tests on non-benchmark conversation domains.
