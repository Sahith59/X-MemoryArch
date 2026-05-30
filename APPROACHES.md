# X-MemoryArch — Retrieval Approach Reference

All approaches are evaluated on three datasets:
- **SQuAD v1.1** — 2,067 passage memories, 200 sampled queries. Factual QA over Wikipedia paragraphs.
- **LoCoMo** — 272 session-level memories (10 conversations), 200 queries. Long-term conversational memory.
- **LongMemEval** — 940 session memories, 200 queries. Multi-session episodic memory recall.

Metrics: **R@5** (primary), MRR@10, NDCG@10, p50 latency.
Plan targets: R@5 ≥ 0.80 · MRR@10 ≥ 0.78 · NDCG@10 ≥ 0.55

---

## A1 — Rule-based (BM25 + Entity)

**What it does:** Pure keyword search using SQLite FTS5 (BM25 ranking) with entity boosting. No embeddings, no ML models. Zero-shot baseline.

**Pipeline:** Query → BM25 FTS5 search → entity match boost → top-10

**Cost:** $0. Latency: ~1-4ms.

**When to use:** Exact keyword queries, structured lookups, debugging retrieval pipeline.

**Results:**
| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.025 | 0.025 |
| LoCoMo | 0.060 | 0.055 |
| LongMemEval | 0.035 | 0.035 |
| **Aggregate** | **0.040** | **0.038** |

**Why it scores so low:** Conversational memories don't share exact keywords with questions. Someone who "loves hiking" won't be retrieved by searching "outdoor activities."

---

## A2 — Local LLM (Ollama + HyDE)

**What it does:** Uses a locally-run Ollama model (llama3.1) to generate a hypothetical memory (HyDE), then embeds it with `nomic-embed-text` and retrieves by cosine similarity.

**Pipeline:** Query → Ollama generates hypothetical memory → nomic-embed-text embedding → cosine search → top-10

**Cost:** $0 (local). Latency: high (~5-30s per query depending on hardware).

**When to use:** Privacy-first deployments with no cloud dependency.

**Results:** Requires Ollama running locally with llama3.1 and nomic-embed-text models. Skipped in most benchmark runs (`--skip-ollama`).

---

## A3 — Cloud LLM (Claude Haiku + Contextual Embeddings + HyDE)

**What it does:** Two Claude API enhancements: (1) generates a short context tag per memory before embedding to improve semantic precision, (2) uses HyDE at query time — generates a hypothetical memory and averages its embedding with the raw query embedding.

**Pipeline:** Memory → Claude generates context prefix → GTE-small embeds (prefix + content) → stored. Query → HyDE via Claude → GTE-small embedding → cosine search → top-10

**Cost:** ~$2-4 per 200-query run (HyDE = 1 Claude Haiku call per query).

**When to use:** Full cloud quality tier in production.

**Results (with all-MiniLM-L6-v2, pre-GTE upgrade):**
| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.895 | 0.786 |
| LoCoMo | 0.555 | 0.398 |
| LongMemEval | 0.680 | 0.511 |
| **Aggregate** | **0.710** | **0.565** |

---

## A4 — Hybrid Full Pipeline (BM25 + Dense + Entity + RRF + Graph + Reranker + MMR + Cloud HyDE)

**What it does:** The full production retrieval stack. Combines BM25 keyword, dense cosine, and entity matching into a single candidate pool via Reciprocal Rank Fusion (RRF). Expands candidates via graph links. Applies weighted ranking by intent, then cross-encoder reranking, then MMR for diversity. Query is HyDE-augmented.

**Pipeline:**
1. BM25 FTS5 → candidates
2. Dense cosine (GTE-small, contextual embeddings) → candidates
3. Entity matching → candidates
4. RRF fusion → unified ranked pool
5. Graph expansion (1-hop neighbours)
6. Intent classification → weighted ranking
7. Cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2, top-20)
8. MMR diversification (λ=0.70)
9. HyDE-augmented query vector throughout

**Cost:** ~$2-4 per 200-query run (same as A3 — HyDE dominates cost).

**When to use:** Production cloud path. Best accuracy when API latency is acceptable.

**Added in:** Phase 2.5 (weighted ranking, reranker, MMR). Reranker enabled by default in sub-phase 2.9b.

---

## A4r — Hybrid + Reranker Diagnostic (GTE, no HyDE)

**What it does:** Identical to A4's full hybrid pipeline but with a plain GTE-small cosine query instead of HyDE. No Claude API calls. Used to isolate the reranker's contribution without cloud cost.

**Pipeline:** Same as A4 but Query → GTE-small embedding (no HyDE) → rest of pipeline unchanged.

**Cost:** $0. Latency: ~30-40ms (includes cross-encoder pass).

**When to use:** Diagnostic runs to validate reranker impact without burning API budget.

**Results:**
| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.925 | 0.866 |
| LoCoMo | 0.565 | 0.472 |
| LongMemEval | 0.695 | 0.565 |
| **Aggregate** | **0.728** | **0.635** |

**Key finding:** A4r beats old A4+HyDE (0.728 vs 0.715) even without HyDE — the reranker alone recovers more than HyDE adds on session-level memories. But A5 still dominates (0.768) — representation quality (fact decomposition) beats algorithmic improvements on sessions.

---

## A5 — Extracted Facts (GTE dense, no HyDE)

**What it does:** Uses Claude Haiku to decompose each session memory into 3-5 standalone atomic facts before indexing. Each fact is embedded independently. At retrieval time, a plain cosine search finds the most relevant individual facts, which are mapped back to their parent sessions.

**Pipeline:**
1. Claude Haiku extracts 3-5 facts per session (cached — run once, $0 after)
2. GTE-small embeds each fact independently
3. Query → GTE-small embedding → cosine search over fact corpus → top-10 facts
4. Map retrieved fact UUIDs → parent session IDs for evaluation

**Cost:** ~$0.50 one-time for fact extraction across all 3 datasets. $0 per retrieval after cache.

**Why this matters:** Short atomic facts embed much more precisely than long messy session blobs. A query about "what Alice likes to eat" matches a 10-word fact exactly, not a 1200-character session that mentions food once in passing.

**Results:**
| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.920 | 0.819 |
| LoCoMo | 0.615 | 0.504 |
| LongMemEval | 0.770 | 0.656 |
| **Aggregate** | **0.768** | **0.663** |

**Added in:** Sub-phase 2.8b. This is the technique Mem0 uses to hit 92.5% on LoCoMo — fact decomposition is the architectural unlock, not algorithm tuning.

---

## A5r — Extracted Facts + Cross-Encoder Reranker + Session-MMR (GTE)

**What it does:** Extends A5 with two post-retrieval improvements: (1) cross-encoder reranker scores each (query, fact) pair precisely — much better than cosine for short facts; (2) session-diversity MMR caps 2 facts per parent session to maximise coverage across different memories rather than returning 5 facts from the same session.

**Pipeline:**
1. Same fact extraction and embedding as A5 (uses A5's cache)
2. Query → GTE-small embedding → cosine search → top-20 candidate facts
3. Cross-encoder reranker scores all (query, fact) pairs, reorders
4. Session-MMR: greedy selection, max 2 facts per session, pick top-10
5. Map selected fact UUIDs → parent sessions for evaluation

**Cost:** $0 per retrieval (facts cached, cross-encoder is local). ~$0.50 one-time fact extraction if not already cached.

**Latency:** ~50-100ms (cross-encoder adds ~30-60ms over A5's 9-35ms).

**Added in:** Sub-phase 2.9b (enhanced). The real 2.9b deliverable — applying the reranker to our best approach, not just A4.

**Note on LoCoMo:** A5r regresses slightly on LoCoMo (-0.025 vs A5) because `ms-marco-MiniLM-L-6-v2` was trained on passage retrieval (MS MARCO), not conversational data. GTE cosine similarity handles LoCoMo's short conversational turns better than the reranker. Fix: conversation-tuned cross-encoder (Phase 3).

**Results:**
| Dataset | R@5 | MRR@10 |
|---|---|---|
| SQuAD | 0.955 | 0.880 |
| LoCoMo | 0.590 | 0.519 |
| LongMemEval | 0.815 | 0.680 |
| **Aggregate** | **0.787** | **0.693** |

---

## A3(mh) / A4(mh) / A5r(mh) — Multi-HyDE Variants (2.9c) — NEGATIVE EXPERIMENT

**What it does:** Instead of 1 HyDE call per query, generates 3 hypothetical documents from different angles, embeds each, and **averages** all vectors (including the plain query) into one composite vector.

**Result after full 3-dataset smoke test:**

| Approach | SQuAD R@5 | LoCoMo R@5 | LME R@5 | Agg R@5 |
|---|---|---|---|---|
| A5r baseline (no Multi-HyDE) | **0.955** | **0.590** | **0.815** | **0.787** |
| A5r + Multi-HyDE | 0.940 | 0.590 | 0.805 | 0.778 |
| Delta | -0.015 | 0.000 | -0.010 | **-0.009** |

Same pattern for A3 and A4 — Multi-HyDE consistently hurt or was flat across all datasets and approaches.

**Why it failed:** Averaging 3 hypothetical embeddings produces a blurry centroid vector that sits between interpretations in semantic space, weakening the query signal. A single sharp query embedding outperforms a diluted average.

**The right solution for ambiguous queries (Phase 3):** Multi-query retrieval — run N independent searches and merge via RRF. Preserves signal sharpness per search instead of blending it away.

**Status:** Kept as `--multi-hyde` opt-in flag. Disabled by default.

---

## A5(te3l) / A5r(te3l) — OpenAI text-embedding-3-large (2.9d Cloud Path)

**What it does:** Replaces local GTE embeddings with OpenAI `text-embedding-3-large` (3072-dim) for the cloud path. Same A5/A5r pipeline — only the embedding function changes.

**Cost:** ~$0.01–0.05 per 200-query run (embedding API calls per query). Fact extraction remains cached ($0 after first run).

**Key finding: A5 beats A5r when using te3l.** The ms-marco reranker hurts at high embedding quality — 3072-dim OpenAI embeddings already rank correctly; passage-biased reranking introduces noise for 12-17 word atomic facts.

**Results (te3l vs gtel comparison):**
| Dataset | A4r gtel | A4r te3l | A5 gtel | A5 te3l | A5r gtel | A5r te3l |
|---|---|---|---|---|---|---|
| SQuAD | 0.940 | 0.930 | 0.920 | **0.950** | 0.955 | 0.950 |
| LoCoMo | 0.580 | 0.590 | 0.615 | **0.620** | 0.595 | 0.590 |
| LongMemEval | 0.725 | 0.750 | 0.770 | **0.800** | 0.825 | 0.810 |
| **Aggregate R@5** | 0.748 | 0.757 | 0.768 | **0.790** | **0.792** | 0.783 |
| **Aggregate MRR@10** | 0.649 | 0.651 | 0.656 | 0.686 | 0.692 | **0.694** |
| **p50 latency** | 43ms | 243ms | 30ms | 216ms | 47ms | 281ms |

**Recommended production paths:**
- **Local/privacy**: `gtel` + A5r → 0.792 R@5, ~47ms p50 (no API cost)
- **Cloud quality**: `te3l` + A5 → 0.790 R@5, ~216ms p50 (skip reranker)

---

## Approach Evolution Summary

```
A1   (BM25 keyword)                          → 0.040 aggregate R@5
A3   (Cloud LLM + HyDE, MiniLM)             → 0.710
A4   (Full hybrid + HyDE, no reranker)       → 0.715
A4r  (Full hybrid + reranker, no HyDE)       → 0.728   ← reranker adds +0.013
A5   (Fact decomposition, GTE-small)         → 0.768   ← representation >> algorithm
A5r  (Facts + reranker + session-MMR, gtel)  → 0.792   ← reranker on facts adds +0.024
                                                          LongMemEval 0.825 ✓ beats Zep (71.2%)
A5r+mh (Facts + reranker + Multi-HyDE)      → 0.778   ← -0.014 vs A5r(gtel), Multi-HyDE fails
A5   (te3l cloud path, OpenAI 3072-dim)      → 0.790   ← best cloud path; reranker hurts here
```

Plan target R@5 ≥ 0.80 hit on SQuAD (0.955), LongMemEval (0.825).
LoCoMo (0.590-0.620) still below target — requires Phase 3 (conversation-tuned reranker + multi-query retrieval).

**The key architectural insights:**
1. Representation quality (what you store) matters more than retrieval algorithm. A5 outperforms A4 by +5.3% simply by decomposing sessions into atomic facts.
2. Reranker benefit depends on embedding quality. With weak embeddings (GTE-small), the ms-marco reranker rescues precision. With strong embeddings (te3l 3072-dim), it introduces noise — skip the reranker in the cloud path.

---

## Running the Benchmark

```bash
# $0 diagnostic — A1, A4r, A5, A5r with gtel (local, facts already cached)
python3 -u benchmark_4approaches.py --skip-ollama --skip-cloud

# Cloud path — te3l (requires OPENAI_API_KEY, ~$0.05 per 200-query run)
python3 -u benchmark_4approaches.py --skip-ollama --skip-cloud --embed-model te3l

# Full smoke test — all approaches including A3/A4 with HyDE (~$2-3)
python3 -u benchmark_4approaches.py --skip-ollama

# With Multi-HyDE enabled (experimental, negative result, ~$8-10, ~3h runtime)
python3 -u benchmark_4approaches.py --skip-ollama --multi-hyde

# Specific embedding models
python3 -u benchmark_4approaches.py --embed-model gtel   # GTE-large (local, recommended)
python3 -u benchmark_4approaches.py --embed-model te3l   # OpenAI 3072-dim (cloud best)
python3 -u benchmark_4approaches.py --embed-model bge    # BGE-small
python3 -u benchmark_4approaches.py --embed-model minilm # original MiniLM

# Full dataset publishable run (~$12)
python3 -u benchmark_4approaches.py --skip-ollama --n 10000
```

---

## Competitor Targets

| Competitor | Dataset | Their Score | Our Status |
|---|---|---|---|
| Letta | LoCoMo | 74.0% | A5r targets this |
| Zep CE | LongMemEval | 71.2% | Already beaten by A5r (81.5%) |
| Mem0 | LoCoMo | 92.5% | Phase 3 target |
| Mem0 | LongMemEval | 94.4% | Phase 3 target |
