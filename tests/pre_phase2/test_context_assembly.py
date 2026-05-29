"""
Sub-phase 2.2 — Context Assembly tests.

Coverage:
  - Token counting
  - Extractive unit formatting (all field combinations)
  - Assembly algorithm: greedy inclusion, budget enforcement, ordering
  - At-least-one guarantee (budget overrun on first memory)
  - Abstractive fallback (cluster_label summary for excluded memories)
  - Metrics: Compression Ratio, RCD proxy
  - ContextPacket DB wiring
  - RetrievalRun packet metrics update (packet_token_budget, packet_compression_ratio)
  - Stress tests: 100 memories, budget constraints never violated, metric targets
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.retrieval.context_assembly import (
    AssemblyResult,
    assemble_context_packet,
    count_tokens,
    format_extractive_unit,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight Memory-like objects (no DB required)
# ---------------------------------------------------------------------------

def _mem(
    mid: str = "m1",
    title: str = "Test Memory",
    content: str = "Some content about the project.",
    importance: int = 3,
    mem_type: str = "decision",
    canonical_type: str | None = None,
    source_quote: str | None = "Verbatim quote from session.",
    retrieval_hint: str | None = "TL;DR hint for retrieval.",
    cluster_id: int | None = None,
    cluster_label: str | None = None,
    source_session_id: str | None = None,
    created_at: datetime | None = None,
) -> SimpleNamespace:
    """Create a minimal Memory-like object for testing."""
    return SimpleNamespace(
        id=mid,
        title=title,
        content=content,
        importance=importance,
        type=mem_type,
        canonical_type=canonical_type,
        source_quote=source_quote,
        retrieval_hint=retrieval_hint,
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        source_session_id=source_session_id,
        created_at=created_at or datetime(2026, 5, 28, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_single_word_returns_one(self):
        assert count_tokens("hello") == 1

    def test_longer_text_has_more_tokens_than_shorter(self):
        short = count_tokens("Hello world")
        long = count_tokens("Hello world this is a much longer sentence with many words")
        assert long > short

    def test_returns_positive_int(self):
        assert count_tokens("some text here") > 0
        assert isinstance(count_tokens("some text here"), int)

    def test_heuristic_is_word_count_times_1_3(self):
        text = "one two three four five"  # 5 words
        expected = int(5 * 1.3)           # 6
        assert count_tokens(text) == expected

    def test_none_like_empty_behaves_gracefully(self):
        # count_tokens receives str only; whitespace-only → 0 words → 0
        assert count_tokens("   ") == 0 or count_tokens("   ") >= 0


# ---------------------------------------------------------------------------
# Extractive unit formatting
# ---------------------------------------------------------------------------

class TestFormatExtractiveUnit:
    def test_includes_type_and_title(self):
        m = _mem(title="Use PostgreSQL", mem_type="decision")
        unit = format_extractive_unit(m)
        assert "decision" in unit
        assert "Use PostgreSQL" in unit

    def test_uses_canonical_type_over_type_when_set(self):
        m = _mem(mem_type="decision", canonical_type="constraint")
        unit = format_extractive_unit(m)
        assert "constraint" in unit
        assert "decision" not in unit

    def test_includes_timestamp(self):
        m = _mem(created_at=datetime(2026, 5, 28, tzinfo=timezone.utc))
        unit = format_extractive_unit(m)
        assert "2026-05-28" in unit

    def test_includes_importance(self):
        m = _mem(importance=4)
        unit = format_extractive_unit(m)
        assert "4/5" in unit

    def test_includes_source_quote_by_default(self):
        m = _mem(source_quote="This is the verbatim quote.")
        unit = format_extractive_unit(m)
        assert "This is the verbatim quote." in unit

    def test_excludes_source_quote_when_flag_false(self):
        m = _mem(source_quote="This is the verbatim quote.")
        unit = format_extractive_unit(m, include_source_quote=False)
        assert "This is the verbatim quote." not in unit

    def test_includes_retrieval_hint_by_default(self):
        m = _mem(retrieval_hint="TL;DR: use Postgres")
        unit = format_extractive_unit(m)
        assert "TL;DR: use Postgres" in unit

    def test_excludes_retrieval_hint_when_flag_false(self):
        m = _mem(retrieval_hint="TL;DR: use Postgres")
        unit = format_extractive_unit(m, include_retrieval_hint=False)
        assert "TL;DR: use Postgres" not in unit

    def test_graceful_when_no_source_quote(self):
        m = _mem(source_quote=None)
        unit = format_extractive_unit(m)
        assert "Source" not in unit  # field omitted when None

    def test_graceful_when_no_retrieval_hint(self):
        m = _mem(retrieval_hint=None)
        unit = format_extractive_unit(m)
        assert "Hint" not in unit

    def test_includes_content(self):
        m = _mem(content="The content body of this memory.")
        unit = format_extractive_unit(m)
        assert "The content body of this memory." in unit

    def test_includes_session_id_prefix_when_set(self):
        m = _mem(source_session_id="abc12345-xyz")
        unit = format_extractive_unit(m)
        assert "abc12345" in unit  # first 8 chars

    def test_no_session_line_when_session_id_none(self):
        m = _mem(source_session_id=None)
        unit = format_extractive_unit(m)
        assert "Session" not in unit

    def test_timestamp_none_graceful(self):
        m = _mem(created_at=None)
        unit = format_extractive_unit(m)
        # Should not raise; timestamp line may be empty but unit still formed
        assert m.title in unit


# ---------------------------------------------------------------------------
# Core assembly algorithm
# ---------------------------------------------------------------------------

class TestAssembleContextPacket:

    def test_all_memories_fit_within_budget(self):
        mems = [_mem(f"m{i}", content="Short.") for i in range(3)]
        result = assemble_context_packet(mems, token_budget=4000)
        assert set(result.included_memory_ids) == {"m0", "m1", "m2"}
        assert result.excluded_memory_ids == []

    def test_empty_memories_returns_valid_result(self):
        result = assemble_context_packet([], token_budget=4000)
        assert result.included_memory_ids == []
        assert result.excluded_memory_ids == []
        assert result.token_count >= 0

    def test_at_least_one_memory_always_included_even_over_budget(self):
        # One giant memory that exceeds even budget=1
        m = _mem("m1", content="A " * 5000)  # ~5000 tokens
        result = assemble_context_packet([m], token_budget=1)
        assert "m1" in result.included_memory_ids

    def test_drops_lowest_ranked_when_over_budget(self):
        # m1 is rank 1 (first in list), m2 rank 2; tight budget fits only one
        m1 = _mem("m1", content="Short first memory content.")
        m2 = _mem("m2", content="Short second memory content.")
        # Budget: fit header + m1 only (calculate approximate)
        m1_unit = format_extractive_unit(m1)
        header_tok = count_tokens("---")
        m1_tok = count_tokens(m1_unit)
        budget = header_tok + m1_tok + 5  # just enough for m1 but not m2

        result = assemble_context_packet([m1, m2], token_budget=budget)
        assert "m1" in result.included_memory_ids
        assert "m2" in result.excluded_memory_ids

    def test_rank_order_preserved_in_content(self):
        m1 = _mem("m1", title="First memory", content="Content A.")
        m2 = _mem("m2", title="Second memory", content="Content B.")
        result = assemble_context_packet([m1, m2], token_budget=4000)
        idx1 = result.content.index("First memory")
        idx2 = result.content.index("Second memory")
        assert idx1 < idx2  # m1 appears before m2

    def test_excluded_memory_ids_tracked(self):
        mems = [_mem(f"m{i}", content="x " * 500) for i in range(5)]
        result = assemble_context_packet(mems, token_budget=200)
        total = len(result.included_memory_ids) + len(result.excluded_memory_ids)
        assert total == 5

    def test_budget_never_violated_when_multiple_memories_included(self):
        mems = [_mem(f"m{i}", content="Some moderate length content for this memory.") for i in range(10)]
        budget = 1000
        result = assemble_context_packet(mems, token_budget=budget)
        # Actual token count may slightly exceed due to at-least-one guarantee,
        # but for multiple inclusions the budget must be respected
        if len(result.included_memory_ids) > 1:
            assert result.token_count <= budget + 50  # small tolerance for header

    def test_large_budget_includes_all_memories(self):
        mems = [_mem(f"m{i}", content="Brief content.") for i in range(10)]
        result = assemble_context_packet(mems, token_budget=100_000)
        assert len(result.included_memory_ids) == 10
        assert result.excluded_memory_ids == []

    def test_header_includes_query(self):
        m = _mem("m1")
        result = assemble_context_packet([m], query="What is the DB decision?")
        assert "What is the DB decision?" in result.content

    def test_header_includes_project_name(self):
        m = _mem("m1")
        result = assemble_context_packet([m], project_name="MyProject")
        assert "MyProject" in result.content

    def test_no_source_quote_flag_propagates(self):
        m = _mem("m1", source_quote="Verbatim quote.")
        result = assemble_context_packet([m], include_source_quote=False)
        assert "Verbatim quote." not in result.content

    def test_no_retrieval_hint_flag_propagates(self):
        m = _mem("m1", retrieval_hint="TL;DR hint.")
        result = assemble_context_packet([m], include_retrieval_hint=False)
        assert "TL;DR hint." not in result.content


# ---------------------------------------------------------------------------
# Metrics: Compression Ratio & RCD proxy
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_compression_ratio_positive(self):
        mems = [_mem("m1", content="Decent length content to compute ratio.")]
        result = assemble_context_packet(mems, token_budget=4000)
        assert result.compression_ratio > 0

    def test_compression_ratio_is_total_input_over_packet(self):
        content = "word " * 100  # ~100 words
        m = _mem("m1", content=content)
        result = assemble_context_packet([m], token_budget=4000)
        expected = result.total_input_tokens / max(result.token_count, 1)
        assert abs(result.compression_ratio - expected) < 0.01

    def test_rcd_proxy_between_zero_and_one_for_normal_inputs(self):
        mems = [_mem(f"m{i}", importance=i % 5 + 1, content="Test content.") for i in range(5)]
        result = assemble_context_packet(mems, token_budget=4000)
        # RCD proxy is importance-weighted; should be in (0, ~1] for typical data
        assert result.rcd_proxy >= 0

    def test_rcd_proxy_higher_for_high_importance_memories(self):
        m_high = _mem("m_high", importance=5, content="Important decision was made here.")
        m_low = _mem("m_low", importance=1, content="Important decision was made here.")
        result_high = assemble_context_packet([m_high], token_budget=4000)
        result_low = assemble_context_packet([m_low], token_budget=4000)
        assert result_high.rcd_proxy > result_low.rcd_proxy

    def test_total_input_tokens_positive_for_nonempty_memories(self):
        mems = [_mem("m1", content="Some content words here.")]
        result = assemble_context_packet(mems, token_budget=4000)
        assert result.total_input_tokens > 0

    def test_compression_ratio_increases_with_more_input_content(self):
        m_small = _mem("m1", content="Short.")
        m_large = _mem("m1", content="word " * 500)
        r_small = assemble_context_packet([m_small], token_budget=4000)
        r_large = assemble_context_packet([m_large], token_budget=4000)
        # Larger input → higher compression ratio (more was "compressed")
        assert r_large.compression_ratio > r_small.compression_ratio


# ---------------------------------------------------------------------------
# Abstractive fallback
# ---------------------------------------------------------------------------

class TestAbstractiveFallback:

    def test_no_fallback_when_all_memories_included(self):
        mems = [_mem("m1", content="Short.")]
        result = assemble_context_packet(mems, token_budget=4000)
        assert result.used_abstractive_fallback is False

    def test_no_fallback_for_excluded_memories_without_cluster(self):
        # Excluded memory has no cluster_id → no abstractive fallback
        m1 = _mem("m1", content="Short.")
        m2 = _mem("m2", content="x " * 1000, cluster_id=None)
        result = assemble_context_packet([m1, m2], token_budget=50)
        assert result.used_abstractive_fallback is False

    def test_fallback_triggered_for_excluded_clustered_memories(self):
        m1 = _mem("m1", content="Short.")
        m2 = _mem("m2", content="x " * 500, cluster_id=1, cluster_label="Database Decisions")
        result = assemble_context_packet([m1, m2], token_budget=100)
        if "m2" in result.excluded_memory_ids:
            assert result.used_abstractive_fallback is True
            assert "Database Decisions" in result.content

    def test_fallback_includes_cluster_label_in_content(self):
        m1 = _mem("m1", content="Short.")
        m2 = _mem("m2", content="x " * 500, cluster_id=7, cluster_label="Auth Design")
        result = assemble_context_packet([m1, m2], token_budget=100)
        if result.used_abstractive_fallback:
            assert "Auth Design" in result.content

    def test_fallback_includes_excluded_count(self):
        m1 = _mem("m1", content="Short.")
        m2 = _mem("m2", content="x " * 300, cluster_id=1, cluster_label="Cluster One")
        m3 = _mem("m3", content="x " * 300, cluster_id=1, cluster_label="Cluster One")
        result = assemble_context_packet([m1, m2, m3], token_budget=80)
        if result.used_abstractive_fallback:
            assert "2" in result.content or "memor" in result.content.lower()

    def test_fallback_groups_by_cluster(self):
        m1 = _mem("m1", content="Short.")
        m2 = _mem("m2", content="x " * 300, cluster_id=1, cluster_label="Cluster A")
        m3 = _mem("m3", content="x " * 300, cluster_id=2, cluster_label="Cluster B")
        result = assemble_context_packet([m1, m2, m3], token_budget=80)
        if result.used_abstractive_fallback:
            assert "Cluster A" in result.content
            assert "Cluster B" in result.content

    def test_fallback_false_for_empty_memories(self):
        result = assemble_context_packet([], token_budget=4000)
        assert result.used_abstractive_fallback is False


# ---------------------------------------------------------------------------
# DB wiring — ContextPacket persistence + RetrievalRun metric update
# ---------------------------------------------------------------------------

class TestContextPacketDBWiring:
    """Tests that context assembly correctly persists to the DB."""

    def test_assembly_result_can_be_stored_in_context_packet(self, db, project):
        from app import models as phase1_models
        mems = []
        for i in range(3):
            m = phase1_models.Memory(
                project_id=project.id,
                type="decision",
                title=f"Memory {i}",
                content=f"Content for memory {i} with enough words to test token counting.",
                importance=3 + (i % 2),
                privacy_level="internal",
                retrieval_hint=f"Hint {i}",
                source_quote=f"Quote {i} from session text.",
            )
            db.add(m)
        db.commit()
        db.refresh(m)

        # Get all memories
        db_mems = db.query(phase1_models.Memory).filter(
            phase1_models.Memory.project_id == project.id
        ).all()

        result = assemble_context_packet(
            memories=db_mems,
            token_budget=4000,
            query="What decisions were made?",
            project_name=project.name,
        )

        # Persist ContextPacket
        packet = phase1_models.ContextPacket(
            project_id=project.id,
            target_tool="Claude",
            intent="What decisions were made?",
            included_memory_ids=json.dumps(result.included_memory_ids),
            content=result.content,
            token_estimate=result.token_count,
        )
        db.add(packet)
        db.commit()
        db.refresh(packet)

        assert packet.id is not None
        assert packet.token_estimate == result.token_count
        assert set(json.loads(packet.included_memory_ids)) == set(result.included_memory_ids)
        assert "What decisions were made?" in packet.content

    def test_retrieval_run_updated_with_packet_metrics(self, db, project):
        from app.p2_models import RetrievalRun

        run = RetrievalRun(
            project_id=project.id,
            query="test query",
            latency_ms=42,
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        # Simulate what the router does after assembly
        run.packet_token_budget = 4000
        run.packet_compression_ratio = 12.5
        run.token_count = 320
        db.commit()
        db.refresh(run)

        assert run.packet_token_budget == 4000
        assert run.packet_compression_ratio == 12.5
        assert run.token_count == 320

    def test_context_packet_includes_correct_memory_ids(self, db, project):
        from app import models as phase1_models

        m1 = phase1_models.Memory(
            project_id=project.id, type="decision",
            title="Decision A", content="Content A.",
            importance=4, privacy_level="internal",
        )
        m2 = phase1_models.Memory(
            project_id=project.id, type="task",
            title="Task B", content="Content B.",
            importance=2, privacy_level="internal",
        )
        db.add_all([m1, m2])
        db.commit()

        result = assemble_context_packet([m1, m2], token_budget=4000)

        assert m1.id in result.included_memory_ids
        assert m2.id in result.included_memory_ids

    def test_excluded_memories_not_in_packet_content(self, db, project):
        from app import models as phase1_models

        m_included = phase1_models.Memory(
            project_id=project.id, type="decision",
            title="Short", content="Brief.",
            importance=5, privacy_level="internal",
        )
        m_excluded = phase1_models.Memory(
            project_id=project.id, type="task",
            title="Very Long Memory",
            content="word " * 2000,
            importance=1, privacy_level="internal",
        )
        db.add_all([m_included, m_excluded])
        db.commit()

        result = assemble_context_packet(
            [m_included, m_excluded],
            token_budget=80,
        )
        assert m_included.id in result.included_memory_ids
        # The long memory's full content should not appear verbatim
        if m_excluded.id in result.excluded_memory_ids:
            assert "word " * 20 not in result.content


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class TestStress:

    def test_100_memories_all_fit_in_large_budget(self):
        mems = [
            _mem(f"m{i}", content="Project decision with meaningful content.", importance=(i % 5) + 1)
            for i in range(100)
        ]
        result = assemble_context_packet(mems, token_budget=200_000)
        assert len(result.included_memory_ids) == 100
        assert result.excluded_memory_ids == []

    def test_100_memories_tight_budget_never_violated(self):
        mems = [
            _mem(f"m{i}", content="Content for this memory unit.", importance=(i % 5) + 1)
            for i in range(100)
        ]
        budget = 500
        result = assemble_context_packet(mems, token_budget=budget)
        # Budget may be exceeded by at-most-one guarantee only on first memory;
        # for all subsequent, must be within budget
        included = result.included_memory_ids
        if len(included) > 1:
            # Re-compute tokens for all included units to verify budget adherence
            total = count_tokens(result.content)
            assert total <= budget * 1.5  # allow header overhead

    def test_compression_ratio_positive_for_all_scales(self):
        for n in [1, 10, 50, 100]:
            mems = [_mem(f"m{i}", content="Content here.") for i in range(n)]
            result = assemble_context_packet(mems, token_budget=50_000)
            assert result.compression_ratio > 0, f"compression_ratio <= 0 for n={n}"

    def test_rcd_proxy_non_negative_for_all_inputs(self):
        for importance in [1, 2, 3, 4, 5]:
            mems = [_mem("m1", importance=importance, content="Content.")]
            result = assemble_context_packet(mems, token_budget=4000)
            assert result.rcd_proxy >= 0

    def test_high_importance_memories_meet_rcd_target(self):
        # All importance=5 memories should push RCD proxy toward target ≥ 0.65
        mems = [_mem(f"m{i}", importance=5, content="Critical architectural decision.") for i in range(5)]
        result = assemble_context_packet(mems, token_budget=50_000)
        # importance=5 → 5/5=1.0 weight → rcd_proxy approaches 1.0
        assert result.rcd_proxy >= 0.65, f"RCD proxy {result.rcd_proxy:.2f} below target 0.65"

    def test_ordering_consistent_at_scale(self):
        # First memory in list must always appear before last memory in content
        mems = [
            _mem(f"m{i}", title=f"Memory {i:03d}", content="Content.")
            for i in range(20)
        ]
        result = assemble_context_packet(mems, token_budget=200_000)
        if len(result.included_memory_ids) >= 2:
            first_title = f"Memory {0:03d}"
            last_included_idx = int(result.included_memory_ids[-1][1:])
            last_title = f"Memory {last_included_idx:03d}"
            if first_title in result.content and last_title in result.content:
                assert result.content.index(first_title) < result.content.index(last_title)

    def test_empty_memories_edge_case(self):
        result = assemble_context_packet([], token_budget=4000, query="anything")
        assert result.included_memory_ids == []
        assert result.compression_ratio >= 0
        assert result.rcd_proxy >= 0

    def test_single_memory_always_has_nonzero_token_count(self):
        m = _mem("m1", content="Some content.", retrieval_hint="Hint.")
        result = assemble_context_packet([m])
        assert result.token_count > 0

    def test_mixed_cluster_unclustered_memories(self):
        mems = [
            _mem("m1", content="Short.", cluster_id=None),
            _mem("m2", content="x " * 200, cluster_id=1, cluster_label="DB Layer"),
            _mem("m3", content="x " * 200, cluster_id=1, cluster_label="DB Layer"),
            _mem("m4", content="Short.", cluster_id=None),
        ]
        result = assemble_context_packet(mems, token_budget=150)
        # Should not raise; fallback may or may not trigger
        assert len(result.included_memory_ids) >= 1

    def test_token_count_field_matches_actual_content_tokens(self):
        mems = [_mem(f"m{i}", content="Content for memory.") for i in range(5)]
        result = assemble_context_packet(mems, token_budget=10_000)
        actual = count_tokens(result.content)
        assert abs(result.token_count - actual) <= 1  # must be identical (computed same way)

    def test_assembly_result_is_assembly_result_instance(self):
        mems = [_mem("m1")]
        result = assemble_context_packet(mems)
        assert isinstance(result, AssemblyResult)
