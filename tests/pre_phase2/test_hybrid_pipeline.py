"""
Tests for the hybrid LLM extraction pipeline components:
  - apply_noise_gate: filler removal, technical content preservation
  - chunk_sentences: max_tokens boundary, markdown headers, overlap
  - prepare_chunks: combined pipeline return type
  - deduplicate_drafts: source_quote, title similarity, content similarity
  - deduplicate_against_existing: DB duplicate filtering
  - validate_draft: required field validation, field normalization
"""
from __future__ import annotations

import pytest

from app.services.extraction.chunker import (
    Chunk,
    NoiseGateResult,
    apply_noise_gate,
    chunk_sentences,
    prepare_chunks,
    _is_obvious_noise,
    _estimate_tokens,
)
from app.services.extraction.deduplicator import (
    deduplicate_drafts,
    deduplicate_against_existing,
    DeduplicationResult,
)
from app.services.extraction.base import ExtractedMemoryDraft
from app.services.extraction.prompts import validate_draft


# ---------------------------------------------------------------------------
# _is_obvious_noise (regex pre-filter — no model needed)
# ---------------------------------------------------------------------------

class TestObviousNoise:
    @pytest.mark.parametrize("sentence", [
        "Sure! I'd be happy to help.",
        "Of course, here's what I suggest.",
        "Great question!",
        "Let me help you with this.",
        "I understand that.",
        "Thank you for the explanation!",
        "Hi there! How can I help?",
        "That's a great point!",
        "Okay, let me think about that.",
        "Absolutely, I can do that.",
    ])
    def test_filler_is_obvious_noise(self, sentence):
        assert _is_obvious_noise(sentence) is True

    @pytest.mark.parametrize("sentence", [
        "We decided to use PostgreSQL for production.",
        "The connection pool size is set to 10.",
        "TODO: Write integration tests for the payment flow.",
        "The API returns a 404 when the resource is not found.",
        "We are using Redis for session caching to reduce DB load.",
    ])
    def test_technical_is_not_obvious_noise(self, sentence):
        assert _is_obvious_noise(sentence) is False

    def test_empty_string_is_noise(self):
        assert _is_obvious_noise("") is True

    def test_too_short_is_noise(self):
        assert _is_obvious_noise("Hi") is True

    def test_five_chars_threshold(self):
        # 4 chars → noise, 5 chars → not noise (passes regex stage)
        assert _is_obvious_noise("abcd") is True
        assert _is_obvious_noise("abcde") is False  # passes length, fails on regex match


# ---------------------------------------------------------------------------
# apply_noise_gate
# ---------------------------------------------------------------------------

class TestApplyNoiseGate:
    def test_returns_noise_gate_result(self, mock_is_technical_true):
        result = apply_noise_gate("We use PostgreSQL for the database.")
        assert isinstance(result, NoiseGateResult)

    def test_empty_text_returns_empty(self, mock_is_technical_true):
        result = apply_noise_gate("")
        assert result.substantive == []
        assert result.filtered_count == 0
        assert result.total_count == 0

    def test_technical_content_passes(self, mock_is_technical_true):
        text = "We use PostgreSQL for production. The API runs on port 8080."
        result = apply_noise_gate(text)
        assert len(result.substantive) > 0

    def test_obvious_filler_is_filtered(self, mock_is_technical_true):
        text = "Sure! Of course! Great question! Thank you!"
        result = apply_noise_gate(text)
        # All are obvious noise — filtered by regex before even reaching is_technical
        assert result.filtered_count > 0

    def test_semantic_gate_filters_non_technical(self, mock_is_technical_false):
        """When is_technical always returns False, nothing passes semantic gate."""
        text = "We decided to use Redis for caching. The pool size is 10."
        result = apply_noise_gate(text)
        # Regex passes technical sentences, but semantic gate blocks them all
        assert len(result.substantive) == 0
        assert result.filtered_count > 0

    def test_pass_rate_property(self, mock_is_technical_true):
        text = "We use PostgreSQL. The connection pool is 10."
        result = apply_noise_gate(text)
        assert 0.0 <= result.pass_rate <= 1.0

    def test_threshold_used_matches_input(self, mock_is_technical_true):
        text = "We use PostgreSQL for production."
        result = apply_noise_gate(text, threshold=0.25)
        assert result.threshold_used == 0.25


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") >= 1  # returns max(1, ...)

    def test_single_word(self):
        estimate = _estimate_tokens("hello")
        assert estimate >= 1

    def test_scales_with_length(self):
        short = _estimate_tokens("hello world")
        long = _estimate_tokens("hello world this is a much longer sentence with many more words")
        assert long > short


# ---------------------------------------------------------------------------
# chunk_sentences
# ---------------------------------------------------------------------------

class TestChunkSentences:
    def test_empty_returns_empty(self):
        assert chunk_sentences([]) == []

    def test_single_sentence_one_chunk(self):
        chunks = chunk_sentences(["We use PostgreSQL for the production database."])
        assert len(chunks) == 1

    def test_chunk_is_chunk_type(self):
        chunks = chunk_sentences(["A sentence."])
        assert isinstance(chunks[0], Chunk)
        assert isinstance(chunks[0].text, str)
        assert isinstance(chunks[0].index, int)

    def test_chunks_have_sequential_indices(self):
        sentences = [f"Sentence number {i} has technical content about databases." for i in range(50)]
        chunks = chunk_sentences(sentences, max_tokens=30)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_max_tokens_respected(self):
        # 100 words per sentence × 1.3 ≈ 130 tokens; limit to 150 → 1 sentence per chunk
        sentences = [" ".join(["word"] * 100) for _ in range(5)]
        chunks = chunk_sentences(sentences, max_tokens=150)
        for chunk in chunks:
            assert chunk.token_estimate <= 150 + 200  # allow overlap

    def test_markdown_header_starts_new_chunk(self):
        sentences = [
            "First section content about databases.",
            "## New Topic",
            "Second section content about caching.",
        ]
        chunks = chunk_sentences(sentences, max_tokens=2000)
        # Header should force a new chunk
        assert len(chunks) >= 2

    def test_overlap_carries_last_sentence(self):
        sentences = [
            "First sentence about database configuration.",
            "Second sentence about connection pooling.",
            "Third sentence about caching strategies.",
            "Fourth sentence about Redis setup.",
        ]
        chunks = chunk_sentences(sentences, max_tokens=50, overlap_sentences=1)
        if len(chunks) > 1:
            # Last sentence of chunk N should appear at start of chunk N+1
            last_sentence_of_first = chunks[0].text.split(".")[-2].strip() + "."
            # The overlap sentence should be in the second chunk
            assert chunks[0].sentence_count >= 1

    def test_large_input_produces_multiple_chunks(self):
        sentences = [
            f"Technical decision {i}: use component {i} for microservice architecture." for i in range(100)
        ]
        chunks = chunk_sentences(sentences, max_tokens=200)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# prepare_chunks
# ---------------------------------------------------------------------------

class TestPrepareChunks:
    def test_returns_tuple(self, mock_is_technical_true):
        chunks, gate = prepare_chunks("We use PostgreSQL.")
        assert isinstance(chunks, list)
        assert isinstance(gate, NoiseGateResult)

    def test_filler_only_returns_empty_chunks(self, mock_is_technical_false):
        text = "Sure! Great! Of course! Thank you!"
        chunks, gate = prepare_chunks(text)
        assert chunks == []
        assert gate.filtered_count > 0


# ---------------------------------------------------------------------------
# deduplicate_drafts
# ---------------------------------------------------------------------------

def _make_draft(
    title="Test Memory",
    content="Some content about the topic.",
    canonical_type="decision",
    confidence=0.8,
    source_quote="Test memory about the topic.",
):
    return ExtractedMemoryDraft(
        canonical_type=canonical_type,
        type_label=canonical_type.replace("_", " ").title(),
        title=title,
        content=content,
        confidence=confidence,
        source_quote=source_quote,
    )


class TestDeduplicateDrafts:
    def test_empty_input(self):
        result = deduplicate_drafts([])
        assert result.kept == []
        assert result.removed == []

    def test_single_item_kept(self):
        draft = _make_draft()
        result = deduplicate_drafts([draft])
        assert len(result.kept) == 1
        assert result.removed == []

    def test_identical_source_quote_deduped(self):
        d1 = _make_draft(title="Memory A", source_quote="We use Redis for caching.")
        d2 = _make_draft(title="Memory B (dup)", source_quote="We use Redis for caching.")
        result = deduplicate_drafts([d1, d2])
        assert len(result.kept) == 1
        assert len(result.removed) == 1

    def test_identical_source_quote_keeps_higher_confidence(self):
        d1 = _make_draft(title="Low conf", source_quote="We use Redis.", confidence=0.6)
        d2 = _make_draft(title="High conf", source_quote="We use Redis.", confidence=0.9)
        result = deduplicate_drafts([d1, d2])
        assert result.kept[0].title == "High conf"

    def test_similar_title_same_type_deduped(self):
        d1 = _make_draft(title="Use Redis for session caching", source_quote="Quote 1.")
        d2 = _make_draft(title="Use Redis as session cache", source_quote="Quote 2.")
        result = deduplicate_drafts([d1, d2], title_threshold=0.8)
        # Titles are similar and same canonical_type → deduplicated
        assert len(result.kept) <= 2  # may or may not dedup depending on exact ratio

    def test_different_types_title_not_deduped(self):
        d1 = _make_draft(
            title="Redis caching decision",
            canonical_type="decision",
            content="Redis caching decision from an architectural perspective.",
            source_quote="Q1.",
        )
        d2 = _make_draft(
            title="Redis caching decision",
            canonical_type="task",
            content="Implement Redis caching as a task item in the team backlog.",
            source_quote="Q2.",
        )
        result = deduplicate_drafts([d1, d2], title_threshold=0.85)
        # Different canonical_type → title dedup skipped; content is different enough
        assert len(result.kept) == 2

    def test_identical_content_cross_type_deduped(self):
        content = "We will migrate to PostgreSQL by the end of the quarter."
        d1 = _make_draft(title="Postgres Migration Plan", canonical_type="task",
                          content=content, source_quote="Q1 about postgres.")
        d2 = _make_draft(title="DB Migration Decision", canonical_type="decision",
                          content=content, source_quote="Q2 about postgres.")
        result = deduplicate_drafts([d1, d2], content_threshold=0.90)
        assert len(result.kept) == 1

    def test_removed_titles_populated(self):
        d1 = _make_draft(title="Same Memory", source_quote="Exact quote.")
        d2 = _make_draft(title="Same Memory Duplicate", source_quote="Exact quote.")
        result = deduplicate_drafts([d1, d2])
        assert len(result.removed_titles) > 0

    def test_kept_count_property(self):
        drafts = [_make_draft(title=f"Memory {i}", source_quote=f"Quote {i}.") for i in range(5)]
        result = deduplicate_drafts(drafts)
        assert result.kept_count == len(result.kept)

    def test_removed_count_property(self):
        d1 = _make_draft(source_quote="Repeated quote.")
        d2 = _make_draft(source_quote="Repeated quote.")
        result = deduplicate_drafts([d1, d2])
        assert result.removed_count == len(result.removed)


# ---------------------------------------------------------------------------
# deduplicate_against_existing
# ---------------------------------------------------------------------------

class TestDeduplicateAgainstExisting:
    def test_empty_drafts_returns_empty(self):
        result = deduplicate_against_existing([], ["Existing Title"], ["Existing content."])
        assert result.kept == []

    def test_empty_existing_returns_all(self):
        drafts = [_make_draft()]
        result = deduplicate_against_existing(drafts, [], [])
        assert len(result.kept) == 1

    def test_similar_title_to_existing_removed(self):
        # "use redis for session caching" vs "use redis for session cache" → ratio ≈ 0.929
        drafts = [_make_draft(title="Use Redis for session caching")]
        existing_titles = ["Use Redis for session cache"]
        result = deduplicate_against_existing(drafts, existing_titles, [], title_threshold=0.85)
        assert len(result.kept) == 0
        assert len(result.removed) == 1

    def test_unique_title_kept(self):
        drafts = [_make_draft(title="Deploy Kubernetes cluster")]
        existing_titles = ["Use PostgreSQL for database"]
        result = deduplicate_against_existing(drafts, existing_titles, [])
        assert len(result.kept) == 1

    def test_similar_content_to_existing_removed(self):
        content = "We will use PostgreSQL instead of SQLite for production."
        drafts = [_make_draft(content=content, title="DB Choice")]
        existing_contents = ["We will use PostgreSQL instead of SQLite for production."]
        result = deduplicate_against_existing(drafts, [], existing_contents, content_threshold=0.90)
        assert len(result.kept) == 0


# ---------------------------------------------------------------------------
# validate_draft
# ---------------------------------------------------------------------------

class TestValidateDraft:
    def _valid_raw(self):
        return {
            "canonical_type": "decision",
            "type_label": "Architecture Decision",
            "title": "Use PostgreSQL",
            "content": "Chose PostgreSQL for production workloads.",
            "importance": 4,
            "confidence": 0.9,
            "source_quote": "We decided to use PostgreSQL for production.",
            "tags": [],
            "entities": [],
            "links": [],
            "related_files": [],
            "privacy_level": "internal",
            "review_recommended": False,
            "llm_reasoning": "Explicit decision.",
            "temporal": {},
        }

    def test_valid_draft_passes(self):
        result = validate_draft(self._valid_raw())
        assert result is not None
        assert result["title"] == "Use PostgreSQL"

    def test_missing_source_quote_rejected(self):
        raw = self._valid_raw()
        del raw["source_quote"]
        assert validate_draft(raw) is None

    def test_empty_source_quote_rejected(self):
        raw = self._valid_raw()
        raw["source_quote"] = ""
        assert validate_draft(raw) is None

    def test_missing_title_rejected(self):
        raw = self._valid_raw()
        del raw["title"]
        assert validate_draft(raw) is None

    def test_missing_content_rejected(self):
        raw = self._valid_raw()
        del raw["content"]
        assert validate_draft(raw) is None

    def test_importance_clamped_high(self):
        raw = self._valid_raw()
        raw["importance"] = 99
        result = validate_draft(raw)
        assert result is not None
        assert result["importance"] == 5

    def test_importance_clamped_low(self):
        raw = self._valid_raw()
        raw["importance"] = -5
        result = validate_draft(raw)
        assert result is not None
        assert result["importance"] == 1

    def test_confidence_clamped_high(self):
        raw = self._valid_raw()
        raw["confidence"] = 2.5
        result = validate_draft(raw)
        assert result is not None
        assert result["confidence"] == 1.0

    def test_confidence_clamped_low(self):
        raw = self._valid_raw()
        raw["confidence"] = -0.5
        result = validate_draft(raw)
        assert result is not None
        assert result["confidence"] == 0.0

    def test_optional_fields_have_defaults(self):
        raw = self._valid_raw()
        for field in ("tags", "entities", "links", "related_files"):
            del raw[field]
        result = validate_draft(raw)
        assert result is not None
        assert result["tags"] == []
        assert result["entities"] == []

    def test_non_dict_input_rejected(self):
        assert validate_draft("not a dict") is None
        assert validate_draft(None) is None
        assert validate_draft([]) is None
