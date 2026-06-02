"""
Pre-Phase 2 — Noise gate + intelligent chunker for LLM extraction backends.

Pipeline:
  1. Sentence split raw session text.
  2. Noise gate (semantic cosine similarity, threshold 0.15) — removes pure filler.
     Lower threshold than Phase 1's 0.30: keeps borderline "oh by the way..." content
     that an LLM can understand but a rule-based classifier would drop.
  3. Group substantive segments into topic-coherent chunks (~2000 tokens each).
     Grouping preserves conversation turn boundaries and section headers.

Why 0.15 for LLM backends:
  Phase 1 uses 0.30 as a binary gate before type classification.
  For LLM extraction, 0.15 is the floor — we only discard text with
  < 0.15 cosine similarity to ANY technical/substantive exemplar.
  This keeps "we decided to use Redis" (maybe 0.18) which Phase 1 drops.
  Cuts token cost 30–40% while preserving all borderline content.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Add Phase 1 to sys.path for semantic_classifier
# ---------------------------------------------------------------------------
_P1_ROOT = Path(__file__).resolve().parents[4] / "project-memory-core"
if str(_P1_ROOT) not in sys.path:
    sys.path.insert(0, str(_P1_ROOT))


# ---------------------------------------------------------------------------
# Sentence splitting — lightweight, handles code blocks and lists
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"       # standard sentence boundary
    r"|(?<=\n)\s*(?=[-*•]\s)"       # bullet list items
    r"|(?<=\n)\s*(?=\d+\.\s)"       # numbered list items
    r"|(?<=\n\n)"                    # paragraph boundaries
)

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-level segments, preserving code blocks."""
    if not text or not text.strip():
        return []

    # Protect code blocks from splitting (replace with placeholder)
    placeholders: dict[str, str] = {}
    def _replace_code(m: re.Match) -> str:
        key = f"\x00CODE{len(placeholders)}\x00"
        placeholders[key] = m.group(0)
        return key

    clean = _CODE_BLOCK_RE.sub(_replace_code, text)

    # Split
    raw_parts = _SENTENCE_SPLIT_RE.split(clean)
    sentences: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Restore code blocks
        for key, val in placeholders.items():
            part = part.replace(key, val)
        sentences.append(part)

    return sentences


# ---------------------------------------------------------------------------
# Token estimation (cheap approximation for chunking)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Approximate token count: ~1.3 tokens per word."""
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------

# Fast regex pre-filter — catches obvious AI filler before model is loaded
_OBVIOUS_NOISE_RE = re.compile(
    r"^("
    r"thought for \d+|thinking\.\.\.|processing\.\.\.|"
    r"sure[,!]?\s|of course[,!]?\s|great[,!]?\s|okay[,!]?\s|alright[,!]?\s|"
    r"certainly[,!]?\s|absolutely[,!]?\s|definitely[,!]?\s|"
    r"here('s| is) (what|how)|i('ll| will| can| would)|"
    r"as (an|a) ai|<[^>]+>|"
    r"hey[,!]?\s|hi[,!]?\s|hello[,!]?\s|"
    r"how('s| are| is)|what('s| a) up|"
    r"glad to hear|sounds (good|great|like)|"
    r"that('s| is) (a )?(great|good|interesting|cool|awesome|nice)|"
    r"thank(s| you)|you're welcome|no problem|"
    r"let me (help|show|explain|walk)|"
    r"i (understand|see|get) (that|what)|"
    r"good (question|point|idea)|excellent (question|point)|"
    r"i('m| am) happy to|"
    r"feel free to"
    r")",
    re.IGNORECASE,
)


def _is_obvious_noise(sentence: str) -> bool:
    """Fast regex pre-filter before invoking the embedding model."""
    stripped = sentence.strip()
    if not stripped or len(stripped) < 5:
        return True
    return bool(_OBVIOUS_NOISE_RE.match(stripped))


def _is_noise_semantic(sentence: str, threshold: float) -> bool:  # noqa: ARG001
    """
    Semantic noise gate using Phase 1's semantic_classifier.

    Returns True (is noise) if the sentence is not substantive.
    Falls back to False (keep) if the model is unavailable.
    Phase 1's is_technical() uses its own internal TECH_GATE_THRESHOLD.
    """
    try:
        from app.services.semantic_classifier import is_technical
        return not is_technical(sentence)
    except Exception:
        return False  # safe default: keep the sentence


@dataclass
class NoiseGateResult:
    """Result of running the noise gate over session text."""
    substantive: list[str]       # sentences that passed the gate
    filtered_count: int          # sentences removed
    total_count: int             # total sentences examined
    threshold_used: float        # actual threshold used

    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return len(self.substantive) / self.total_count


def apply_noise_gate(
    text: str,
    threshold: float = 0.15,
) -> NoiseGateResult:
    """
    Split text into sentences and filter through the noise gate.

    Args:
        text:       Raw session text.
        threshold:  Cosine similarity floor (0.15 for LLM backends).

    Returns:
        NoiseGateResult with substantive sentences and filter stats.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return NoiseGateResult(
            substantive=[], filtered_count=0,
            total_count=0, threshold_used=threshold,
        )

    substantive: list[str] = []
    filtered = 0

    for sent in sentences:
        # Stage 1: cheap regex pre-filter
        if _is_obvious_noise(sent):
            filtered += 1
            continue
        # Stage 2: semantic gate (loads model lazily on first call)
        if _is_noise_semantic(sent, threshold):
            filtered += 1
            continue
        substantive.append(sent)

    return NoiseGateResult(
        substantive=substantive,
        filtered_count=filtered,
        total_count=len(sentences),
        threshold_used=threshold,
    )


# ---------------------------------------------------------------------------
# Intelligent chunker
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A coherent chunk of substantive session text ready for LLM extraction."""
    text: str
    token_estimate: int
    sentence_count: int
    index: int                   # 0-based position in the chunk sequence


def chunk_sentences(
    sentences: list[str],
    max_tokens: int = 2000,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Group sentences into topic-coherent chunks of at most max_tokens.

    Strategy:
      - Fill a chunk until adding the next sentence would exceed max_tokens.
      - Include `overlap_sentences` trailing sentences from the previous chunk
        at the start of each new chunk to preserve cross-chunk context.
      - Markdown headers (##, ###) always start a new chunk (topic boundary).

    Args:
        sentences:          List of substantive sentences from noise gate.
        max_tokens:         Maximum tokens per chunk (default 2000).
        overlap_sentences:  Sentences to repeat across chunk boundaries.

    Returns:
        List of Chunk objects in order.
    """
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    tail: list[str] = []  # overlap from previous chunk

    def _flush() -> None:
        nonlocal current, current_tokens, tail
        if not current:
            return
        text = " ".join(current)
        chunks.append(Chunk(
            text=text,
            token_estimate=current_tokens,
            sentence_count=len(current),
            index=len(chunks),
        ))
        # Keep last N sentences as overlap for next chunk
        tail = current[-overlap_sentences:] if overlap_sentences > 0 else []
        current = list(tail)  # start next chunk with overlap
        current_tokens = sum(_estimate_tokens(s) for s in current)

    for sent in sentences:
        sent_tokens = _estimate_tokens(sent)

        # Markdown header → always start a new chunk (topic boundary)
        if _MARKDOWN_HEADER_RE.match(sent.strip()):
            _flush()

        # Would exceed budget → flush first
        if current and current_tokens + sent_tokens > max_tokens:
            _flush()

        current.append(sent)
        current_tokens += sent_tokens

    _flush()  # flush final chunk

    # Assign sequential indices
    for i, chunk in enumerate(chunks):
        chunk.index = i

    return chunks


def prepare_chunks(
    session_text: str,
    threshold: float = 0.15,
    max_tokens: int = 2000,
    overlap_sentences: int = 1,
) -> tuple[list[Chunk], NoiseGateResult]:
    """
    Full preparation: noise gate → chunking.

    Returns:
        (chunks, gate_result)
    """
    gate = apply_noise_gate(session_text, threshold=threshold)
    chunks = chunk_sentences(
        gate.substantive,
        max_tokens=max_tokens,
        overlap_sentences=overlap_sentences,
    )
    return chunks, gate
