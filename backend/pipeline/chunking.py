"""Chunking of extracted PDF pages for embedding/retrieval.

Text is split recursively — paragraphs (``\\n\\n``), then sentences, then a
hard cut at word boundaries — into character-budgeted chunks with a
word-aligned overlap prepended from the previous chunk. Chunks never start or
end mid-word. Each table is serialized as one GitHub-markdown table chunk
(``chunk_type == "table"``) and is never split, regardless of size.
"""
from __future__ import annotations

import logging
import re
import uuid

logger = logging.getLogger(__name__)

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_text(text: str, budget: int) -> list[str]:
    """Recursively split ``text`` into stripped pieces of <= ``budget`` chars.

    Order of attack: paragraphs -> sentences -> hard cut at word boundaries.
    A single word longer than the budget is kept intact (never mid-word).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= budget:
        return [text]

    paragraphs = [p for p in _PARAGRAPH_RE.split(text) if p.strip()]
    if len(paragraphs) > 1:
        return _merge_units(paragraphs, budget, "\n\n")

    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    if len(sentences) > 1:
        return _merge_units(sentences, budget, " ")

    words = text.split()
    if len(words) > 1:
        return _merge_units(words, budget, " ")

    logger.warning(
        "Single word of %d chars exceeds chunk budget %d; keeping it intact",
        len(text),
        budget,
    )
    return [text]


def _merge_units(units: list[str], budget: int, sep: str) -> list[str]:
    """Greedily pack units into pieces of <= ``budget`` chars.

    Units that individually exceed the budget are recursively split at the
    next-finer granularity via :func:`_split_text`.
    """
    pieces: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > budget:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_text(unit, budget))
            continue
        candidate = f"{current}{sep}{unit}" if current else unit
        if len(candidate) <= budget:
            current = candidate
        else:
            pieces.append(current)
            current = unit
    if current:
        pieces.append(current)
    return pieces


def _word_aligned_tail(text: str, budget: int) -> str:
    """Return a suffix of ``text`` of <= ``budget`` chars starting on a word.

    Returns "" when no word boundary falls within the budget.
    """
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text.strip()
    tail = text[-budget:]
    if not text[-budget - 1].isspace():
        # Cut landed mid-word — advance past the partial word.
        match = re.search(r"\s", tail)
        if match is None:
            return ""
        tail = tail[match.end():]
    return tail.strip()


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend a word-aligned tail of each piece onto its successor."""
    if overlap <= 0 or len(pieces) <= 1:
        return list(pieces)
    result = [pieces[0]]
    for previous, piece in zip(pieces, pieces[1:]):
        tail = _word_aligned_tail(previous, overlap)
        result.append(f"{tail} {piece}" if tail else piece)
    return result


def _clean_cell(cell: object) -> str:
    """Make a cell safe for a single markdown table row."""
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").replace("|", "\\|").strip()


def _table_to_markdown(table: list[list[object]]) -> str:
    """Serialize a table (list of rows of cells) as a GitHub-markdown table.

    The first row is treated as the header. Ragged rows are padded. Returns
    "" for an empty table.
    """
    rows = [[_clean_cell(cell) for cell in row] for row in table if row]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _make_chunk(text: str, page: dict, chunk_type: str) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "page_number": int(page.get("page_number", 0)),
        "source_file": str(page.get("source_file", "")),
        "chunk_type": chunk_type,
    }


def chunk_pages(
    pages: list[dict], chunk_size: int = 800, overlap: int = 100
) -> list[dict]:
    """Chunk extracted pages into embedding-ready pieces.

    Args:
        pages: page dicts from ``pdf_loader.extract_pdf`` (or compatible).
        chunk_size: maximum characters per text chunk (overlap included).
        overlap: characters of word-aligned overlap shared with the previous
            chunk of the same page.

    Returns:
        List of ``{"chunk_id", "text", "page_number", "source_file",
        "chunk_type"}`` dicts, ``chunk_type`` in {"text", "table"}. Table
        chunks are one markdown table each and are never split (they may
        exceed ``chunk_size``).

    Raises:
        ValueError: if ``chunk_size <= 0`` or not ``0 <= overlap < chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size, got "
            f"overlap={overlap}, chunk_size={chunk_size}"
        )

    # Base pieces leave room for the prepended overlap tail plus one
    # separator space, so final text chunks never exceed chunk_size.
    base_budget = max(1, chunk_size - overlap - 1) if overlap else chunk_size

    chunks: list[dict] = []
    for page in pages:
        text = page.get("text") or ""
        pieces = _apply_overlap(_split_text(text, base_budget), overlap)
        for piece in pieces:
            chunks.append(_make_chunk(piece, page, "text"))

        for table in page.get("tables") or []:
            markdown = _table_to_markdown(table)
            if not markdown:
                continue
            chunks.append(_make_chunk(markdown, page, "table"))

    logger.info(
        "Chunked %d page(s) into %d chunk(s) (chunk_size=%d, overlap=%d)",
        len(pages),
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks
