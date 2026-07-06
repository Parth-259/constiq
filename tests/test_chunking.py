"""Hermetic tests for backend.pipeline.chunking (pure functions, no I/O)."""
from __future__ import annotations

import uuid

import pytest

from backend.pipeline.chunking import chunk_pages


def _page(
    text: str = "",
    tables: list | None = None,
    page_number: int = 1,
    source_file: str = "doc.pdf",
) -> dict:
    return {
        "page_number": page_number,
        "text": text,
        "tables": tables if tables is not None else [],
        "source_file": source_file,
    }


def _long_text() -> str:
    """8 paragraphs of 5 sentences each, every word globally unique so a
    mid-word cut would produce a token not present in the source vocabulary."""
    sentences = [
        f"Sentence num{i:04d} covers material tok{i:04d} and spec ref{i:04d}."
        for i in range(40)
    ]
    paragraphs = [" ".join(sentences[i : i + 5]) for i in range(0, 40, 5)]
    return "\n\n".join(paragraphs)


def test_chunk_size_respected_and_multiple_chunks() -> None:
    chunks = chunk_pages([_page(text=_long_text())], chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["chunk_type"] == "text"
        assert 0 < len(chunk["text"]) <= 200


def test_no_mid_word_starts_or_ends() -> None:
    text = _long_text()
    vocabulary = set(text.split())
    chunks = chunk_pages([_page(text=text)], chunk_size=200, overlap=50)
    for chunk in chunks:
        assert chunk["text"] == chunk["text"].strip()
        # Every whitespace-delimited token must exist verbatim in the source:
        # a mid-word start or end would create a fragment not in the vocabulary.
        assert set(chunk["text"].split()) <= vocabulary


def test_overlap_shared_between_consecutive_chunks() -> None:
    chunks = chunk_pages([_page(text=_long_text())], chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        first_word = current["text"].split()[0]
        assert first_word in set(previous["text"].split())


def test_hard_cut_on_one_giant_sentence() -> None:
    # No sentence punctuation anywhere -> forces the word-boundary hard cut.
    text = " ".join(f"token{i:04d}" for i in range(120))
    vocabulary = set(text.split())
    chunks = chunk_pages([_page(text=text)], chunk_size=120, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 120
        assert set(chunk["text"].split()) <= vocabulary


def test_table_serialized_as_markdown_chunk() -> None:
    table = [
        ["Material", "Qty", "Unit"],
        ["Fe500D TMT", "120", "tonne"],
        ["OPC 53", "800", "bag"],
    ]
    chunks = chunk_pages(
        [_page(text="Some intro text.", tables=[table], page_number=3)]
    )
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) == 1
    lines = table_chunks[0]["text"].split("\n")
    assert lines[0] == "| Material | Qty | Unit |"
    assert lines[1] == "| --- | --- | --- |"
    assert "| Fe500D TMT | 120 | tonne |" in lines
    assert "| OPC 53 | 800 | bag |" in lines
    assert table_chunks[0]["page_number"] == 3


def test_large_table_never_split() -> None:
    header = ["Item description column", "Quantity column", "Remarks column"]
    rows = [
        [f"Ready mix concrete grade M{20 + i} for slab pour {i}", str(100 + i), f"remark {i}"]
        for i in range(40)
    ]
    table = [header] + rows
    chunks = chunk_pages([_page(tables=[table])], chunk_size=200, overlap=50)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) == 1  # never split, even though it dwarfs chunk_size
    assert len(table_chunks[0]["text"]) > 200
    # header + separator + 40 body rows
    assert len(table_chunks[0]["text"].split("\n")) == 42


def test_table_with_none_cells_does_not_crash() -> None:
    table = [["Material", None], [None, "120"]]
    chunks = chunk_pages([_page(tables=[table])])
    assert len(chunks) == 1
    assert chunks[0]["text"].split("\n")[0] == "| Material |  |"


def test_metadata_survives_and_ids_are_uuid4() -> None:
    pages = [
        _page(text=_long_text(), page_number=2, source_file="tender.pdf"),
        _page(
            text="Short note.",
            tables=[[["A", "B"], ["1", "2"]]],
            page_number=7,
            source_file="annex.pdf",
        ),
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=50)
    assert chunks, "expected chunks from non-empty pages"
    by_source: dict[str, set[int]] = {}
    for chunk in chunks:
        assert set(chunk.keys()) == {
            "chunk_id",
            "text",
            "page_number",
            "source_file",
            "chunk_type",
        }
        assert chunk["chunk_type"] in {"text", "table"}
        parsed = uuid.UUID(chunk["chunk_id"])
        assert parsed.version == 4
        by_source.setdefault(chunk["source_file"], set()).add(chunk["page_number"])
    assert by_source == {"tender.pdf": {2}, "annex.pdf": {7}}
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_page_yields_no_chunks() -> None:
    assert chunk_pages([_page(text="   \n\n  ")]) == []
    assert chunk_pages([]) == []


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        chunk_pages([_page(text="hello world")], chunk_size=0)
    with pytest.raises(ValueError):
        chunk_pages([_page(text="hello world")], chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_pages([_page(text="hello world")], chunk_size=100, overlap=-1)
