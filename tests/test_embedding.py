"""Tests for backend.pipeline.embedding.

Hermetic: Chroma is pointed at a per-test tmp_path via monkeypatch of
config.CHROMA_PATH (and a reset of the lazy client cache) before first use.
The real local sentence-transformers model is used — it is free and offline.
"""
from __future__ import annotations

import subprocess
import sys
import uuid

import pytest

from backend import config
from backend.pipeline import embedding


def _chunk(text: str, page_number: int = 1, source_file: str = "tender.pdf") -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "page_number": page_number,
        "source_file": source_file,
        "chunk_type": "text",
    }


PROJECT_A_TEXTS = [
    "Supply of Fe500D TMT steel reinforcement bars conforming to IS 1786, 120 tonnes.",
    "OPC 53 grade cement, 800 bags, delivery to site by August 2026.",
    "Double-glazed curtain wall glass panels, 1200 square metres for facade works.",
]
PROJECT_B_TEXT = "Hydraulic excavator rental for earthworks, 3 machines for 6 months."


@pytest.fixture()
def emb(tmp_path, monkeypatch):
    """embedding module wired to a fresh Chroma store under tmp_path."""
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    # Reset the lazy client cache so the tmp_path store is actually used;
    # monkeypatch restores the previous value on teardown.
    monkeypatch.setattr(embedding, "_client", None)
    return embedding


def _index_both_projects(emb) -> tuple[list[dict], dict]:
    chunks_a = [_chunk(text) for text in PROJECT_A_TEXTS]
    chunk_b = _chunk(PROJECT_B_TEXT, source_file="other_tender.pdf")
    assert emb.index_chunks(chunks_a, project_id="A") == 3
    assert emb.index_chunks([chunk_b], project_id="B") == 1
    return chunks_a, chunk_b


def test_import_does_not_load_torch_or_chroma() -> None:
    """Importing the module must not pull in torch/chromadb (lazy loading)."""
    code = (
        "import sys; import backend.pipeline.embedding; "
        "assert 'torch' not in sys.modules, 'torch loaded at import time'; "
        "assert 'chromadb' not in sys.modules, 'chromadb loaded at import time'; "
        "assert 'sentence_transformers' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(config.PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_retrieve_is_scoped_to_project(emb) -> None:
    chunks_a, chunk_b = _index_both_projects(emb)

    out = emb.retrieve("excavator earthworks machinery", project_id="A", top_k=10)
    texts = [r["text"] for r in out["results"]]
    assert len(texts) == 3  # every A chunk, never B's
    assert chunk_b["text"] not in texts
    assert all(r["source_file"] == "tender.pdf" for r in out["results"])

    # Result shape per contract.
    for r in out["results"]:
        assert set(r) == {
            "text", "page_number", "source_file", "chunk_type",
            "source_type", "distance",
        }
        assert isinstance(r["distance"], float)
        assert r["source_type"] == "tender"
        assert r["chunk_type"] == "text"
        assert r["page_number"] == 1


def test_retrieve_relevant_match_is_confident_and_sorted(emb) -> None:
    _index_both_projects(emb)
    out = emb.retrieve(PROJECT_A_TEXTS[0], project_id="A", top_k=3)
    assert out["low_confidence"] is False
    distances = [r["distance"] for r in out["results"]]
    assert distances == sorted(distances)
    # Near-identical text: best hit is the steel chunk with a tiny distance.
    assert out["results"][0]["text"] == PROJECT_A_TEXTS[0]
    assert out["results"][0]["distance"] < config.MIN_CONFIDENCE_DISTANCE


def test_reindexing_same_chunks_does_not_grow_collection(emb) -> None:
    chunks_a, _ = _index_both_projects(emb)
    collection = emb.get_collection()
    assert collection.count() == 4

    # Re-ingest the exact same chunks: upsert must not create duplicates.
    assert emb.index_chunks(chunks_a, project_id="A") == 3
    assert collection.count() == 4

    out = emb.retrieve("steel cement glass", project_id="A", top_k=10)
    assert len(out["results"]) == 3


def test_empty_project_returns_low_confidence(emb) -> None:
    _index_both_projects(emb)
    out = emb.retrieve("Fe500D TMT steel", project_id="NO-SUCH-PROJECT", top_k=5)
    assert out == {"results": [], "low_confidence": True}


def test_empty_collection_returns_low_confidence(emb) -> None:
    out = emb.retrieve("anything at all", project_id="A", top_k=5)
    assert out["results"] == []
    assert out["low_confidence"] is True


def test_index_chunks_empty_list_returns_zero(emb) -> None:
    assert emb.index_chunks([], project_id="A") == 0
