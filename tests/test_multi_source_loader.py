"""Hermetic tests for backend.pipeline.multi_source_loader.

chunking/embedding are replaced with fake modules injected into sys.modules
(the loader imports them lazily), so no embedding model or Chroma store is
ever touched. anthropic is mocked; only tmp_path files are read.
"""
from __future__ import annotations

import sys
import types
import uuid
from types import SimpleNamespace
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models import Base, IngestedDocument, InspectionFinding
from backend.pipeline import multi_source_loader as loader

EXPECTED_KEYS = {
    "project_id",
    "filename",
    "source_type",
    "pages_processed",
    "chunks_indexed",
    "tables_found",
    "requirements_extracted",
}


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def fake_pipeline(monkeypatch):
    """Inject fake chunking/embedding modules so no model ever loads."""

    def fake_chunk_pages(pages, chunk_size=800, overlap=100):
        return [
            {
                "chunk_id": str(uuid.uuid4()),
                "text": page["text"],
                "page_number": page["page_number"],
                "source_file": page["source_file"],
                "chunk_type": "text",
            }
            for page in pages
        ]

    index_chunks = mock.MagicMock(side_effect=lambda chunks, project_id, source_type="tender": len(chunks))

    fake_chunking = types.ModuleType("backend.pipeline.chunking")
    fake_chunking.chunk_pages = mock.MagicMock(side_effect=fake_chunk_pages)
    fake_embedding = types.ModuleType("backend.pipeline.embedding")
    fake_embedding.index_chunks = index_chunks

    monkeypatch.setitem(sys.modules, "backend.pipeline.chunking", fake_chunking)
    monkeypatch.setitem(sys.modules, "backend.pipeline.embedding", fake_embedding)
    return SimpleNamespace(
        chunk_pages=fake_chunking.chunk_pages, index_chunks=index_chunks
    )


def _write_txt(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _assert_contract_dict(result: dict, project_id: str, filename: str, source_type: str):
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["project_id"] == project_id
    assert result["filename"] == filename
    assert result["source_type"] == source_type
    assert isinstance(result["pages_processed"], int)
    assert isinstance(result["chunks_indexed"], int)
    assert isinstance(result["tables_found"], int)
    assert isinstance(result["requirements_extracted"], int)


def _assert_document_row(db_session, project_id: str, filename: str, source_type: str):
    doc = (
        db_session.query(IngestedDocument)
        .filter(IngestedDocument.filename == filename)
        .one()
    )
    assert doc.project_id == project_id
    assert doc.source_type == source_type
    assert doc.pages == 1
    assert doc.chunks == 1
    assert doc.tables_found == 0
    assert doc.size_bytes > 0
    return doc


def test_load_tender_txt(tmp_path, db_session, fake_pipeline, monkeypatch):
    path = _write_txt(tmp_path, "tender.txt", "Concrete shall be M40 grade.")
    fake_reqs = [mock.MagicMock(), mock.MagicMock()]
    extract = mock.MagicMock(return_value=fake_reqs)
    monkeypatch.setattr(loader.extraction, "extract_requirements", extract)

    result = loader.load_document(path, "PRJ-TEST", "tender", db_session)

    _assert_contract_dict(result, "PRJ-TEST", "tender.txt", "tender")
    assert result["pages_processed"] == 1
    assert result["chunks_indexed"] == 1
    assert result["tables_found"] == 0
    assert result["requirements_extracted"] == 2

    # Indexed with the right source_type; extraction got the chunks.
    _, index_kwargs = fake_pipeline.index_chunks.call_args
    assert index_kwargs["source_type"] == "tender"
    extract.assert_called_once()
    chunks_arg = extract.call_args.args[0]
    assert chunks_arg[0]["text"] == "Concrete shall be M40 grade."

    _assert_document_row(db_session, "PRJ-TEST", "tender.txt", "tender")


def test_load_change_request_txt(tmp_path, db_session, fake_pipeline, monkeypatch):
    path = _write_txt(tmp_path, "change.txt", "Steel changed to Fe500D.")
    process = mock.MagicMock(return_value=[mock.MagicMock()])
    extract = mock.MagicMock()
    monkeypatch.setattr(loader.extraction, "process_change_request", process)
    monkeypatch.setattr(loader.extraction, "extract_requirements", extract)

    result = loader.load_document(path, "PRJ-TEST", "change_request", db_session)

    _assert_contract_dict(result, "PRJ-TEST", "change.txt", "change_request")
    assert result["requirements_extracted"] == 1
    process.assert_called_once()
    extract.assert_not_called()
    _assert_document_row(db_session, "PRJ-TEST", "change.txt", "change_request")


def test_load_meeting_notes_txt_indexes_only(tmp_path, db_session, fake_pipeline, monkeypatch):
    path = _write_txt(tmp_path, "minutes.txt", "Discussed slab pour schedule.")
    extract = mock.MagicMock()
    process = mock.MagicMock()
    monkeypatch.setattr(loader.extraction, "extract_requirements", extract)
    monkeypatch.setattr(loader.extraction, "process_change_request", process)

    with mock.patch("anthropic.Anthropic") as anthropic_cls:
        result = loader.load_document(path, "PRJ-TEST", "meeting_notes", db_session)

    _assert_contract_dict(result, "PRJ-TEST", "minutes.txt", "meeting_notes")
    assert result["requirements_extracted"] == 0
    extract.assert_not_called()
    process.assert_not_called()
    anthropic_cls.assert_not_called()
    _, index_kwargs = fake_pipeline.index_chunks.call_args
    assert index_kwargs["source_type"] == "meeting_notes"
    _assert_document_row(db_session, "PRJ-TEST", "minutes.txt", "meeting_notes")


def test_load_inspection_report_with_api_key(tmp_path, db_session, fake_pipeline, monkeypatch):
    path = _write_txt(
        tmp_path, "inspection.txt", "Honeycombing observed in Block A slab."
    )
    monkeypatch.setattr(loader.config, "ANTHROPIC_API_KEY", "test-key")

    block = SimpleNamespace(
        type="tool_use",
        name="record_findings",
        input={
            "findings": [
                {
                    "location": "Block A slab",
                    "defect_description": "Honeycombing observed",
                    "severity": "high",
                    "source_page": 1,
                }
            ]
        },
        id="toolu_test",
    )
    client = mock.MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[block], stop_reason="tool_use"
    )

    with mock.patch("anthropic.Anthropic", return_value=client):
        result = loader.load_document(path, "PRJ-TEST", "inspection_report", db_session)

    _assert_contract_dict(result, "PRJ-TEST", "inspection.txt", "inspection_report")
    assert result["requirements_extracted"] == 0

    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "record_findings"}

    findings = db_session.query(InspectionFinding).all()
    assert len(findings) == 1
    assert findings[0].location == "Block A slab"
    assert findings[0].defect_description == "Honeycombing observed"
    assert findings[0].severity == "high"
    assert findings[0].project_id == "PRJ-TEST"
    assert findings[0].source_file == "inspection.txt"
    _assert_document_row(db_session, "PRJ-TEST", "inspection.txt", "inspection_report")


def test_load_inspection_report_without_api_key_skips_llm(
    tmp_path, db_session, fake_pipeline, monkeypatch, caplog
):
    path = _write_txt(tmp_path, "inspection.txt", "Cracks near column C4.")
    monkeypatch.setattr(loader.config, "ANTHROPIC_API_KEY", "")

    with mock.patch("anthropic.Anthropic") as anthropic_cls:
        with caplog.at_level("WARNING", logger="backend.pipeline.multi_source_loader"):
            result = loader.load_document(
                path, "PRJ-TEST", "inspection_report", db_session
            )

    _assert_contract_dict(result, "PRJ-TEST", "inspection.txt", "inspection_report")
    anthropic_cls.assert_not_called()
    assert db_session.query(InspectionFinding).count() == 0
    assert any("ANTHROPIC_API_KEY" in r.message for r in caplog.records)
    # Document row is still recorded even when the LLM step is skipped.
    _assert_document_row(db_session, "PRJ-TEST", "inspection.txt", "inspection_report")


def test_unknown_source_type_raises(tmp_path, db_session, fake_pipeline):
    path = _write_txt(tmp_path, "doc.txt", "hello")
    with pytest.raises(ValueError, match="source_type"):
        loader.load_document(path, "PRJ-TEST", "invoice", db_session)


def test_unsupported_extension_raises(tmp_path, db_session, fake_pipeline):
    path = tmp_path / "doc.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extension"):
        loader.load_document(str(path), "PRJ-TEST", "tender", db_session)
