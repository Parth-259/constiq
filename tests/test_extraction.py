"""Hermetic tests for backend.pipeline.extraction (LLM SDKs fully mocked)."""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models import Base, ExtractedRequirement
from backend.pipeline import extraction

TENDER_TEXT = (
    "Concrete shall be M40 grade. Steel shall be Fe500D. "
    "Completion period 18 months."
)

CANNED_REQUIREMENTS = [
    {
        "material": "Concrete",
        "grade": "M40",
        "quantity": None,
        "unit": None,
        "deadline": None,
        "certification": None,
        "source_page": 1,
        "confidence": "high",
    },
    {
        "material": "Steel",
        "grade": "Fe500D",
        "quantity": None,
        "unit": None,
        "deadline": None,
        "certification": None,
        "source_page": 1,
        "confidence": "high",
    },
    {
        "material": "Overall project completion",
        "grade": None,
        "quantity": None,
        "unit": None,
        "deadline": "18 months",
        "certification": None,
        "source_page": 1,
        "confidence": "medium",
    },
]


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


def _chunk(text: str, chunk_type: str = "text", page: int = 1) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text,
        "page_number": page,
        "source_file": "tender.pdf",
        "chunk_type": chunk_type,
    }


def _tool_use_response(requirements: list[dict]):
    """Build a canned messages.create response with one tool_use block."""
    block = SimpleNamespace(
        type="tool_use",
        name="record_requirements",
        input={"requirements": requirements},
        id="toolu_test",
    )
    return SimpleNamespace(content=[block], stop_reason="tool_use")


def _mock_client(requirements: list[dict]) -> mock.MagicMock:
    client = mock.MagicMock()
    client.messages.create.return_value = _tool_use_response(requirements)
    return client


def test_extract_requirements_persists_three_rows_with_nulls(db_session, monkeypatch):
    monkeypatch.setattr(extraction.config, "ANTHROPIC_API_KEY", "test-key")
    client = _mock_client(CANNED_REQUIREMENTS)
    with mock.patch("anthropic.Anthropic", return_value=client) as cls:
        rows = extraction.extract_requirements(
            [_chunk(TENDER_TEXT)], "PRJ-TEST", db_session
        )
        cls.assert_called_once_with(api_key="test-key")

    assert len(rows) == 3
    assert client.messages.create.call_count == 1

    # Forced tool-use call shape.
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "record_requirements"}
    assert call_kwargs["tools"][0]["name"] == "record_requirements"
    assert TENDER_TEXT in call_kwargs["messages"][0]["content"]

    persisted = (
        db_session.query(ExtractedRequirement)
        .order_by(ExtractedRequirement.id)
        .all()
    )
    assert len(persisted) == 3

    concrete, steel, completion = persisted
    assert concrete.material == "Concrete"
    assert concrete.grade == "M40"
    assert concrete.quantity is None
    assert concrete.unit is None
    assert concrete.deadline is None
    assert concrete.certification is None
    assert concrete.project_id == "PRJ-TEST"
    assert concrete.source_file == "tender.pdf"
    assert concrete.source_page == 1
    assert concrete.confidence == "high"

    assert steel.material == "Steel"
    assert steel.grade == "Fe500D"
    assert steel.quantity is None
    assert steel.deadline is None

    assert completion.deadline == "18 months"
    assert completion.grade is None
    assert completion.quantity is None
    assert completion.confidence == "medium"


def test_invalid_items_are_logged_and_skipped(db_session, monkeypatch, caplog):
    monkeypatch.setattr(extraction.config, "ANTHROPIC_API_KEY", "test-key")
    items = [
        CANNED_REQUIREMENTS[0],
        {"grade": "Fe500", "source_page": 1, "confidence": "high"},  # no material
        {"material": "Cement", "source_page": 1, "confidence": "certain"},  # bad enum
    ]
    client = _mock_client(items)
    with mock.patch("anthropic.Anthropic", return_value=client):
        with caplog.at_level("WARNING", logger="backend.pipeline.extraction"):
            rows = extraction.extract_requirements(
                [_chunk(TENDER_TEXT)], "PRJ-TEST", db_session
            )

    assert len(rows) == 1
    assert rows[0].material == "Concrete"
    assert db_session.query(ExtractedRequirement).count() == 1
    assert any("Skipping invalid requirement" in r.message for r in caplog.records)


def test_table_chunks_are_skipped(db_session, monkeypatch):
    monkeypatch.setattr(extraction.config, "ANTHROPIC_API_KEY", "test-key")
    client = _mock_client(CANNED_REQUIREMENTS)
    with mock.patch("anthropic.Anthropic", return_value=client):
        rows = extraction.extract_requirements(
            [_chunk("| a | b |", chunk_type="table")], "PRJ-TEST", db_session
        )
    assert rows == []
    client.messages.create.assert_not_called()


def test_missing_api_key_returns_empty_without_calling_api(db_session, monkeypatch, caplog):
    monkeypatch.setattr(extraction.config, "ANTHROPIC_API_KEY", "")
    with mock.patch("anthropic.Anthropic") as cls:
        with caplog.at_level("WARNING", logger="backend.pipeline.extraction"):
            rows = extraction.extract_requirements(
                [_chunk(TENDER_TEXT)], "PRJ-TEST", db_session
            )
    assert rows == []
    cls.assert_not_called()
    assert any("ANTHROPIC_API_KEY" in r.message for r in caplog.records)
    assert db_session.query(ExtractedRequirement).count() == 0


def test_api_error_is_caught_and_returns_empty(db_session, monkeypatch):
    monkeypatch.setattr(extraction.config, "ANTHROPIC_API_KEY", "test-key")
    client = mock.MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    with mock.patch("anthropic.Anthropic", return_value=client):
        rows = extraction.extract_requirements(
            [_chunk(TENDER_TEXT)], "PRJ-TEST", db_session
        )
    assert rows == []
    assert db_session.query(ExtractedRequirement).count() == 0


def test_process_change_request_supersedes_matching_material(db_session, monkeypatch):
    # Seed one existing Fe500 steel requirement.
    old = ExtractedRequirement(
        project_id="PRJ-TEST",
        source_file="tender.pdf",
        material="TMT steel",
        grade="Fe500",
        quantity=100.0,
        unit="tonne",
        deadline="2026-06-01",
        certification="IS 1786",
        source_page=2,
        confidence="high",
    )
    db_session.add(old)
    db_session.commit()

    monkeypatch.setattr(extraction.config, "ANTHROPIC_API_KEY", "test-key")
    change = [
        {
            "material": "Steel",
            "grade": "Fe500D",
            "quantity": None,
            "unit": None,
            "deadline": "2026-09-30",
            "certification": None,
            "source_page": 1,
            "confidence": "high",
        }
    ]
    client = _mock_client(change)
    with mock.patch("anthropic.Anthropic", return_value=client):
        new_rows = extraction.process_change_request(
            [_chunk("Steel changed to Fe500D. New deadline 2026-09-30.")],
            "PRJ-TEST",
            db_session,
        )

    assert len(new_rows) == 1
    new = new_rows[0]

    all_rows = db_session.query(ExtractedRequirement).order_by(ExtractedRequirement.id).all()
    assert len(all_rows) == 2

    db_session.refresh(old)
    assert old.superseded_by == new.id
    assert new.superseded_by is None
    assert new.grade == "Fe500D"
    assert new.deadline == "2026-09-30"

    current = extraction.get_current_requirements("PRJ-TEST", db_session)
    assert [r.id for r in current] == [new.id]
    assert current[0].grade == "Fe500D"


def test_change_request_does_not_supersede_other_projects_or_materials(
    db_session, monkeypatch
):
    other_project = ExtractedRequirement(
        project_id="PRJ-OTHER",
        material="TMT steel",
        grade="Fe500",
        source_page=1,
        confidence="high",
    )
    other_material = ExtractedRequirement(
        project_id="PRJ-TEST",
        material="OPC 53 cement",
        grade="OPC 53",
        source_page=1,
        confidence="high",
    )
    db_session.add_all([other_project, other_material])
    db_session.commit()

    monkeypatch.setattr(extraction.config, "ANTHROPIC_API_KEY", "test-key")
    change = [
        {
            "material": "Steel",
            "grade": "Fe500D",
            "source_page": 1,
            "confidence": "high",
        }
    ]
    client = _mock_client(change)
    with mock.patch("anthropic.Anthropic", return_value=client):
        extraction.process_change_request(
            [_chunk("Steel changed to Fe500D.")], "PRJ-TEST", db_session
        )

    db_session.refresh(other_project)
    db_session.refresh(other_material)
    assert other_project.superseded_by is None
    assert other_material.superseded_by is None


def test_requirement_model_rejects_bad_confidence():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        extraction.RequirementModel.model_validate(
            {"material": "Steel", "source_page": 1, "confidence": "very high"}
        )
