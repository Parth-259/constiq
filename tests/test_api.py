"""Hermetic tests for the ConstructIQ FastAPI layer.

- In-memory SQLite (StaticPool) injected via app.dependency_overrides[get_db].
- CONSTRUCTIQ_SKIP_SEED=1 is set BEFORE importing backend.main so the startup
  hook never touches the project data/ directory.
- Heavy pipeline pieces (embedding.retrieve, multi_source_loader.load_document,
  the anthropic client) are patched — no real model/API/Chroma usage.
- The backend.agent.* modules are built by another workstream; when they are
  not importable yet, real ModuleType stubs are installed so the lazy imports
  inside the routers resolve, and each test monkeypatches the functions it
  needs (which works identically once the real modules exist).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from datetime import date, timedelta
from types import SimpleNamespace

os.environ["CONSTRUCTIQ_SKIP_SEED"] = "1"

import anthropic
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def _unimplemented(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("stubbed in tests — monkeypatch this function")


class _StubInvalidTransitionError(ValueError):
    pass


def _install_stub(name: str, attrs: dict) -> types.ModuleType:
    """Import the real module if available, else install a stub ModuleType."""
    try:
        return importlib.import_module(name)
    except Exception:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        parent_name, _, child = name.rpartition(".")
        setattr(importlib.import_module(parent_name), child, module)
        return module


tools_mod = _install_stub(
    "backend.agent.tools",
    {
        "vendor_lookup": _unimplemented,
        "get_market_reference_price": _unimplemented,
        "check_compliance": _unimplemented,
        "calculate_risk": _unimplemented,
        "vendor_discovery": _unimplemented,
        "vendor_evaluation": _unimplemented,
        "recommend_vendor": _unimplemented,
    },
)
negotiation_mod = _install_stub(
    "backend.agent.negotiation",
    {
        "start_negotiation": _unimplemented,
        "run_negotiation_round": _unimplemented,
        "run_full_negotiation": _unimplemented,
        "approve_negotiation": _unimplemented,
        "decline_negotiation": _unimplemented,
        "get_negotiation_state": _unimplemented,
    },
)
purchase_order_mod = _install_stub(
    "backend.agent.purchase_order", {"generate_po": _unimplemented}
)
tracking_mod = _install_stub(
    "backend.agent.tracking",
    {
        "InvalidTransitionError": _StubInvalidTransitionError,
        "update_tracking_status": _unimplemented,
        "get_po_timeline": _unimplemented,
    },
)
agent_mod = _install_stub("backend.agent.agent", {"run_agent": _unimplemented})

from backend import config  # noqa: E402
from backend.db.models import (  # noqa: E402
    Base,
    ExtractedRequirement,
    IngestedDocument,
    Negotiation,
    PurchaseOrder,
    Vendor,
)
from backend.db.session import get_db  # noqa: E402
from backend.main import app  # noqa: E402

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def db_session():
    db = TestingSessionLocal()
    yield db
    db.close()


def _make_vendor(db, **overrides) -> Vendor:
    defaults = dict(
        name="Shakti TMT Industries",
        materials_supplied=json.dumps(["Fe500D TMT steel", "Fe500 TMT steel"]),
        location="Nagpur",
        region="Central India",
        rating=4.4,
        avg_delivery_days=10,
        historical_on_time_pct=92.0,
        typical_min_qty=20.0,
        typical_max_qty=500.0,
        price_index=1.02,
        negotiation_flexibility=0.4,
    )
    defaults.update(overrides)
    vendor = Vendor(**defaults)
    db.add(vendor)
    db.commit()
    return vendor


def _make_requirement(db, **overrides) -> ExtractedRequirement:
    defaults = dict(
        project_id="PRJ-TEST",
        source_file="tender.pdf",
        material="TMT steel",
        grade="Fe500D",
        quantity=120.0,
        unit="tonne",
        deadline="2026-08-10",
        source_page=3,
        confidence="high",
    )
    defaults.update(overrides)
    requirement = ExtractedRequirement(**defaults)
    db.add(requirement)
    db.commit()
    return requirement


# ---------------------------------------------------------------- health ----


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------- ingest ----


def test_ingest_txt_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_load_document(
        file_path: str,
        project_id: str,
        source_type: str,
        db_session,
        original_filename: str | None = None,
    ):
        captured["file_path"] = file_path
        captured["existed"] = os.path.exists(file_path)
        captured["original_filename"] = original_filename
        with open(file_path, "rb") as handle:
            captured["content"] = handle.read()
        return {
            "project_id": project_id,
            "filename": original_filename or os.path.basename(file_path),
            "source_type": source_type,
            "pages_processed": 1,
            "chunks_indexed": 3,
            "tables_found": 0,
            "requirements_extracted": 2,
        }

    monkeypatch.setattr(
        "backend.pipeline.multi_source_loader.load_document", fake_load_document
    )
    response = client.post(
        "/api/ingest",
        files={"file": ("site notes.txt", b"Fe500D TMT steel, 120 tonne", "text/plain")},
        data={"project_id": "PRJ-TEST", "source_type": "tender"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "PRJ-TEST"
    assert body["source_type"] == "tender"
    assert body["pages_processed"] == 1
    assert body["chunks_indexed"] == 3
    assert body["requirements_extracted"] == 2
    # The original upload name is passed through for display/citations.
    assert captured["original_filename"] == "site notes.txt"
    assert body["filename"] == "site notes.txt"
    # Temp file preserved the suffix, held the bytes, and was cleaned up.
    assert captured["file_path"].endswith(".txt")
    assert captured["existed"] is True
    assert captured["content"] == b"Fe500D TMT steel, 120 tonne"
    assert not os.path.exists(captured["file_path"])


def test_ingest_rejects_bad_extension() -> None:
    response = client.post(
        "/api/ingest",
        files={"file": ("payload.exe", b"MZ...", "application/octet-stream")},
        data={"project_id": "PRJ-TEST"},
    )
    assert response.status_code == 400
    assert "extension" in response.json()["detail"].lower()


def test_ingest_rejects_pdf_without_magic_bytes() -> None:
    response = client.post(
        "/api/ingest",
        files={"file": ("report.pdf", b"this is not a pdf", "application/pdf")},
        data={"project_id": "PRJ-TEST"},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_ingest_rejects_bad_source_type() -> None:
    response = client.post(
        "/api/ingest",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"project_id": "PRJ-TEST", "source_type": "blueprint"},
    )
    assert response.status_code == 400
    assert "source_type" in response.json()["detail"]


# ------------------------------------------------------------------- ask ----


RETRIEVAL_RESULTS = [
    {
        "text": "Fe500D TMT steel, 120 tonnes required by August 2026.",
        "page_number": 3,
        "source_file": "tender.pdf",
        "chunk_type": "text",
        "source_type": "tender",
        "distance": 0.21,
    },
    {
        "text": "OPC 53 cement, 800 bags, IS 12269 certified.",
        "page_number": 7,
        "source_file": "tender.pdf",
        "chunk_type": "text",
        "source_type": "tender",
        "distance": 0.35,
    },
]


def _patch_retrieve(monkeypatch: pytest.MonkeyPatch, results, low_confidence=False) -> None:
    def fake_retrieve(query: str, project_id: str, top_k: int = 5) -> dict:
        assert top_k == 5
        return {"results": results, "low_confidence": low_confidence}

    monkeypatch.setattr("backend.pipeline.embedding.retrieve", fake_retrieve)


def _patch_anthropic_answer(monkeypatch: pytest.MonkeyPatch, answer_text: str) -> dict:
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=answer_text)])

    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda api_key: SimpleNamespace(messages=SimpleNamespace(create=fake_create)),
    )
    return captured


def test_ask_answers_from_context_with_cited_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    _patch_retrieve(monkeypatch, RETRIEVAL_RESULTS, low_confidence=False)
    captured = _patch_anthropic_answer(
        monkeypatch, "The tender requires 120 tonnes of Fe500D TMT steel (p.3)."
    )

    response = client.post(
        "/api/ask", json={"project_id": "PRJ-TEST", "question": "How much steel?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "Fe500D" in body["answer"]
    assert body["low_confidence"] is False
    # Only the cited page (3) is returned as a source.
    assert [(s["source_file"], s["page_number"]) for s in body["sources"]] == [
        ("tender.pdf", 3)
    ]
    assert body["sources"][0]["snippet"].startswith("Fe500D TMT steel")
    assert captured["model"] == config.MODEL_SONNET
    # Context blocks are labelled [source_file p.N].
    assert "[tender.pdf p.3]" in captured["messages"][0]["content"]


def test_ask_falls_back_to_all_sources_and_caveats_low_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    _patch_retrieve(monkeypatch, RETRIEVAL_RESULTS, low_confidence=True)
    _patch_anthropic_answer(monkeypatch, "No page citations in this answer.")

    response = client.post(
        "/api/ask", json={"project_id": "PRJ-TEST", "question": "How much steel?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["low_confidence"] is True
    assert body["answer"].startswith("Note:")
    assert len(body["sources"]) == 2  # fallback: all retrieved chunks


def test_ask_without_api_key_still_returns_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    _patch_retrieve(monkeypatch, RETRIEVAL_RESULTS, low_confidence=False)

    response = client.post(
        "/api/ask", json={"project_id": "PRJ-TEST", "question": "How much steel?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "ANTHROPIC_API_KEY" in body["answer"]
    assert len(body["sources"]) == 2


def test_ask_anthropic_error_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    _patch_retrieve(monkeypatch, RETRIEVAL_RESULTS, low_confidence=False)

    class _Boom(anthropic.AnthropicError):
        pass

    def fake_create(**kwargs):
        raise _Boom("model overloaded")

    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda api_key: SimpleNamespace(messages=SimpleNamespace(create=fake_create)),
    )
    response = client.post(
        "/api/ask", json={"project_id": "PRJ-TEST", "question": "How much steel?"}
    )
    assert response.status_code == 503


def test_ask_empty_question_is_400() -> None:
    response = client.post("/api/ask", json={"project_id": "PRJ-TEST", "question": "  "})
    assert response.status_code == 400


# ------------------------------------------------- vendors / requirements ----


def test_vendors_list(db_session) -> None:
    vendor = _make_vendor(db_session)
    response = client.get("/api/vendors")
    assert response.status_code == 200
    vendors = response.json()["vendors"]
    assert len(vendors) == 1
    assert vendors[0]["id"] == vendor.id
    assert vendors[0]["name"] == "Shakti TMT Industries"
    assert vendors[0]["materials_supplied"] == ["Fe500D TMT steel", "Fe500 TMT steel"]


def test_requirements_list_returns_current_only(db_session) -> None:
    old = _make_requirement(db_session, material="TMT steel", grade="Fe415")
    new = _make_requirement(db_session, material="TMT steel", grade="Fe500D")
    old.superseded_by = new.id
    db_session.commit()

    response = client.get("/api/requirements", params={"project_id": "PRJ-TEST"})
    assert response.status_code == 200
    requirements = response.json()["requirements"]
    assert [r["id"] for r in requirements] == [new.id]
    assert requirements[0]["grade"] == "Fe500D"


def test_documents_list(db_session) -> None:
    db_session.add(
        IngestedDocument(
            project_id="PRJ-TEST", filename="tender.pdf", source_type="tender",
            pages=12, chunks=40, tables_found=3, size_bytes=1024,
        )
    )
    db_session.commit()
    response = client.get("/api/documents", params={"project_id": "PRJ-TEST"})
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 1
    assert documents[0]["filename"] == "tender.pdf"


def test_discovery_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "internal_matches": [{"name": "Shakti TMT Industries", "verified": True}],
        "web_matches": [],
        "web_search_succeeded": False,
    }
    monkeypatch.setattr(
        tools_mod, "vendor_discovery", lambda material, grade, hint, db: payload
    )
    response = client.post("/api/discovery", json={"material": "TMT steel"})
    assert response.status_code == 200
    assert response.json() == payload


def test_recommend_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_recommend(requirement_id: int, deadline_days_remaining: int, db) -> dict:
        captured["args"] = (requirement_id, deadline_days_remaining)
        return {"recommended_vendor": {"id": 1}, "compliance_status": "compliant"}

    monkeypatch.setattr(tools_mod, "recommend_vendor", fake_recommend)
    response = client.post("/api/recommend", json={"requirement_id": 4})
    assert response.status_code == 200
    assert response.json()["compliance_status"] == "compliant"
    assert captured["args"] == (4, 30)  # default deadline_days_remaining


# ------------------------------------------------------------------ risk ----


def test_risk_cards_shape(db_session, monkeypatch: pytest.MonkeyPatch) -> None:
    vendor = _make_vendor(db_session)
    requirement = _make_requirement(db_session, deadline="2026-08-10")
    vendor_dict = vendor.to_dict()
    captured: dict = {}

    monkeypatch.setattr(
        tools_mod, "vendor_lookup", lambda material, grade, db: [vendor_dict]
    )
    monkeypatch.setattr(
        tools_mod, "get_market_reference_price", lambda material, grade, db: 55000.0
    )

    def fake_calculate_risk(requirement_id, vendor_id, deadline_days_remaining, db):
        captured["risk_args"] = (requirement_id, vendor_id, deadline_days_remaining)
        return {
            "score": 72,
            "label": "HIGH",
            "factors": {
                "lead_time_pressure": 1.2,
                "reliability_factor": 0.08,
                "order_size_factor": 0.0,
            },
            "explanation": "Lead time of 10 days is tight against the deadline.",
        }

    monkeypatch.setattr(tools_mod, "calculate_risk", fake_calculate_risk)

    response = client.get("/api/risk/PRJ-TEST")
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "PRJ-TEST"
    assert body["total_risk_score"] == 72
    assert body["active_mitigations"] == 1
    assert len(body["cards"]) == 1
    card = body["cards"][0]
    assert card["requirement"]["id"] == requirement.id
    assert card["vendor"]["id"] == vendor.id
    assert card["score"] == 72
    assert card["label"] == "HIGH"
    assert card["factors"]["lead_time_pressure"] == 1.2
    assert card["est_value"] == pytest.approx(120.0 * 55000.0)
    assert card["est_delivery"] == (date.today() + timedelta(days=10)).isoformat()
    # ISO deadline parsed into real days remaining (floor 1).
    expected_days = max(1, (date(2026, 8, 10) - date.today()).days)
    assert captured["risk_args"] == (requirement.id, vendor.id, expected_days)


def test_risk_no_vendor_card(db_session, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_requirement(db_session, material="Aerogel insulation", grade=None)
    monkeypatch.setattr(tools_mod, "vendor_lookup", lambda material, grade, db: [])
    monkeypatch.setattr(
        tools_mod, "get_market_reference_price", lambda material, grade, db: None
    )
    response = client.get("/api/risk/PRJ-TEST")
    assert response.status_code == 200
    body = response.json()
    card = body["cards"][0]
    assert card["vendor"] is None
    assert card["score"] == 0
    assert card["label"] == "NO_VENDOR"
    assert "No vendor found" in card["explanation"]
    assert card["est_delivery"] is None
    assert body["total_risk_score"] == 0
    assert body["active_mitigations"] == 0


# ----------------------------------------------------------- negotiation ----


def test_negotiation_start_run_approve_flow(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor = _make_vendor(db_session)
    requirement = _make_requirement(db_session)
    status_box = {"status": "in_progress"}

    def fake_state(negotiation_id: int, db) -> dict:
        return {
            "negotiation": {
                "id": negotiation_id,
                "requirement_id": requirement.id,
                "vendor_id": vendor.id,
                "status": status_box["status"],
                "final_price": 54120.0,
                "vendor_name": vendor.name,
            },
            "rounds": [
                {"round_number": 1, "actor": "buyer", "offered_price": 52250.0,
                 "message": "We open at Rs 52,250 per tonne."},
            ],
        }

    def fake_start(requirement_id: int, vendor_id: int, db) -> SimpleNamespace:
        assert (requirement_id, vendor_id) == (requirement.id, vendor.id)
        return SimpleNamespace(id=11)

    def fake_run_full(negotiation_id: int, db) -> dict:
        status_box["status"] = "pending_approval"
        return fake_state(negotiation_id, db)

    def fake_approve(negotiation_id: int, db) -> SimpleNamespace:
        if status_box["status"] != "pending_approval":
            raise ValueError("Negotiation is not pending approval.")
        status_box["status"] = "accepted"
        return SimpleNamespace(id=negotiation_id)

    def fake_generate_po(negotiation_id: int, db) -> SimpleNamespace:
        return SimpleNamespace(
            to_dict=lambda: {
                "id": 5,
                "negotiation_id": negotiation_id,
                "po_number": "PO-PRJ-TEST-0001",
                "status": "draft",
                "total_amount": 6494400.0,
            }
        )

    monkeypatch.setattr(negotiation_mod, "start_negotiation", fake_start)
    monkeypatch.setattr(negotiation_mod, "run_full_negotiation", fake_run_full)
    monkeypatch.setattr(negotiation_mod, "approve_negotiation", fake_approve)
    monkeypatch.setattr(negotiation_mod, "get_negotiation_state", fake_state)
    monkeypatch.setattr(purchase_order_mod, "generate_po", fake_generate_po)

    # 1. start
    response = client.post(
        "/api/negotiation/start",
        json={"requirement_id": requirement.id, "vendor_id": vendor.id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["negotiation"]["id"] == 11
    assert body["negotiation"]["status"] == "in_progress"
    assert body["rounds"][0]["actor"] == "buyer"

    # 2. run to convergence
    response = client.post("/api/negotiation/11/run")
    assert response.status_code == 200
    assert response.json()["negotiation"]["status"] == "pending_approval"

    # 3. approve -> state includes the generated PO
    response = client.post("/api/negotiation/11/approve")
    assert response.status_code == 200
    body = response.json()
    assert body["negotiation"]["status"] == "accepted"
    assert body["po"]["po_number"] == "PO-PRJ-TEST-0001"
    assert body["po"]["status"] == "draft"

    # 4. approving twice is a 400 with the ValueError message
    response = client.post("/api/negotiation/11/approve")
    assert response.status_code == 400
    assert response.json()["detail"] == "Negotiation is not pending approval."


def test_negotiation_start_unknown_requirement_is_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_start(requirement_id: int, vendor_id: int, db):
        raise ValueError("Requirement 999 not found")

    monkeypatch.setattr(negotiation_mod, "start_negotiation", fake_start)
    response = client.post(
        "/api/negotiation/start", json={"requirement_id": 999, "vendor_id": 1}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Requirement 999 not found"


def test_negotiations_list_includes_vendor_name(db_session) -> None:
    vendor = _make_vendor(db_session)
    db_session.add(
        Negotiation(
            project_id="PRJ-TEST", requirement_id=1, vendor_id=vendor.id,
            material="TMT steel", grade="Fe500D", quantity=120.0, unit="tonne",
            opening_offer=52250.0, target_price=54000.0,
            vendor_asking_price=56100.0, status="pending_approval",
        )
    )
    db_session.commit()
    response = client.get("/api/negotiations", params={"project_id": "PRJ-TEST"})
    assert response.status_code == 200
    negotiations = response.json()["negotiations"]
    assert len(negotiations) == 1
    assert negotiations[0]["vendor_name"] == vendor.name
    assert negotiations[0]["status"] == "pending_approval"


# -------------------------------------------------------------------- PO ----


def _make_po(db, **overrides) -> PurchaseOrder:
    defaults = dict(
        project_id="PRJ-TEST",
        negotiation_id=1,
        po_number="PO-PRJ-TEST-0001",
        vendor_id=1,
        material="TMT steel",
        grade="Fe500D",
        quantity=120.0,
        unit="tonne",
        unit_price=54120.0,
        total_amount=6494400.0,
        status="draft",
    )
    defaults.update(overrides)
    po = PurchaseOrder(**defaults)
    db.add(po)
    db.commit()
    return po


def test_po_list_includes_vendor_name(db_session) -> None:
    vendor = _make_vendor(db_session)
    _make_po(db_session, vendor_id=vendor.id)
    response = client.get("/api/po", params={"project_id": "PRJ-TEST"})
    assert response.status_code == 200
    purchase_orders = response.json()["purchase_orders"]
    assert len(purchase_orders) == 1
    assert purchase_orders[0]["po_number"] == "PO-PRJ-TEST-0001"
    assert purchase_orders[0]["vendor_name"] == vendor.name


def test_po_status_transition(db_session, monkeypatch: pytest.MonkeyPatch) -> None:
    po = _make_po(db_session)

    def fake_update(po_id: int, new_status: str, note, db) -> dict:
        row = db.get(PurchaseOrder, po_id)
        row.status = new_status
        db.commit()
        return {"po": row.to_dict(), "note": note}

    monkeypatch.setattr(tracking_mod, "update_tracking_status", fake_update)
    response = client.post(
        f"/api/po/{po.id}/status", json={"status": "sent", "note": "Emailed to vendor"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == po.id
    assert body["status"] == "sent"


def test_po_invalid_status_transition_is_400(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    po = _make_po(db_session, status="completed", po_number="PO-PRJ-TEST-0002")
    reason = "Cannot move a completed PO back to 'draft' — transitions are forward-only."

    def fake_update(po_id: int, new_status: str, note, db):
        raise tracking_mod.InvalidTransitionError(reason)

    monkeypatch.setattr(tracking_mod, "update_tracking_status", fake_update)
    response = client.post(f"/api/po/{po.id}/status", json={"status": "draft"})
    assert response.status_code == 400
    assert response.json()["detail"] == reason


def test_po_timeline(db_session, monkeypatch: pytest.MonkeyPatch) -> None:
    po = _make_po(db_session)
    events = [
        {"id": 1, "purchase_order_id": po.id, "status": "draft",
         "timestamp": "2026-07-01T10:00:00", "note": None},
        {"id": 2, "purchase_order_id": po.id, "status": "sent",
         "timestamp": "2026-07-02T09:00:00", "note": "Emailed"},
    ]
    monkeypatch.setattr(tracking_mod, "get_po_timeline", lambda po_id, db: events)
    response = client.get(f"/api/po/{po.id}/timeline")
    assert response.status_code == 200
    assert response.json() == {"timeline": events}


def test_po_download_missing_is_404(db_session) -> None:
    response = client.get("/api/po/999/download")
    assert response.status_code == 404


# ----------------------------------------------------------------- stats ----


def test_stats_shape(db_session) -> None:
    _make_po(db_session, status="completed", total_amount=500000.0,
             po_number="PO-PRJ-TEST-0001")
    _make_po(db_session, status="sent", total_amount=250000.0,
             po_number="PO-PRJ-TEST-0002")
    db_session.add(
        Negotiation(
            project_id="PRJ-TEST", requirement_id=1, vendor_id=1,
            opening_offer=100.0, target_price=110.0, status="pending_approval",
        )
    )
    db_session.commit()

    response = client.get("/api/stats", params={"project_id": "PRJ-TEST"})
    assert response.status_code == 200
    body = response.json()
    assert body["total_po_value"] == pytest.approx(750000.0)
    assert body["pending_approval"] == 1
    assert body["in_transit"] == 1
    assert body["completion_rate"] == pytest.approx(50.0)
    assert body["po_count"] == 2


def test_stats_empty_project(db_session) -> None:
    response = client.get("/api/stats", params={"project_id": "PRJ-EMPTY"})
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "total_po_value": 0.0,
        "pending_approval": 0,
        "in_transit": 0,
        "completion_rate": 0.0,
        "po_count": 0,
    }


# ----------------------------------------------------------------- agent ----


def test_agent_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    result = {
        "answer": "Recommended vendor: Shakti TMT Industries (via recommend_vendor).",
        "tool_calls": [
            {"tool": "recommend_vendor", "input": {"requirement_id": 1},
             "output_summary": "Shakti TMT Industries, risk 42 MEDIUM"}
        ],
        "final": True,
    }

    def fake_run_agent(question: str, project_id: str, db) -> dict:
        assert question == "Who should supply the TMT steel?"
        assert project_id == "PRJ-TEST"
        return result

    monkeypatch.setattr(agent_mod, "run_agent", fake_run_agent)
    response = client.post(
        "/api/agent/ask",
        json={"project_id": "PRJ-TEST", "question": "Who should supply the TMT steel?"},
    )
    assert response.status_code == 200
    assert response.json() == result


def test_agent_ask_empty_question_is_400() -> None:
    response = client.post(
        "/api/agent/ask", json={"project_id": "PRJ-TEST", "question": ""}
    )
    assert response.status_code == 400


def test_agent_ask_llm_provider_error_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider API failures (e.g. invalid key, overload) surface as 503, not 500."""
    from backend import llm

    def fake_run_agent(question: str, project_id: str, db) -> dict:
        raise llm.LLMError("anthropic tools loop failed: invalid x-api-key")

    monkeypatch.setattr(agent_mod, "run_agent", fake_run_agent)
    response = client.post(
        "/api/agent/ask",
        json={"project_id": "PRJ-TEST", "question": "Who should supply the TMT steel?"},
    )
    assert response.status_code == 503
    assert "retry" in response.json()["detail"].lower()
