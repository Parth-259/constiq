"""Hermetic tests for backend.agent.purchase_order and backend.agent.tracking.

In-memory SQLite; config.PO_DIR monkeypatched to tmp_path so no PDF ever
lands in the project data/ directory. No LLM involvement in these modules.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.agent import purchase_order as po_mod
from backend.agent import tracking as tracking_mod
from backend.agent.tracking import InvalidTransitionError
from backend.db.models import (
    Base,
    ExtractedRequirement,
    Negotiation,
    PurchaseOrder,
    TrackingEvent,
    Vendor,
    VendorQuote,
)

PROJECT_ID = "PRJ-T"
FINAL_PRICE = 1018.30
QUANTITY = 100.0


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def tmp_po_dir(tmp_path, monkeypatch):
    """PDFs must go to a temp dir, never the real config.PO_DIR."""
    monkeypatch.setattr(config, "PO_DIR", tmp_path)
    return tmp_path


def _seed_negotiation(db_session, status: str = "accepted", final_price: float | None = FINAL_PRICE):
    vendor = Vendor(
        name="Shakti Steel Traders",
        materials_supplied='["Fe500D TMT steel"]',
        location="Pune",
        contact_email="sales@shakti.example",
        avg_delivery_days=12,
        historical_on_time_pct=92.0,
    )
    db_session.add(vendor)
    db_session.flush()
    requirement = ExtractedRequirement(
        project_id=PROJECT_ID,
        source_file="seed",
        material="Fe500D TMT steel",
        grade="Fe500D",
        quantity=QUANTITY,
        unit="tonne",
        source_page=1,
        confidence="high",
    )
    db_session.add(requirement)
    db_session.flush()
    quote = VendorQuote(
        project_id=PROJECT_ID,
        vendor_id=vendor.id,
        material="Fe500D TMT steel",
        grade="Fe500D",
        quoted_price=1000.0,
        unit="tonne",
        payment_terms="Net 45",
    )
    negotiation = Negotiation(
        project_id=PROJECT_ID,
        requirement_id=requirement.id,
        vendor_id=vendor.id,
        material="Fe500D TMT steel",
        grade="Fe500D",
        quantity=QUANTITY,
        unit="tonne",
        opening_offer=950.0,
        target_price=1015.0,
        vendor_asking_price=1100.0,
        final_price=final_price,
        status=status,
    )
    db_session.add_all([quote, negotiation])
    db_session.commit()
    return negotiation, vendor


def _make_po(db_session, vendor_id: int, negotiation_id: int, status: str = "draft", seq: int = 1):
    po = PurchaseOrder(
        project_id=PROJECT_ID,
        negotiation_id=negotiation_id,
        po_number=f"PO-{PROJECT_ID}-{seq:04d}",
        vendor_id=vendor_id,
        material="Fe500D TMT steel",
        grade="Fe500D",
        quantity=QUANTITY,
        unit="tonne",
        unit_price=FINAL_PRICE,
        total_amount=QUANTITY * FINAL_PRICE,
        status=status,
    )
    db_session.add(po)
    db_session.commit()
    return po


# --------------------------------------------------------------------------
# purchase_order.generate_po
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["in_progress", "pending_approval", "stalled", "declined"])
def test_generate_po_refuses_non_accepted(db_session, status):
    negotiation, _vendor = _seed_negotiation(db_session, status=status)
    with pytest.raises(
        ValueError,
        match=(
            r"Purchase orders can only be generated from an accepted "
            rf"negotiation \(current status: {status}\)"
        ),
    ):
        po_mod.generate_po(negotiation.id, db_session)
    # Nothing persisted, no PDF written.
    assert db_session.query(PurchaseOrder).count() == 0


def test_generate_po_succeeds_on_accepted(db_session, tmp_po_dir):
    negotiation, vendor = _seed_negotiation(db_session, status="accepted")
    po = po_mod.generate_po(negotiation.id, db_session)

    assert po.po_number == f"PO-{PROJECT_ID}-0001"
    assert po.status == "draft"
    assert po.unit_price == pytest.approx(FINAL_PRICE)
    assert po.total_amount == pytest.approx(QUANTITY * FINAL_PRICE)
    assert po.payment_terms == "Net 45"  # from the vendor's matching quote
    assert po.delivery_date == date.today() + timedelta(days=vendor.avg_delivery_days)

    # PDF actually exists on disk, inside the temp PO_DIR.
    pdf_path = Path(po.pdf_path)
    assert pdf_path.exists()
    assert pdf_path.parent == tmp_po_dir
    assert pdf_path.name == f"{po.po_number}.pdf"
    assert pdf_path.stat().st_size > 0
    assert pdf_path.read_bytes().startswith(b"%PDF")

    # Initial "draft" tracking event recorded.
    timeline = tracking_mod.get_po_timeline(po.id, db_session)
    assert len(timeline) == 1
    assert timeline[0]["status"] == "draft"


def test_generate_po_numbers_are_sequential_per_project(db_session):
    first_neg, vendor = _seed_negotiation(db_session, status="accepted")
    second_neg = Negotiation(
        project_id=PROJECT_ID,
        requirement_id=first_neg.requirement_id,
        vendor_id=vendor.id,
        material="Fe500D TMT steel",
        grade="Fe500D",
        quantity=50.0,
        unit="tonne",
        opening_offer=950.0,
        target_price=1015.0,
        vendor_asking_price=1100.0,
        final_price=1005.0,
        status="accepted",
    )
    db_session.add(second_neg)
    db_session.commit()

    first_po = po_mod.generate_po(first_neg.id, db_session)
    second_po = po_mod.generate_po(second_neg.id, db_session)
    assert first_po.po_number == f"PO-{PROJECT_ID}-0001"
    assert second_po.po_number == f"PO-{PROJECT_ID}-0002"


def test_generate_po_defaults_payment_terms_without_matching_quote(db_session):
    negotiation, _vendor = _seed_negotiation(db_session, status="accepted")
    db_session.query(VendorQuote).delete()
    db_session.commit()

    po = po_mod.generate_po(negotiation.id, db_session)
    assert po.payment_terms == "Net 30"


# --------------------------------------------------------------------------
# tracking.update_tracking_status / get_po_timeline
# --------------------------------------------------------------------------

def test_forward_transition_ok_and_recorded(db_session):
    negotiation, vendor = _seed_negotiation(db_session)
    po = _make_po(db_session, vendor.id, negotiation.id, status="draft")

    result = tracking_mod.update_tracking_status(po.id, "sent", "Emailed to vendor", db_session)
    assert result["status"] == "sent"
    assert po.status == "sent"

    timeline = tracking_mod.get_po_timeline(po.id, db_session)
    assert timeline[-1]["status"] == "sent"
    assert timeline[-1]["note"] == "Emailed to vendor"


def test_backward_transition_raises(db_session):
    negotiation, vendor = _seed_negotiation(db_session)
    po = _make_po(db_session, vendor.id, negotiation.id, status="sent")

    with pytest.raises(InvalidTransitionError, match="forward"):
        tracking_mod.update_tracking_status(po.id, "draft", None, db_session)
    assert po.status == "sent"  # unchanged
    # No event appended for the rejected transition.
    assert tracking_mod.get_po_timeline(po.id, db_session) == []


def test_invalid_transition_error_is_a_value_error(db_session):
    assert issubclass(InvalidTransitionError, ValueError)


def test_cancelled_allowed_from_sent(db_session):
    negotiation, vendor = _seed_negotiation(db_session)
    po = _make_po(db_session, vendor.id, negotiation.id, status="sent")

    result = tracking_mod.update_tracking_status(po.id, "cancelled", "Vendor unresponsive", db_session)
    assert result["status"] == "cancelled"
    timeline = tracking_mod.get_po_timeline(po.id, db_session)
    assert timeline[-1]["status"] == "cancelled"


def test_cancelled_disallowed_from_completed(db_session):
    negotiation, vendor = _seed_negotiation(db_session)
    po = _make_po(db_session, vendor.id, negotiation.id, status="completed")

    with pytest.raises(InvalidTransitionError, match="completed"):
        tracking_mod.update_tracking_status(po.id, "cancelled", None, db_session)
    assert po.status == "completed"


def test_unknown_status_raises(db_session):
    negotiation, vendor = _seed_negotiation(db_session)
    po = _make_po(db_session, vendor.id, negotiation.id, status="draft")

    with pytest.raises(InvalidTransitionError, match="not a recognised"):
        tracking_mod.update_tracking_status(po.id, "teleported", None, db_session)


def test_timeline_sorted_ascending_over_full_lifecycle(db_session):
    negotiation, vendor = _seed_negotiation(db_session)
    po = _make_po(db_session, vendor.id, negotiation.id, status="draft")

    for status in ["sent", "accepted", "delivered", "completed"]:
        tracking_mod.update_tracking_status(po.id, status, None, db_session)

    timeline = tracking_mod.get_po_timeline(po.id, db_session)
    assert [event["status"] for event in timeline] == [
        "sent",
        "accepted",
        "delivered",
        "completed",
    ]
    timestamps = [event["timestamp"] for event in timeline]
    assert timestamps == sorted(timestamps)

    # And a completed PO is terminal for any further transition.
    with pytest.raises(InvalidTransitionError):
        tracking_mod.update_tracking_status(po.id, "sent", None, db_session)
