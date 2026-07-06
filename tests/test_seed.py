"""Hermetic tests for backend.db.seed_vendors (in-memory SQLite only)."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.db.models import Base, ExtractedRequirement, Vendor, VendorQuote
from backend.db.seed_vendors import (
    DEMO_PROJECT_ID,
    DEMO_SOURCE_FILE,
    seed_all,
    seed_demo_requirements,
    seed_quotes,
    seed_vendors,
)


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


def test_seed_all_returns_counts(db_session: Session) -> None:
    counts = seed_all(db_session)
    assert set(counts) == {"vendors", "quotes", "requirements"}
    assert 16 <= counts["vendors"] <= 18
    assert counts["requirements"] == 4
    # 2-3 quotes per vendor
    assert 2 * counts["vendors"] <= counts["quotes"] <= 3 * counts["vendors"]
    # counts reflect what actually landed in the DB
    assert db_session.query(Vendor).count() == counts["vendors"]
    assert db_session.query(VendorQuote).count() == counts["quotes"]
    assert db_session.query(ExtractedRequirement).count() == counts["requirements"]


def test_seed_all_is_idempotent(db_session: Session) -> None:
    first = seed_all(db_session)
    vendors_before = db_session.query(Vendor).count()
    quotes_before = db_session.query(VendorQuote).count()
    reqs_before = db_session.query(ExtractedRequirement).count()

    second = seed_all(db_session)
    assert second == {"vendors": 0, "quotes": 0, "requirements": 0}
    assert db_session.query(Vendor).count() == vendors_before == first["vendors"]
    assert db_session.query(VendorQuote).count() == quotes_before
    assert db_session.query(ExtractedRequirement).count() == reqs_before


def test_at_least_one_vendor_supplies_fe500d(db_session: Session) -> None:
    seed_vendors(db_session)
    vendors = db_session.query(Vendor).all()
    fe500d_suppliers = [
        v for v in vendors
        if any("fe500d" in m.lower() for m in v.materials_list())
    ]
    assert fe500d_suppliers, "expected at least one Fe500D TMT supplier"


def test_quotes_exist_for_fe500d_material(db_session: Session) -> None:
    """Market-reference-style sanity: quotes are queryable for Fe500D."""
    seed_vendors(db_session)
    seed_quotes(db_session)
    quotes = db_session.query(VendorQuote).all()
    fe500d_quotes = [q for q in quotes if "fe500d" in q.material.lower()]
    assert fe500d_quotes, "expected VendorQuote rows for Fe500D material"
    for quote in fe500d_quotes:
        assert quote.quoted_price > 0
        assert quote.project_id == DEMO_PROJECT_ID


def test_quotes_only_for_supplied_materials_and_deterministic_prices(
    db_session: Session,
) -> None:
    seed_vendors(db_session)
    seed_quotes(db_session)
    vendors_by_id = {v.id: v for v in db_session.query(Vendor).all()}
    quotes = db_session.query(VendorQuote).order_by(VendorQuote.id).all()
    for quote in quotes:
        vendor = vendors_by_id[quote.vendor_id]
        assert quote.material in vendor.materials_list()

    # Determinism: a fresh DB seeded the same way yields identical prices.
    engine2 = create_engine("sqlite://")
    Base.metadata.create_all(engine2)
    factory2 = sessionmaker(bind=engine2, autoflush=False, expire_on_commit=False)
    with factory2() as session2:
        seed_vendors(session2)
        seed_quotes(session2)
        prices2 = [
            q.quoted_price
            for q in session2.query(VendorQuote).order_by(VendorQuote.id).all()
        ]
    assert [q.quoted_price for q in quotes] == prices2
    engine2.dispose()


def test_vendor_min_qty_less_than_max_qty(db_session: Session) -> None:
    seed_vendors(db_session)
    for vendor in db_session.query(Vendor).all():
        assert vendor.typical_min_qty < vendor.typical_max_qty, vendor.name


def test_vendor_attribute_ranges_and_flexibility_spread(db_session: Session) -> None:
    seed_vendors(db_session)
    vendors = db_session.query(Vendor).all()
    for v in vendors:
        assert 68.0 <= v.historical_on_time_pct <= 98.0, v.name
        assert 3 <= v.avg_delivery_days <= 45, v.name
        assert 0.92 <= v.price_index <= 1.15, v.name
        assert 0.05 <= v.negotiation_flexibility <= 0.8, v.name
        assert 3.0 <= v.rating <= 5.0, v.name
    flexibilities = [v.negotiation_flexibility for v in vendors]
    assert min(flexibilities) < 0.15, "expected at least one stubborn vendor"
    assert max(flexibilities) > 0.6, "expected at least one flexible vendor"


def test_demo_requirements_match_contract(db_session: Session) -> None:
    seed_demo_requirements(db_session)
    rows = (
        db_session.query(ExtractedRequirement)
        .filter(
            ExtractedRequirement.project_id == DEMO_PROJECT_ID,
            ExtractedRequirement.source_file == DEMO_SOURCE_FILE,
        )
        .all()
    )
    assert len(rows) == 4
    by_grade = {r.grade: r for r in rows}

    fe500d = by_grade["Fe500D"]
    assert fe500d.quantity == 120.0
    assert fe500d.unit == "tonne"
    assert fe500d.deadline == "2026-08-10"
    assert "tmt" in fe500d.material.lower()

    opc53 = by_grade["OPC 53"]
    assert opc53.quantity == 800.0
    assert opc53.unit == "bag"
    assert "cement" in opc53.material.lower()

    m40 = by_grade["M40"]
    assert m40.quantity == 350.0
    assert m40.unit == "cum"
    assert "ready-mix concrete" in m40.material.lower()

    glass = by_grade[None]
    assert glass.quantity == 1200.0
    assert glass.unit == "sqm"
    assert "curtain wall glass" in glass.material.lower()

    assert all(r.superseded_by is None for r in rows)
