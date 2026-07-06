"""Hermetic tests for backend.agent.tools.

Uses an in-memory SQLite database with hand-seeded fixture vendors, quotes,
and requirements. External services (Tavily) are mocked — no network calls.
"""
from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend import config
from backend.agent import tools
from backend.db.models import Base, ExtractedRequirement, Vendor, VendorQuote


# ---------------------------------------------------------------------------
# Fixtures (small hand-seeded dataset — deliberately NOT backend.db.seed_vendors)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_external_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call real APIs even if the developer has keys in .env."""
    monkeypatch.setattr(config, "TAVILY_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "", raising=False)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _add_vendor(session: Session, name: str, materials: list[str], **kwargs) -> Vendor:
    vendor = Vendor(name=name, materials_supplied=json.dumps(materials), **kwargs)
    session.add(vendor)
    session.flush()
    return vendor


def _add_requirement(session: Session, material: str, **kwargs) -> ExtractedRequirement:
    requirement = ExtractedRequirement(
        project_id="PRJ-TEST-001", material=material, **kwargs
    )
    session.add(requirement)
    session.flush()
    return requirement


def _add_quote(
    session: Session, vendor_id: int, material: str, price: float, grade: str | None = None
) -> VendorQuote:
    quote = VendorQuote(
        project_id="PRJ-TEST-001",
        vendor_id=vendor_id,
        material=material,
        grade=grade,
        quoted_price=price,
        quantity=100.0,
        unit="tonne",
    )
    session.add(quote)
    session.flush()
    return quote


@pytest.fixture()
def seeded(db_session: Session) -> dict:
    v_exact = _add_vendor(
        db_session,
        "Deccan TMT Works",
        ["Fe500D TMT steel"],
        historical_on_time_pct=88.0,
        avg_delivery_days=12,
        typical_min_qty=20.0,
        typical_max_qty=300.0,
        price_index=1.0,
    )
    v_alt = _add_vendor(
        db_session,
        "Bharat Steels",
        ["Fe500 TMT steel"],
        historical_on_time_pct=98.0,
        avg_delivery_days=6,
        typical_min_qty=10.0,
        typical_max_qty=1000.0,
        price_index=0.95,
    )
    v_cement = _add_vendor(
        db_session,
        "Ganga Cement",
        ["OPC 53 cement", "PPC cement"],
        historical_on_time_pct=92.0,
        avg_delivery_days=8,
        typical_min_qty=100.0,
        typical_max_qty=5000.0,
    )
    v_glass = _add_vendor(
        db_session,
        "Crystal Glazing",
        ["Double-glazed curtain wall glass"],
        historical_on_time_pct=90.0,
        avg_delivery_days=25,
        typical_min_qty=100.0,
        typical_max_qty=5000.0,
        price_index=1.1,
    )

    # Quotes: TMT steel market data exists; glass has NO quotes at all.
    _add_quote(db_session, v_exact.id, "Fe500D TMT steel", 62000.0, grade="Fe500D")
    _add_quote(db_session, v_exact.id, "Fe500D TMT steel", 64000.0, grade="Fe500D")
    _add_quote(db_session, v_alt.id, "Fe500 TMT steel", 58000.0, grade="Fe500")
    _add_quote(db_session, v_cement.id, "OPC 53 cement", 420.0, grade="OPC 53")

    r_tmt = _add_requirement(
        db_session,
        "TMT steel",
        grade="Fe500D",
        quantity=120.0,
        unit="tonne",
        deadline="2026-08-10",
    )
    r_glass = _add_requirement(
        db_session, "curtain wall glass", grade=None, quantity=1200.0, unit="sqm"
    )
    r_copper = _add_requirement(db_session, "Copper pipes", grade=None, quantity=50.0)
    db_session.commit()

    return {
        "v_exact": v_exact,
        "v_alt": v_alt,
        "v_cement": v_cement,
        "v_glass": v_glass,
        "r_tmt": r_tmt,
        "r_glass": r_glass,
        "r_copper": r_copper,
    }


# ---------------------------------------------------------------------------
# vendor_lookup / get_market_reference_price
# ---------------------------------------------------------------------------

def test_vendor_lookup_case_insensitive_substring(db_session, seeded):
    results = tools.vendor_lookup("tmt STEEL", None, db_session)
    names = {r["name"] for r in results}
    assert names == {"Deccan TMT Works", "Bharat Steels"}


def test_vendor_lookup_exact_grade_first_and_no_match(db_session, seeded):
    results = tools.vendor_lookup("TMT steel", "Fe500D", db_session)
    assert results[0]["name"] == "Deccan TMT Works"
    assert tools.vendor_lookup("Copper pipes", None, db_session) == []


def test_market_reference_price(db_session, seeded):
    assert tools.get_market_reference_price("TMT steel", "Fe500D", db_session) == pytest.approx(
        63000.0
    )
    # No grade filter -> all three TMT quotes averaged.
    assert tools.get_market_reference_price("TMT steel", None, db_session) == pytest.approx(
        (62000.0 + 64000.0 + 58000.0) / 3
    )
    assert tools.get_market_reference_price("Copper pipes", None, db_session) is None


# ---------------------------------------------------------------------------
# check_compliance — all three statuses
# ---------------------------------------------------------------------------

def test_compliance_compliant_exact_grade(db_session, seeded):
    result = tools.check_compliance(seeded["r_tmt"].id, db_session)
    assert result["status"] == "compliant"
    assert result["requirement"]["material"] == "TMT steel"
    names = {v["name"] for v in result["matching_vendors"]}
    assert names == {"Deccan TMT Works"}
    assert "Fe500D" in result["explanation"]


def test_compliance_alternate_grade_names_both_grades(db_session, seeded):
    result = tools.check_compliance(
        seeded["r_tmt"].id, db_session, vendor_id=seeded["v_alt"].id
    )
    assert result["status"] == "non_compliant_alternate_available"
    assert [v["name"] for v in result["matching_vendors"]] == ["Bharat Steels"]
    assert "Fe500 is not interchangeable with Fe500D per IS 1786" in result["explanation"]


def test_compliance_no_vendor_found(db_session, seeded):
    result = tools.check_compliance(seeded["r_copper"].id, db_session)
    assert result["status"] == "no_vendor_found"
    assert result["matching_vendors"] == []
    assert "Copper pipes" in result["explanation"]


def test_compliance_no_grade_requirement_is_compliant(db_session, seeded):
    result = tools.check_compliance(seeded["r_glass"].id, db_session)
    assert result["status"] == "compliant"
    assert result["matching_vendors"][0]["name"] == "Crystal Glazing"


# ---------------------------------------------------------------------------
# calculate_risk — LOW / MEDIUM / HIGH with real numbers in the explanation
# ---------------------------------------------------------------------------

def test_risk_low(db_session, seeded):
    # lead 6/30=0.2 -> 10; reliability (100-98)/100=0.02 -> 0.7; size in range -> 0.
    result = tools.calculate_risk(seeded["r_tmt"].id, seeded["v_alt"].id, 30, db_session)
    assert result["score"] == 11
    assert result["label"] == "LOW"
    assert result["factors"]["lead_time_pressure"] == pytest.approx(0.2)
    assert result["factors"]["order_size_factor"] == 0.0
    assert "6 days" in result["explanation"]
    assert "30 days" in result["explanation"]
    assert "98%" in result["explanation"]


def test_risk_medium(db_session, seeded):
    # lead 12/15=0.8 -> 40; reliability 0.12 -> 4.2; size 0. Total 44.
    result = tools.calculate_risk(seeded["r_tmt"].id, seeded["v_exact"].id, 15, db_session)
    assert result["score"] == 44
    assert result["label"] == "MEDIUM"
    assert "12 days" in result["explanation"]
    assert "15 days" in result["explanation"]
    assert "88%" in result["explanation"]


def test_risk_high_lead_time_capped(db_session, seeded):
    # lead 12/5=2.4 capped at 2.0 -> 100; clamped to 100 overall.
    result = tools.calculate_risk(seeded["r_tmt"].id, seeded["v_exact"].id, 5, db_session)
    assert result["score"] == 100
    assert result["label"] == "HIGH"
    assert result["factors"]["lead_time_pressure"] == pytest.approx(2.0)
    assert "12 days" in result["explanation"]
    assert "5 days" in result["explanation"]


def test_risk_order_size_factor_outside_range(db_session, seeded):
    big = _add_requirement(
        db_session, "TMT steel", grade="Fe500D", quantity=5000.0, unit="tonne"
    )
    db_session.commit()
    # qty 5000 > max 300 -> relative distance (5000-300)/300 capped at 1.
    result = tools.calculate_risk(big.id, seeded["v_exact"].id, 30, db_session)
    assert result["factors"]["order_size_factor"] == 1.0


# ---------------------------------------------------------------------------
# vendor_discovery — Tavily mocked, never called for real
# ---------------------------------------------------------------------------

def test_discovery_tavily_raises_still_returns_internal(db_session, seeded, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key-for-test")
    fake_tavily = ModuleType("tavily")
    fake_tavily.TavilyClient = mock.Mock(side_effect=RuntimeError("network down"))
    with mock.patch.dict(sys.modules, {"tavily": fake_tavily}):
        result = tools.vendor_discovery("TMT steel", "Fe500D", "Mumbai", db_session)
    assert result["web_search_succeeded"] is False
    assert result["web_matches"] == []
    names = {v["name"] for v in result["internal_matches"]}
    assert names == {"Deccan TMT Works", "Bharat Steels"}
    assert all(v["verified"] is True for v in result["internal_matches"])


def test_discovery_missing_key_skips_web(db_session, seeded):
    result = tools.vendor_discovery("TMT steel", "Fe500D", "Mumbai", db_session)
    assert result["web_search_succeeded"] is False
    assert result["web_matches"] == []
    assert len(result["internal_matches"]) == 2


def test_discovery_success_fixed_query_and_unverified_web_matches(
    db_session, seeded, monkeypatch
):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "fake-key-for-test")
    client = mock.Mock()
    client.search.return_value = {
        "results": [
            {
                "title": "Acme Steel Traders",
                "url": "https://example.com/acme",
                "content": "Leading TMT supplier in Mumbai.",
            }
        ]
    }
    fake_tavily = ModuleType("tavily")
    fake_tavily.TavilyClient = mock.Mock(return_value=client)
    with mock.patch.dict(sys.modules, {"tavily": fake_tavily}):
        result = tools.vendor_discovery("TMT steel", "Fe500D", "Mumbai", db_session)

    assert client.search.call_args[0][0] == "TMT steel Fe500D supplier Mumbai India"
    assert result["web_search_succeeded"] is True
    assert len(result["web_matches"]) == 1
    web = result["web_matches"][0]
    assert web == {
        "name": "Acme Steel Traders",
        "snippet": "Leading TMT supplier in Mumbai.",
        "source_url": "https://example.com/acme",
        "verified": False,
    }
    # No fabricated numeric vendor fields on web results.
    assert "historical_on_time_pct" not in web
    assert "price_index" not in web


# ---------------------------------------------------------------------------
# vendor_evaluation
# ---------------------------------------------------------------------------

def test_evaluation_ranking_known_inputs(db_session, seeded):
    candidates = tools.vendor_lookup("TMT steel", "Fe500D", db_session)
    results = tools.vendor_evaluation(candidates, "TMT steel", "Fe500D", 120.0, db_session)

    assert [r["name"] for r in results] == ["Bharat Steels", "Deccan TMT Works"]
    alt, exact = results
    # Bharat: reliability .98, price 1-|58000-63000|/63000, capacity 1.
    assert alt["reliability_score"] == pytest.approx(0.98)
    assert alt["price_score"] == pytest.approx(1 - 5000 / 63000, abs=1e-4)
    assert alt["capacity_score"] == pytest.approx(1.0)
    assert alt["evaluation_score"] == pytest.approx(
        0.4 * 0.98 + 0.4 * (1 - 5000 / 63000) + 0.2 * 1.0, abs=1e-3
    )
    # Deccan: quotes average exactly at the market reference -> price 1.0.
    assert exact["price_score"] == pytest.approx(1.0)
    assert exact["evaluation_score"] == pytest.approx(0.952, abs=1e-3)
    assert alt["evaluation_score"] > exact["evaluation_score"]
    assert "summary" in alt and alt["summary"]


def test_evaluation_no_market_data_defaults_price_score(db_session, seeded):
    candidates = tools.vendor_lookup("curtain wall glass", None, db_session)
    results = tools.vendor_evaluation(
        candidates, "curtain wall glass", None, 1200.0, db_session
    )
    assert len(results) == 1
    glass = results[0]
    assert glass["price_score"] == 0.5
    assert "note" in glass and "0.5" in glass["note"]
    assert glass["evaluation_score"] == pytest.approx(
        0.4 * 0.9 + 0.4 * 0.5 + 0.2 * 1.0, abs=1e-3
    )


def test_evaluation_unverified_candidates_appended_unscored(db_session, seeded):
    candidates = tools.vendor_lookup("TMT steel", "Fe500D", db_session)
    web_candidate = {
        "name": "Acme Steel Traders",
        "snippet": "web result",
        "source_url": "https://example.com/acme",
        "verified": False,
    }
    results = tools.vendor_evaluation(
        candidates + [web_candidate], "TMT steel", "Fe500D", 120.0, db_session
    )
    assert results[-1]["name"] == "Acme Steel Traders"
    assert results[-1]["evaluation_score"] is None
    assert "note" in results[-1]


# ---------------------------------------------------------------------------
# recommend_vendor
# ---------------------------------------------------------------------------

def test_recommend_prefers_compliant_over_higher_scoring_alternate(db_session, seeded):
    # Bharat Steels (Fe500, alternate) out-scores Deccan (Fe500D, compliant) on
    # evaluation, but compliance must win the ranking.
    result = tools.recommend_vendor(seeded["r_tmt"].id, 30, db_session)
    assert result["recommended_vendor"]["id"] == seeded["v_exact"].id
    assert result["compliance_status"] == "compliant"
    assert result["risk_label"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(result["risk_score"], int)
    assert "12 days" in result["risk_explanation"]
    assert result["evaluation_summary"]
    assert result["overall_reason"]
    # Runner-up is the alternate-grade vendor.
    assert len(result["alternatives"]) == 1
    alt = result["alternatives"][0]
    assert alt["vendor"]["id"] == seeded["v_alt"].id
    assert alt["compliance_status"] == "non_compliant_alternate_available"
    assert alt["evaluation_score"] > result["recommended_vendor"]["evaluation_score"]


def test_recommend_no_vendor_possible(db_session, seeded):
    result = tools.recommend_vendor(seeded["r_copper"].id, 30, db_session)
    assert result["recommended_vendor"] is None
    assert result["reason"].startswith("no_recommendation_possible")


def test_recommend_unknown_requirement(db_session, seeded):
    result = tools.recommend_vendor(99999, 30, db_session)
    assert result["recommended_vendor"] is None
    assert result["reason"].startswith("no_recommendation_possible")
