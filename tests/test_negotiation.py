"""Hermetic tests for backend.agent.negotiation.

- In-memory SQLite, hand-seeded vendors/requirements (no project data/ touched).
- backend.agent.tools is stubbed if the real module is not present yet, and
  its two functions are always monkeypatched — negotiation math must not
  depend on their real implementations.
- anthropic is never called for real: either the API key is forced to "" or
  sys.modules["anthropic"] is replaced with a mock.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# The tools module is built by another team; provide a stub when absent so
# backend.agent.negotiation's lazy imports resolve.
try:  # pragma: no cover - depends on build order
    import backend.agent.tools  # noqa: F401
except ImportError:  # pragma: no cover
    _tools_stub = types.ModuleType("backend.agent.tools")
    _tools_stub.get_market_reference_price = (  # type: ignore[attr-defined]
        lambda material, grade, db_session: None
    )
    _tools_stub.calculate_risk = (  # type: ignore[attr-defined]
        lambda requirement_id, vendor_id, deadline_days_remaining, db_session: {
            "score": 0,
            "label": "LOW",
            "factors": {},
            "explanation": "stub",
        }
    )
    sys.modules["backend.agent.tools"] = _tools_stub
    import backend.agent as _agent_pkg

    _agent_pkg.tools = _tools_stub  # type: ignore[attr-defined]

from backend import config
from backend.agent import negotiation as neg_mod
from backend.db.models import (
    Base,
    ExtractedRequirement,
    NegotiationRound,
    Vendor,
    VendorQuote,
)

MARKET_REF = 1000.0
RISK_RESULT = {"score": 50, "label": "MEDIUM", "factors": {}, "explanation": "mocked"}
# With market 1000, discount 5%, risk 50 and the 3% urgency premium:
EXPECTED_OPENING = 950.0
EXPECTED_TARGET = 1015.0


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
def no_api_key(monkeypatch):
    """Default every test to the no-key template path (no anthropic import)."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")


def _patch_tools(monkeypatch, market: float | None = MARKET_REF, risk: dict = RISK_RESULT) -> None:
    import backend.agent.tools as tools

    monkeypatch.setattr(tools, "get_market_reference_price", lambda *a, **k: market)
    monkeypatch.setattr(tools, "calculate_risk", lambda *a, **k: dict(risk))


def _seed(db_session, flexibility: float, price_index: float):
    vendor = Vendor(
        name="Test Steel Co",
        materials_supplied='["Fe500D TMT steel"]',
        location="Pune",
        avg_delivery_days=10,
        historical_on_time_pct=90.0,
        typical_min_qty=10.0,
        typical_max_qty=500.0,
        price_index=price_index,
        negotiation_flexibility=flexibility,
    )
    requirement = ExtractedRequirement(
        project_id="PRJ-T",
        source_file="seed",
        material="Fe500D TMT steel",
        grade="Fe500D",
        quantity=100.0,
        unit="tonne",
        deadline="2026-08-10",
        source_page=1,
        confidence="high",
    )
    db_session.add_all([vendor, requirement])
    db_session.commit()
    return requirement, vendor


def _expected_rounds(
    opening: float, asking: float, target: float, flexibility: float, max_rounds: int
) -> tuple[list[tuple[str, float]], str, float | None]:
    """Independent reference implementation of the contract's concession math."""
    rounds: list[tuple[str, float]] = [("buyer", opening)]
    last_buyer: float = opening
    last_vendor: float | None = None
    status = "in_progress"
    final: float | None = None
    while status == "in_progress" and len(rounds) < 50:
        actor = "vendor" if rounds[-1][0] == "buyer" else "buyer"
        if actor == "vendor":
            if last_vendor is None:
                price = asking
            else:
                price = last_vendor - abs(last_vendor - last_buyer) * flexibility
            last_vendor = price
        else:
            assert last_vendor is not None
            price = min(last_buyer + abs(last_vendor - last_buyer) * 0.4, target)
            last_buyer = price
        rounds.append((actor, price))
        gap = abs(last_vendor - last_buyer)
        if gap / opening < config.NEGOTIATION_CONVERGENCE_PCT:
            status = "pending_approval"
            final = round((last_vendor + last_buyer) / 2, 2)
        elif sum(1 for a, _ in rounds if a == "vendor") >= max_rounds:
            status = "stalled"
    return rounds, status, final


def test_start_negotiation_creates_opening_round(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    _patch_tools(monkeypatch)

    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)

    assert negotiation.status == "in_progress"
    assert negotiation.opening_offer == pytest.approx(EXPECTED_OPENING)
    assert negotiation.vendor_asking_price == pytest.approx(1100.0)
    assert negotiation.target_price == pytest.approx(EXPECTED_TARGET)
    assert negotiation.max_rounds == config.NEGOTIATION_MAX_ROUNDS

    rounds = (
        db_session.query(NegotiationRound)
        .filter_by(negotiation_id=negotiation.id)
        .all()
    )
    assert len(rounds) == 1
    assert rounds[0].round_number == 1
    assert rounds[0].actor == "buyer"
    assert rounds[0].offered_price == pytest.approx(EXPECTED_OPENING)
    assert rounds[0].message  # template narration present without an API key


def test_flexible_vendor_converges_to_pending_approval(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    _patch_tools(monkeypatch)

    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)
    state = neg_mod.run_full_negotiation(negotiation.id, db_session)

    assert state["negotiation"]["status"] == "pending_approval"
    assert state["negotiation"]["vendor_name"] == "Test Steel Co"

    rounds = state["rounds"]
    # Every round stored: contiguous numbering, buyer first, strict alternation.
    assert [r["round_number"] for r in rounds] == list(range(1, len(rounds) + 1))
    assert rounds[0]["actor"] == "buyer"
    for previous, current in zip(rounds, rounds[1:]):
        assert current["actor"] != previous["actor"]
    assert all(r["message"] for r in rounds)

    vendor_turns = sum(1 for r in rounds if r["actor"] == "vendor")
    assert 0 < vendor_turns <= negotiation.max_rounds

    # final_price is the 2dp midpoint of the two last stated prices.
    last_vendor = [r["offered_price"] for r in rounds if r["actor"] == "vendor"][-1]
    last_buyer = [r["offered_price"] for r in rounds if r["actor"] == "buyer"][-1]
    assert abs(last_vendor - last_buyer) / EXPECTED_OPENING < config.NEGOTIATION_CONVERGENCE_PCT
    assert state["negotiation"]["final_price"] == pytest.approx(
        round((last_vendor + last_buyer) / 2, 2)
    )

    # Round rows really are in the DB, one per state entry.
    db_rounds = (
        db_session.query(NegotiationRound)
        .filter_by(negotiation_id=negotiation.id)
        .count()
    )
    assert db_rounds == len(rounds)


def test_stubborn_vendor_stalls_at_exactly_max_rounds(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.05, price_index=1.12)
    _patch_tools(monkeypatch)

    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)
    state = neg_mod.run_full_negotiation(negotiation.id, db_session)

    assert state["negotiation"]["status"] == "stalled"
    assert state["negotiation"]["final_price"] is None
    vendor_turns = sum(1 for r in state["rounds"] if r["actor"] == "vendor")
    assert vendor_turns == negotiation.max_rounds  # exactly max_rounds vendor turns
    # Buyer opens, so a stall after N vendor turns means 2N rounds total.
    assert len(state["rounds"]) == 2 * negotiation.max_rounds


def test_offered_prices_follow_formula_regardless_of_llm_text(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    _patch_tools(monkeypatch)

    # Force the LLM path with a mocked anthropic module that returns nonsense.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key-not-real")
    llm_text = "Nine crore rupees, final offer, take it or leave it!!!"
    fake_client = MagicMock()
    fake_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text=llm_text)]
    )
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)
    state = neg_mod.run_full_negotiation(negotiation.id, db_session)

    expected, expected_status, expected_final = _expected_rounds(
        opening=EXPECTED_OPENING,
        asking=1100.0,
        target=EXPECTED_TARGET,
        flexibility=0.7,
        max_rounds=negotiation.max_rounds,
    )
    got = [(r["actor"], r["offered_price"]) for r in state["rounds"]]
    assert len(got) == len(expected)
    for (got_actor, got_price), (exp_actor, exp_price) in zip(got, expected):
        assert got_actor == exp_actor
        assert got_price == pytest.approx(exp_price)
    assert state["negotiation"]["status"] == expected_status == "pending_approval"
    assert state["negotiation"]["final_price"] == pytest.approx(expected_final)

    # The mocked LLM was used for narration only and never leaked into any
    # price. Its text omits the computed price, so every round falls back to
    # the deterministic template, which states the price verbatim.
    assert fake_client.messages.create.called
    for r in state["rounds"]:
        assert r["message"] != llm_text
        assert f"₹{r['offered_price']:,.2f}" in r["message"]


def test_approve_pending_approval_becomes_accepted(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    _patch_tools(monkeypatch)
    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)
    neg_mod.run_full_negotiation(negotiation.id, db_session)
    assert negotiation.status == "pending_approval"

    approved = neg_mod.approve_negotiation(negotiation.id, db_session)
    assert approved.status == "accepted"
    assert approved.final_price is not None


def test_approve_stalled_raises_value_error(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.05, price_index=1.12)
    _patch_tools(monkeypatch)
    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)
    neg_mod.run_full_negotiation(negotiation.id, db_session)
    assert negotiation.status == "stalled"

    with pytest.raises(ValueError, match="stalled"):
        neg_mod.approve_negotiation(negotiation.id, db_session)


def test_approve_in_progress_raises_value_error(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    _patch_tools(monkeypatch)
    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)

    with pytest.raises(ValueError, match="in_progress"):
        neg_mod.approve_negotiation(negotiation.id, db_session)


def test_decline_sets_declined(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    _patch_tools(monkeypatch)
    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)

    declined = neg_mod.decline_negotiation(negotiation.id, db_session)
    assert declined.status == "declined"


def test_market_ref_falls_back_to_vendor_quote(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    db_session.add(
        VendorQuote(
            project_id="PRJ-T",
            vendor_id=vendor.id,
            material="Fe500D TMT steel",
            grade="Fe500D",
            quoted_price=900.0,
            unit="tonne",
        )
    )
    db_session.commit()
    _patch_tools(monkeypatch, market=None)  # no market reference price

    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)
    assert negotiation.vendor_asking_price == pytest.approx(900.0 * 1.10)
    assert negotiation.opening_offer == pytest.approx(
        900.0 * (1 - config.NEGOTIATION_OPENING_DISCOUNT)
    )


def test_start_negotiation_without_any_price_raises(db_session, monkeypatch):
    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    _patch_tools(monkeypatch, market=None)  # and the vendor has no quotes

    with pytest.raises(ValueError, match="[Nn]o market reference"):
        neg_mod.start_negotiation(requirement.id, vendor.id, db_session)


def test_narrate_rejects_paraphrased_price(db_session, monkeypatch):
    """Validate that paraphrased or rounded prices are rejected in favor of template."""
    from backend import llm as llm_mod

    requirement, vendor = _seed(db_session, flexibility=0.7, price_index=1.10)
    _patch_tools(monkeypatch)

    # Enable LLM with mocked response that paraphrases the price
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key-not-real")

    # The computed price is ₹950.00, but LLM returns a paraphrased version
    paraphrased_text = "We would like to offer ₹95,000 (approximately) for your materials."
    fake_client = MagicMock()
    fake_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text=paraphrased_text)]
    )
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    negotiation = neg_mod.start_negotiation(requirement.id, vendor.id, db_session)

    # The opening_offer is ₹950.00, so the template should contain "₹950.00"
    # because the LLM text contains paraphrased price "₹95,000" instead
    opening_round = (
        db_session.query(NegotiationRound)
        .filter_by(negotiation_id=negotiation.id, round_number=1)
        .one()
    )

    # The message should be the template fallback because it rejected the paraphrased price
    assert f"₹{negotiation.opening_offer:,.2f}" in opening_round.message
    assert opening_round.message != paraphrased_text
