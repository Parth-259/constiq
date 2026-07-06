"""Vendor price negotiation engine for ConstructIQ.

All pricing is DETERMINISTIC — the concession math below is the single source
of truth for every number stored in a :class:`NegotiationRound`. The LLM
(``backend.llm``, tier "fast") is used ONLY to phrase an already-computed
price as one short in-character line; with no LLM provider configured a
formatted template is used instead.

Concession rules (see CONTRACT.md):
- Round 1 is always the buyer's opening offer (created by ``start_negotiation``).
- Vendor's first counter is exactly ``vendor_asking_price``; afterwards the
  vendor concedes ``gap * negotiation_flexibility`` per turn.
- Buyer moves up ``gap * 0.4`` per turn, hard-capped at ``target_price``.
- After each round: converged when ``gap / opening_offer <
  config.NEGOTIATION_CONVERGENCE_PCT``; stalled when the vendor has used
  ``max_rounds`` turns without convergence.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from backend import config, llm
from backend.db.models import (
    ExtractedRequirement,
    Negotiation,
    NegotiationRound,
    Vendor,
    VendorQuote,
)

logger = logging.getLogger(__name__)

# Buyer closes 40% of the current gap on each counter-offer (per contract).
BUYER_COUNTER_STEP: float = 0.4
# Urgency premium: a maximally risky order concedes up to 3% above market.
RISK_CONCESSION_PCT: float = 0.03


def _format_inr(amount: float) -> str:
    """Format an INR amount for narration, e.g. ₹63,500.00."""
    return f"₹{amount:,.2f}"


def _vendor_avg_quote(vendor_id: int, material: str, db_session: Session) -> float | None:
    """Average of the vendor's own quotes, preferring material-family matches."""
    quotes = db_session.query(VendorQuote).filter(VendorQuote.vendor_id == vendor_id).all()
    material_lower = material.lower()
    matching = [
        q
        for q in quotes
        if material_lower in q.material.lower() or q.material.lower() in material_lower
    ]
    pool = matching or quotes
    if not pool:
        return None
    return sum(q.quoted_price for q in pool) / len(pool)


def _template_message(
    actor: str, price: float, negotiation: Negotiation, first_turn: bool
) -> str:
    """Deterministic narration used when no API key is configured (or on error)."""
    price_str = _format_inr(price)
    material = negotiation.material or "the requested material"
    unit = negotiation.unit or "unit"
    if actor == "buyer":
        if first_turn:
            return (
                f"We would like to open at {price_str} per {unit} for {material}, "
                "in line with current market reference pricing."
            )
        return (
            f"We can stretch to {price_str} per {unit}, but that is the limit "
            f"our project budget allows for {material}."
        )
    if first_turn:
        return (
            f"Our asking price for {material} is {price_str} per {unit}, "
            "reflecting current input costs."
        )
    return (
        f"We can come down to {price_str} — that is close to our floor "
        "given current input costs."
    )


def _narrate(
    actor: str,
    price: float,
    negotiation: Negotiation,
    vendor: Vendor | None,
    first_turn: bool,
) -> str:
    """Phrase the already-computed ``price`` as one in-character line.

    The LLM never influences the number — it only words it. Any failure (or no
    configured LLM provider) falls back to the deterministic template.
    """
    fallback = _template_message(actor, price, negotiation, first_turn)
    if not llm.is_configured():
        return fallback
    if actor == "buyer":
        role = "a procurement manager for a construction company (the buyer)"
    else:
        vendor_name = vendor.name if vendor is not None else "a construction materials vendor"
        role = f"a sales representative for {vendor_name} (the vendor)"
    offer_kind = "opening offer" if first_turn else "counter-offer"
    try:
        text = llm.complete(
            (
                f"You are role-playing {role} in a price negotiation for "
                "construction materials. Reply with exactly ONE short, "
                "professional, in-character sentence that states the given "
                "offer price verbatim. Never invent or alter the number."
            ),
            (
                f"State this {offer_kind}: {_format_inr(price)} per "
                f"{negotiation.unit} for {negotiation.quantity:g} "
                f"{negotiation.unit} of {negotiation.material}."
            ),
            max_tokens=120,
            tier="fast",
        )
        text = text.strip()
        # The narration must state the deterministically computed price
        # verbatim; anything else (empty, paraphrased, rounded) is rejected.
        if not text or _format_inr(price) not in text:
            logger.warning(
                "LLM narration omitted or altered the computed price %s; "
                "using template fallback",
                _format_inr(price),
            )
            return fallback
        return text
    except llm.LLMError:
        logger.warning("LLM narration failed; using template fallback", exc_info=True)
        return fallback


def _get_rounds(negotiation_id: int, db_session: Session) -> list[NegotiationRound]:
    return (
        db_session.query(NegotiationRound)
        .filter(NegotiationRound.negotiation_id == negotiation_id)
        .order_by(NegotiationRound.round_number.asc())
        .all()
    )


def _last_price(rounds: list[NegotiationRound], actor: str) -> float | None:
    for rnd in reversed(rounds):
        if rnd.actor == actor:
            return rnd.offered_price
    return None


def start_negotiation(requirement_id: int, vendor_id: int, db_session: Session) -> Negotiation:
    """Create a negotiation and its opening (buyer) round.

    market_ref: :func:`get_market_reference_price`, falling back to the
    vendor's own average quote; ``ValueError`` when neither exists.
    """
    from backend.agent.tools import calculate_risk, get_market_reference_price

    requirement = db_session.get(ExtractedRequirement, requirement_id)
    if requirement is None:
        raise ValueError(f"Requirement {requirement_id} not found")
    vendor = db_session.get(Vendor, vendor_id)
    if vendor is None:
        raise ValueError(f"Vendor {vendor_id} not found")

    market_ref = get_market_reference_price(requirement.material, requirement.grade, db_session)
    if market_ref is None:
        market_ref = _vendor_avg_quote(vendor_id, requirement.material, db_session)
        if market_ref is not None:
            logger.info(
                "No market reference price for %r; falling back to vendor %s average quote %.2f",
                requirement.material,
                vendor_id,
                market_ref,
            )
    if market_ref is None:
        raise ValueError(
            f"No market reference price or vendor quotes available for "
            f"'{requirement.material}' — cannot start a negotiation"
        )

    vendor_asking_price = market_ref * vendor.price_index
    opening_offer = market_ref * (1 - config.NEGOTIATION_OPENING_DISCOUNT)

    try:
        risk = calculate_risk(requirement_id, vendor_id, config.DEFAULT_DEADLINE_DAYS, db_session)
        risk_score = float(risk.get("score", 0))
    except Exception:  # noqa: BLE001 — risk failure must not block negotiating
        logger.warning(
            "calculate_risk failed for requirement %s / vendor %s; assuming score 0",
            requirement_id,
            vendor_id,
            exc_info=True,
        )
        risk_score = 0.0

    # Urgency concedes more: riskier orders get a slightly higher buyer ceiling.
    target_price = market_ref * (1 + RISK_CONCESSION_PCT * risk_score / 100)

    negotiation = Negotiation(
        project_id=requirement.project_id,
        requirement_id=requirement_id,
        vendor_id=vendor_id,
        material=requirement.material,
        grade=requirement.grade,
        quantity=requirement.quantity or 0.0,
        unit=requirement.unit or "tonne",
        opening_offer=opening_offer,
        target_price=target_price,
        vendor_asking_price=vendor_asking_price,
        status="in_progress",
        max_rounds=config.NEGOTIATION_MAX_ROUNDS,
    )
    db_session.add(negotiation)
    db_session.flush()

    message = _narrate("buyer", opening_offer, negotiation, vendor, first_turn=True)
    opening_round = NegotiationRound(
        negotiation_id=negotiation.id,
        round_number=1,
        actor="buyer",
        offered_price=opening_offer,
        message=message,
    )
    db_session.add(opening_round)
    db_session.commit()
    logger.info(
        "Negotiation %s started: opening %.2f, asking %.2f, target %.2f",
        negotiation.id,
        opening_offer,
        vendor_asking_price,
        target_price,
    )
    return negotiation


def run_negotiation_round(negotiation_id: int, db_session: Session) -> NegotiationRound:
    """Play one round (alternating actor) with deterministic pricing.

    After the round is stored, convergence and stall conditions are evaluated
    and the negotiation status updated accordingly.
    """
    negotiation = db_session.get(Negotiation, negotiation_id)
    if negotiation is None:
        raise ValueError(f"Negotiation {negotiation_id} not found")
    if negotiation.status != "in_progress":
        raise ValueError(
            f"Negotiation {negotiation_id} is not in progress "
            f"(status: {negotiation.status})"
        )

    vendor = db_session.get(Vendor, negotiation.vendor_id)
    rounds = _get_rounds(negotiation_id, db_session)
    if not rounds:
        raise ValueError(f"Negotiation {negotiation_id} has no opening round")

    actor = "vendor" if rounds[-1].actor == "buyer" else "buyer"
    last_vendor = _last_price(rounds, "vendor")
    last_buyer = _last_price(rounds, "buyer")
    first_turn = False

    if actor == "vendor":
        if last_vendor is None:
            first_turn = True
            new_price = negotiation.vendor_asking_price
        else:
            gap = abs(last_vendor - (last_buyer if last_buyer is not None else negotiation.opening_offer))
            flexibility = vendor.negotiation_flexibility if vendor is not None else 0.3
            new_price = last_vendor - gap * flexibility
        last_vendor = new_price
    else:
        if last_buyer is None or last_vendor is None:
            # Cannot happen with a buyer-first opening round; defensive only.
            raise ValueError(
                f"Negotiation {negotiation_id} is in an inconsistent state: "
                "buyer counter requested before both sides have offered"
            )
        gap = abs(last_vendor - last_buyer)
        # Never counter above the vendor's current ask (a cheap vendor can open
        # below our offer) nor above the target ceiling.
        new_price = min(
            last_buyer + gap * BUYER_COUNTER_STEP,
            negotiation.target_price,
            last_vendor,
        )
        last_buyer = new_price

    message = _narrate(actor, new_price, negotiation, vendor, first_turn=first_turn)
    new_round = NegotiationRound(
        negotiation_id=negotiation_id,
        round_number=len(rounds) + 1,
        actor=actor,
        offered_price=new_price,
        message=message,
    )
    db_session.add(new_round)
    db_session.flush()

    # Stop conditions, evaluated AFTER the new round is recorded.
    if last_vendor is not None and last_buyer is not None and negotiation.opening_offer:
        gap = abs(last_vendor - last_buyer)
        # Crossed offers (vendor asking <= buyer offer, e.g. a vendor whose
        # price index undercuts the opening discount) are an immediate deal:
        # both sides already agree, so treat it exactly like convergence.
        crossed = last_vendor <= last_buyer
        if crossed or gap / negotiation.opening_offer < config.NEGOTIATION_CONVERGENCE_PCT:
            negotiation.status = "pending_approval"
            negotiation.final_price = round((last_vendor + last_buyer) / 2, 2)
            logger.info(
                "Negotiation %s converged at %.2f (gap %.2f) — pending approval",
                negotiation_id,
                negotiation.final_price,
                gap,
            )
    if negotiation.status == "in_progress":
        vendor_turns = sum(1 for r in rounds if r.actor == "vendor") + (
            1 if actor == "vendor" else 0
        )
        if vendor_turns >= negotiation.max_rounds:
            negotiation.status = "stalled"
            logger.info(
                "Negotiation %s stalled after %s vendor turns without convergence",
                negotiation_id,
                vendor_turns,
            )

    db_session.commit()
    return new_round


def run_full_negotiation(negotiation_id: int, db_session: Session) -> dict:
    """Run rounds until the negotiation leaves ``in_progress``; return the state."""
    negotiation = db_session.get(Negotiation, negotiation_id)
    if negotiation is None:
        raise ValueError(f"Negotiation {negotiation_id} not found")

    max_iterations = 2 * negotiation.max_rounds + 2  # runaway guard
    iterations = 0
    while negotiation.status == "in_progress" and iterations < max_iterations:
        run_negotiation_round(negotiation_id, db_session)
        iterations += 1
    if negotiation.status == "in_progress":
        logger.warning(
            "Negotiation %s hit the %s-iteration guard while still in progress",
            negotiation_id,
            max_iterations,
        )
    return get_negotiation_state(negotiation_id, db_session)


def approve_negotiation(negotiation_id: int, db_session: Session) -> Negotiation:
    """pending_approval -> accepted; anything else is a ValueError."""
    negotiation = db_session.get(Negotiation, negotiation_id)
    if negotiation is None:
        raise ValueError(f"Negotiation {negotiation_id} not found")
    if negotiation.status != "pending_approval":
        raise ValueError(
            f"Only negotiations pending approval can be approved "
            f"(current status: {negotiation.status})"
        )
    negotiation.status = "accepted"
    db_session.commit()
    logger.info("Negotiation %s approved at %.2f", negotiation_id, negotiation.final_price or 0.0)
    return negotiation


def decline_negotiation(negotiation_id: int, db_session: Session) -> Negotiation:
    """Mark the negotiation declined."""
    negotiation = db_session.get(Negotiation, negotiation_id)
    if negotiation is None:
        raise ValueError(f"Negotiation {negotiation_id} not found")
    negotiation.status = "declined"
    db_session.commit()
    logger.info("Negotiation %s declined", negotiation_id)
    return negotiation


def get_negotiation_state(negotiation_id: int, db_session: Session) -> dict:
    """Full state: negotiation dict (+ vendor_name) and every stored round."""
    negotiation = db_session.get(Negotiation, negotiation_id)
    if negotiation is None:
        raise ValueError(f"Negotiation {negotiation_id} not found")
    vendor = db_session.get(Vendor, negotiation.vendor_id)
    negotiation_dict = negotiation.to_dict()
    negotiation_dict["vendor_name"] = vendor.name if vendor is not None else ""
    rounds = _get_rounds(negotiation_id, db_session)
    return {
        "negotiation": negotiation_dict,
        "rounds": [r.to_dict() for r in rounds],
    }
