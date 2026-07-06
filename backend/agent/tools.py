"""Agent tool functions for ConstructIQ procurement.

Pure functions over the vendor/requirement database. They are called both by
the LLM agent loop (as tool implementations) and directly by the API routers,
so they must stay deterministic and degrade gracefully when external API keys
(Tavily) are missing.
"""
from __future__ import annotations

import logging
import re
from statistics import fmean
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import config
from backend.db.models import ExtractedRequirement, Vendor, VendorQuote

logger = logging.getLogger(__name__)

RISK_LABEL_LOW = "LOW"
RISK_LABEL_MEDIUM = "MEDIUM"
RISK_LABEL_HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Matching helpers (case-insensitive material-family and grade matching)
# ---------------------------------------------------------------------------

def _norm_text(value: str | None) -> str:
    """Lowercase and collapse whitespace for case-insensitive comparison."""
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _norm_grade(value: str | None) -> str:
    """Normalize a grade designation: 'Fe 500-D' -> 'fe500d'."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _material_matches(requirement_material: str, vendor_entry: str) -> bool:
    """True when either normalized material string contains the other."""
    a = _norm_text(requirement_material)
    b = _norm_text(vendor_entry)
    if not a or not b:
        return False
    return a in b or b in a


def _entry_grade_tokens(entry: str) -> set[str]:
    """Grade-like tokens of an entry: words plus adjacent-word merges.

    'OPC 53 cement' -> {'opc', '53', 'cement', 'opc53', '53cement'} so that
    both 'OPC 53' and 'OPC53' style grade strings can be matched exactly
    without letting 'Fe500' match 'Fe500D'.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", entry.lower()) if t]
    grams: set[str] = set(tokens)
    for i in range(len(tokens) - 1):
        grams.add(tokens[i] + tokens[i + 1])
    return grams


def _entry_has_grade(entry: str, grade: str) -> bool:
    """True when the normalized grade appears as an exact token in the entry."""
    normalized = _norm_grade(grade)
    return bool(normalized) and normalized in _entry_grade_tokens(entry)


def _vendor_supplies(vendor: Vendor, material: str) -> bool:
    """True when any of the vendor's materials matches the material family."""
    return any(_material_matches(material, entry) for entry in vendor.materials_list())


def _vendor_has_exact_grade(vendor: Vendor, material: str, grade: str) -> bool:
    """True when one vendor entry matches the material family AND the grade."""
    return any(
        _material_matches(material, entry) and _entry_has_grade(entry, grade)
        for entry in vendor.materials_list()
    )


def _alternate_grade_label(vendor: Vendor, material: str) -> str | None:
    """Best-effort grade descriptor left after removing the material from a
    matching vendor entry, e.g. entry 'Fe500 TMT steel' minus 'TMT steel'
    -> 'Fe500'; None when nothing grade-like remains."""
    target = _norm_text(material)
    for entry in vendor.materials_list():
        lowered = _norm_text(entry)
        idx = lowered.find(target)
        if idx == -1:
            continue
        remainder = (entry[:idx] + entry[idx + len(target):]).strip(" ,-/()")
        if remainder:
            return remainder
    return None


def _grading_standard(material: str) -> str:
    """Name the Indian Standard governing grades for the material family."""
    lowered = _norm_text(material)
    if "tmt" in lowered or "steel" in lowered or "rebar" in lowered or lowered.startswith("fe"):
        return "IS 1786"
    if "cement" in lowered or "opc" in lowered or "ppc" in lowered:
        return "IS 269"
    if "concrete" in lowered or "rmc" in lowered:
        return "IS 456"
    return "the applicable IS standard"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _order_size_factor(
    quantity: float | None, min_qty: float, max_qty: float
) -> float:
    """0 when quantity is unknown or inside [min, max]; otherwise the relative
    distance outside the range, capped at 1."""
    if quantity is None:
        return 0.0
    if min_qty <= quantity <= max_qty:
        return 0.0
    if quantity < min_qty:
        if min_qty <= 0:
            return 0.0
        return min(1.0, (min_qty - quantity) / min_qty)
    if max_qty <= 0:
        return 1.0
    return min(1.0, (quantity - max_qty) / max_qty)


def _get_requirement(requirement_id: int, db_session: Session) -> ExtractedRequirement:
    requirement = db_session.get(ExtractedRequirement, requirement_id)
    if requirement is None:
        raise ValueError(f"Requirement {requirement_id} not found")
    return requirement


def _get_vendor(vendor_id: int, db_session: Session) -> Vendor:
    vendor = db_session.get(Vendor, vendor_id)
    if vendor is None:
        raise ValueError(f"Vendor {vendor_id} not found")
    return vendor


def _vendor_quotes_for_material(
    vendor_id: int, material: str, grade: str | None, db_session: Session
) -> list[VendorQuote]:
    """Vendor's quotes matching the material family, preferring exact-grade quotes."""
    quotes = [
        q
        for q in db_session.scalars(
            select(VendorQuote).where(VendorQuote.vendor_id == vendor_id)
        )
        if _material_matches(material, q.material)
    ]
    if grade:
        graded = [
            q
            for q in quotes
            if (q.grade and _norm_grade(q.grade) == _norm_grade(grade))
            or _entry_has_grade(q.material, grade)
        ]
        if graded:
            return graded
    return quotes


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def vendor_lookup(material: str, grade: str | None, db_session: Session) -> list[dict]:
    """Find vendors in the internal directory that supply a given construction material, listing exact-grade suppliers first."""
    vendors = [
        v for v in db_session.scalars(select(Vendor)) if _vendor_supplies(v, material)
    ]
    if grade:
        vendors.sort(
            key=lambda v: (0 if _vendor_has_exact_grade(v, material, grade) else 1, v.name)
        )
    else:
        vendors.sort(key=lambda v: v.name)
    logger.info(
        "vendor_lookup material=%r grade=%r -> %d match(es)", material, grade, len(vendors)
    )
    return [v.to_dict() for v in vendors]


def get_market_reference_price(
    material: str, grade: str | None, db_session: Session
) -> float | None:
    """Compute the market reference price (INR per unit) for a material as the average of matching vendor quotes, or None when no quotes exist."""
    quotes = [
        q
        for q in db_session.scalars(select(VendorQuote))
        if _material_matches(material, q.material)
    ]
    if grade:
        graded = [
            q
            for q in quotes
            if (q.grade and _norm_grade(q.grade) == _norm_grade(grade))
            or _entry_has_grade(q.material, grade)
        ]
        if graded:
            quotes = graded
    if not quotes:
        logger.info("No quotes found for material=%r grade=%r", material, grade)
        return None
    price = float(fmean(q.quoted_price for q in quotes))
    logger.info(
        "Market reference for %r (grade=%r) = %.2f from %d quote(s)",
        material,
        grade,
        price,
        len(quotes),
    )
    return price


def check_compliance(
    requirement_id: int, db_session: Session, vendor_id: int | None = None
) -> dict:
    """Check whether vendors can supply a requirement's material in the exact required grade, flagging non-interchangeable alternate grades."""
    requirement = _get_requirement(requirement_id, db_session)
    if vendor_id is not None:
        candidates = [_get_vendor(vendor_id, db_session)]
    else:
        candidates = list(db_session.scalars(select(Vendor)))

    family = [v for v in candidates if _vendor_supplies(v, requirement.material)]
    result: dict[str, Any] = {"requirement": requirement.to_dict()}

    if not family:
        result["status"] = "no_vendor_found"
        result["matching_vendors"] = []
        scope = "The selected vendor does not supply" if vendor_id is not None else "No vendor in the directory supplies"
        result["explanation"] = (
            f"{scope} {requirement.material}"
            + (f" (grade {requirement.grade})" if requirement.grade else "")
            + "."
        )
        logger.info("check_compliance req=%d -> no_vendor_found", requirement_id)
        return result

    if not requirement.grade:
        result["status"] = "compliant"
        result["matching_vendors"] = [v.to_dict() for v in family]
        names = ", ".join(v.name for v in family)
        result["explanation"] = (
            f"The requirement specifies no grade, and {len(family)} vendor(s) "
            f"supply {requirement.material}: {names}."
        )
        logger.info("check_compliance req=%d -> compliant (no grade)", requirement_id)
        return result

    exact = [
        v for v in family if _vendor_has_exact_grade(v, requirement.material, requirement.grade)
    ]
    if exact:
        result["status"] = "compliant"
        result["matching_vendors"] = [v.to_dict() for v in exact]
        names = ", ".join(v.name for v in exact)
        result["explanation"] = (
            f"{len(exact)} vendor(s) supply {requirement.material} in the exact "
            f"required grade {requirement.grade}: {names}."
        )
        logger.info("check_compliance req=%d -> compliant", requirement_id)
        return result

    result["status"] = "non_compliant_alternate_available"
    result["matching_vendors"] = [v.to_dict() for v in family]
    standard = _grading_standard(requirement.material)
    alt_grade = next(
        (
            label
            for label in (
                _alternate_grade_label(v, requirement.material) for v in family
            )
            if label
        ),
        None,
    )
    if alt_grade:
        result["explanation"] = (
            f"No vendor offers {requirement.material} in the required grade "
            f"{requirement.grade}; the closest available alternative is {alt_grade}. "
            f"{alt_grade} is not interchangeable with {requirement.grade} per "
            f"{standard} grading."
        )
    else:
        names = ", ".join(v.name for v in family)
        result["explanation"] = (
            f"No vendor offers {requirement.material} in the required grade "
            f"{requirement.grade}; {names} supply the same material family without "
            f"a stated grade, and an ungraded product is not interchangeable with "
            f"{requirement.grade} per {standard} grading."
        )
    logger.info(
        "check_compliance req=%d -> non_compliant_alternate_available", requirement_id
    )
    return result


def calculate_risk(
    requirement_id: int,
    vendor_id: int,
    deadline_days_remaining: int,
    db_session: Session,
) -> dict:
    """Score the delivery risk (0-100 with LOW/MEDIUM/HIGH label) of sourcing a requirement from a vendor given the days remaining to deadline."""
    requirement = _get_requirement(requirement_id, db_session)
    vendor = _get_vendor(vendor_id, db_session)

    days = max(int(deadline_days_remaining), 1)
    lead_time_pressure = min(vendor.avg_delivery_days / days, 2.0)
    reliability_factor = (100.0 - vendor.historical_on_time_pct) / 100.0
    order_size_factor = _order_size_factor(
        requirement.quantity, vendor.typical_min_qty, vendor.typical_max_qty
    )

    raw_score = (
        lead_time_pressure * config.RISK_WEIGHT_LEAD_TIME
        + reliability_factor * config.RISK_WEIGHT_RELIABILITY
        + order_size_factor * config.RISK_WEIGHT_ORDER_SIZE
    )
    score = int(round(_clamp(raw_score, 0.0, 100.0)))
    if score <= 33:
        label = RISK_LABEL_LOW
    elif score <= 66:
        label = RISK_LABEL_MEDIUM
    else:
        label = RISK_LABEL_HIGH

    if requirement.quantity is None:
        qty_phrase = "the order quantity is unspecified"
    elif order_size_factor == 0:
        qty_phrase = (
            f"the order quantity of {requirement.quantity:g} {requirement.unit or 'units'} "
            f"is within the vendor's typical range"
        )
    else:
        qty_phrase = (
            f"the order quantity of {requirement.quantity:g} {requirement.unit or 'units'} "
            f"falls outside the vendor's typical range of "
            f"{vendor.typical_min_qty:g}-{vendor.typical_max_qty:g}"
        )

    explanation = (
        f"{vendor.name} typically delivers in {vendor.avg_delivery_days} days against "
        f"{deadline_days_remaining} days remaining (lead-time pressure "
        f"{lead_time_pressure:.2f}), has a {vendor.historical_on_time_pct:.0f}% "
        f"historical on-time rate (reliability factor {reliability_factor:.2f}), and "
        f"{qty_phrase} (order-size factor {order_size_factor:.2f}); overall risk "
        f"{score}/100 = {label}."
    )
    logger.info(
        "calculate_risk req=%d vendor=%d days=%d -> %d (%s)",
        requirement_id,
        vendor_id,
        deadline_days_remaining,
        score,
        label,
    )
    return {
        "score": score,
        "label": label,
        "factors": {
            "lead_time_pressure": round(lead_time_pressure, 4),
            "reliability_factor": round(reliability_factor, 4),
            "order_size_factor": round(order_size_factor, 4),
            "avg_delivery_days": vendor.avg_delivery_days,
            "deadline_days_remaining": deadline_days_remaining,
            "historical_on_time_pct": vendor.historical_on_time_pct,
            "quantity": requirement.quantity,
        },
        "explanation": explanation,
    }


def vendor_discovery(
    material: str, grade: str | None, location_hint: str, db_session: Session
) -> dict:
    """Discover suppliers for a material: verified matches from the internal directory plus unverified web results from a single Tavily search."""
    internal_matches: list[dict] = []
    for vendor_dict in vendor_lookup(material, grade, db_session):
        entry = dict(vendor_dict)
        entry["verified"] = True
        internal_matches.append(entry)

    web_matches: list[dict] = []
    web_search_succeeded = False
    query = f"{material} {grade or ''} supplier {location_hint} India"

    if not config.TAVILY_API_KEY:
        logger.info("TAVILY_API_KEY not set; skipping web discovery")
    else:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=config.TAVILY_API_KEY)
            response = client.search(query, max_results=5)
            results = response.get("results", []) if isinstance(response, dict) else []
            for item in results[:5]:
                web_matches.append(
                    {
                        "name": item.get("title", ""),
                        "snippet": item.get("content", ""),
                        "source_url": item.get("url", ""),
                        "verified": False,
                    }
                )
            web_search_succeeded = True
            logger.info("Tavily search %r -> %d web match(es)", query, len(web_matches))
        except Exception:  # noqa: BLE001 - degrade gracefully on any Tavily failure
            logger.warning("Tavily web discovery failed for query %r", query, exc_info=True)
            web_matches = []
            web_search_succeeded = False

    return {
        "internal_matches": internal_matches,
        "web_matches": web_matches,
        "web_search_succeeded": web_search_succeeded,
    }


def vendor_evaluation(
    candidates: list[dict],
    material: str,
    grade: str | None,
    quantity: float | None,
    db_session: Session,
) -> list[dict]:
    """Score verified vendor candidates on reliability, price competitiveness, and capacity fit, returning them sorted by weighted evaluation score."""
    market_ref = get_market_reference_price(material, grade, db_session)

    scored: list[dict] = []
    unverified: list[dict] = []
    for candidate in candidates:
        entry = dict(candidate)
        is_verified = bool(entry.get("verified", entry.get("id") is not None))
        if not is_verified:
            entry["evaluation_score"] = None
            entry["note"] = (
                "Unverified web result — not scored; verify credentials before engaging."
            )
            unverified.append(entry)
            continue

        vendor = (
            db_session.get(Vendor, entry["id"]) if entry.get("id") is not None else None
        )
        on_time = (
            vendor.historical_on_time_pct
            if vendor is not None
            else float(entry.get("historical_on_time_pct", 0.0))
        )
        reliability_score = _clamp(on_time / 100.0, 0.0, 1.0)

        notes: list[str] = []
        vendor_avg_quote: float | None = None
        if vendor is not None:
            quotes = _vendor_quotes_for_material(vendor.id, material, grade, db_session)
            if quotes:
                vendor_avg_quote = float(fmean(q.quoted_price for q in quotes))

        if market_ref is None:
            price_score = 0.5
            notes.append(
                f"No market price data available for {material}; price score "
                f"defaulted to 0.5."
            )
        else:
            if vendor_avg_quote is None:
                price_index = (
                    vendor.price_index
                    if vendor is not None
                    else float(entry.get("price_index", 1.0))
                )
                vendor_avg_quote = market_ref * price_index
            price_score = _clamp(
                1.0 - abs(vendor_avg_quote - market_ref) / market_ref, 0.0, 1.0
            )

        if vendor is not None:
            capacity_score = 1.0 - _order_size_factor(
                quantity, vendor.typical_min_qty, vendor.typical_max_qty
            )
        else:
            capacity_score = 1.0

        evaluation_score = (
            config.EVAL_WEIGHT_RELIABILITY * reliability_score
            + config.EVAL_WEIGHT_PRICE * price_score
            + config.EVAL_WEIGHT_CAPACITY * capacity_score
        )

        entry["reliability_score"] = round(reliability_score, 4)
        entry["price_score"] = round(price_score, 4)
        entry["capacity_score"] = round(capacity_score, 4)
        entry["evaluation_score"] = round(evaluation_score, 4)
        entry["summary"] = (
            f"{entry.get('name', 'Vendor')} scores {evaluation_score:.2f} overall "
            f"(reliability {reliability_score:.2f}, price {price_score:.2f}, "
            f"capacity {capacity_score:.2f}) for {material}"
            + (f" grade {grade}" if grade else "")
            + "."
        )
        if notes:
            entry["note"] = " ".join(notes)
        scored.append(entry)

    scored.sort(key=lambda e: e["evaluation_score"], reverse=True)
    logger.info(
        "vendor_evaluation material=%r grade=%r scored=%d unverified=%d",
        material,
        grade,
        len(scored),
        len(unverified),
    )
    return scored + unverified


def recommend_vendor(
    requirement_id: int, deadline_days_remaining: int, db_session: Session
) -> dict:
    """Recommend the best vendor for a requirement by combining discovery, evaluation, grade compliance, and delivery risk, with up to two ranked alternatives."""
    requirement = db_session.get(ExtractedRequirement, requirement_id)
    if requirement is None:
        return {
            "recommended_vendor": None,
            "reason": f"no_recommendation_possible: requirement {requirement_id} not found",
        }

    discovery = vendor_discovery(requirement.material, requirement.grade, "", db_session)
    internal = discovery["internal_matches"]
    if not internal:
        return {
            "recommended_vendor": None,
            "reason": (
                f"no_recommendation_possible: no vendor in the directory supplies "
                f"{requirement.material}"
            ),
        }

    evaluated = vendor_evaluation(
        internal, requirement.material, requirement.grade, requirement.quantity, db_session
    )

    ranked: list[dict] = []
    for candidate in evaluated:
        if candidate.get("evaluation_score") is None or candidate.get("id") is None:
            continue  # unverified or unscorable candidates cannot be recommended
        compliance = check_compliance(
            requirement_id, db_session, vendor_id=candidate["id"]
        )
        status = compliance["status"]
        if status == "no_vendor_found":
            continue  # skip risk call entirely for no-match candidates
        risk = calculate_risk(
            requirement_id, candidate["id"], deadline_days_remaining, db_session
        )
        ranked.append(
            {
                "candidate": candidate,
                "compliance_status": status,
                "compliance_explanation": compliance["explanation"],
                "risk": risk,
            }
        )

    if not ranked:
        return {
            "recommended_vendor": None,
            "reason": (
                f"no_recommendation_possible: no candidate vendor can supply "
                f"{requirement.material}"
                + (f" (grade {requirement.grade})" if requirement.grade else "")
            ),
        }

    ranked.sort(
        key=lambda r: (
            0 if r["compliance_status"] == "compliant" else 1,
            -(r["candidate"]["evaluation_score"] or 0.0),
            r["risk"]["score"],
        )
    )
    top = ranked[0]
    top_candidate = top["candidate"]
    compliance_phrase = (
        "meets the exact grade requirement"
        if top["compliance_status"] == "compliant"
        else "offers the closest available alternate grade"
    )
    overall_reason = (
        f"{top_candidate.get('name', 'Vendor')} {compliance_phrase} for "
        f"{requirement.material}"
        + (f" ({requirement.grade})" if requirement.grade else "")
        + f", ranks best on evaluation ({top_candidate['evaluation_score']:.2f}) among "
        f"{len(ranked)} candidate(s), and carries {top['risk']['label']} delivery risk "
        f"({top['risk']['score']}/100)."
    )
    if top["compliance_status"] != "compliant":
        overall_reason += f" Note: {top['compliance_explanation']}"

    alternatives = [
        {
            "vendor": r["candidate"],
            "compliance_status": r["compliance_status"],
            "evaluation_score": r["candidate"]["evaluation_score"],
            "risk_score": r["risk"]["score"],
            "risk_label": r["risk"]["label"],
        }
        for r in ranked[1:3]
    ]

    logger.info(
        "recommend_vendor req=%d -> vendor=%s (%s)",
        requirement_id,
        top_candidate.get("id"),
        top["compliance_status"],
    )
    return {
        "recommended_vendor": top_candidate,
        "compliance_status": top["compliance_status"],
        "evaluation_summary": top_candidate.get("summary", ""),
        "risk_score": top["risk"]["score"],
        "risk_label": top["risk"]["label"],
        "risk_explanation": top["risk"]["explanation"],
        "overall_reason": overall_reason,
        "alternatives": alternatives,
    }
