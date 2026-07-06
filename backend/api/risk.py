"""GET /api/risk/{project_id} — per-requirement risk cards for the dashboard.

For each current (non-superseded) requirement:

1. candidate vendors via ``backend.agent.tools.vendor_lookup``;
2. best vendor picked recommend-lite: compliance-first (exact-grade token
   match, same rules as ``tools.check_compliance``), then by highest
   ``tools.vendor_evaluation`` score (0.4/0.4/0.2 weights);
3. risk via ``backend.agent.tools.calculate_risk`` — the days-remaining input
   is parsed from the requirement deadline when it is an ISO ``YYYY-MM-DD``
   date (floored at 1 day), otherwise ``config.DEFAULT_DEADLINE_DAYS``;
4. est_value = quantity x market reference price (when both known);
   est_delivery = today + vendor.avg_delivery_days.

Requirements with no matching vendor get a "NO_VENDOR" card with score 0.
Cards are sorted by score descending.
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import config
from backend.db.models import ExtractedRequirement
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

INTERNAL_ERROR_DETAIL = "Internal error — see server logs"

_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _deadline_days_remaining(deadline: str | None) -> int:
    """Days until an ISO YYYY-MM-DD deadline (floor 1); default otherwise."""
    if deadline and _ISO_DATE_PATTERN.match(deadline.strip()):
        try:
            deadline_date = date.fromisoformat(deadline.strip())
        except ValueError:
            return config.DEFAULT_DEADLINE_DAYS
        return max(1, (deadline_date - date.today()).days)
    return config.DEFAULT_DEADLINE_DAYS


def _is_compliant(vendor: dict, material: str, grade: str | None) -> bool:
    """True when a vendor entry matches the material family AND the exact grade.

    Uses the same exact-token grade matching as ``tools.check_compliance`` so
    e.g. 'Fe500' never matches a 'Fe500D TMT steel' entry.
    """
    if not grade:
        return True
    from backend.agent import tools

    return any(
        tools._material_matches(material, str(entry))
        and tools._entry_has_grade(str(entry), grade)
        for entry in vendor.get("materials_supplied", []) or []
    )


def _pick_best(candidates: list[dict], requirement: ExtractedRequirement, db: Session) -> dict:
    """Recommend-lite: compliance-first, then best 0.4/0.4/0.2 evaluation score."""
    from backend.agent import tools

    evaluated = tools.vendor_evaluation(
        candidates, requirement.material, requirement.grade, requirement.quantity, db
    )
    scored = [v for v in evaluated if v.get("evaluation_score") is not None]
    pool = scored or evaluated or candidates
    compliant = [
        v for v in pool if _is_compliant(v, requirement.material, requirement.grade)
    ]
    bucket = compliant or pool
    return max(bucket, key=lambda v: float(v.get("evaluation_score") or 0.0))


@router.get("/risk/{project_id}")
def project_risk(project_id: str, db: Session = Depends(get_db)) -> dict:
    """Risk cards for every current requirement of the project."""
    try:
        from backend.agent import tools

        requirements = (
            db.query(ExtractedRequirement)
            .filter(
                ExtractedRequirement.project_id == project_id,
                ExtractedRequirement.superseded_by.is_(None),
            )
            .order_by(ExtractedRequirement.id)
            .all()
        )

        cards: list[dict] = []
        for requirement in requirements:
            candidates = tools.vendor_lookup(requirement.material, requirement.grade, db)
            market_ref = tools.get_market_reference_price(
                requirement.material, requirement.grade, db
            )
            est_value: float | None = None
            if requirement.quantity and market_ref:
                est_value = round(float(requirement.quantity) * float(market_ref), 2)

            if not candidates:
                grade_suffix = f" ({requirement.grade})" if requirement.grade else ""
                cards.append(
                    {
                        "requirement": requirement.to_dict(),
                        "vendor": None,
                        "score": 0,
                        "label": "NO_VENDOR",
                        "factors": {},
                        "explanation": (
                            f"No vendor found supplying {requirement.material}"
                            f"{grade_suffix} — procurement is blocked until one "
                            "is sourced."
                        ),
                        "est_value": est_value,
                        "est_delivery": None,
                    }
                )
                continue

            vendor = _pick_best(candidates, requirement, db)
            days_remaining = _deadline_days_remaining(requirement.deadline)
            risk = tools.calculate_risk(
                requirement.id, int(vendor["id"]), days_remaining, db
            )
            est_delivery = (
                date.today() + timedelta(days=int(vendor.get("avg_delivery_days", 0) or 0))
            ).isoformat()
            cards.append(
                {
                    "requirement": requirement.to_dict(),
                    "vendor": vendor,
                    "score": int(risk.get("score", 0)),
                    "label": str(risk.get("label", "LOW")),
                    "factors": risk.get("factors", {}),
                    "explanation": str(risk.get("explanation", "")),
                    "est_value": est_value,
                    "est_delivery": est_delivery,
                }
            )

        cards.sort(key=lambda card: card["score"], reverse=True)
        scored = [card for card in cards if card["label"] != "NO_VENDOR"]
        total_risk_score = (
            int(round(sum(card["score"] for card in scored) / len(scored)))
            if scored
            else 0
        )
        return {
            "project_id": project_id,
            "cards": cards,
            "total_risk_score": total_risk_score,
            "active_mitigations": sum(1 for card in cards if card["label"] == "HIGH"),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("/risk failed for project %s", project_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None
