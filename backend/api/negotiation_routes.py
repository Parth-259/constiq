"""Negotiation endpoints — thin wrappers over backend.agent.negotiation.

Routes (mounted under /api by main.py):

- POST /negotiation/start          {requirement_id, vendor_id} -> state dict
- POST /negotiation/{id}/run       -> run_full_negotiation state
- POST /negotiation/{id}/round     -> single round, then state
- GET  /negotiation/{id}           -> state
- GET  /negotiations?project_id=   -> {"negotiations": [neg dict + vendor_name]}
- POST /negotiation/{id}/approve   -> state + "po" (PO auto-generated)
- POST /negotiation/{id}/decline   -> state

ValueError (unknown id, bad status transition, no price data) => HTTP 400
with the exact message. The agent modules are imported lazily so this router
imports cleanly while they are still being built, and tests can substitute
them.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.models import Negotiation, PurchaseOrder, Vendor
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

INTERNAL_ERROR_DETAIL = "Internal error — see server logs"


class StartNegotiationRequest(BaseModel):
    requirement_id: int
    vendor_id: int


@router.post("/negotiation/start")
def start_negotiation(
    request: StartNegotiationRequest, db: Session = Depends(get_db)
) -> dict:
    """Open a negotiation (round 1 = buyer opening offer) and return its state."""
    try:
        from backend.agent import negotiation

        record = negotiation.start_negotiation(
            request.requirement_id, request.vendor_id, db
        )
        return negotiation.get_negotiation_state(record.id, db)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception(
            "/negotiation/start failed (requirement=%s vendor=%s)",
            request.requirement_id,
            request.vendor_id,
        )
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.post("/negotiation/{negotiation_id}/run")
def run_negotiation(negotiation_id: int, db: Session = Depends(get_db)) -> dict:
    """Run rounds until the negotiation converges, stalls or needs approval."""
    try:
        from backend.agent import negotiation

        return negotiation.run_full_negotiation(negotiation_id, db)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/negotiation/%s/run failed", negotiation_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.post("/negotiation/{negotiation_id}/round")
def run_single_round(negotiation_id: int, db: Session = Depends(get_db)) -> dict:
    """Advance the negotiation by exactly one round, then return the state."""
    try:
        from backend.agent import negotiation

        negotiation.run_negotiation_round(negotiation_id, db)
        return negotiation.get_negotiation_state(negotiation_id, db)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/negotiation/%s/round failed", negotiation_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.get("/negotiation/{negotiation_id}")
def get_negotiation(negotiation_id: int, db: Session = Depends(get_db)) -> dict:
    """Current negotiation state (negotiation + all rounds)."""
    try:
        from backend.agent import negotiation

        return negotiation.get_negotiation_state(negotiation_id, db)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/negotiation/%s failed", negotiation_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.get("/negotiations")
def list_negotiations(project_id: str, db: Session = Depends(get_db)) -> dict:
    """All negotiations for a project, each enriched with the vendor name."""
    try:
        rows = (
            db.query(Negotiation)
            .filter(Negotiation.project_id == project_id)
            .order_by(Negotiation.id.desc())
            .all()
        )
        vendor_names = {vendor.id: vendor.name for vendor in db.query(Vendor).all()}
        negotiations = []
        for row in rows:
            item = row.to_dict()
            item["vendor_name"] = vendor_names.get(
                row.vendor_id, f"Vendor #{row.vendor_id}"
            )
            negotiations.append(item)
        return {"negotiations": negotiations}
    except Exception:
        logger.exception("/negotiations failed for project %s", project_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.post("/negotiation/{negotiation_id}/approve")
def approve_negotiation(negotiation_id: int, db: Session = Depends(get_db)) -> dict:
    """Human approval gate: accept the negotiated price and auto-generate a PO."""
    try:
        from backend.agent import negotiation, purchase_order

        row = db.get(Negotiation, negotiation_id)
        # Reject a missing/zero quantity BEFORE committing "accepted": the PO
        # generation would refuse it anyway, and failing early keeps the
        # negotiation approvable once a quantity is stated.
        if row is not None and not (row.quantity and row.quantity > 0):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Negotiation {negotiation_id} has no stated quantity — "
                    "the source requirement did not state one; set a quantity "
                    "before approving and generating a purchase order"
                ),
            )
        # Recovery path: a previous approve committed status="accepted" but PO
        # generation then failed (PDF/disk/collision). Retrying must generate
        # the missing PO rather than 400 on the already-accepted status.
        already_accepted_without_po = (
            row is not None
            and row.status == "accepted"
            and db.query(PurchaseOrder)
            .filter(PurchaseOrder.negotiation_id == negotiation_id)
            .first()
            is None
        )
        if already_accepted_without_po:
            logger.warning(
                "Negotiation %s is accepted but has no purchase order — "
                "recovering by generating the PO now",
                negotiation_id,
            )
        else:
            negotiation.approve_negotiation(negotiation_id, db)
        po = purchase_order.generate_po(negotiation_id, db)
        state = negotiation.get_negotiation_state(negotiation_id, db)
        state["po"] = po.to_dict()
        return state
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/negotiation/%s/approve failed", negotiation_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.post("/negotiation/{negotiation_id}/decline")
def decline_negotiation(negotiation_id: int, db: Session = Depends(get_db)) -> dict:
    """Decline the negotiation and return its final state."""
    try:
        from backend.agent import negotiation

        negotiation.decline_negotiation(negotiation_id, db)
        return negotiation.get_negotiation_state(negotiation_id, db)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/negotiation/%s/decline failed", negotiation_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None
