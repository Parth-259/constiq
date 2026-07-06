"""Purchase-order endpoints — thin wrappers over the PO/tracking modules.

Routes (mounted under /api by main.py):

- GET  /po?project_id=        -> {"purchase_orders": [po dict + vendor_name]}
- GET  /po/{id}/download      -> FileResponse with the generated PDF
- GET  /po/{id}/timeline      -> {"timeline": [tracking events asc]}
- POST /po/{id}/status        -> updated po dict (400 on invalid transition)
- GET  /stats?project_id=     -> dashboard stats

``InvalidTransitionError`` / ``ValueError`` => HTTP 400 with str(e).
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.models import Negotiation, PurchaseOrder, Vendor
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

INTERNAL_ERROR_DETAIL = "Internal error — see server logs"


class StatusUpdateRequest(BaseModel):
    status: str
    note: str | None = None


@router.get("/po")
def list_purchase_orders(project_id: str, db: Session = Depends(get_db)) -> dict:
    """All purchase orders for a project, each enriched with the vendor name."""
    try:
        rows = (
            db.query(PurchaseOrder)
            .filter(PurchaseOrder.project_id == project_id)
            .order_by(PurchaseOrder.id.desc())
            .all()
        )
        vendor_names = {vendor.id: vendor.name for vendor in db.query(Vendor).all()}
        purchase_orders = []
        for row in rows:
            item = row.to_dict()
            item["vendor_name"] = vendor_names.get(
                row.vendor_id, f"Vendor #{row.vendor_id}"
            )
            purchase_orders.append(item)
        return {"purchase_orders": purchase_orders}
    except Exception:
        logger.exception("/po failed for project %s", project_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.get("/po/{po_id}/download")
def download_purchase_order(po_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """Download the generated PO PDF."""
    try:
        po = db.get(PurchaseOrder, po_id)
        if po is None:
            raise HTTPException(
                status_code=404, detail=f"Purchase order {po_id} not found."
            )
        if not po.pdf_path or not os.path.exists(po.pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"PDF for {po.po_number} was not found on disk.",
            )
        return FileResponse(
            po.pdf_path,
            media_type="application/pdf",
            filename=f"{po.po_number}.pdf",
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("/po/%s/download failed", po_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.get("/po/{po_id}/timeline")
def purchase_order_timeline(po_id: int, db: Session = Depends(get_db)) -> dict:
    """Append-only tracking history for one PO, oldest first."""
    try:
        from backend.agent import tracking

        return {"timeline": tracking.get_po_timeline(po_id, db)}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/po/%s/timeline failed", po_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.post("/po/{po_id}/status")
def update_purchase_order_status(
    po_id: int, request: StatusUpdateRequest, db: Session = Depends(get_db)
) -> dict:
    """Advance a PO's status (forward-only; 400 with reason when invalid)."""
    try:
        from backend.agent import tracking

        try:
            tracking.update_tracking_status(po_id, request.status, request.note, db)
        except tracking.InvalidTransitionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        po = db.get(PurchaseOrder, po_id)
        if po is None:
            raise HTTPException(
                status_code=404, detail=f"Purchase order {po_id} not found."
            )
        return po.to_dict()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/po/%s/status failed", po_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.get("/stats")
def project_stats(project_id: str, db: Session = Depends(get_db)) -> dict:
    """Dashboard stats for one project.

    completion_rate = POs completed or delivered / total POs * 100 (0.0 when
    the project has no POs at all).
    """
    try:
        pos = (
            db.query(PurchaseOrder)
            .filter(PurchaseOrder.project_id == project_id)
            .all()
        )
        po_count = len(pos)
        total_po_value = round(sum(float(po.total_amount or 0.0) for po in pos), 2)
        pending_approval = (
            db.query(Negotiation)
            .filter(
                Negotiation.project_id == project_id,
                Negotiation.status == "pending_approval",
            )
            .count()
        )
        in_transit = sum(1 for po in pos if po.status in ("sent", "accepted"))
        completed = sum(1 for po in pos if po.status in ("completed", "delivered"))
        completion_rate = round(completed / po_count * 100.0, 2) if po_count else 0.0
        return {
            "total_po_value": total_po_value,
            "pending_approval": pending_approval,
            "in_transit": in_transit,
            "completion_rate": completion_rate,
            "po_count": po_count,
        }
    except Exception:
        logger.exception("/stats failed for project %s", project_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None
