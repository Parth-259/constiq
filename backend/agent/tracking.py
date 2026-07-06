"""Purchase-order lifecycle tracking for ConstructIQ.

Transitions are forward-only along ``models.PO_STATUS_ORDER``
(draft → sent → accepted → delivered → completed); ``cancelled`` is allowed
from any non-completed state. Every accepted transition appends an immutable
:class:`TrackingEvent`, so the timeline is an honest audit trail.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from backend.db.models import (
    PO_STATUS_ORDER,
    PO_STATUSES,
    PurchaseOrder,
    TrackingEvent,
)

logger = logging.getLogger(__name__)


class InvalidTransitionError(ValueError):
    """Raised when a PO status change violates the forward-only lifecycle."""


def update_tracking_status(
    po_id: int, new_status: str, note: str | None, db_session: Session
) -> dict:
    """Apply a forward-only status transition and append a TrackingEvent.

    Returns the updated purchase order as a dict. Raises
    :class:`InvalidTransitionError` (a ``ValueError``) with a plain-English
    reason for any disallowed transition.
    """
    po = db_session.get(PurchaseOrder, po_id)
    if po is None:
        raise ValueError(f"Purchase order {po_id} not found")

    if new_status not in PO_STATUSES:
        raise InvalidTransitionError(
            f"'{new_status}' is not a recognised purchase order status; "
            f"valid statuses are: {', '.join(PO_STATUSES)}."
        )

    current = po.status
    if new_status == "cancelled":
        if current == "completed":
            raise InvalidTransitionError(
                f"Purchase order {po.po_number} is already completed and can "
                "no longer be cancelled."
            )
        if current == "cancelled":
            raise InvalidTransitionError(
                f"Purchase order {po.po_number} is already cancelled."
            )
    else:
        if current == "cancelled":
            raise InvalidTransitionError(
                f"Purchase order {po.po_number} was cancelled; a cancelled "
                f"order cannot move to '{new_status}'."
            )
        current_index = PO_STATUS_ORDER.index(current) if current in PO_STATUS_ORDER else -1
        new_index = PO_STATUS_ORDER.index(new_status)
        if new_index <= current_index:
            raise InvalidTransitionError(
                f"Cannot move purchase order {po.po_number} from '{current}' "
                f"to '{new_status}': status may only move forward "
                f"({' → '.join(PO_STATUS_ORDER)})."
            )

    po.status = new_status
    event = TrackingEvent(purchase_order_id=po.id, status=new_status, note=note)
    db_session.add(event)
    db_session.commit()
    logger.info(
        "PO %s: %s -> %s%s",
        po.po_number,
        current,
        new_status,
        f" ({note})" if note else "",
    )
    return po.to_dict()


def get_po_timeline(po_id: int, db_session: Session) -> list[dict]:
    """All tracking events for a PO, oldest first."""
    events = (
        db_session.query(TrackingEvent)
        .filter(TrackingEvent.purchase_order_id == po_id)
        .order_by(TrackingEvent.timestamp.asc(), TrackingEvent.id.asc())
        .all()
    )
    return [event.to_dict() for event in events]
