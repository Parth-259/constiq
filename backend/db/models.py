"""SQLAlchemy models for ConstructIQ.

Design rules carried through every model here:
- Requirements are never UPDATEd in place: a change request inserts a new row
  and points the old row's `superseded_by` at it (honest audit trail).
- Negotiation stores every round, not just the final number.
- Tracking events are append-only; PO status transitions are forward-only.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utc_isoformat(value: datetime | None) -> str | None:
    """Serialize a stored datetime with an explicit UTC offset.

    Columns store naive UTC (``datetime.utcnow``); serializing them without a
    timezone suffix makes ``new Date(...)`` in browsers parse them as LOCAL
    time, shifting every displayed timestamp by the viewer's UTC offset.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class Vendor(Base):
    __tablename__ = "vendors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # JSON-encoded list of strings, e.g. ["Fe500D TMT steel", "OPC 53 cement"]
    materials_supplied: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    location: Mapped[str] = mapped_column(String(120), default="")
    region: Mapped[str] = mapped_column(String(120), default="India")
    contact_email: Mapped[str] = mapped_column(String(200), default="")
    contact_phone: Mapped[str] = mapped_column(String(60), default="")
    rating: Mapped[float] = mapped_column(Float, default=4.0)  # 0-5, for directory UI
    avg_delivery_days: Mapped[int] = mapped_column(Integer, default=14)
    historical_on_time_pct: Mapped[float] = mapped_column(Float, default=90.0)  # 0-100
    typical_order_size: Mapped[str] = mapped_column(String(120), default="")  # free text
    typical_min_qty: Mapped[float] = mapped_column(Float, default=0.0)
    typical_max_qty: Mapped[float] = mapped_column(Float, default=1e9)
    price_index: Mapped[float] = mapped_column(Float, default=1.0)  # 1.0 = market avg
    negotiation_flexibility: Mapped[float] = mapped_column(Float, default=0.3)  # 0-1

    def materials_list(self) -> list[str]:
        try:
            return json.loads(self.materials_supplied)
        except (TypeError, ValueError):
            return []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "materials_supplied": self.materials_list(),
            "location": self.location,
            "region": self.region,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "rating": self.rating,
            "avg_delivery_days": self.avg_delivery_days,
            "historical_on_time_pct": self.historical_on_time_pct,
            "typical_order_size": self.typical_order_size,
            "price_index": self.price_index,
            "negotiation_flexibility": self.negotiation_flexibility,
        }


class VendorQuote(Base):
    __tablename__ = "vendor_quotes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(String(80), index=True)
    vendor_id: Mapped[int] = mapped_column(ForeignKey("vendors.id"), index=True)
    material: Mapped[str] = mapped_column(String(200))
    grade: Mapped[str | None] = mapped_column(String(80), nullable=True)
    quoted_price: Mapped[float] = mapped_column(Float)  # per unit, INR
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(40), default="tonne")
    delivery_days: Mapped[int] = mapped_column(Integer, default=14)
    payment_terms: Mapped[str] = mapped_column(String(200), default="Net 30")
    valid_until: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "vendor_id": self.vendor_id,
            "material": self.material,
            "grade": self.grade,
            "quoted_price": self.quoted_price,
            "quantity": self.quantity,
            "unit": self.unit,
            "delivery_days": self.delivery_days,
            "payment_terms": self.payment_terms,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }


class ExtractedRequirement(Base):
    __tablename__ = "extracted_requirements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(String(80), index=True)
    source_file: Mapped[str] = mapped_column(String(300), default="")
    material: Mapped[str] = mapped_column(String(200))
    grade: Mapped[str | None] = mapped_column(String(80), nullable=True)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    deadline: Mapped[str | None] = mapped_column(String(120), nullable=True)  # free text
    certification: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_page: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[str] = mapped_column(String(10), default="medium")  # high|medium|low
    # Never UPDATE a requirement in place — new row + link the old one here.
    superseded_by: Mapped[int | None] = mapped_column(
        ForeignKey("extracted_requirements.id"), nullable=True
    )
    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "source_file": self.source_file,
            "material": self.material,
            "grade": self.grade,
            "quantity": self.quantity,
            "unit": self.unit,
            "deadline": self.deadline,
            "certification": self.certification,
            "source_page": self.source_page,
            "confidence": self.confidence,
            "superseded_by": self.superseded_by,
            "created_date": _utc_isoformat(self.created_date),
        }


class InspectionFinding(Base):
    __tablename__ = "inspection_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(String(80), index=True)
    source_file: Mapped[str] = mapped_column(String(300), default="")
    location: Mapped[str] = mapped_column(String(200), default="")
    defect_description: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(10), default="medium")  # low|medium|high
    source_page: Mapped[int] = mapped_column(Integer, default=0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "source_file": self.source_file,
            "location": self.location,
            "defect_description": self.defect_description,
            "severity": self.severity,
            "source_page": self.source_page,
        }


class IngestedDocument(Base):
    __tablename__ = "ingested_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(String(80), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    source_type: Mapped[str] = mapped_column(String(40), default="tender")
    pages: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    tables_found: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "filename": self.filename,
            "source_type": self.source_type,
            "pages": self.pages,
            "chunks": self.chunks,
            "tables_found": self.tables_found,
            "size_bytes": self.size_bytes,
            "created_date": _utc_isoformat(self.created_date),
        }


NEGOTIATION_STATUSES = ("in_progress", "pending_approval", "accepted", "stalled", "declined")


class Negotiation(Base):
    __tablename__ = "negotiations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(String(80), index=True)
    requirement_id: Mapped[int] = mapped_column(ForeignKey("extracted_requirements.id"))
    vendor_id: Mapped[int] = mapped_column(ForeignKey("vendors.id"))
    material: Mapped[str] = mapped_column(String(200), default="")
    grade: Mapped[str | None] = mapped_column(String(80), nullable=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(40), default="tonne")
    opening_offer: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    vendor_asking_price: Mapped[float] = mapped_column(Float, default=0.0)
    final_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    max_rounds: Mapped[int] = mapped_column(Integer, default=4)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "requirement_id": self.requirement_id,
            "vendor_id": self.vendor_id,
            "material": self.material,
            "grade": self.grade,
            "quantity": self.quantity,
            "unit": self.unit,
            "opening_offer": self.opening_offer,
            "target_price": self.target_price,
            "vendor_asking_price": self.vendor_asking_price,
            "final_price": self.final_price,
            "status": self.status,
            "max_rounds": self.max_rounds,
            "created_date": _utc_isoformat(self.created_date),
        }


class NegotiationRound(Base):
    __tablename__ = "negotiation_rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    negotiation_id: Mapped[int] = mapped_column(ForeignKey("negotiations.id"), index=True)
    round_number: Mapped[int] = mapped_column(Integer)
    actor: Mapped[str] = mapped_column(String(10))  # "buyer" | "vendor"
    offered_price: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(Text, default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "negotiation_id": self.negotiation_id,
            "round_number": self.round_number,
            "actor": self.actor,
            "offered_price": self.offered_price,
            "message": self.message,
            "timestamp": _utc_isoformat(self.timestamp),
        }


PO_STATUSES = ("draft", "sent", "accepted", "delivered", "completed", "cancelled")
# Forward-only transition order (cancelled allowed from any non-completed state).
PO_STATUS_ORDER = ["draft", "sent", "accepted", "delivered", "completed"]


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[str] = mapped_column(String(80), index=True)
    negotiation_id: Mapped[int] = mapped_column(ForeignKey("negotiations.id"))
    po_number: Mapped[str] = mapped_column(String(80), unique=True)
    vendor_id: Mapped[int] = mapped_column(ForeignKey("vendors.id"))
    material: Mapped[str] = mapped_column(String(200))
    grade: Mapped[str | None] = mapped_column(String(80), nullable=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(40), default="tonne")
    unit_price: Mapped[float] = mapped_column(Float)
    total_amount: Mapped[float] = mapped_column(Float)
    delivery_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    payment_terms: Mapped[str] = mapped_column(String(200), default="Net 30")
    status: Mapped[str] = mapped_column(String(20), default="draft")
    pdf_path: Mapped[str] = mapped_column(String(400), default="")
    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "negotiation_id": self.negotiation_id,
            "po_number": self.po_number,
            "vendor_id": self.vendor_id,
            "material": self.material,
            "grade": self.grade,
            "quantity": self.quantity,
            "unit": self.unit,
            "unit_price": self.unit_price,
            "total_amount": self.total_amount,
            "delivery_date": self.delivery_date.isoformat() if self.delivery_date else None,
            "payment_terms": self.payment_terms,
            "status": self.status,
            "pdf_path": self.pdf_path,
            "created_date": _utc_isoformat(self.created_date),
        }


class TrackingEvent(Base):
    __tablename__ = "tracking_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    purchase_order_id: Mapped[int] = mapped_column(ForeignKey("purchase_orders.id"), index=True)
    status: Mapped[str] = mapped_column(String(20))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "purchase_order_id": self.purchase_order_id,
            "status": self.status,
            "timestamp": _utc_isoformat(self.timestamp),
            "note": self.note,
        }
