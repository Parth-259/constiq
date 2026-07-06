"""Vendor directory, requirements/documents listings, discovery + recommend.

Routes (mounted under /api by main.py):

- GET  /vendors                -> {"vendors": [Vendor.to_dict()...]}
- GET  /requirements?project_id= -> current (non-superseded) requirements
- GET  /documents?project_id=  -> ingested document records
- POST /discovery              -> backend.agent.tools.vendor_discovery dict
- POST /recommend              -> backend.agent.tools.recommend_vendor dict
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.models import ExtractedRequirement, IngestedDocument, Vendor
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

INTERNAL_ERROR_DETAIL = "Internal error — see server logs"


class DiscoveryRequest(BaseModel):
    material: str
    grade: str | None = None
    location_hint: str = ""


class RecommendRequest(BaseModel):
    requirement_id: int
    deadline_days_remaining: int = 30


@router.get("/vendors")
def list_vendors(db: Session = Depends(get_db)) -> dict:
    """All vendors in the directory."""
    try:
        vendors = db.query(Vendor).order_by(Vendor.name).all()
        return {"vendors": [vendor.to_dict() for vendor in vendors]}
    except Exception:
        logger.exception("/vendors failed")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.get("/requirements")
def list_requirements(project_id: str, db: Session = Depends(get_db)) -> dict:
    """Current (non-superseded) extracted requirements for a project."""
    try:
        rows = (
            db.query(ExtractedRequirement)
            .filter(
                ExtractedRequirement.project_id == project_id,
                ExtractedRequirement.superseded_by.is_(None),
            )
            .order_by(ExtractedRequirement.id)
            .all()
        )
        return {"requirements": [row.to_dict() for row in rows]}
    except Exception:
        logger.exception("/requirements failed for project %s", project_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.get("/documents")
def list_documents(project_id: str, db: Session = Depends(get_db)) -> dict:
    """Ingested-document records for a project (most recent first)."""
    try:
        rows = (
            db.query(IngestedDocument)
            .filter(IngestedDocument.project_id == project_id)
            .order_by(IngestedDocument.id.desc())
            .all()
        )
        return {"documents": [row.to_dict() for row in rows]}
    except Exception:
        logger.exception("/documents failed for project %s", project_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.post("/discovery")
def discovery(request: DiscoveryRequest, db: Session = Depends(get_db)) -> dict:
    """Internal + web vendor discovery for a material/grade."""
    try:
        material = request.material.strip()
        if not material:
            raise HTTPException(status_code=400, detail="material must not be empty.")

        from backend.agent import tools

        return tools.vendor_discovery(
            material, request.grade, request.location_hint or "", db
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/discovery failed for material %r", request.material)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None


@router.post("/recommend")
def recommend(request: RecommendRequest, db: Session = Depends(get_db)) -> dict:
    """End-to-end vendor recommendation for one requirement."""
    try:
        from backend.agent import tools

        return tools.recommend_vendor(
            request.requirement_id, request.deadline_days_remaining, db
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("/recommend failed for requirement %s", request.requirement_id)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None
