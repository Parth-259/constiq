"""POST /api/ingest — upload a construction document into the pipeline.

Validates the upload (extension whitelist, PDF magic bytes), stages the bytes
in a NamedTemporaryFile that preserves the suffix, hands the path to
``backend.pipeline.multi_source_loader.load_document`` and always cleans the
temp file up in a ``finally`` block.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.pipeline import multi_source_loader

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
PDF_MAGIC = b"%PDF"
INTERNAL_ERROR_DETAIL = "Internal error — see server logs"


def _safe_stem(filename: str) -> str:
    """Filesystem-safe version of the uploaded file's stem (for temp naming)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:60] or "upload"


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    source_type: str = Form("tender"),
    db: Session = Depends(get_db),
) -> dict:
    """Ingest one uploaded document and return the load_document summary."""
    try:
        filename = file.filename or ""
        extension = os.path.splitext(filename)[1].lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file extension {extension or '(none)'!r} — "
                    "allowed extensions are: pdf, docx, txt."
                ),
            )
        if not project_id.strip():
            raise HTTPException(status_code=400, detail="project_id must not be empty.")
        if source_type not in multi_source_loader.SOURCE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown source_type {source_type!r} — expected one of "
                    f"{list(multi_source_loader.SOURCE_TYPES)}."
                ),
            )

        content = await file.read()
        if extension == "pdf" and not content.startswith(PDF_MAGIC):
            raise HTTPException(
                status_code=400,
                detail=(
                    "File has a .pdf extension but does not look like a PDF "
                    "(missing %PDF magic bytes)."
                ),
            )

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{_safe_stem(filename)}__",
                suffix=f".{extension}",
                delete=False,
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            result = multi_source_loader.load_document(
                tmp_path,
                project_id.strip(),
                source_type,
                db,
                original_filename=filename,
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.warning("Could not remove temp file %s", tmp_path)
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        # Known-bad input surfaced by the pipeline (bad source_type, bad file).
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Ingestion failed for %r", file.filename)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None
