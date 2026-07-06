"""LLM-backed extraction of procurement requirements from document chunks.

Design rules:
- One :func:`backend.llm.extract_json` call (forced structured output) per
  batch of up to :data:`BATCH_SIZE` text chunks — guaranteed JSON, no free text.
- The system prompt forbids inference: anything not explicitly stated in the
  text must come back as ``null``.
- Every item is validated through :class:`RequirementModel`; invalid items are
  logged and skipped, never raised.
- Requirements are never UPDATEd in place. A change request inserts new rows
  and points the superseded rows' ``superseded_by`` at them (audit trail).
- With no LLM provider configured (no ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY``)
  the functions degrade gracefully: they log a warning and return ``[]``.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import pydantic
from sqlalchemy.orm import Session

from backend import config, llm  # noqa: F401 — config kept for test monkeypatching
from backend.db.models import ExtractedRequirement

logger = logging.getLogger(__name__)

# Max text chunks sent to the model in a single messages.create call.
BATCH_SIZE = 6

TOOL_NAME = "record_requirements"

REQUIREMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "material": {
                        "type": "string",
                        "description": "Material or item being procured.",
                    },
                    "grade": {
                        "type": ["string", "null"],
                        "description": "Grade/spec (e.g. Fe500D, M40) or null.",
                    },
                    "quantity": {
                        "type": ["number", "null"],
                        "description": "Numeric quantity or null if not stated.",
                    },
                    "unit": {
                        "type": ["string", "null"],
                        "description": "Unit for the quantity or null.",
                    },
                    "deadline": {
                        "type": ["string", "null"],
                        "description": "Deadline exactly as stated, or null.",
                    },
                    "certification": {
                        "type": ["string", "null"],
                        "description": "Required certification/standard or null.",
                    },
                    "source_page": {
                        "type": "integer",
                        "description": "Page number the requirement came from.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": ["material", "source_page", "confidence"],
            },
        }
    },
    "required": ["requirements"],
}

SYSTEM_PROMPT = (
    "You are a meticulous construction-procurement analyst. You are given "
    "excerpts of a construction document, each tagged with its page number. "
    "Extract ONLY requirements that are explicitly stated in the text. "
    "Never infer, guess, or fill in a value that is not stated — use null "
    "for any field the text does not explicitly provide (grade, quantity, "
    "unit, deadline, certification). Do not invent requirements. If the "
    "text contains nothing procurement-relevant, record an empty "
    "requirements list. Always respond via the record_requirements tool."
)


class RequirementModel(pydantic.BaseModel):
    """Validated shape of a single extracted requirement."""

    material: str
    grade: str | None = None
    quantity: float | None = None
    unit: str | None = None
    deadline: str | None = None
    certification: str | None = None
    source_page: int
    confidence: Literal["high", "medium", "low"]


def _batch_message(batch: list[dict]) -> str:
    """Render a batch of chunks into one user message, page-tagged."""
    parts = [
        f"[page {chunk.get('page_number', 0)} | {chunk.get('source_file', '')}]\n"
        f"{chunk.get('text', '')}"
        for chunk in batch
    ]
    return (
        "Extract the explicitly stated procurement requirements from the "
        "following document excerpts:\n\n" + "\n\n---\n\n".join(parts)
    )


def _payload_items(payload: dict | None) -> list[dict]:
    """Pull the requirements list out of an extract_json payload."""
    if payload is None:
        return []
    items = payload.get("requirements", [])
    if isinstance(items, list):
        return items
    logger.warning("Malformed structured-output payload from model: %r", payload)
    return []


def _validate_items(items: list[dict]) -> list[RequirementModel]:
    """Validate raw tool items via pydantic; log and skip invalid ones."""
    valid: list[RequirementModel] = []
    for item in items:
        try:
            valid.append(RequirementModel.model_validate(item))
        except pydantic.ValidationError as exc:
            logger.warning("Skipping invalid requirement item %r: %s", item, exc)
    return valid


def extract_requirements(
    chunks: list[dict], project_id: str, db_session: Session
) -> list[ExtractedRequirement]:
    """Extract requirements via forced structured output and persist them.

    Table chunks are skipped (their serialized markdown confuses per-line
    extraction and is indexed for retrieval instead). Returns the persisted
    :class:`ExtractedRequirement` rows. With no LLM provider configured this
    logs a warning and returns ``[]`` — deterministic callers keep working.
    """
    if not llm.is_configured():
        logger.warning(
            "No LLM provider configured (ANTHROPIC_API_KEY / GEMINI_API_KEY "
            "not set) — skipping requirement extraction for project %s "
            "(0 requirements recorded).",
            project_id,
        )
        return []

    text_chunks = [c for c in chunks if c.get("chunk_type") != "table"]
    if not text_chunks:
        logger.info("No text chunks to extract from for project %s", project_id)
        return []

    rows: list[ExtractedRequirement] = []

    for start in range(0, len(text_chunks), BATCH_SIZE):
        batch = text_chunks[start : start + BATCH_SIZE]
        page_to_file = {
            c.get("page_number", 0): c.get("source_file", "") for c in batch
        }
        default_file = batch[0].get("source_file", "")
        payload = llm.extract_json(
            SYSTEM_PROMPT,
            _batch_message(batch),
            REQUIREMENTS_SCHEMA,
            tool_name=TOOL_NAME,
        )
        if payload is None:
            logger.warning(
                "LLM extraction returned nothing for project %s (batch %d)",
                project_id,
                start // BATCH_SIZE,
            )
            continue

        for model_item in _validate_items(_payload_items(payload)):
            rows.append(
                ExtractedRequirement(
                    project_id=project_id,
                    source_file=page_to_file.get(model_item.source_page, default_file),
                    material=model_item.material,
                    grade=model_item.grade,
                    quantity=model_item.quantity,
                    unit=model_item.unit,
                    deadline=model_item.deadline,
                    certification=model_item.certification,
                    source_page=model_item.source_page,
                    confidence=model_item.confidence,
                )
            )

    if rows:
        db_session.add_all(rows)
        db_session.commit()
    logger.info(
        "Extracted %d requirement(s) for project %s from %d text chunk(s)",
        len(rows),
        project_id,
        len(text_chunks),
    )
    return rows


def _same_material_family(material_a: str, material_b: str) -> bool:
    """Case-insensitive substring match in either direction."""
    a = (material_a or "").strip().lower()
    b = (material_b or "").strip().lower()
    if not a or not b:
        return False
    return a in b or b in a


def process_change_request(
    chunks: list[dict], project_id: str, db_session: Session
) -> list[ExtractedRequirement]:
    """Extract requirements from a change request and supersede matching rows.

    For each newly extracted requirement, any existing non-superseded row in
    the same project whose material is in the same family (case-insensitive
    substring either direction; grade may differ) gets its ``superseded_by``
    pointed at the new row. Old rows are never updated in place or deleted.
    """
    existing = (
        db_session.query(ExtractedRequirement)
        .filter(
            ExtractedRequirement.project_id == project_id,
            ExtractedRequirement.superseded_by.is_(None),
        )
        .all()
    )

    new_rows = extract_requirements(chunks, project_id, db_session)
    if not new_rows:
        return []

    superseded_count = 0
    for new_row in new_rows:
        for old_row in existing:
            if old_row.superseded_by is not None:
                continue  # already superseded within this change request
            if _same_material_family(old_row.material, new_row.material):
                old_row.superseded_by = new_row.id
                superseded_count += 1
                logger.info(
                    "Requirement #%d (%s %s) superseded by #%d (%s %s)",
                    old_row.id,
                    old_row.material,
                    old_row.grade,
                    new_row.id,
                    new_row.material,
                    new_row.grade,
                )
    db_session.commit()
    logger.info(
        "Change request for project %s: %d new requirement(s), %d superseded",
        project_id,
        len(new_rows),
        superseded_count,
    )
    return new_rows


def get_current_requirements(
    project_id: str, db_session: Session
) -> list[ExtractedRequirement]:
    """Return the non-superseded requirements for a project."""
    return (
        db_session.query(ExtractedRequirement)
        .filter(
            ExtractedRequirement.project_id == project_id,
            ExtractedRequirement.superseded_by.is_(None),
        )
        .order_by(ExtractedRequirement.id)
        .all()
    )
