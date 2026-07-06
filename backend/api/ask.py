"""POST /api/ask — retrieval-augmented Q&A over the ingested documents.

Retrieves the top-5 chunks for the project, builds labelled context blocks
(``[source_file p.N]``) and asks the configured LLM (via ``backend.llm``,
tier "smart") to answer ONLY from that context, citing page numbers.
Degrades gracefully:

- no LLM provider configured -> HTTP 200 with an explanatory answer, the
  retrieved sources are still returned;
- LLM API errors             -> HTTP 503;
- no retrieval hits          -> the standard "not covered" answer.
"""
from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import llm
from backend.pipeline import embedding

logger = logging.getLogger(__name__)

router = APIRouter()

INTERNAL_ERROR_DETAIL = "Internal error — see server logs"
NOT_COVERED = "This isn't covered in the uploaded documents."
LOW_CONFIDENCE_PREFIX = (
    "Note: the retrieved passages are only loosely related to your question, "
    "so treat this answer with caution. "
)
MISSING_KEY_ANSWER = (
    "I can't generate an answer because no LLM provider is configured on the "
    "server (set ANTHROPIC_API_KEY or GEMINI_API_KEY). The most relevant "
    "passages from the uploaded documents are listed in the sources below."
)
SYSTEM_PROMPT = (
    "You are ConstructIQ's construction-document Q&A assistant. Answer ONLY "
    "from the provided context blocks — never from outside knowledge. Each "
    "context block is labelled like [file.pdf p.3] with its source file and "
    "page number. If the context does not contain the answer, reply exactly: "
    '"This isn\'t covered in the uploaded documents." Always cite the page '
    "numbers you used, e.g. (p.3)."
)

_PAGE_PATTERN = re.compile(r"(?:\bp|\bpg|\bpage)\.?\s*(\d+)", re.IGNORECASE)
_SNIPPET_CHARS = 240


class AskRequest(BaseModel):
    project_id: str
    question: str


class SourceItem(BaseModel):
    source_file: str
    page_number: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    low_confidence: bool


def _snippet(text: str) -> str:
    text = " ".join(str(text).split())
    return text[:_SNIPPET_CHARS] + ("…" if len(text) > _SNIPPET_CHARS else "")


def _sources_from(results: list[dict]) -> list[SourceItem]:
    """Deduped (file, page) source items, in retrieval order."""
    seen: set[tuple[str, int]] = set()
    items: list[SourceItem] = []
    for result in results:
        key = (str(result["source_file"]), int(result["page_number"]))
        if key in seen:
            continue
        seen.add(key)
        items.append(
            SourceItem(
                source_file=key[0],
                page_number=key[1],
                snippet=_snippet(result.get("text", "")),
            )
        )
    return items


def _cited_pages(answer: str) -> set[int]:
    """Page numbers the answer text mentions (p.3 / page 3 / pg 3)."""
    return {int(match) for match in _PAGE_PATTERN.findall(answer)}


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question strictly from the project's indexed documents."""
    try:
        question = request.question.strip()
        project_id = request.project_id.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty.")
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id must not be empty.")

        retrieval = embedding.retrieve(question, project_id, top_k=5)
        results: list[dict] = retrieval.get("results", []) or []
        low_confidence = bool(retrieval.get("low_confidence", False))

        if not results:
            return AskResponse(answer=NOT_COVERED, sources=[], low_confidence=low_confidence)

        if not llm.is_configured():
            logger.warning("No LLM provider configured — returning retrieval-only answer.")
            return AskResponse(
                answer=MISSING_KEY_ANSWER,
                sources=_sources_from(results),
                low_confidence=low_confidence,
            )

        context = "\n\n".join(
            f"[{r['source_file']} p.{r['page_number']}]\n{r['text']}" for r in results
        )
        try:
            answer = llm.complete(
                SYSTEM_PROMPT,
                f"Context blocks:\n\n{context}\n\nQuestion: {question}",
                max_tokens=1024,
                tier="smart",
            )
        except llm.LLMError as exc:
            logger.exception("LLM API error during /ask")
            raise HTTPException(
                status_code=503,
                detail="LLM provider error — please retry shortly.",
            ) from exc

        if not answer:
            answer = NOT_COVERED

        cited = _cited_pages(answer)
        cited_results = [r for r in results if int(r["page_number"]) in cited]
        # Fall back to all retrieved chunks when no page citation was detected.
        sources = _sources_from(cited_results or results)

        if low_confidence:
            answer = LOW_CONFIDENCE_PREFIX + answer
        return AskResponse(answer=answer, sources=sources, low_confidence=low_confidence)
    except HTTPException:
        raise
    except Exception:
        logger.exception("/ask failed")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None
