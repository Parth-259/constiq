"""POST /api/agent/ask — full agentic Q&A via backend.agent.agent.run_agent.

The agent module is imported lazily inside the handler so this router can be
imported (and the rest of the API served) even while the agent package is
still being built, and so tests can substitute it.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend import llm
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

INTERNAL_ERROR_DETAIL = "Internal error — see server logs"


class AgentAskRequest(BaseModel):
    project_id: str
    question: str


@router.post("/agent/ask")
def agent_ask(request: AgentAskRequest, db: Session = Depends(get_db)) -> dict:
    """Run the procurement agent loop and return its answer + tool trace."""
    try:
        question = request.question.strip()
        project_id = request.project_id.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question must not be empty.")
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id must not be empty.")

        from backend.agent import agent as agent_module

        return agent_module.run_agent(question, project_id, db)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except llm.LLMError as exc:
        logger.exception("/agent/ask LLM provider error")
        raise HTTPException(
            status_code=503,
            detail="LLM provider error — please retry shortly.",
        ) from exc
    except Exception:
        logger.exception("/agent/ask failed")
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR_DETAIL) from None
