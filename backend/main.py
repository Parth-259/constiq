"""ConstructIQ FastAPI application.

Mounts every API router under ``/api``, serves the static frontend from
``frontend/`` at ``/`` (mounted LAST so API routes win), and on startup
creates the SQLite tables and seeds the synthetic vendor demo data.

Set ``CONSTRUCTIQ_SKIP_SEED=1`` to skip the startup init/seed entirely —
used by the hermetic test suite so it never touches the project data/ dir.

Run with: ``uvicorn backend.main:app --reload --port 8000``.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.api import (
    agent_routes,
    ask,
    ingest,
    negotiation_routes,
    po_routes,
    risk,
    vendors,
)
from backend.db import session

logger = logging.getLogger(__name__)

app = FastAPI(title="ConstructIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        config.FRONTEND_ORIGIN,
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api")
app.include_router(ask.router, prefix="/api")
app.include_router(agent_routes.router, prefix="/api")
app.include_router(vendors.router, prefix="/api")
app.include_router(risk.router, prefix="/api")
app.include_router(negotiation_routes.router, prefix="/api")
app.include_router(po_routes.router, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.on_event("startup")
def startup() -> None:
    """Create tables and seed the synthetic demo data (skippable for tests)."""
    if os.getenv("CONSTRUCTIQ_SKIP_SEED") == "1":
        logger.info("CONSTRUCTIQ_SKIP_SEED=1 — skipping init_db() and seeding.")
        return
    session.init_db()
    try:
        from backend.db import seed_vendors

        db = session.SessionLocal()
        try:
            counts = seed_vendors.seed_all(db)
            logger.info("Vendor seed check complete: %s", counts)
        finally:
            db.close()
    except Exception:
        logger.exception("Vendor seeding failed — continuing without seed data.")
    logger.info("Chroma vector store path: %s", config.CHROMA_PATH)


# Static frontend LAST so /api/* routes always win. check_dir=False lets the
# app import/start even before the frontend directory has been created.
app.mount(
    "/",
    StaticFiles(directory=str(config.PROJECT_ROOT / "frontend"), html=True, check_dir=False),
    name="frontend",
)
