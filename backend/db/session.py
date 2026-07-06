"""Engine/session factory for the SQLite store."""
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend import config
from backend.db.models import Base

logger = logging.getLogger(__name__)

engine = create_engine(
    f"sqlite:///{config.DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(engine)
    logger.info("SQLite tables ensured at %s", config.DB_PATH)


def get_db():
    """FastAPI dependency — yields a session and always closes it."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
