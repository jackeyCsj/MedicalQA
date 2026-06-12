from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from config import settings


engine_kwargs = {"pool_pre_ping": True}
if settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
    if ":memory:" in settings.DATABASE_URL:
        engine_kwargs["poolclass"] = StaticPool

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables for the learning/demo setup.

    Alembic files are included for a migration path, but this keeps first-run setup simple.
    """
    from backend.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
