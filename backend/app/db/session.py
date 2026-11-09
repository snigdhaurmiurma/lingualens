"""Database engine, session management, and configuration for LinguaLens.

Supports dual-engine architecture:
- SQLite (file or :memory:) for lightweight, zero-dependency local development and CI.
- PostgreSQL + pgvector for scalable production vector indexing.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class DatabaseSettings(BaseSettings):
    """Database configuration settings loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: str = "sqlite:///./lingualens.db"
    ECHO_SQL: bool = False
    POOL_PRE_PING: bool = True


db_settings = DatabaseSettings()


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy entities in LinguaLens."""

    pass


def get_engine(database_url: str | None = None, echo: bool | None = None) -> Engine:
    """Create and configure a SQLAlchemy Engine."""
    url = database_url or db_settings.DATABASE_URL
    echo_flag = echo if echo is not None else db_settings.ECHO_SQL

    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        url,
        echo=echo_flag,
        connect_args=connect_args,
        pool_pre_ping=db_settings.POOL_PRE_PING if not url.startswith("sqlite") else False,
    )

    # Enable foreign key constraint enforcement in SQLite
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


# Default engine and sessionmaker
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency helper yielding a scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
