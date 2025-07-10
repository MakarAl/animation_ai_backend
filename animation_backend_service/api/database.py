"""Database connection and SQLAlchemy models for Animation AI backend.

This module now connects to the Supabase-hosted Postgres database instead of
the previous local SQLite file.  All ORM models are aligned with the tables
that already exist in Supabase, therefore **NO** automatic table creation is
performed here (we rely on Supabase migrations / dashboard for schema).
"""

import os
import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Text,
    JSON,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import settings

# ---------------------------------------------------------------------------
# Database engine / session setup
# ---------------------------------------------------------------------------

DATABASE_URL = settings.SUPABASE_DB_URL

# Standard Postgres URL: postgresql+psycopg2://user:pass@host:port/dbname
# We deliberately **do not** pass `connect_args` here; SQLAlchemy will choose
# sensible defaults for Postgres.

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory for FastAPI dependency
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base
Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM models (mirroring Supabase tables)
# ---------------------------------------------------------------------------


class Job(Base):
    """SQLAlchemy mirror of Supabase `jobs` table.

    NOTE: The actual table and indexes are managed by Supabase migrations.  Do
    **NOT** call `Base.metadata.create_all` – we only need the model for ORM
    convenience inside the API & worker code.
    """

    __tablename__ = "jobs"

    # Primary key – Supabase uses UUID (generated server-side). We still allow
    # Python to set a default UUID when inserting via the API.
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Ownership / context
    user_id = Column(UUID(as_uuid=True), nullable=False)
    project_id = Column(UUID(as_uuid=True), nullable=False)
    scene_id = Column(UUID(as_uuid=True), nullable=False)

    # State & metadata
    status = Column(String, nullable=False, default="PENDING")
    params = Column(JSON, nullable=False, default=dict)  # JSONB in Postgres
    error_message = Column(Text)

    # File URLs
    keyframe_0_url = Column(Text)
    keyframe_1_url = Column(Text)
    result_url = Column(Text)
    result_urls = Column(JSON)  # list of URLs for sequence jobs
    gif_url = Column(Text)

    # Timeline mapping (for sequence jobs)
    slots = Column(JSON)  # list of ints

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_started_at = Column(DateTime)
    completed_at = Column(DateTime)

# IMPORTANT: we intentionally **do not** run `Base.metadata.create_all(engine)`.
# Supabase already owns the schema; running DDL from the app could conflict.


# ---------------------------------------------------------------------------
# Session dependency for FastAPI routes
# ---------------------------------------------------------------------------


def get_db():
    """Yield a DB session for a request and ensure it’s closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
