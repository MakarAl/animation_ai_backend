from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from .config import settings

# For now, we'll use a simple SQLite database for local testing
# TODO: Replace with Supabase PostgreSQL connection
DATABASE_URL = "sqlite:///./test.db"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class Job(Base):
    """Database model for interpolation jobs"""
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    keyframe_0_url = Column(String)
    keyframe_1_url = Column(String)
    result_url = Column(String)
    result_urls = Column(Text)  # JSON-encoded list of URLs for sequence jobs
    slots = Column(Text)        # JSON-encoded list of slot indices for sequence jobs
    gif_url = Column(String)    # URL or path to generated GIF (if any)
    params = Column(String, nullable=False, default="{}")  # JSON as string for SQLite
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_started_at = Column(DateTime)
    completed_at = Column(DateTime)


# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
