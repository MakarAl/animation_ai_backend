from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class InterpolateRequest(BaseModel):
    """Request model for interpolation job creation"""
    size: int = Field(default=1440, ge=256, le=4096, description="Maximum image dimension")
    vector_cleanup: bool = Field(default=False, description="Enable vector cleanup")
    no_edge_sharpen: bool = Field(default=True, description="Disable edge sharpening")
    uniform_thin: bool = Field(default=False, description="Enable uniform thickness")


class InterpolateSequenceRequest(BaseModel):
    """Request model for sequence interpolation job creation"""
    timesteps: List[float] = Field(..., description="List of t values (0 < t < 1) at which to generate inbetweens")
    size: int = Field(default=1440, ge=256, le=4096, description="Maximum image dimension")
    vector_cleanup: bool = Field(default=False, description="Enable vector cleanup")
    no_edge_sharpen: bool = Field(default=True, description="Disable edge sharpening")
    uniform_thin: bool = Field(default=False, description="Enable uniform thickness")
    create_gif: bool = Field(default=False, description="Whether to create a GIF of the sequence")
    gif_duration: float = Field(default=0.05, description="Duration per frame in the GIF (seconds)")


class JobCreate(BaseModel):
    """Model for creating a new job"""
    user_id: str
    keyframe_0_url: str
    keyframe_1_url: str
    params: InterpolateRequest


class JobResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: JobStatus
    created_at: datetime
    processing_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None


class JobSequenceResponse(BaseModel):
    """Response model for sequence job status"""
    job_id: str
    status: JobStatus
    created_at: datetime
    processing_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_urls: Optional[List[str]] = None  # List of generated image URLs
    slots: Optional[List[int]] = None        # List of slot indices for each result_url
    gif_url: Optional[str] = None
    error_message: Optional[str] = None


class JobCreateResponse(BaseModel):
    """Response model for job creation"""
    job_id: str
    message: str = "Job created successfully"
