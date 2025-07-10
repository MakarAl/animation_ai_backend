from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
import json
import uuid
import os
from datetime import datetime

from ..database import get_db, Job
from ..supabase_client import upload_bytes_to_bucket
from ..schemas import JobCreateResponse, JobResponse, InterpolateRequest, InterpolateSequenceRequest, JobSequenceResponse
from ..config import settings

# Import Celery task - fix the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from worker.tasks import interpolate_task, interpolate_sequence_task
    print("✅ Celery task import successful")
except ImportError as e:
    print(f"❌ Celery task import failed: {e}")
    # Create a mock task for testing
    def interpolate_task(*args, **kwargs):
        pass
    def interpolate_sequence_task(*args, **kwargs):
        pass

router = APIRouter(prefix="/api/v1", tags=["jobs"])


@router.post("/interpolate", response_model=JobCreateResponse)
async def create_interpolation_job(
    keyframe_0: UploadFile = File(...),
    keyframe_1: UploadFile = File(...),
    size: int = Form(1440),
    vector_cleanup: bool = Form(False),
    no_edge_sharpen: bool = Form(True),
    uniform_thin: bool = Form(False),
    project_id: str = Form(...),
    scene_id: str = Form(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Create a new interpolation job.
    """
    try:
        # Validate file types
        if not keyframe_0.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="keyframe_0 must be an image file")
        if not keyframe_1.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="keyframe_1 must be an image file")
        
        # Create job parameters
        params = {
            "size": size,
            "vector_cleanup": vector_cleanup,
            "no_edge_sharpen": no_edge_sharpen,
            "uniform_thin": uniform_thin
        }
        
        # Generate job ID first
        job_id = str(uuid.uuid4())

        # Upload keyframes to Supabase Storage (bucket: frames)
        key0_bytes = await keyframe_0.read()
        key1_bytes = await keyframe_1.read()

        keyframe_0_url_public = upload_bytes_to_bucket(
            bucket="frames",
            content=key0_bytes,
            mime_type=keyframe_0.content_type,
            user_scope=user_id,
            extension=keyframe_0.filename.split(".")[-1],
        )
        keyframe_1_url_public = upload_bytes_to_bucket(
            bucket="frames",
            content=key1_bytes,
            mime_type=keyframe_1.content_type,
            user_scope=user_id,
            extension=keyframe_1.filename.split(".")[-1],
        )

        # Create job record in database using Supabase URLs
        job = Job(
            id=job_id,
            user_id=user_id,
            project_id=project_id,
            scene_id=scene_id,
            status="PENDING",
            keyframe_0_url=keyframe_0_url_public,
            keyframe_1_url=keyframe_1_url_public,
            params=json.dumps(params),
            created_at=datetime.utcnow(),
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Optionally save local scratch copies for the worker
        os.makedirs("temp", exist_ok=True)
        with open(f"temp/{job_id}_keyframe_0.jpg", "wb") as f:
            f.write(key0_bytes)
        with open(f"temp/{job_id}_keyframe_1.jpg", "wb") as f:
            f.write(key1_bytes)
        
        # Start background task
        try:
            interpolate_task.delay(job_id)
            print(f"✅ Task queued for job {job_id}")
        except Exception as e:
            print(f"⚠️  Task queuing failed: {e}")
            # Continue without the task for now
        
        return JobCreateResponse(
            job_id=job_id,
            message="Interpolation job created successfully"
        )
        
    except Exception as e:
        print(f"❌ Error in create_interpolation_job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpolate_sequence", response_model=JobCreateResponse)
async def create_interpolation_sequence_job(
    keyframe_0: UploadFile = File(...),
    keyframe_1: UploadFile = File(...),
    timesteps: str = Form(...),  # JSON-encoded list of floats
    size: int = Form(1440),
    vector_cleanup: bool = Form(False),
    no_edge_sharpen: bool = Form(True),
    uniform_thin: bool = Form(False),
    create_gif: bool = Form(False),
    gif_duration: float = Form(0.05),
    slots: Optional[str] = Form(None),  # JSON-encoded list of ints
    project_id: str = Form(...),
    scene_id: str = Form(...),
    user_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Create a new sequence interpolation job.
    """
    try:
        # Validate file types
        if not keyframe_0.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="keyframe_0 must be an image file")
        if not keyframe_1.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="keyframe_1 must be an image file")
        # Parse timesteps
        try:
            timesteps_list = json.loads(timesteps)
            assert isinstance(timesteps_list, list) and all(isinstance(t, float) or isinstance(t, int) for t in timesteps_list)
            timesteps_list = [float(t) for t in timesteps_list]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid timesteps: {e}")
        
        # Parse slots if provided
        slots_list = None
        if slots is not None:
            try:
                slots_list = json.loads(slots)
                assert isinstance(slots_list, list) and all(isinstance(s, int) for s in slots_list)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid slots: {e}")

        # Create job parameters
        params = {
            "timesteps": timesteps_list,
            "size": size,
            "vector_cleanup": vector_cleanup,
            "no_edge_sharpen": no_edge_sharpen,
            "uniform_thin": uniform_thin,
            "create_gif": create_gif,
            "gif_duration": gif_duration,
            "slots": slots_list,
        }
        
        # Generate job ID first
        job_id = str(uuid.uuid4())

        # Upload keyframes to Supabase Storage
        key0_bytes = await keyframe_0.read()
        key1_bytes = await keyframe_1.read()

        keyframe_0_url_public = upload_bytes_to_bucket(
            bucket="frames",
            content=key0_bytes,
            mime_type=keyframe_0.content_type,
            user_scope=user_id,
            extension=keyframe_0.filename.split(".")[-1],
        )
        keyframe_1_url_public = upload_bytes_to_bucket(
            bucket="frames",
            content=key1_bytes,
            mime_type=keyframe_1.content_type,
            user_scope=user_id,
            extension=keyframe_1.filename.split(".")[-1],
        )

        # Create job record
        job = Job(
            id=job_id,
            user_id=user_id,
            project_id=project_id,
            scene_id=scene_id,
            status="PENDING",
            keyframe_0_url=keyframe_0_url_public,
            keyframe_1_url=keyframe_1_url_public,
            params=json.dumps(params),
            created_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        # Save local scratch copies
        os.makedirs("temp", exist_ok=True)
        with open(f"temp/{job_id}_keyframe_0.jpg", "wb") as f:
            f.write(key0_bytes)
        with open(f"temp/{job_id}_keyframe_1.jpg", "wb") as f:
            f.write(key1_bytes)
        # Start background task
        try:
            interpolate_sequence_task.delay(job_id)
            print(f"✅ Sequence task queued for job {job_id}")
        except Exception as e:
            print(f"⚠️  Sequence task queuing failed: {e}")
        return JobCreateResponse(
            job_id=job_id,
            message="Sequence interpolation job created successfully"
        )
    except Exception as e:
        print(f"❌ Error in create_interpolation_sequence_job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Get the status of a specific job (single or sequence).
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # If result_urls is present, return JobSequenceResponse
    try:
        result_urls = json.loads(job.result_urls) if job.result_urls else None
    except Exception:
        result_urls = None
    # Try to get slots from job.slots (if present), or from job.params (if stored there)
    slots = None
    if hasattr(job, 'slots') and job.slots:
        try:
            slots = json.loads(job.slots)
        except Exception:
            slots = None
    if slots is None:
        try:
            params = json.loads(job.params)
            slots = params.get('slots')
        except Exception:
            slots = None
    if result_urls:
        return JobSequenceResponse(
            job_id=str(job.id),
            status=job.status,
            created_at=job.created_at,
            processing_started_at=job.processing_started_at,
            completed_at=job.completed_at,
            result_urls=result_urls,
            slots=slots,
            gif_url=job.gif_url,
            error_message=job.error_message
        )
    # Otherwise, return JobResponse
    return JobResponse(
        job_id=str(job.id),
        status=job.status,
        created_at=job.created_at,
        processing_started_at=job.processing_started_at,
        completed_at=job.completed_at,
        result_url=job.result_url,
        error_message=job.error_message
    )


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str, db: Session = Depends(get_db)):
    """
    Get the result file for a completed job.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Job is not completed. Current status: {job.status}")
    
    if not job.result_url or not os.path.exists(job.result_url):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    # TODO: Return file from Supabase storage instead of local file
    from fastapi.responses import RedirectResponse
    # Redirect to the Supabase public URL for the result image
    return RedirectResponse(job.result_url)
