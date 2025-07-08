import time
import os
import sys
import json
from datetime import datetime
from celery import current_task
from .celery_app import celery_app

# Add the parent directory to the path to import database models and TPS wrapper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'wrappers'))

from api.database import SessionLocal, Job
from api.config import settings
from tps_inbetween_wrapper import TPSInbetweenWrapper


@celery_app.task(bind=True)
def interpolate_task(self, job_id: str):
    """
    Background task to perform interpolation using TPS Inbetween model.
    """
    db = SessionLocal()
    
    try:
        # Get job details
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Parse job parameters
        params = json.loads(job.params)
        size = params.get('size', 1440)
        vector_cleanup = params.get('vector_cleanup', False)
        no_edge_sharpen = params.get('no_edge_sharpen', True)
        uniform_thin = params.get('uniform_thin', False)
        
        # Update job status to PROCESSING
        job.status = "PROCESSING"
        job.processing_started_at = datetime.utcnow()
        db.commit()
        
        # Initialize TPS wrapper
        print(f"Initializing TPS wrapper for job {job_id}...")
        tps_wrapper = TPSInbetweenWrapper(
            device='cpu',  # TODO: Add GPU support
            no_edge_sharpen=no_edge_sharpen,
            vector_cleanup=vector_cleanup,
            uniform_thin=uniform_thin
        )
        
        # Get input file paths
        img0_path = job.keyframe_0_url
        img1_path = job.keyframe_1_url
        
        # Verify input files exist
        if not os.path.exists(img0_path):
            raise FileNotFoundError(f"Input image 0 not found: {img0_path}")
        if not os.path.exists(img1_path):
            raise FileNotFoundError(f"Input image 1 not found: {img1_path}")
        
        # Generate output path
        output_dir = "temp"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{job_id}_result.png")
        
        # Perform interpolation
        print(f"Starting interpolation for job {job_id}...")
        result_path = tps_wrapper.interpolate(
            img0_path=img0_path,
            img1_path=img1_path,
            output_path=output_path,
            num_frames=1,
            max_image_size=size,
            create_gif=True,
            gif_duration=0.05
        )
        
        if result_path is None:
            raise RuntimeError("TPS interpolation failed - no result generated")
        
        # Update job status to COMPLETED
        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow()
        job.result_url = result_path
        db.commit()
        
        print(f"Interpolation completed for job {job_id}: {result_path}")
        return {"status": "success", "job_id": job_id, "result_path": result_path}
        
    except Exception as e:
        print(f"Error in interpolation task for job {job_id}: {str(e)}")
        
        # Update job status to FAILED
        if 'job' in locals() and job:
            job.status = "FAILED"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
        
        # Re-raise the exception
        raise
    finally:
        db.close()


@celery_app.task(bind=True)
def interpolate_sequence_task(self, job_id: str):
    """
    Background task to perform sequence interpolation using TPS Inbetween model.
    """
    db = SessionLocal()
    try:
        # Get job details
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        # Parse job parameters
        params = json.loads(job.params)
        timesteps = params.get('timesteps', None)
        size = params.get('size', 1440)
        vector_cleanup = params.get('vector_cleanup', False)
        no_edge_sharpen = params.get('no_edge_sharpen', True)
        uniform_thin = params.get('uniform_thin', False)
        create_gif = params.get('create_gif', False)
        gif_duration = params.get('gif_duration', 0.05)
        slots = params.get('slots', None)
        if not timesteps or not isinstance(timesteps, list):
            raise ValueError("timesteps must be a non-empty list of floats")
        # Update job status to PROCESSING
        job.status = "PROCESSING"
        job.processing_started_at = datetime.utcnow()
        db.commit()
        # Initialize TPS wrapper
        print(f"Initializing TPS wrapper for sequence job {job_id}...")
        tps_wrapper = TPSInbetweenWrapper(
            device='cpu',  # TODO: Add GPU support
            no_edge_sharpen=no_edge_sharpen,
            vector_cleanup=vector_cleanup,
            uniform_thin=uniform_thin
        )
        # Get input file paths
        img0_path = job.keyframe_0_url
        img1_path = job.keyframe_1_url
        # Verify input files exist
        if not os.path.exists(img0_path):
            raise FileNotFoundError(f"Input image 0 not found: {img0_path}")
        if not os.path.exists(img1_path):
            raise FileNotFoundError(f"Input image 1 not found: {img1_path}")
        # Generate output path
        output_dir = "temp"
        os.makedirs(output_dir, exist_ok=True)
        # Perform sequence interpolation
        print(f"Starting sequence interpolation for job {job_id}...")
        output_paths, gif_path = tps_wrapper.interpolate_sequence(
            img0_path=img0_path,
            img1_path=img1_path,
            output_dir=output_dir,
            timesteps=timesteps,
            create_gif=create_gif,
            gif_duration=gif_duration,
            max_image_size=size
        )
        if not output_paths or len(output_paths) == 0:
            raise RuntimeError("TPS sequence interpolation failed - no results generated")
        # Update job status to COMPLETED
        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow()
        job.result_urls = json.dumps(output_paths)
        if slots is not None:
            job.slots = json.dumps(slots)
        job.gif_url = gif_path
        db.commit()
        print(f"Sequence interpolation completed for job {job_id}: {output_paths}")
        return {"status": "success", "job_id": job_id, "result_paths": output_paths, "gif_path": gif_path}
    except Exception as e:
        print(f"Error in sequence interpolation task for job {job_id}: {str(e)}")
        # Update job status to FAILED
        if 'job' in locals() and job:
            job.status = "FAILED"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()
