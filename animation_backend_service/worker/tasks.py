import time
import os
import sys
import json
from datetime import datetime
from celery import current_task
from .celery_app import celery_app

# New imports for Supabase integration and HTTP download
import requests

# Supabase helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ensure api package resolvable
from api.supabase_client import (
    upload_bytes_to_bucket,
    insert_uploaded_file,
    insert_timeline_slot,
)

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
        
        # Parse job parameters (job.params may be dict or JSON string)
        if isinstance(job.params, dict):
            params = job.params
        else:
            params = json.loads(job.params or "{}")
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
        
        # Download keyframes from Supabase URLs if not local
        def download_if_url(url, local_name):
            if url.startswith("http://") or url.startswith("https://"):
                r = requests.get(url)
                r.raise_for_status()
                with open(local_name, "wb") as f:
                    f.write(r.content)
                return local_name
            return url

        os.makedirs("temp", exist_ok=True)
        img0_path = download_if_url(job.keyframe_0_url, f"temp/{job_id}_keyframe_0.jpg")
        img1_path = download_if_url(job.keyframe_1_url, f"temp/{job_id}_keyframe_1.jpg")

        # No os.path.exists checks needed; download_if_url guarantees file or raises

        # Generate output path
        output_dir = "temp"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{job_id}_result.png")
        
        # Perform interpolation
        print(f"Starting interpolation for job {job_id}...")
        result_path = tps_wrapper.interpolate(
            img0_path=img0_path,  # Always use local file path
            img1_path=img1_path,  # Always use local file path
            output_path=output_path,
            num_frames=1,
            max_image_size=size,
            create_gif=True,
            gif_duration=0.05
        )
        
        if result_path is None:
            raise RuntimeError("TPS interpolation failed - no result generated")
        
        # Upload result image to Supabase Storage bucket 'inbetweens'
        with open(result_path, "rb") as f:
            result_bytes = f.read()

        public_result_url = upload_bytes_to_bucket(
            bucket="inbetweens",
            content=result_bytes,
            mime_type="image/png",
            user_scope=job.user_id,
            extension="png",
        )

        # Insert into uploaded_files table
        file_id = insert_uploaded_file(
            scene_id=str(job.scene_id),
            original_id=str(job.id),
            name=f"interpolated_{job.id}.png",
            url=public_result_url,
        )

        # Determine slot indices
        slots_list = None
        if job.slots is not None:
            slots_list = job.slots if isinstance(job.slots, list) else json.loads(job.slots)
        if not slots_list:
            slots_list = params.get("slots") if params else None
        if not slots_list:
            slots_list = [0]  # fallback

        for slot_index in slots_list:
            insert_timeline_slot(
                project_id=str(job.project_id),
                slot_index=slot_index,
                file_id=file_id,
            )

        # Update job status to COMPLETED with Supabase URL
        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow()
        job.result_url = public_result_url
        db.commit()
        
        print(f"Interpolation completed for job {job_id}: {result_path}")
        return {"status": "success", "job_id": job_id, "result_url": public_result_url}
        
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
        # Download keyframes from Supabase URLs if not local
        def download_if_url(url, local_name):
            if url.startswith("http://") or url.startswith("https://"):
                r = requests.get(url)
                r.raise_for_status()
                with open(local_name, "wb") as f:
                    f.write(r.content)
                return local_name
            return url

        os.makedirs("temp", exist_ok=True)
        img0_path = download_if_url(job.keyframe_0_url, f"temp/{job.id}_keyframe_0.jpg")
        img1_path = download_if_url(job.keyframe_1_url, f"temp/{job.id}_keyframe_1.jpg")
        # No os.path.exists checks needed; download_if_url guarantees file or raises
        # Generate output path
        output_dir = "temp"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Starting sequence interpolation for job {job_id}...")
        output_paths, gif_path_local = tps_wrapper.interpolate_sequence(
            img0_path=img0_path,  # Always use local file path
            img1_path=img1_path,  # Always use local file path
            output_dir=output_dir,
            timesteps=timesteps,
            create_gif=create_gif,
            gif_duration=gif_duration,
            max_image_size=size,
        )

        if not output_paths:
            raise RuntimeError("TPS sequence interpolation failed - no results generated")

        # Upload each generated frame to Supabase and create DB records
        public_urls: list[str] = []
        for idx, local_path in enumerate(output_paths):
            with open(local_path, "rb") as f:
                img_bytes = f.read()

            public_url = upload_bytes_to_bucket(
                bucket="inbetweens",
                content=img_bytes,
                mime_type="image/png",
                user_scope=job.user_id,
                extension="png",
            )
            public_urls.append(public_url)

            # Map slot index if provided
            slot_index = (
                slots[idx] if slots and idx < len(slots) else idx  # fallback sequential
            )

            # Insert uploaded_files & timeline slot
            file_id = insert_uploaded_file(
                scene_id=str(job.scene_id),
                original_id=str(job.id),
                name=f"interpolated_{job.id}_{idx}.png",
                url=public_url,
            )
            insert_timeline_slot(
                project_id=str(job.project_id),
                slot_index=slot_index,
                file_id=file_id,
            )

        # Upload GIF if created
        gif_public_url = None
        if create_gif and gif_path_local and os.path.exists(gif_path_local):
            with open(gif_path_local, "rb") as gf:
                gif_bytes = gf.read()
            gif_public_url = upload_bytes_to_bucket(
                bucket="inbetweens",
                content=gif_bytes,
                mime_type="image/gif",
                user_scope=job.user_id,
                extension="gif",
            )

        # Update job record
        job.status = "COMPLETED"
        job.completed_at = datetime.utcnow()
        job.result_urls = public_urls
        if slots:
            job.slots = slots
        job.gif_url = gif_public_url
        db.commit()

        print(f"Sequence interpolation completed for job {job_id}: {public_urls}")
        return {
            "status": "success",
            "job_id": job_id,
            "result_urls": public_urls,
            "gif_url": gif_public_url,
        }
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
