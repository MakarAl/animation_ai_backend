from celery import Celery
import sys
import os

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings

# Create Celery app
celery_app = Celery(
    "animation_backend",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["worker.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_concurrency=settings.WORKER_CONCURRENCY,
)
