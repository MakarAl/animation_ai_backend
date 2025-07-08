import json
import uuid
from datetime import datetime
from api.database import SessionLocal, Job

def test_job_creation():
    db = SessionLocal()
    try:
        # Create job parameters
        params = {
            "size": 512,
            "vector_cleanup": False,
            "no_edge_sharpen": True,
            "uniform_thin": False
        }
        
        # Create job record in database
        job = Job(
            id=str(uuid.uuid4()),
            user_id="test-user",
            status="PENDING",
            keyframe_0_url="temp/test_keyframe_0.jpg",
            keyframe_1_url="temp/test_keyframe_1.jpg",
            params=json.dumps(params),
            created_at=datetime.utcnow()
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        print(f"✅ Job created successfully: {job.id}")
        print(f"Status: {job.status}")
        print(f"Params: {job.params}")
        
        return job.id
        
    except Exception as e:
        print(f"❌ Error creating job: {e}")
        return None
    finally:
        db.close()

if __name__ == "__main__":
    test_job_creation()
