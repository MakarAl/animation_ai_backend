# Animation AI Backend Service

A FastAPI-based backend service for frame interpolation using the TPS Inbetween model, designed to work with a Lovable frontend and Supabase.

## 🏗️ Architecture

- **FastAPI**: REST API server
- **Celery + Redis**: Asynchronous task processing
- **SQLite**: Local database (can be replaced with Supabase PostgreSQL)
- **Supabase**: Authentication, storage, and database (production)
- **TPS Inbetween Model**: Frame interpolation engine

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis (for Celery)
- Docker (optional)

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Redis:**
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:7-alpine
   
   # Or install Redis locally
   brew install redis  # macOS
   redis-server
   ```

3. **Start the API server:**
   ```bash
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **Start the Celery worker (in another terminal):**
   ```bash
   celery -A worker.celery_app worker --loglevel=info --concurrency=1
   ```

5. **Test the API:**
   ```bash
   python test_api.py
   ```

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://127.0.0.1:8000/docs
- **ReDoc documentation**: http://127.0.0.1:8000/redoc

## �� API Endpoints

### POST /api/v1/interpolate
Create a new interpolation job.

**Parameters:**
- `keyframe_0`: First input image (file upload)
- `keyframe_1`: Second input image (file upload)
- `size`: Maximum image dimension (default: 1440)
- `vector_cleanup`: Enable vector cleanup (default: false)
- `no_edge_sharpen`: Disable edge sharpening (default: true)
- `uniform_thin`: Enable uniform thickness (default: false)

**Response:**
```json
{
  "job_id": "uuid-string",
  "message": "Interpolation job created successfully"
}
```

### GET /api/v1/jobs/{job_id}
Get the status of a specific job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "PENDING|PROCESSING|COMPLETED|FAILED",
  "created_at": "2024-01-01T00:00:00",
  "processing_started_at": "2024-01-01T00:00:00",
  "completed_at": "2024-01-01T00:00:00",
  "result_url": "https://example.com/result.png",
  "error_message": "Error details if failed"
}
```

## 🔧 Configuration

Environment variables in `.env`:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Application Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKER_CONCURRENCY=1

# Storage Configuration
STORAGE_BUCKET=inbetweens
```

## 📁 Project Structure

```
animation_backend_service/
├── api/                    # FastAPI application
│   ├── main.py            # Main application
│   ├── config.py          # Configuration
│   ├── database.py        # Database models
│   ├── schemas.py         # Pydantic schemas
│   └── routers/           # API routes
│       └── jobs.py        # Job endpoints
├── worker/                # Celery worker
│   ├── celery_app.py      # Celery configuration
│   └── tasks.py           # Background tasks
├── temp/                  # Temporary file storage
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker orchestration
├── Dockerfile            # Docker image
└── .env                  # Environment variables
```

## 🔄 Current Status

### ✅ Implemented
- FastAPI server with basic endpoints
- Celery task queue setup
- SQLite database with job tracking
- File upload handling
- Mock interpolation task
- Docker configuration

### 🚧 In Progress
- TPS model integration
- Supabase storage integration
- Authentication middleware
- Real interpolation processing

### 📋 Next Steps
1. Integrate TPS wrapper from parent directory
2. Add Supabase storage for file uploads
3. Implement authentication with Supabase Auth
4. Add real-time job status updates
5. Optimize for production deployment

## 🧪 Testing

Run the test script:
```bash
python test_api.py
```

Or use curl:
```bash
# Test root endpoint
curl http://127.0.0.1:8000/

# Test health endpoint
curl http://127.0.0.1:8000/health

# Test interpolation endpoint (requires image files)
curl -X POST http://127.0.0.1:8000/api/v1/interpolate \
  -F "keyframe_0=@image1.jpg" \
  -F "keyframe_1=@image2.jpg" \
  -F "size=720"
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

2. **Redis connection error:**
   - Ensure Redis is running: `redis-cli ping`
   - Check Redis URL in `.env`

3. **Import errors:**
   - Ensure you're in the correct directory
   - Check Python path and dependencies

4. **Database errors:**
   - SQLite database is created automatically
   - Check file permissions for `test.db`

## 📄 License

This project is part of the Animation AI Backend system.
