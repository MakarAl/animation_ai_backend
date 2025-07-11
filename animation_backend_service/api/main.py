from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .routers import jobs
from .config import settings
import os, pprint; pprint.pp(settings.SUPABASE_DB_URL)


# Create FastAPI app
app = FastAPI(
    title="Animation AI Backend",
    description="Backend service for frame interpolation using TPS Inbetween model",
    version="1.0.0"
)

# Static file serving removed - now using Supabase Storage for all file serving


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(jobs.router)

@app.get("/")
async def root():
    return {"message": "Animation AI Backend API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
