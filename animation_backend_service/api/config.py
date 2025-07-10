from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_KEY: str
    SUPABASE_DB_URL: str  # SQLAlchemy-compatible Postgres URI
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Application Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    WORKER_CONCURRENCY: int = 1
    
    # Storage Configuration
    STORAGE_BUCKET: str = "inbetweens"
    
    class Config:
        env_file = "api/.env"
        case_sensitive = False


# Create global settings instance
settings = Settings()
