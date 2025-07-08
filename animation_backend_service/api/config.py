from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Supabase Configuration
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Application Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    worker_concurrency: int = 1
    
    # Storage Configuration
    storage_bucket: str = "inbetweens"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()
