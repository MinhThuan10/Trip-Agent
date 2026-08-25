from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    
    DATABASE_URL: str
    COLLECTION_NAME: str
    EMBEDDING_MODEL: str
    RANKING_MODEL: str
    
    # LM Studio / Local LLM configuration
    LLM_API_BASE: str
    LLM_API_KEY: str
    LLM_MODEL_NAME: str

    
    # External Flight Search API
    FLIGHT_SEARCH_API_URL: str

    CHAT_API_KEY: str

    # Langfuse
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_BASE_URL: str

    VITE_API_BASE_URL:str

    model_config = SettingsConfigDict(env_file=".env" if os.path.exists(".env") else None, env_file_encoding="utf-8")
    
settings = Settings()
