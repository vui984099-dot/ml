"""
Configuration management for the Amazon product recommendation system.
"""

import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501
    
    # Database Configuration
    database_url: str = "sqlite:///./data/products.db"
    
    # Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gpt-3.5-turbo"
    
    # API Keys
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # FAISS Configuration
    faiss_index_type: str = "HNSW"
    faiss_m: int = 32
    faiss_ef_construction: int = 200
    faiss_ef_search: int = 64
    
    # Data Processing Configuration
    max_chunks_per_product: int = 100
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    
    # Retrieval Configuration
    retrieval_top_k: int = 50
    rerank_top_k: int = 10
    qa_max_chunks: int = 5
    
    # Dataset Configuration
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    target_category: str = "Electronics"
    max_products: int = 20000
    max_reviews: int = 100000
    min_helpful_votes: int = 1
    
    # Paths
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    parquet_dir: str = "./data/parquet"
    index_dir: str = "./data/indexes"
    models_dir: str = "./models"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings