"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.models.schemas import Base
from src.database import get_db
from src.api.main import app


@pytest.fixture(scope="session")
def temp_db():
    """Create a temporary database for testing."""
    temp_db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db_path = temp_db_file.name
    temp_db_file.close()
    
    # Create test database
    engine = create_engine(f"sqlite:///{temp_db_path}")
    Base.metadata.create_all(bind=engine)
    
    yield temp_db_path
    
    # Cleanup
    os.unlink(temp_db_path)


@pytest.fixture
def test_db_session(temp_db):
    """Create a test database session."""
    engine = create_engine(f"sqlite:///{temp_db}")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_client(test_db_session):
    """Create a test client with dependency overrides."""
    
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_products():
    """Sample product data for testing."""
    return [
        {
            "asin": "B08N5WRWNW",
            "title": "Echo Dot (4th Gen) | Smart speaker with Alexa",
            "brand": "Amazon",
            "category": "Electronics",
            "price": 49.99,
            "description": "Meet Echo Dot - Our most popular smart speaker with a fabric design.",
            "image_url": "https://example.com/image1.jpg",
            "avg_rating": 4.7,
            "num_reviews": 15420
        },
        {
            "asin": "B08F7PTF53",
            "title": "Fire TV Stick 4K Max streaming device", 
            "brand": "Amazon",
            "category": "Electronics",
            "price": 54.99,
            "description": "Fire TV Stick 4K Max streaming device, supports Wi-Fi 6.",
            "image_url": "https://example.com/image2.jpg",
            "avg_rating": 4.6,
            "num_reviews": 8932
        }
    ]


@pytest.fixture
def sample_reviews():
    """Sample review data for testing."""
    return [
        {
            "review_id": "R1234567890",
            "asin": "B08N5WRWNW",
            "review_text": "Great little device! The sound quality is surprisingly good.",
            "rating": 5,
            "helpful_votes": 12,
            "reviewer_id": "user123"
        },
        {
            "review_id": "R1234567891", 
            "asin": "B08N5WRWNW",
            "review_text": "Good value for money. The size is perfect for my nightstand.",
            "rating": 4,
            "helpful_votes": 8,
            "reviewer_id": "user456"
        }
    ]


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    from src.models.schemas import ChunkMetadata
    
    return [
        ChunkMetadata(
            chunk_id="chunk_1",
            asin="B08N5WRWNW",
            type="review",
            text="Great little device! The sound quality is surprisingly good for such a small speaker.",
            start_char=0,
            end_char=85,
            helpful_votes=12,
            rating=5,
            faiss_id=0
        ),
        ChunkMetadata(
            chunk_id="chunk_2", 
            asin="B08F7PTF53",
            type="description",
            text="Fire TV Stick 4K Max streaming device, supports Wi-Fi 6, Dolby Vision, HDR.",
            start_char=0,
            end_char=75,
            helpful_votes=0,
            rating=None,
            faiss_id=1
        )
    ]