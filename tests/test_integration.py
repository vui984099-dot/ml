"""
Integration tests for the complete system.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.etl.data_loader import AmazonDataLoader
from src.etl.text_processor import TextProcessor
from src.indexing.embeddings import EmbeddingGenerator
from src.models.schemas import ChunkMetadata


@pytest.mark.integration
def test_complete_data_pipeline():
    """Test the complete data processing pipeline."""
    # Use demo data for testing
    loader = AmazonDataLoader()
    processor = TextProcessor()
    
    # Load demo data
    demo_data = loader._generate_demo_data()
    
    # Process products and reviews
    products_df = loader.process_products(demo_data)
    reviews_df = loader.process_reviews(demo_data)
    
    assert len(products_df) > 0
    assert len(reviews_df) > 0
    
    # Process into chunks
    chunks = processor.process_all_text(products_df, reviews_df)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, ChunkMetadata) for chunk in chunks)


@pytest.mark.integration 
def test_embedding_generation_pipeline():
    """Test embedding generation with real text."""
    generator = EmbeddingGenerator()
    
    # Test with sample texts
    sample_texts = [
        "This is a great product with excellent battery life",
        "The sound quality is amazing and crystal clear",
        "Easy to set up and use, very intuitive interface"
    ]
    
    embeddings = generator.encode_texts(sample_texts)
    
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == generator.embedding_dim
    assert not np.isnan(embeddings).any()
    
    # Test that embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


@pytest.mark.integration
def test_search_pipeline():
    """Test search pipeline with mock data."""
    from src.indexing.faiss_index import VectorSearchEngine
    import numpy as np
    
    # Create sample chunks
    chunks = [
        ChunkMetadata(
            chunk_id=f"chunk_{i}",
            asin=f"ASIN_{i}",
            type="review",
            text=f"This is test review text number {i} about product quality",
            start_char=0,
            end_char=50,
            helpful_votes=i * 2,
            rating=4 + (i % 2),
            faiss_id=i
        ) for i in range(5)
    ]
    
    # Test vector search engine
    search_engine = VectorSearchEngine()
    
    # Build index with sample chunks
    search_engine.build_index_from_chunks(chunks)
    
    # Test search
    results = search_engine.search_similar_chunks("product quality", top_k=3)
    
    assert len(results) <= 3
    assert all(hasattr(chunk, 'similarity_score') for chunk in results)


@pytest.mark.integration
def test_api_health_check():
    """Test API health check functionality."""
    from src.api.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    required_fields = ["status", "timestamp", "database_status", "index_status", "model_status"]
    for field in required_fields:
        assert field in data


@pytest.mark.integration
def test_end_to_end_qa_flow():
    """Test end-to-end Q&A flow with mock components."""
    from src.qa.qa_engine import QAEngine
    from src.models.schemas import QARequest
    
    # Mock the retrieval engine to return sample chunks
    with patch('src.qa.qa_engine.RetrievalEngine') as mock_retrieval:
        mock_engine = Mock()
        mock_chunks = [
            ChunkMetadata(
                chunk_id="chunk_1",
                asin="A1",
                type="review",
                text="The battery life is excellent, lasts all day",
                start_char=0,
                end_char=44,
                helpful_votes=10,
                rating=5
            )
        ]
        mock_engine.get_chunks_for_qa.return_value = mock_chunks
        mock_retrieval.return_value = mock_engine
        
        qa_engine = QAEngine()
        request = QARequest(question="How is the battery life?")
        
        response = qa_engine.answer_question(request)
        
        assert response.question == "How is the battery life?"
        assert len(response.answer) > 0
        assert 0.0 <= response.confidence_score <= 1.0


@pytest.mark.integration
def test_system_components_compatibility():
    """Test that all system components can work together."""
    # Test imports work correctly
    from src.config import settings
    from src.database import init_database
    from src.models.schemas import Product, Review, TextChunk
    
    # Test settings are accessible
    assert settings.embedding_model is not None
    assert settings.database_url is not None
    
    # Test database models are compatible
    assert hasattr(Product, '__tablename__')
    assert hasattr(Review, '__tablename__')
    assert hasattr(TextChunk, '__tablename__')


def test_error_handling():
    """Test error handling in various components."""
    from src.etl.data_loader import AmazonDataLoader
    
    loader = AmazonDataLoader()
    
    # Test with invalid data
    result = loader.clean_text(None)
    assert result == ""
    
    result = loader.parse_price("invalid_price")
    assert result is None
    
    result = loader.normalize_category([])
    assert result == "Unknown"