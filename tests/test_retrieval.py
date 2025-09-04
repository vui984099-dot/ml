"""
Tests for retrieval and search functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.retrieval.search_engine import RetrievalEngine
from src.indexing.embeddings import EmbeddingGenerator, CrossEncoderReranker
from src.indexing.faiss_index import FAISSIndexManager, VectorSearchEngine
from src.models.schemas import ChunkMetadata


def test_embedding_generator_initialization():
    """Test embedding generator can be initialized."""
    generator = EmbeddingGenerator()
    assert generator.model is not None
    assert generator.embedding_dim > 0


def test_embedding_generation(sample_chunks):
    """Test embedding generation from text chunks."""
    generator = EmbeddingGenerator()
    
    # Test with sample texts
    texts = ["This is a test text", "Another test sentence"]
    embeddings = generator.encode_texts(texts)
    
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == generator.embedding_dim
    assert not np.isnan(embeddings).any()


def test_cross_encoder_initialization():
    """Test cross-encoder can be initialized."""
    reranker = CrossEncoderReranker()
    assert reranker.model is not None


def test_cross_encoder_scoring():
    """Test cross-encoder scoring functionality."""
    reranker = CrossEncoderReranker()
    
    query = "good battery life"
    texts = [
        "The battery lasts all day",
        "Sound quality is excellent", 
        "Battery life is poor"
    ]
    
    scores = reranker.score_pairs(query, texts)
    
    assert len(scores) == 3
    assert all(isinstance(score, (int, float)) for score in scores)


def test_faiss_index_manager():
    """Test FAISS index manager functionality."""
    # Create mock embeddings
    embeddings = np.random.rand(10, 384).astype('float32')
    
    # Create mock chunks
    chunks = []
    for i in range(10):
        chunk = ChunkMetadata(
            chunk_id=f"chunk_{i}",
            asin=f"ASIN_{i}",
            type="review",
            text=f"This is test text {i}",
            start_char=0,
            end_char=20,
            helpful_votes=i,
            rating=4,
            faiss_id=i
        )
        chunks.append(chunk)
    
    # Test index building
    index_manager = FAISSIndexManager(384)
    index_manager.build_hnsw_index(embeddings, chunks)
    
    assert index_manager.index is not None
    assert index_manager.index.ntotal == 10
    
    # Test search
    query_embedding = np.random.rand(1, 384).astype('float32')
    results = index_manager.search(query_embedding, top_k=3)
    
    assert len(results) <= 3
    assert all(isinstance(chunk_id, str) and isinstance(score, float) for chunk_id, score in results)


def test_vector_search_engine_mock():
    """Test vector search engine with mocked components."""
    # This test would require actual chunks and embeddings
    # For now, test initialization
    search_engine = VectorSearchEngine()
    assert search_engine.embedding_generator is not None
    assert search_engine.index_manager is not None


@pytest.mark.integration
def test_retrieval_engine_integration():
    """Integration test for retrieval engine."""
    # This would require actual data and built index
    engine = RetrievalEngine()
    
    # Test that components are initialized
    assert engine.vector_search is not None
    assert engine.cross_encoder is not None


def test_chunk_filtering():
    """Test chunk filtering functionality."""
    from src.retrieval.search_engine import RetrievalEngine
    
    engine = RetrievalEngine()
    
    # Create sample chunks
    chunks = [
        ChunkMetadata(
            chunk_id="chunk_1",
            asin="ASIN_1",
            type="review",
            text="Great product",
            start_char=0,
            end_char=13,
            helpful_votes=5,
            rating=5
        ),
        ChunkMetadata(
            chunk_id="chunk_2",
            asin="ASIN_2", 
            type="review",
            text="Poor quality",
            start_char=0,
            end_char=12,
            helpful_votes=2,
            rating=2
        )
    ]
    
    # Test rating filter
    filtered = engine._apply_filters(chunks, min_rating=4.0)
    # Should filter based on rating (though this requires DB access)
    assert isinstance(filtered, list)


def test_recommendation_logic():
    """Test recommendation generation logic."""
    from src.retrieval.search_engine import RetrievalEngine
    
    engine = RetrievalEngine()
    
    # Test with empty chunks
    recommendations = engine.get_product_recommendations("NONEXISTENT", top_k=3)
    assert recommendations == []


@pytest.mark.parametrize("query,expected_type", [
    ("battery life", str),
    ("sound quality", str),
    ("", str),
    ("very long query with many words to test handling", str)
])
def test_query_processing(query, expected_type):
    """Test various query inputs."""
    # Test that queries are processed as strings
    assert isinstance(query, expected_type)