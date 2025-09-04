"""
Tests for recommendation functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.recommendations.recommender import ProductRecommender
from src.models.schemas import RecommendationRequest, RecommendationResponse, ProductResponse


def test_recommendation_request_validation():
    """Test recommendation request validation."""
    # Query-based request
    request = RecommendationRequest(query="wireless headphones", top_k=5)
    assert request.query == "wireless headphones"
    assert request.top_k == 5
    assert request.asin is None
    
    # Similar products request
    request = RecommendationRequest(asin="B123456789", top_k=6)
    assert request.asin == "B123456789"
    assert request.query is None
    
    # Popular products request (no query or asin)
    request = RecommendationRequest(top_k=10)
    assert request.asin is None
    assert request.query is None


def test_recommender_initialization():
    """Test recommender can be initialized."""
    with patch('src.recommendations.recommender.RetrievalEngine'):
        recommender = ProductRecommender()
        assert recommender.retrieval_engine is not None


def test_query_relevance_scoring():
    """Test query relevance scoring algorithm."""
    with patch('src.recommendations.recommender.RetrievalEngine'):
        recommender = ProductRecommender()
        
        # Sample products
        products = [
            ProductResponse(
                asin="A1",
                title="Wireless Bluetooth Headphones",
                brand="Sony",
                category="Electronics",
                price=99.99,
                description="High-quality wireless headphones with great sound",
                image_url="",
                avg_rating=4.5,
                num_reviews=200,
                created_at="2023-01-01T00:00:00"
            ),
            ProductResponse(
                asin="A2", 
                title="Gaming Mouse",
                brand="Logitech",
                category="Electronics",
                price=49.99,
                description="Precision gaming mouse",
                image_url="",
                avg_rating=4.2,
                num_reviews=150,
                created_at="2023-01-01T00:00:00"
            )
        ]
        
        # Test relevance scoring
        scores = recommender._calculate_query_relevance_scores("wireless headphones", products)
        
        assert len(scores) == 2
        assert scores[0] > scores[1]  # First product should be more relevant


def test_collaborative_filtering_logic():
    """Test collaborative filtering recommendation logic."""
    with patch('src.recommendations.recommender.RetrievalEngine'):
        recommender = ProductRecommender()
        
        # Test with empty user history
        recommendations = recommender.get_collaborative_recommendations([], top_k=5)
        assert recommendations == []


def test_recommendation_response_structure():
    """Test recommendation response structure."""
    products = [
        ProductResponse(
            asin="A1",
            title="Test Product",
            brand="Test Brand",
            category="Electronics",
            price=99.99,
            description="Test description",
            image_url="",
            avg_rating=4.5,
            num_reviews=100,
            created_at="2023-01-01T00:00:00"
        )
    ]
    
    response = RecommendationResponse(
        products=products,
        recommendation_type="similar",
        scores=[0.95]
    )
    
    assert len(response.products) == 1
    assert response.recommendation_type == "similar"
    assert len(response.scores) == 1
    assert response.scores[0] == 0.95


@pytest.mark.parametrize("rec_type,expected_field", [
    ("similar", "asin"),
    ("query_based", "query"),
    ("popular", None)
])
def test_recommendation_types(rec_type, expected_field):
    """Test different recommendation types."""
    if rec_type == "similar":
        request = RecommendationRequest(asin="B123456789")
        assert request.asin is not None
    elif rec_type == "query_based":
        request = RecommendationRequest(query="test query")
        assert request.query is not None
    else:  # popular
        request = RecommendationRequest()
        assert request.asin is None
        assert request.query is None


def test_exclude_asins_functionality():
    """Test ASIN exclusion in recommendations."""
    request = RecommendationRequest(
        query="test",
        exclude_asins=["A1", "A2"],
        top_k=5
    )
    
    assert "A1" in request.exclude_asins
    assert "A2" in request.exclude_asins
    assert len(request.exclude_asins) == 2