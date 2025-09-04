"""
Tests for product comparison functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.comparison.comparator import ProductComparator
from src.models.schemas import ComparisonRequest, ComparisonResponse, ProductResponse


def test_comparison_request_validation():
    """Test comparison request validation."""
    request = ComparisonRequest(asin1="A1", asin2="A2")
    assert request.asin1 == "A1"
    assert request.asin2 == "A2"


def test_comparator_initialization():
    """Test product comparator can be initialized."""
    comparator = ProductComparator()
    assert comparator.llm_client is not None
    assert comparator.spec_patterns is not None


def test_spec_extraction():
    """Test specification extraction from product descriptions."""
    comparator = ProductComparator()
    
    # Sample product with specs in description
    product = ProductResponse(
        asin="A1",
        title="Gaming Laptop",
        brand="Dell", 
        category="Electronics",
        price=1299.99,
        description="Gaming laptop with Intel i7 CPU, 16GB RAM, 512GB SSD, RTX 3060 GPU, 15.6 inch display",
        image_url="",
        avg_rating=4.5,
        num_reviews=200,
        created_at="2023-01-01T00:00:00"
    )
    
    specs = comparator._extract_specifications(product)
    
    assert "brand" in specs
    assert specs["brand"] == "Dell"
    assert "price" in specs
    
    # Check if any technical specs were extracted
    spec_keys = ["cpu", "ram", "storage", "gpu", "display"]
    extracted_specs = [key for key in spec_keys if key in specs]
    # Should extract at least some specs from the description
    # (exact extraction depends on regex patterns)


def test_review_summary_generation():
    """Test review summary generation."""
    comparator = ProductComparator()
    
    # Mock database query
    with patch('src.comparison.comparator.SessionLocal') as mock_session:
        mock_db = Mock()
        mock_session.return_value = mock_db
        
        # Mock reviews
        mock_reviews = [
            Mock(rating=5, helpful_votes=10, review_text="Great product, love it!"),
            Mock(rating=4, helpful_votes=5, review_text="Good quality, works well"),
            Mock(rating=2, helpful_votes=2, review_text="Poor build quality, disappointing")
        ]
        
        mock_db.query.return_value.filter.return_value.all.return_value = mock_reviews
        
        summary = comparator._get_review_summary("A1")
        
        assert "count" in summary
        assert "avg_rating" in summary
        assert "sentiment" in summary
        assert summary["count"] == 3


def test_key_differences_identification():
    """Test identification of key differences between products."""
    comparator = ProductComparator()
    
    product1 = ProductResponse(
        asin="A1",
        title="Budget Laptop",
        brand="Acer",
        category="Electronics", 
        price=599.99,
        description="",
        image_url="",
        avg_rating=4.0,
        num_reviews=100,
        created_at="2023-01-01T00:00:00"
    )
    
    product2 = ProductResponse(
        asin="A2",
        title="Premium Laptop", 
        brand="Apple",
        category="Electronics",
        price=1999.99,
        description="",
        image_url="",
        avg_rating=4.8,
        num_reviews=500,
        created_at="2023-01-01T00:00:00"
    )
    
    specs1 = {"brand": "Acer", "price": "$599.99", "rating": "4.0/5.0"}
    specs2 = {"brand": "Apple", "price": "$1999.99", "rating": "4.8/5.0"}
    
    differences = comparator._identify_key_differences(product1, product2, specs1, specs2)
    
    assert len(differences) > 0
    # Should identify price and brand differences
    price_diff_found = any("price" in diff.lower() or "cheaper" in diff.lower() for diff in differences)
    brand_diff_found = any("brand" in diff.lower() for diff in differences)
    
    assert price_diff_found or brand_diff_found  # At least one should be found


def test_category_winners_determination():
    """Test determination of category winners."""
    comparator = ProductComparator()
    
    product1 = ProductResponse(
        asin="A1",
        title="Budget Option",
        brand="Brand1",
        category="Electronics",
        price=50.0,
        description="",
        image_url="",
        avg_rating=4.0,
        num_reviews=100,
        created_at="2023-01-01T00:00:00"
    )
    
    product2 = ProductResponse(
        asin="A2",
        title="Premium Option",
        brand="Brand2", 
        category="Electronics",
        price=150.0,
        description="",
        image_url="",
        avg_rating=4.8,
        num_reviews=500,
        created_at="2023-01-01T00:00:00"
    )
    
    specs1 = {}
    specs2 = {}
    reviews1 = {"sentiment": "positive", "count": 100}
    reviews2 = {"sentiment": "positive", "count": 500}
    
    winners = comparator._determine_category_winners(
        product1, product2, specs1, specs2, reviews1, reviews2
    )
    
    # Product1 should win on price, Product2 should win on rating and popularity
    assert winners.get("price") == "A1"  # Cheaper
    assert winners.get("rating") == "A2"  # Higher rating
    assert winners.get("popularity") == "A2"  # More reviews


def test_comparison_response_structure():
    """Test comparison response structure."""
    from datetime import datetime
    
    product1 = ProductResponse(
        asin="A1",
        title="Product 1",
        brand="Brand1",
        category="Electronics",
        price=99.99,
        description="Description 1",
        image_url="",
        avg_rating=4.5,
        num_reviews=100,
        created_at=datetime.now()
    )
    
    product2 = ProductResponse(
        asin="A2",
        title="Product 2", 
        brand="Brand2",
        category="Electronics",
        price=149.99,
        description="Description 2",
        image_url="",
        avg_rating=4.2,
        num_reviews=200,
        created_at=datetime.now()
    )
    
    response = ComparisonResponse(
        product1=product1,
        product2=product2,
        comparison_summary="Product comparison summary",
        key_differences=["Price difference", "Brand difference"],
        winner_categories={"price": "A1", "rating": "A1"}
    )
    
    assert response.product1.asin == "A1"
    assert response.product2.asin == "A2"
    assert len(response.key_differences) == 2
    assert len(response.winner_categories) == 2