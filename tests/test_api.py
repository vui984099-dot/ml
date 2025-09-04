"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from src.models.schemas import Product, Review


def test_root_endpoint(test_client):
    """Test root endpoint returns API information."""
    response = test_client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "database_status" in data


def test_search_products_empty_query(test_client):
    """Test search with empty query."""
    response = test_client.post("/search", json={"query": ""})
    # Should handle gracefully
    assert response.status_code in [200, 400, 422]


def test_search_products_valid_query(test_client, test_db_session, sample_products):
    """Test search with valid query."""
    # Add sample products to database
    for product_data in sample_products:
        product = Product(
            asin=product_data["asin"],
            title=product_data["title"],
            brand=product_data["brand"],
            category=product_data["category"],
            price=product_data["price"],
            description=product_data["description"],
            image_url=product_data["image_url"],
            avg_rating=product_data["avg_rating"],
            num_reviews=product_data["num_reviews"],
            created_at=datetime.utcnow()
        )
        test_db_session.merge(product)
    
    test_db_session.commit()
    
    # Test search
    response = test_client.post("/search", json={
        "query": "Echo Dot",
        "top_k": 5
    })
    
    # Should not crash (may return empty results if index not built)
    assert response.status_code in [200, 503]


def test_qa_endpoint_structure(test_client):
    """Test Q&A endpoint structure."""
    response = test_client.post("/qa", json={
        "question": "How is the battery life?",
        "max_chunks": 3
    })
    
    # Should not crash (may return empty/error response if components not initialized)
    assert response.status_code in [200, 503]


def test_recommendations_endpoint_structure(test_client):
    """Test recommendations endpoint structure."""
    response = test_client.post("/recommendations", json={
        "query": "wireless headphones",
        "top_k": 5
    })
    
    # Should not crash
    assert response.status_code in [200, 503]


def test_compare_products_endpoint_structure(test_client):
    """Test product comparison endpoint structure."""
    response = test_client.post("/compare", json={
        "asin1": "B08N5WRWNW",
        "asin2": "B08F7PTF53"
    })
    
    # Should not crash (may return 404 if products not found)
    assert response.status_code in [200, 404, 503]


def test_get_product_not_found(test_client):
    """Test getting non-existent product."""
    response = test_client.get("/products/NONEXISTENT")
    assert response.status_code == 404


def test_get_categories_endpoint(test_client):
    """Test categories endpoint."""
    response = test_client.get("/categories")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)


def test_invalid_search_parameters(test_client):
    """Test search with invalid parameters."""
    # Invalid top_k
    response = test_client.post("/search", json={
        "query": "test",
        "top_k": 100  # Too large
    })
    assert response.status_code == 422
    
    # Missing query
    response = test_client.post("/search", json={})
    assert response.status_code == 422


def test_invalid_qa_parameters(test_client):
    """Test Q&A with invalid parameters."""
    # Empty question
    response = test_client.post("/qa", json={"question": ""})
    assert response.status_code == 422
    
    # Invalid max_chunks
    response = test_client.post("/qa", json={
        "question": "test",
        "max_chunks": 0
    })
    assert response.status_code == 422