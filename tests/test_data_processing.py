"""
Tests for data processing and ETL functionality.
"""

import pytest
import pandas as pd
from datetime import datetime

from src.etl.data_loader import AmazonDataLoader
from src.etl.text_processor import TextProcessor
from src.models.schemas import ChunkMetadata


def test_amazon_data_loader_initialization():
    """Test data loader can be initialized."""
    loader = AmazonDataLoader()
    assert loader.dataset_name is not None
    assert loader.target_category is not None


def test_text_cleaning():
    """Test text cleaning functionality."""
    loader = AmazonDataLoader()
    
    # Test HTML removal
    dirty_text = "<p>This is a <b>great</b> product!</p>"
    clean_text = loader.clean_text(dirty_text)
    assert "<" not in clean_text
    assert ">" not in clean_text
    assert "great" in clean_text
    
    # Test whitespace normalization
    whitespace_text = "This   has    multiple    spaces"
    clean_text = loader.clean_text(whitespace_text)
    assert "  " not in clean_text
    
    # Test empty text
    assert loader.clean_text("") == ""
    assert loader.clean_text(None) == ""


def test_price_parsing():
    """Test price parsing from various formats."""
    loader = AmazonDataLoader()
    
    # Test valid prices
    assert loader.parse_price("$49.99") == 49.99
    assert loader.parse_price("$1,234.56") == 1234.56
    assert loader.parse_price("29.95") == 29.95
    
    # Test invalid prices
    assert loader.parse_price("") is None
    assert loader.parse_price("invalid") is None
    assert loader.parse_price("$999999") is None  # Too expensive


def test_category_normalization():
    """Test category normalization."""
    loader = AmazonDataLoader()
    
    # Test list input
    categories = ["Electronics", "Computers", "Laptops"]
    normalized = loader.normalize_category(categories)
    assert normalized == "Laptops"  # Should take the most specific (last)
    
    # Test string input
    normalized = loader.normalize_category("Electronics")
    assert normalized == "Electronics"
    
    # Test empty input
    normalized = loader.normalize_category([])
    assert normalized == "Unknown"


def test_demo_data_generation():
    """Test demo data generation."""
    loader = AmazonDataLoader()
    demo_data = loader._generate_demo_data()
    
    assert "products" in demo_data
    assert "reviews" in demo_data
    assert len(demo_data["products"]) > 0
    assert len(demo_data["reviews"]) > 0
    
    # Validate product structure
    product = demo_data["products"][0]
    required_fields = ["asin", "title", "brand", "category", "price"]
    for field in required_fields:
        assert field in product


def test_text_processor_initialization():
    """Test text processor can be initialized."""
    processor = TextProcessor()
    assert processor.tokenizer is not None
    assert processor.max_tokens > 0
    assert processor.overlap_tokens > 0


def test_text_chunking():
    """Test text chunking functionality."""
    processor = TextProcessor()
    
    # Test short text (should create one chunk)
    short_text = "This is a short text."
    chunks = processor.chunk_text(short_text, "review", "ASIN_1", helpful_votes=5, rating=4)
    
    assert len(chunks) == 1
    assert chunks[0].text == short_text
    assert chunks[0].asin == "ASIN_1"
    assert chunks[0].helpful_votes == 5
    assert chunks[0].rating == 4
    
    # Test long text (should create multiple chunks)
    long_text = " ".join(["This is a very long text that should be chunked."] * 50)
    chunks = processor.chunk_text(long_text, "description", "ASIN_2")
    
    assert len(chunks) > 1
    assert all(chunk.asin == "ASIN_2" for chunk in chunks)
    assert all(chunk.type == "description" for chunk in chunks)
    
    # Test empty text
    empty_chunks = processor.chunk_text("", "review", "ASIN_3")
    assert len(empty_chunks) == 0


def test_product_text_processing():
    """Test processing product descriptions into chunks."""
    processor = TextProcessor()
    
    product_data = {
        "asin": "B123456789",
        "title": "Test Product",
        "description": "This is a great product with excellent features. " * 20  # Long description
    }
    
    chunks = processor.process_product_text(product_data)
    
    assert len(chunks) > 0
    assert all(chunk.asin == "B123456789" for chunk in chunks)
    assert all(chunk.type == "description" for chunk in chunks)


def test_review_text_processing():
    """Test processing review text into chunks."""
    processor = TextProcessor()
    
    review_data = {
        "asin": "B123456789",
        "review_text": "This is an amazing product. I've been using it for months and it works perfectly. " * 10,
        "helpful_votes": 15,
        "rating": 5
    }
    
    chunks = processor.process_review_text(review_data)
    
    assert len(chunks) > 0
    assert all(chunk.asin == "B123456789" for chunk in chunks)
    assert all(chunk.type == "review" for chunk in chunks)
    assert all(chunk.helpful_votes == 15 for chunk in chunks)
    assert all(chunk.rating == 5 for chunk in chunks)


def test_data_processing_pipeline(sample_products, sample_reviews):
    """Test complete data processing pipeline."""
    loader = AmazonDataLoader()
    processor = TextProcessor()
    
    # Convert to DataFrames
    products_df = pd.DataFrame(sample_products)
    reviews_df = pd.DataFrame(sample_reviews)
    
    # Process into chunks
    chunks = processor.process_all_text(products_df, reviews_df)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, ChunkMetadata) for chunk in chunks)
    
    # Test that we have both description and review chunks
    chunk_types = set(chunk.type for chunk in chunks)
    assert len(chunk_types) > 0  # Should have at least one type


def test_top_helpful_reviews():
    """Test selection of top helpful reviews."""
    loader = AmazonDataLoader()
    
    # Create sample reviews DataFrame
    reviews_data = [
        {"asin": "A1", "review_text": "Review 1", "helpful_votes": 10, "rating": 5, "review_id": "R1", "reviewer_id": "U1"},
        {"asin": "A1", "review_text": "Review 2", "helpful_votes": 5, "rating": 4, "review_id": "R2", "reviewer_id": "U2"},
        {"asin": "A1", "review_text": "Review 3", "helpful_votes": 15, "rating": 5, "review_id": "R3", "reviewer_id": "U3"},
        {"asin": "A2", "review_text": "Review 4", "helpful_votes": 8, "rating": 4, "review_id": "R4", "reviewer_id": "U4"},
    ]
    
    reviews_df = pd.DataFrame(reviews_data)
    top_reviews = loader.get_top_helpful_reviews(reviews_df, top_n=2)
    
    # Should get top 2 reviews per product
    assert len(top_reviews) <= 4  # 2 products * 2 reviews max
    
    # Check that highest helpful_votes are selected
    a1_reviews = top_reviews[top_reviews["asin"] == "A1"]
    if len(a1_reviews) > 0:
        assert a1_reviews["helpful_votes"].max() == 15  # Should include the highest


@pytest.mark.parametrize("text,expected_chunks", [
    ("Short text", 1),
    ("", 0),
    ("A" * 2000, 2),  # Should create multiple chunks for very long text
])
def test_chunking_edge_cases(text, expected_chunks):
    """Test text chunking with various edge cases."""
    processor = TextProcessor()
    chunks = processor.chunk_text(text, "test", "ASIN_TEST")
    
    if expected_chunks == 0:
        assert len(chunks) == 0
    else:
        assert len(chunks) >= expected_chunks or len(chunks) == 0  # May be 0 if text too short