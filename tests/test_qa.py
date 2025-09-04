"""
Tests for Q&A functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.qa.qa_engine import QAEngine, QAEvaluator
from src.qa.llm_client import MockLLMClient, get_llm_client
from src.models.schemas import QARequest, QAResponse, ChunkMetadata


def test_mock_llm_client():
    """Test mock LLM client functionality."""
    client = MockLLMClient()
    
    # Test battery query
    response = client.generate_response("How is the battery life?")
    assert "battery" in response.lower()
    assert len(response) > 50
    
    # Test sound query
    response = client.generate_response("How is the sound quality?")
    assert "sound" in response.lower() or "audio" in response.lower()
    
    # Test general query
    response = client.generate_response("Is this a good product?")
    assert len(response) > 50


def test_get_llm_client():
    """Test LLM client selection logic."""
    client = get_llm_client()
    assert client is not None
    
    # Should return mock client if no API keys available
    response = client.generate_response("test prompt")
    assert isinstance(response, str)


def test_qa_engine_initialization():
    """Test Q&A engine can be initialized."""
    # Mock the retrieval engine to avoid requiring actual index
    with patch('src.qa.qa_engine.RetrievalEngine') as mock_retrieval:
        mock_retrieval.return_value = Mock()
        
        qa_engine = QAEngine()
        assert qa_engine.llm_client is not None


def test_qa_request_validation():
    """Test Q&A request validation."""
    # Valid request
    request = QARequest(question="How is the battery life?", max_chunks=5)
    assert request.question == "How is the battery life?"
    assert request.max_chunks == 5
    
    # Test defaults
    request = QARequest(question="Test question")
    assert request.max_chunks == 5  # Default value


def test_qa_response_structure():
    """Test Q&A response structure."""
    from src.models.schemas import Citation, ProductResponse
    from datetime import datetime
    
    # Create sample response
    response = QAResponse(
        question="Test question",
        answer="Test answer",
        citations=[],
        confidence_score=0.8,
        suggested_products=[]
    )
    
    assert response.question == "Test question"
    assert response.confidence_score == 0.8
    assert isinstance(response.citations, list)
    assert isinstance(response.suggested_products, list)


def test_citation_extraction():
    """Test citation extraction from LLM responses."""
    from src.qa.qa_engine import QAEngine
    
    # Mock components
    with patch('src.qa.qa_engine.RetrievalEngine'), \
         patch('src.qa.qa_engine.get_llm_client'):
        
        qa_engine = QAEngine()
        
        # Test citation extraction
        sample_chunks = [
            ChunkMetadata(
                chunk_id="chunk_1",
                asin="ASIN_1",
                type="review",
                text="Great battery life",
                start_char=0,
                end_char=18,
                helpful_votes=5,
                rating=5
            )
        ]
        
        sample_answer = "The battery life is excellent [1]. Users report good performance."
        
        processed_answer, citations = qa_engine._process_answer(sample_answer, sample_chunks)
        
        assert "[1]" in processed_answer
        assert len(citations) == 1
        assert citations[0].asin == "ASIN_1"


def test_confidence_calculation():
    """Test confidence score calculation."""
    from src.qa.qa_engine import QAEngine
    
    with patch('src.qa.qa_engine.RetrievalEngine'), \
         patch('src.qa.qa_engine.get_llm_client'):
        
        qa_engine = QAEngine()
        
        # High quality chunks
        high_quality_chunks = [
            ChunkMetadata(
                chunk_id="chunk_1",
                asin="ASIN_1", 
                type="review",
                text="Excellent product",
                start_char=0,
                end_char=16,
                helpful_votes=20,
                rating=5
            )
        ]
        
        # Answer with citations
        answer_with_citations = "This is a great product [1]. Users love it [1]."
        
        confidence = qa_engine._calculate_confidence(high_quality_chunks, answer_with_citations)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be relatively high


def test_qa_evaluator():
    """Test Q&A evaluation metrics."""
    # Mock QA engine
    with patch('src.qa.qa_engine.QAEngine') as mock_engine:
        evaluator = QAEvaluator(mock_engine)
        
        # Test groundedness evaluation
        from src.models.schemas import Citation
        
        sample_response = QAResponse(
            question="Test question",
            answer="The product has great battery life according to reviews.",
            citations=[
                Citation(
                    asin="ASIN_1",
                    chunk_id="chunk_1",
                    text_snippet="battery life is excellent and lasts all day",
                    helpful_votes=10,
                    rating=5
                )
            ],
            confidence_score=0.8,
            suggested_products=[]
        )
        
        groundedness = evaluator.evaluate_groundedness(sample_response)
        assert 0.0 <= groundedness <= 1.0
        
        helpfulness = evaluator.evaluate_helpfulness(sample_response)
        assert 0.0 <= helpfulness <= 1.0


@pytest.mark.parametrize("question", [
    "How is the battery life?",
    "What do customers say about sound quality?",
    "Is this product durable?",
    "Does this work well for gaming?",
    "What are the main pros and cons?"
])
def test_question_types(question):
    """Test different types of questions."""
    request = QARequest(question=question)
    assert request.question == question
    assert len(question) > 0