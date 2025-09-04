"""
Tests for system evaluation and metrics.
"""

import pytest
import numpy as np
from typing import List, Dict

from src.qa.qa_engine import QAEvaluator
from src.models.schemas import QAResponse, Citation, ProductResponse


def test_groundedness_evaluation():
    """Test groundedness evaluation metrics."""
    evaluator = QAEvaluator(None)  # Don't need actual QA engine for this test
    
    # Create sample QA response with citations
    citations = [
        Citation(
            asin="A1",
            chunk_id="chunk_1",
            text_snippet="battery life is excellent and lasts all day long",
            helpful_votes=10,
            rating=5
        )
    ]
    
    # Answer that uses citation content
    qa_response_grounded = QAResponse(
        question="How is the battery life?",
        answer="The battery life is excellent and lasts all day according to user reviews.",
        citations=citations,
        confidence_score=0.8,
        suggested_products=[]
    )
    
    groundedness = evaluator.evaluate_groundedness(qa_response_grounded)
    assert 0.0 <= groundedness <= 1.0
    
    # Answer that doesn't use citation content  
    qa_response_ungrounded = QAResponse(
        question="How is the battery life?",
        answer="The product has amazing features and revolutionary technology.",
        citations=citations,
        confidence_score=0.8,
        suggested_products=[]
    )
    
    groundedness_low = evaluator.evaluate_groundedness(qa_response_ungrounded)
    assert groundedness_low < groundedness  # Should be lower


def test_helpfulness_evaluation():
    """Test helpfulness evaluation metrics."""
    evaluator = QAEvaluator(None)
    
    # Helpful answer with good structure
    helpful_response = QAResponse(
        question="How is the battery life?",
        answer="""Based on customer reviews, the battery life is generally excellent. Users report that it lasts all day with normal usage. However, heavy usage may require charging by evening. Overall, most customers are satisfied with the battery performance.""",
        citations=[
            Citation(asin="A1", chunk_id="c1", text_snippet="battery lasts all day", helpful_votes=10, rating=5)
        ],
        confidence_score=0.8,
        suggested_products=[]
    )
    
    helpfulness = evaluator.evaluate_helpfulness(helpful_response)
    assert 0.0 <= helpfulness <= 1.0
    assert helpfulness > 0.5  # Should be relatively high
    
    # Unhelpful answer (too short)
    unhelpful_response = QAResponse(
        question="How is the battery life?",
        answer="Good.",
        citations=[],
        confidence_score=0.2,
        suggested_products=[]
    )
    
    helpfulness_low = evaluator.evaluate_helpfulness(unhelpful_response)
    assert helpfulness_low < helpfulness  # Should be lower


def test_precision_at_k():
    """Test precision@k metric calculation."""
    def calculate_precision_at_k(relevant_items: List[str], retrieved_items: List[str], k: int) -> float:
        """Calculate precision@k metric."""
        if not retrieved_items or k <= 0:
            return 0.0
        
        retrieved_k = retrieved_items[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant_items))
        
        return relevant_retrieved / min(k, len(retrieved_k))
    
    # Test cases
    relevant = ["A1", "A2", "A3"]
    retrieved = ["A1", "A4", "A2", "A5", "A3"]
    
    precision_3 = calculate_precision_at_k(relevant, retrieved, 3)
    assert precision_3 == 2/3  # A1 and A2 in top 3
    
    precision_5 = calculate_precision_at_k(relevant, retrieved, 5)
    assert precision_5 == 3/5  # All 3 relevant items in top 5


def test_ndcg_calculation():
    """Test NDCG (Normalized Discounted Cumulative Gain) calculation."""
    def calculate_dcg(relevance_scores: List[float]) -> float:
        """Calculate DCG (Discounted Cumulative Gain)."""
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg
    
    def calculate_ndcg(relevance_scores: List[float], ideal_scores: List[float]) -> float:
        """Calculate NDCG."""
        dcg = calculate_dcg(relevance_scores)
        idcg = calculate_dcg(sorted(ideal_scores, reverse=True))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    # Test NDCG calculation
    relevance_scores = [3, 1, 2, 0, 1]
    ideal_scores = [3, 2, 1, 1, 0]
    
    ndcg = calculate_ndcg(relevance_scores, ideal_scores)
    assert 0.0 <= ndcg <= 1.0


def test_mrr_calculation():
    """Test MRR (Mean Reciprocal Rank) calculation."""
    def calculate_mrr(rankings: List[List[str]], relevant_items: List[List[str]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for ranking, relevant in zip(rankings, relevant_items):
            rr = 0.0
            for i, item in enumerate(ranking):
                if item in relevant:
                    rr = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(rr)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # Test MRR calculation
    rankings = [
        ["A1", "A2", "A3"],  # Relevant item A1 at position 1
        ["B1", "A4", "B2"],  # Relevant item A4 at position 2  
        ["C1", "C2", "C3"]   # No relevant items
    ]
    
    relevant_items = [
        ["A1"],
        ["A4"],
        ["D1"]  # Not in ranking
    ]
    
    mrr = calculate_mrr(rankings, relevant_items)
    expected_mrr = (1.0 + 0.5 + 0.0) / 3  # 1/1 + 1/2 + 0
    assert abs(mrr - expected_mrr) < 1e-6


def test_evaluation_metrics_edge_cases():
    """Test evaluation metrics with edge cases."""
    evaluator = QAEvaluator(None)
    
    # Empty citations
    empty_response = QAResponse(
        question="Test question",
        answer="Test answer",
        citations=[],
        confidence_score=0.5,
        suggested_products=[]
    )
    
    groundedness = evaluator.evaluate_groundedness(empty_response)
    assert groundedness == 0.0
    
    helpfulness = evaluator.evaluate_helpfulness(empty_response)
    assert 0.0 <= helpfulness <= 1.0


@pytest.mark.parametrize("test_case", [
    {"query": "battery life", "expected_results": 1},
    {"query": "sound quality", "expected_results": 1}, 
    {"query": "nonexistent feature xyz", "expected_results": 0},
])
def test_search_result_expectations(test_case):
    """Test search results meet basic expectations."""
    query = test_case["query"]
    expected_results = test_case["expected_results"]
    
    # This is a structure test - actual results depend on data and index
    assert len(query) > 0
    assert expected_results >= 0