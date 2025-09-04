"""
Question Answering engine with retrieval-augmented generation.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from src.config import settings
from src.models.schemas import (
    ChunkMetadata, QARequest, QAResponse, Citation, 
    ProductResponse, Product
)
from src.retrieval.search_engine import RetrievalEngine
from src.qa.llm_client import get_llm_client, LLMClient
from src.database import SessionLocal

logger = logging.getLogger(__name__)


class QAEngine:
    """Retrieval-Augmented Generation engine for product Q&A."""
    
    def __init__(self):
        self.retrieval_engine = RetrievalEngine()
        self.llm_client = get_llm_client()
        
    def answer_question(self, request: QARequest) -> QAResponse:
        """
        Answer a product question using RAG.
        
        Args:
            request: QA request with question and filters
            
        Returns:
            QA response with answer, citations, and suggestions
        """
        question = request.question
        max_chunks = request.max_chunks
        
        logger.info(f"Answering question: {question}")
        
        # Step 1: Retrieve relevant chunks
        chunks = self.retrieval_engine.get_chunks_for_qa(question, max_chunks)
        
        if not chunks:
            return self._create_no_results_response(question)
        
        # Step 2: Build prompt with sources
        prompt = self._build_qa_prompt(question, chunks)
        
        # Step 3: Generate answer with LLM
        raw_answer = self.llm_client.generate_response(prompt, max_tokens=600)
        
        # Step 4: Process answer and extract citations
        processed_answer, citations = self._process_answer(raw_answer, chunks)
        
        # Step 5: Calculate confidence score
        confidence_score = self._calculate_confidence(chunks, processed_answer)
        
        # Step 6: Get suggested products
        suggested_products = self._get_suggested_products(chunks)
        
        return QAResponse(
            question=question,
            answer=processed_answer,
            citations=citations,
            confidence_score=confidence_score,
            suggested_products=suggested_products
        )
    
    def _build_qa_prompt(self, question: str, chunks: List[ChunkMetadata]) -> str:
        """Build the RAG prompt with sources and instructions."""
        
        # Build sources section
        sources_text = "SOURCES:\n"
        for i, chunk in enumerate(chunks, 1):
            rating_text = f", rating: {chunk.rating}" if chunk.rating else ""
            helpful_text = f", helpful_votes: {chunk.helpful_votes}" if chunk.helpful_votes > 0 else ""
            
            sources_text += f"[{i}] asin: {chunk.asin}{rating_text}{helpful_text}: \"{chunk.text}\"\n\n"
        
        prompt = f"""You are an assistant that answers product questions using ONLY the provided source chunks from Amazon reviews and product descriptions.

{sources_text}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY information from the sources above
2. Provide a balanced and honest assessment
3. Include specific evidence with source citations [1], [2], etc.
4. If sources conflict, acknowledge the disagreement
5. If you cannot answer based on the sources, say so clearly
6. Do not add information not present in the sources

ANSWER FORMAT:
- Start with a direct answer (1-2 sentences)
- Follow with evidence bullets citing source numbers
- End with any important caveats or limitations

ANSWER:"""
        
        return prompt
    
    def _process_answer(self, raw_answer: str, chunks: List[ChunkMetadata]) -> Tuple[str, List[Citation]]:
        """
        Process LLM answer to extract citations and clean up text.
        
        Returns:
            Tuple of (processed_answer, citations_list)
        """
        # Extract citations from answer
        citations = []
        citation_pattern = r'\[(\d+)\]'
        cited_indices = set(re.findall(citation_pattern, raw_answer))
        
        for idx_str in cited_indices:
            try:
                idx = int(idx_str) - 1  # Convert to 0-based index
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    citation = Citation(
                        asin=chunk.asin,
                        review_id=None,  # Could be enhanced to track review IDs
                        chunk_id=chunk.chunk_id,
                        text_snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                        rating=chunk.rating,
                        helpful_votes=chunk.helpful_votes
                    )
                    citations.append(citation)
            except (ValueError, IndexError):
                continue
        
        # Clean up the answer text
        processed_answer = raw_answer.strip()
        
        # Add warning for unverified claims if no citations found
        if not citations and len(processed_answer) > 50:
            processed_answer += "\n\n[Note: This response may contain unverified information]"
        
        return processed_answer, citations
    
    def _calculate_confidence(self, chunks: List[ChunkMetadata], answer: str) -> float:
        """
        Calculate confidence score based on source quality and answer content.
        
        Args:
            chunks: Source chunks used
            answer: Generated answer
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not chunks:
            return 0.0
        
        # Base confidence on source quality
        avg_helpful_votes = sum(chunk.helpful_votes for chunk in chunks) / len(chunks)
        avg_rating = sum(chunk.rating or 3 for chunk in chunks) / len(chunks)
        
        # Normalize helpful votes (assume max ~20 for normalization)
        helpful_score = min(avg_helpful_votes / 20.0, 1.0)
        
        # Normalize rating (1-5 scale)
        rating_score = (avg_rating - 1) / 4.0
        
        # Check for citation markers in answer
        citation_count = len(re.findall(r'\[\d+\]', answer))
        citation_score = min(citation_count / len(chunks), 1.0)
        
        # Combine scores
        confidence = (helpful_score * 0.3 + rating_score * 0.3 + citation_score * 0.4)
        
        return min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    def _get_suggested_products(self, chunks: List[ChunkMetadata]) -> List[ProductResponse]:
        """Get suggested products based on the chunks used in Q&A."""
        if not chunks:
            return []
        
        # Get unique ASINs from chunks
        unique_asins = list(set(chunk.asin for chunk in chunks))
        
        # Query database for product details
        db = SessionLocal()
        try:
            products = db.query(Product).filter(Product.asin.in_(unique_asins)).limit(5).all()
            
            suggested_products = []
            for product in products:
                product_response = ProductResponse(
                    asin=product.asin,
                    title=product.title,
                    brand=product.brand,
                    category=product.category,
                    price=product.price,
                    description=product.description,
                    image_url=product.image_url,
                    avg_rating=product.avg_rating,
                    num_reviews=product.num_reviews,
                    created_at=product.created_at
                )
                suggested_products.append(product_response)
            
            return suggested_products
            
        except Exception as e:
            logger.error(f"Error fetching suggested products: {e}")
            return []
        finally:
            db.close()
    
    def _create_no_results_response(self, question: str) -> QAResponse:
        """Create response when no relevant chunks are found."""
        return QAResponse(
            question=question,
            answer="I don't have enough information in the available reviews and product descriptions to answer your question. Please try rephrasing your question or asking about a different aspect of the products.",
            citations=[],
            confidence_score=0.0,
            suggested_products=[]
        )
    
    def batch_answer_questions(self, questions: List[str]) -> List[QAResponse]:
        """Answer multiple questions efficiently."""
        responses = []
        
        for question in questions:
            request = QARequest(question=question)
            response = self.answer_question(request)
            responses.append(response)
        
        return responses


class QAEvaluator:
    """Evaluates Q&A system performance."""
    
    def __init__(self, qa_engine: QAEngine):
        self.qa_engine = qa_engine
    
    def evaluate_groundedness(self, qa_response: QAResponse) -> float:
        """
        Evaluate how well the answer is grounded in the provided sources.
        
        Returns:
            Groundedness score between 0.0 and 1.0
        """
        answer = qa_response.answer
        citations = qa_response.citations
        
        if not citations:
            return 0.0
        
        # Simple heuristic: check if answer contains information from citations
        citation_texts = [citation.text_snippet.lower() for citation in citations]
        answer_lower = answer.lower()
        
        # Count how many citation concepts appear in the answer
        matches = 0
        total_concepts = 0
        
        for citation_text in citation_texts:
            words = citation_text.split()
            for i in range(len(words) - 2):  # Check 3-word phrases
                phrase = " ".join(words[i:i+3])
                total_concepts += 1
                if phrase in answer_lower:
                    matches += 1
        
        if total_concepts == 0:
            return 0.0
        
        return matches / total_concepts
    
    def evaluate_helpfulness(self, qa_response: QAResponse, expected_topics: List[str] = None) -> float:
        """
        Evaluate how helpful the answer is.
        
        Args:
            qa_response: The QA response to evaluate
            expected_topics: Optional list of topics that should be covered
            
        Returns:
            Helpfulness score between 0.0 and 1.0
        """
        answer = qa_response.answer.lower()
        
        # Check for key indicators of helpful answers
        helpful_indicators = [
            "based on", "according to", "evidence", "reviews show",
            "users report", "customers mention", "however", "but",
            "overall", "in summary", "caveats", "limitations"
        ]
        
        indicator_score = sum(1 for indicator in helpful_indicators if indicator in answer)
        indicator_score = min(indicator_score / len(helpful_indicators), 1.0)
        
        # Check answer length (not too short, not too long)
        length_score = 1.0
        if len(answer) < 50:
            length_score = 0.3
        elif len(answer) > 1000:
            length_score = 0.7
        
        # Check for citations
        citation_score = min(len(qa_response.citations) / 3.0, 1.0)
        
        # Combine scores
        helpfulness = (indicator_score * 0.4 + length_score * 0.3 + citation_score * 0.3)
        
        return helpfulness


def main():
    """Test the Q&A engine."""
    qa_engine = QAEngine()
    
    test_questions = [
        "How is the battery life?",
        "Is the sound quality good?", 
        "Is this product easy to set up?",
        "What do customers say about the build quality?",
        "Is this good value for money?"
    ]
    
    for question in test_questions:
        print(f"\n=== Question: {question} ===")
        
        request = QARequest(question=question, max_chunks=3)
        response = qa_engine.answer_question(request)
        
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Citations: {len(response.citations)}")
        print(f"Suggested Products: {len(response.suggested_products)}")
        
        if response.citations:
            print("Citations:")
            for i, citation in enumerate(response.citations, 1):
                print(f"  {i}. ASIN: {citation.asin} | Votes: {citation.helpful_votes}")


if __name__ == "__main__":
    main()