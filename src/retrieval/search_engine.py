"""
Complete retrieval and reranking pipeline.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
from sqlalchemy.orm import Session

from src.config import settings
from src.models.schemas import ChunkMetadata, RetrievalResult, Product
from src.indexing.faiss_index import VectorSearchEngine
from src.indexing.embeddings import CrossEncoderReranker
from src.database import SessionLocal

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Complete retrieval pipeline with bi-encoder recall and cross-encoder reranking."""
    
    def __init__(self):
        self.vector_search = VectorSearchEngine()
        self.cross_encoder = CrossEncoderReranker()
        self._load_search_components()
    
    def _load_search_components(self):
        """Load pre-built search index and models."""
        try:
            self.vector_search.load_index()
            logger.info("Search components loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Search index not found: {e}")
            logger.warning("Please build the index first: python src/indexing/build_index.py")
    
    def search_and_rerank(self, query: str, top_k: int = None, 
                         category_filter: Optional[str] = None,
                         min_rating: Optional[float] = None) -> List[RetrievalResult]:
        """
        Complete search pipeline: retrieval → reranking → filtering.
        
        Args:
            query: Search query
            top_k: Final number of results to return
            category_filter: Optional category filter
            min_rating: Optional minimum rating filter
            
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or settings.rerank_top_k
        
        # Step 1: Bi-encoder retrieval (top-50)
        similar_chunks = self.vector_search.search_similar_chunks(
            query, top_k=settings.retrieval_top_k
        )
        
        if not similar_chunks:
            logger.warning(f"No results found for query: {query}")
            return []
        
        # Step 2: Apply filters
        filtered_chunks = self._apply_filters(
            similar_chunks, category_filter, min_rating
        )
        
        # Step 3: Cross-encoder reranking
        reranked_chunks = self.cross_encoder.rerank_chunks(
            query, filtered_chunks, top_k=top_k
        )
        
        # Step 4: Convert to RetrievalResult format
        results = []
        for chunk, score in reranked_chunks:
            result = RetrievalResult(
                chunk_id=chunk.chunk_id,
                asin=chunk.asin,
                text=chunk.text,
                score=float(score),
                chunk_type=chunk.type,
                helpful_votes=chunk.helpful_votes,
                rating=chunk.rating
            )
            results.append(result)
        
        return results
    
    def _apply_filters(self, chunks: List[ChunkMetadata], 
                      category_filter: Optional[str] = None,
                      min_rating: Optional[float] = None) -> List[ChunkMetadata]:
        """Apply category and rating filters to chunks."""
        if not category_filter and not min_rating:
            return chunks
        
        filtered_chunks = []
        
        # Get product information for filtering
        db = SessionLocal()
        try:
            for chunk in chunks:
                # Check category filter
                if category_filter:
                    product = db.query(Product).filter(Product.asin == chunk.asin).first()
                    if not product or category_filter.lower() not in product.category.lower():
                        continue
                
                # Check rating filter
                if min_rating and chunk.rating and chunk.rating < min_rating:
                    continue
                
                filtered_chunks.append(chunk)
        finally:
            db.close()
        
        return filtered_chunks
    
    def get_product_recommendations(self, asin: str, top_k: int = 6) -> List[str]:
        """
        Get product recommendations based on similarity.
        
        Args:
            asin: Source product ASIN
            top_k: Number of recommendations
            
        Returns:
            List of recommended product ASINs
        """
        # Get chunks for the source product
        source_chunks = [chunk for chunk in self.vector_search.index_manager.chunks_metadata 
                        if chunk.asin == asin]
        
        if not source_chunks:
            logger.warning(f"No chunks found for product: {asin}")
            return []
        
        # Use the first chunk as representative
        source_text = source_chunks[0].text
        
        # Search for similar chunks
        similar_chunks = self.vector_search.search_similar_chunks(source_text, top_k=top_k*3)
        
        # Group by ASIN and exclude source product
        asin_scores = defaultdict(list)
        for chunk in similar_chunks:
            if chunk.asin != asin:  # Exclude source product
                score = getattr(chunk, 'similarity_score', 0)
                asin_scores[chunk.asin].append(score)
        
        # Calculate average score per product
        product_scores = []
        for product_asin, scores in asin_scores.items():
            avg_score = sum(scores) / len(scores)
            product_scores.append((product_asin, avg_score))
        
        # Sort by score and return top-k
        product_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_asins = [asin for asin, score in product_scores[:top_k]]
        
        return recommended_asins
    
    def search_products_by_query(self, query: str, top_k: int = 10) -> List[str]:
        """
        Search for products matching a query.
        
        Args:
            query: Search query
            top_k: Number of products to return
            
        Returns:
            List of product ASINs
        """
        # Get relevant chunks
        chunks = self.search_and_rerank(query, top_k=top_k*2)
        
        # Group by ASIN and get unique products
        seen_asins = set()
        product_asins = []
        
        for chunk in chunks:
            if chunk.asin not in seen_asins:
                seen_asins.add(chunk.asin)
                product_asins.append(chunk.asin)
                
                if len(product_asins) >= top_k:
                    break
        
        return product_asins
    
    def get_chunks_for_qa(self, query: str, max_chunks: int = 5) -> List[ChunkMetadata]:
        """
        Get high-quality chunks for Q&A with citations.
        
        Args:
            query: Question to answer
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of best chunks for answering the question
        """
        # Search and rerank
        results = self.search_and_rerank(query, top_k=max_chunks)
        
        # Convert back to ChunkMetadata with scores
        qa_chunks = []
        for result in results:
            # Find the original chunk
            for chunk in self.vector_search.index_manager.chunks_metadata:
                if chunk.chunk_id == result.chunk_id:
                    # Add the reranking score
                    chunk.rerank_score = result.score
                    qa_chunks.append(chunk)
                    break
        
        return qa_chunks


def main():
    """Test the complete retrieval pipeline."""
    engine = RetrievalEngine()
    
    test_queries = [
        "battery life performance",
        "sound quality review", 
        "easy to setup",
        "good value for money"
    ]
    
    for query in test_queries:
        print(f"\n=== Query: '{query}' ===")
        
        # Test search and rerank
        results = engine.search_and_rerank(query, top_k=3)
        print(f"Search results ({len(results)}):")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result.score:.3f} | {result.text[:100]}...")
        
        # Test product recommendations (if we have products)
        if results:
            sample_asin = results[0].asin
            recommendations = engine.get_product_recommendations(sample_asin, top_k=3)
            print(f"Recommendations for {sample_asin}: {recommendations}")


if __name__ == "__main__":
    main()