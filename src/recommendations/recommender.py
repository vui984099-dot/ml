"""
Product recommendation engine with multiple recommendation strategies.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from src.config import settings
from src.models.schemas import (
    RecommendationRequest, RecommendationResponse, ProductResponse,
    Product, Review, ChunkMetadata
)
from src.retrieval.search_engine import RetrievalEngine
from src.database import SessionLocal

logger = logging.getLogger(__name__)


class ProductRecommender:
    """Multi-strategy product recommendation system."""
    
    def __init__(self):
        self.retrieval_engine = RetrievalEngine()
    
    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Get product recommendations based on request type.
        
        Args:
            request: Recommendation request with parameters
            
        Returns:
            Recommendation response with products and scores
        """
        if request.asin:
            return self._get_similar_products(request)
        elif request.query:
            return self._get_query_based_recommendations(request)
        else:
            return self._get_popular_products(request)
    
    def _get_similar_products(self, request: RecommendationRequest) -> RecommendationResponse:
        """Get products similar to a given product."""
        asin = request.asin
        top_k = request.top_k
        exclude_asins = set(request.exclude_asins + [asin])
        
        logger.info(f"Getting similar products for ASIN: {asin}")
        
        # Get similar product ASINs using vector similarity
        similar_asins = self.retrieval_engine.get_product_recommendations(asin, top_k * 2)
        
        # Filter out excluded products
        filtered_asins = [a for a in similar_asins if a not in exclude_asins][:top_k]
        
        # Get product details
        products = self._get_products_by_asins(filtered_asins)
        
        # Calculate similarity scores (mock for now)
        scores = [0.9 - (i * 0.1) for i in range(len(products))]
        
        return RecommendationResponse(
            products=products,
            recommendation_type="similar",
            scores=scores
        )
    
    def _get_query_based_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Get products matching a search query."""
        query = request.query
        top_k = request.top_k
        exclude_asins = set(request.exclude_asins)
        
        logger.info(f"Getting query-based recommendations for: {query}")
        
        # Search for relevant products
        product_asins = self.retrieval_engine.search_products_by_query(query, top_k * 2)
        
        # Filter out excluded products
        filtered_asins = [a for a in product_asins if a not in exclude_asins][:top_k]
        
        # Get product details
        products = self._get_products_by_asins(filtered_asins)
        
        # Calculate relevance scores based on query matching
        scores = self._calculate_query_relevance_scores(query, products)
        
        return RecommendationResponse(
            products=products,
            recommendation_type="query_based",
            scores=scores
        )
    
    def _get_popular_products(self, request: RecommendationRequest) -> RecommendationResponse:
        """Get popular products based on ratings and review counts."""
        top_k = request.top_k
        exclude_asins = set(request.exclude_asins)
        
        logger.info("Getting popular products")
        
        db = SessionLocal()
        try:
            # Query for popular products
            products_query = db.query(Product).filter(
                and_(
                    Product.avg_rating >= 4.0,
                    Product.num_reviews >= 10,
                    ~Product.asin.in_(exclude_asins)
                )
            ).order_by(
                Product.avg_rating.desc(),
                Product.num_reviews.desc()
            ).limit(top_k)
            
            products_orm = products_query.all()
            
            # Convert to response format
            products = []
            scores = []
            
            for product in products_orm:
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
                products.append(product_response)
                
                # Calculate popularity score
                rating_score = product.avg_rating / 5.0
                review_score = min(product.num_reviews / 1000.0, 1.0)
                popularity_score = (rating_score * 0.7 + review_score * 0.3)
                scores.append(popularity_score)
            
            return RecommendationResponse(
                products=products,
                recommendation_type="popular",
                scores=scores
            )
            
        finally:
            db.close()
    
    def _get_products_by_asins(self, asins: List[str]) -> List[ProductResponse]:
        """Fetch product details by ASINs from database."""
        if not asins:
            return []
        
        db = SessionLocal()
        try:
            products_orm = db.query(Product).filter(Product.asin.in_(asins)).all()
            
            # Maintain order of input ASINs
            asin_to_product = {p.asin: p for p in products_orm}
            
            products = []
            for asin in asins:
                if asin in asin_to_product:
                    product = asin_to_product[asin]
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
                    products.append(product_response)
            
            return products
            
        finally:
            db.close()
    
    def _calculate_query_relevance_scores(self, query: str, products: List[ProductResponse]) -> List[float]:
        """Calculate relevance scores for products based on query."""
        if not products:
            return []
        
        query_lower = query.lower()
        scores = []
        
        for product in products:
            score = 0.0
            
            # Title relevance
            title_words = product.title.lower().split()
            query_words = query_lower.split()
            title_matches = sum(1 for word in query_words if any(word in title_word for title_word in title_words))
            title_score = title_matches / len(query_words) if query_words else 0
            
            # Description relevance
            desc_score = 0.0
            if product.description:
                desc_lower = product.description.lower()
                desc_matches = sum(1 for word in query_words if word in desc_lower)
                desc_score = desc_matches / len(query_words) if query_words else 0
            
            # Brand relevance
            brand_score = 0.0
            if product.brand and any(word in product.brand.lower() for word in query_words):
                brand_score = 0.2
            
            # Combine scores
            relevance_score = (title_score * 0.6 + desc_score * 0.3 + brand_score * 0.1)
            
            # Boost by rating
            rating_boost = (product.avg_rating or 3.0) / 5.0
            final_score = relevance_score * 0.8 + rating_boost * 0.2
            
            scores.append(final_score)
        
        return scores
    
    def get_collaborative_recommendations(self, user_asins: List[str], top_k: int = 6) -> List[str]:
        """
        Get collaborative filtering recommendations based on user's product history.
        
        Args:
            user_asins: List of ASINs the user has interacted with
            top_k: Number of recommendations
            
        Returns:
            List of recommended ASINs
        """
        if not user_asins:
            return []
        
        # Find products frequently bought/viewed together
        db = SessionLocal()
        try:
            # Get reviews from users who reviewed the input products
            user_ids_query = db.query(Review.reviewer_id).filter(
                Review.asin.in_(user_asins)
            ).distinct()
            
            user_ids = [row[0] for row in user_ids_query.all()]
            
            if not user_ids:
                return []
            
            # Find other products these users liked (rating >= 4)
            other_products_query = db.query(Review.asin).filter(
                and_(
                    Review.reviewer_id.in_(user_ids),
                    Review.rating >= 4,
                    ~Review.asin.in_(user_asins)
                )
            )
            
            other_asins = [row[0] for row in other_products_query.all()]
            
            # Count frequency and return most common
            asin_counts = Counter(other_asins)
            recommended_asins = [asin for asin, count in asin_counts.most_common(top_k)]
            
            return recommended_asins
            
        finally:
            db.close()


def main():
    """Test recommendation engine."""
    recommender = ProductRecommender()
    
    # Test different recommendation types
    test_cases = [
        {
            "type": "popular",
            "request": RecommendationRequest(top_k=3)
        },
        {
            "type": "query_based", 
            "request": RecommendationRequest(query="wireless headphones", top_k=3)
        }
    ]
    
    for test_case in test_cases:
        print(f"\n=== {test_case['type'].upper()} RECOMMENDATIONS ===")
        
        response = recommender.get_recommendations(test_case["request"])
        
        print(f"Type: {response.recommendation_type}")
        print(f"Found {len(response.products)} products:")
        
        for i, (product, score) in enumerate(zip(response.products, response.scores)):
            print(f"{i+1}. {product.title} | Score: {score:.3f} | Rating: {product.avg_rating}")


if __name__ == "__main__":
    main()