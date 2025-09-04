"""
Product comparison engine for side-by-side analysis.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from src.config import settings
from src.models.schemas import (
    ComparisonRequest, ComparisonResponse, ProductResponse,
    Product, Review
)
from src.qa.llm_client import get_llm_client
from src.database import SessionLocal

logger = logging.getLogger(__name__)


class ProductComparator:
    """Handles side-by-side product comparisons."""
    
    def __init__(self):
        self.llm_client = get_llm_client()
        
        # Spec extraction patterns
        self.spec_patterns = {
            "cpu": r"(?:cpu|processor|chipset)[\s:]*([^,\n\.]+)",
            "gpu": r"(?:gpu|graphics|video)[\s:]*([^,\n\.]+)", 
            "ram": r"(?:ram|memory)[\s:]*(\d+\s*gb)",
            "storage": r"(?:storage|ssd|hdd)[\s:]*(\d+\s*(?:gb|tb))",
            "display": r"(?:display|screen|monitor)[\s:]*([^,\n\.]+)",
            "battery": r"(?:battery|mah)[\s:]*([^,\n\.]+)",
            "camera": r"(?:camera|mp|megapixel)[\s:]*([^,\n\.]+)",
            "weight": r"(?:weight|pounds|lbs|kg)[\s:]*([^,\n\.]+)",
            "dimensions": r"(?:dimensions|size)[\s:]*([^,\n\.]+)"
        }
    
    def compare_products(self, request: ComparisonRequest) -> ComparisonResponse:
        """
        Compare two products side by side.
        
        Args:
            request: Comparison request with two ASINs
            
        Returns:
            Detailed comparison response
        """
        asin1, asin2 = request.asin1, request.asin2
        
        logger.info(f"Comparing products: {asin1} vs {asin2}")
        
        # Get product details
        product1 = self._get_product_details(asin1)
        product2 = self._get_product_details(asin2)
        
        if not product1 or not product2:
            raise ValueError("One or both products not found")
        
        # Extract specifications
        specs1 = self._extract_specifications(product1)
        specs2 = self._extract_specifications(product2)
        
        # Get review summaries
        reviews1 = self._get_review_summary(asin1)
        reviews2 = self._get_review_summary(asin2)
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(
            product1, product2, specs1, specs2, reviews1, reviews2
        )
        
        # Identify key differences
        key_differences = self._identify_key_differences(
            product1, product2, specs1, specs2
        )
        
        # Determine winners by category
        winner_categories = self._determine_category_winners(
            product1, product2, specs1, specs2, reviews1, reviews2
        )
        
        return ComparisonResponse(
            product1=product1,
            product2=product2,
            comparison_summary=comparison_summary,
            key_differences=key_differences,
            winner_categories=winner_categories
        )
    
    def _get_product_details(self, asin: str) -> Optional[ProductResponse]:
        """Get detailed product information from database."""
        db = SessionLocal()
        try:
            product = db.query(Product).filter(Product.asin == asin).first()
            
            if not product:
                return None
            
            return ProductResponse(
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
            
        finally:
            db.close()
    
    def _extract_specifications(self, product: ProductResponse) -> Dict[str, str]:
        """Extract technical specifications from product description."""
        specs = {}
        
        if not product.description:
            return specs
        
        desc_lower = product.description.lower()
        
        for spec_name, pattern in self.spec_patterns.items():
            match = re.search(pattern, desc_lower, re.IGNORECASE)
            if match:
                specs[spec_name] = match.group(1).strip()
        
        # Add basic product info as specs
        specs["brand"] = product.brand or "Unknown"
        specs["price"] = f"${product.price:.2f}" if product.price else "N/A"
        specs["rating"] = f"{product.avg_rating:.1f}/5.0" if product.avg_rating else "N/A"
        specs["reviews"] = str(product.num_reviews or 0)
        
        return specs
    
    def _get_review_summary(self, asin: str) -> Dict[str, Any]:
        """Get summary of reviews for a product."""
        db = SessionLocal()
        try:
            reviews = db.query(Review).filter(Review.asin == asin).all()
            
            if not reviews:
                return {"count": 0, "avg_rating": 0, "summary": "No reviews available"}
            
            # Calculate review statistics
            ratings = [r.rating for r in reviews]
            avg_rating = sum(ratings) / len(ratings)
            
            # Get most helpful reviews
            helpful_reviews = sorted(reviews, key=lambda x: x.helpful_votes, reverse=True)[:3]
            
            # Extract common themes (simplified)
            all_text = " ".join([r.review_text for r in helpful_reviews])
            positive_keywords = ["good", "great", "excellent", "love", "amazing", "perfect"]
            negative_keywords = ["bad", "poor", "terrible", "hate", "awful", "worst"]
            
            positive_count = sum(all_text.lower().count(word) for word in positive_keywords)
            negative_count = sum(all_text.lower().count(word) for word in negative_keywords)
            
            sentiment = "positive" if positive_count > negative_count else "mixed" if positive_count == negative_count else "negative"
            
            return {
                "count": len(reviews),
                "avg_rating": avg_rating,
                "sentiment": sentiment,
                "helpful_reviews": [r.review_text[:200] + "..." for r in helpful_reviews],
                "positive_mentions": positive_count,
                "negative_mentions": negative_count
            }
            
        finally:
            db.close()
    
    def _generate_comparison_summary(self, product1: ProductResponse, product2: ProductResponse,
                                   specs1: Dict[str, str], specs2: Dict[str, str],
                                   reviews1: Dict[str, Any], reviews2: Dict[str, Any]) -> str:
        """Generate natural language comparison summary using LLM."""
        
        # Build comparison prompt
        prompt = f"""Compare these two products and provide a balanced summary:

PRODUCT 1: {product1.title}
- Brand: {product1.brand}
- Price: ${product1.price or 'N/A'}
- Rating: {product1.avg_rating or 'N/A'}/5.0 ({product1.num_reviews or 0} reviews)
- Key specs: {', '.join([f'{k}: {v}' for k, v in specs1.items() if k not in ['brand', 'price', 'rating', 'reviews']])}

PRODUCT 2: {product2.title}  
- Brand: {product2.brand}
- Price: ${product2.price or 'N/A'}
- Rating: {product2.avg_rating or 'N/A'}/5.0 ({product2.num_reviews or 0} reviews)
- Key specs: {', '.join([f'{k}: {v}' for k, v in specs2.items() if k not in ['brand', 'price', 'rating', 'reviews']])}

REVIEW INSIGHTS:
Product 1: {reviews1.get('sentiment', 'unknown')} sentiment, {reviews1.get('count', 0)} reviews
Product 2: {reviews2.get('sentiment', 'unknown')} sentiment, {reviews2.get('count', 0)} reviews

Provide a 2-3 sentence summary comparing these products, highlighting key differences and which might be better for different use cases."""
        
        try:
            summary = self.llm_client.generate_response(prompt, max_tokens=300)
            return summary
        except Exception as e:
            logger.error(f"Failed to generate comparison summary: {e}")
            return f"Comparison between {product1.title} and {product2.title}. Both products have different strengths - please review the detailed specifications below."
    
    def _identify_key_differences(self, product1: ProductResponse, product2: ProductResponse,
                                 specs1: Dict[str, str], specs2: Dict[str, str]) -> List[str]:
        """Identify key differences between products."""
        differences = []
        
        # Price difference
        if product1.price and product2.price:
            price_diff = abs(product1.price - product2.price)
            if price_diff > 20:  # Significant price difference
                cheaper = product1.title if product1.price < product2.price else product2.title
                differences.append(f"Price: {cheaper} is significantly cheaper (${price_diff:.2f} difference)")
        
        # Rating difference
        if product1.avg_rating and product2.avg_rating:
            rating_diff = abs(product1.avg_rating - product2.avg_rating)
            if rating_diff > 0.5:
                higher_rated = product1.title if product1.avg_rating > product2.avg_rating else product2.title
                differences.append(f"Rating: {higher_rated} has higher customer ratings")
        
        # Brand difference
        if product1.brand != product2.brand:
            differences.append(f"Brand: {product1.brand} vs {product2.brand}")
        
        # Spec differences
        common_specs = set(specs1.keys()) & set(specs2.keys())
        for spec in common_specs:
            if specs1[spec] != specs2[spec] and spec not in ['brand', 'price', 'rating', 'reviews']:
                differences.append(f"{spec.title()}: {specs1[spec]} vs {specs2[spec]}")
        
        return differences[:5]  # Limit to top 5 differences
    
    def _determine_category_winners(self, product1: ProductResponse, product2: ProductResponse,
                                  specs1: Dict[str, str], specs2: Dict[str, str],
                                  reviews1: Dict[str, Any], reviews2: Dict[str, Any]) -> Dict[str, str]:
        """Determine which product wins in different categories."""
        winners = {}
        
        # Price winner (lower price wins)
        if product1.price and product2.price:
            winners["price"] = product1.asin if product1.price < product2.price else product2.asin
        
        # Rating winner
        if product1.avg_rating and product2.avg_rating:
            winners["rating"] = product1.asin if product1.avg_rating > product2.avg_rating else product2.asin
        
        # Review count winner
        if product1.num_reviews and product2.num_reviews:
            winners["popularity"] = product1.asin if product1.num_reviews > product2.num_reviews else product2.asin
        
        # Review sentiment winner
        sentiment1 = reviews1.get("sentiment", "unknown")
        sentiment2 = reviews2.get("sentiment", "unknown")
        
        if sentiment1 == "positive" and sentiment2 != "positive":
            winners["user_satisfaction"] = product1.asin
        elif sentiment2 == "positive" and sentiment1 != "positive":
            winners["user_satisfaction"] = product2.asin
        
        return winners


def main():
    """Test product comparison functionality."""
    comparator = ProductComparator()
    
    # This would need actual ASINs from the database
    # For demo purposes, we'll show the interface
    print("Product Comparison Engine initialized")
    print("Usage: comparator.compare_products(ComparisonRequest(asin1='...', asin2='...'))")
    
    # Example comparison (would work with real data)
    try:
        # Get some sample ASINs from database
        db = SessionLocal()
        products = db.query(Product).limit(2).all()
        db.close()
        
        if len(products) >= 2:
            request = ComparisonRequest(asin1=products[0].asin, asin2=products[1].asin)
            response = comparator.compare_products(request)
            
            print(f"\nComparison: {response.product1.title} vs {response.product2.title}")
            print(f"Summary: {response.comparison_summary}")
            print(f"Key differences: {len(response.key_differences)}")
            print(f"Category winners: {response.winner_categories}")
        
    except Exception as e:
        logger.info(f"Demo comparison not available: {e}")


if __name__ == "__main__":
    main()