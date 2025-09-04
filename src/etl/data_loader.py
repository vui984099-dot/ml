"""
Data loading utilities for Amazon reviews dataset from HuggingFace.
"""

import os
import re
import pandas as pd
from typing import Dict, List, Optional, Iterator, Tuple
from datasets import load_dataset, Dataset
from datetime import datetime
import logging
from pathlib import Path

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmazonDataLoader:
    """Handles loading and initial processing of Amazon reviews dataset."""
    
    def __init__(self):
        self.dataset_name = settings.dataset_name
        self.target_category = settings.target_category
        self.max_products = settings.max_products
        self.max_reviews = settings.max_reviews
        self.min_helpful_votes = settings.min_helpful_votes
        
    def load_raw_dataset(self, subset: str = "raw_review_All_Beauty") -> Dataset:
        """Load raw dataset from HuggingFace."""
        try:
            logger.info(f"Loading dataset {self.dataset_name}, subset: {subset}")
            dataset = load_dataset(
                self.dataset_name, 
                subset,
                split="full",
                streaming=True
            )
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            # Fallback to a smaller subset for demo
            logger.info("Falling back to demo data generation")
            return self._generate_demo_data()
    
    def _generate_demo_data(self) -> List[Dict]:
        """Generate demo data for testing when HF dataset is unavailable."""
        demo_products = [
            {
                "asin": "B08N5WRWNW",
                "title": "Echo Dot (4th Gen) | Smart speaker with Alexa",
                "brand": "Amazon",
                "category": ["Electronics", "Smart Home"],
                "price": 49.99,
                "description": "Meet Echo Dot - Our most popular smart speaker with a fabric design. It is our most compact smart speaker that fits perfectly into small spaces.",
                "image": "https://m.media-amazon.com/images/I/61s2vZk5SWL._AC_SX522_.jpg",
                "average_rating": 4.7,
                "rating_number": 15420
            },
            {
                "asin": "B08F7PTF53", 
                "title": "Fire TV Stick 4K Max streaming device",
                "brand": "Amazon",
                "category": ["Electronics", "TV & Video"],
                "price": 54.99,
                "description": "Fire TV Stick 4K Max streaming device, supports Wi-Fi 6, Dolby Vision, HDR, HDR10+, and Dolby Atmos audio.",
                "image": "https://m.media-amazon.com/images/I/51TjJOTfslL._AC_SX522_.jpg",
                "average_rating": 4.6,
                "rating_number": 8932
            }
        ]
        
        demo_reviews = [
            {
                "asin": "B08N5WRWNW",
                "review_id": "R1234567890",
                "review_body": "Great little device! The sound quality is surprisingly good for such a small speaker. Alexa responds quickly and accurately. Setup was easy and it connects well to my other smart home devices.",
                "rating": 5,
                "helpful_vote": 12,
                "timestamp": 1640995200,  # 2022-01-01
                "user_id": "user123"
            },
            {
                "asin": "B08N5WRWNW", 
                "review_id": "R1234567891",
                "review_body": "Good value for money. The size is perfect for my nightstand. However, the bass could be better. Alexa works fine but sometimes has trouble with my accent.",
                "rating": 4,
                "helpful_vote": 8,
                "timestamp": 1641081600,  # 2022-01-02
                "user_id": "user456"
            },
            {
                "asin": "B08F7PTF53",
                "review_id": "R1234567892", 
                "review_body": "Excellent streaming quality! 4K content looks amazing and the interface is smooth. WiFi 6 support makes a real difference in my house. Highly recommended for anyone upgrading their TV.",
                "rating": 5,
                "helpful_vote": 15,
                "timestamp": 1641168000,  # 2022-01-03
                "user_id": "user789"
            }
        ]
        
        return {"products": demo_products, "reviews": demo_reviews}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short or very long content
        if len(text) < 10 or len(text) > 5000:
            return ""
            
        return text
    
    def parse_price(self, price_str: str) -> Optional[float]:
        """Parse price from string format."""
        if not price_str:
            return None
            
        # Extract numeric value from price string
        price_match = re.search(r'[\d,]+\.?\d*', str(price_str))
        if price_match:
            try:
                price = float(price_match.group().replace(',', ''))
                return price if 0 < price < 10000 else None
            except ValueError:
                return None
        return None
    
    def normalize_category(self, categories: List[str]) -> str:
        """Normalize category information."""
        if not categories:
            return "Unknown"
            
        # Take the most specific category (usually the last one)
        main_category = categories[-1] if isinstance(categories, list) else str(categories)
        
        # Clean up category name
        main_category = re.sub(r'[^\w\s]', '', main_category).strip()
        
        return main_category or "Unknown"
    
    def process_products(self, raw_data: Dict) -> pd.DataFrame:
        """Process raw product data into normalized DataFrame."""
        products = []
        
        if "products" in raw_data:
            # Demo data format
            for item in raw_data["products"]:
                product = {
                    "asin": item["asin"],
                    "title": self.clean_text(item["title"]),
                    "brand": item.get("brand", "Unknown"),
                    "category": self.normalize_category(item.get("category", ["Electronics"])),
                    "price": item.get("price"),
                    "description": self.clean_text(item.get("description", "")),
                    "image_url": item.get("image", ""),
                    "avg_rating": item.get("average_rating", 0.0),
                    "num_reviews": item.get("rating_number", 0),
                    "created_at": datetime.utcnow()
                }
                products.append(product)
        else:
            # HuggingFace dataset format (when available)
            for item in raw_data:
                if len(products) >= self.max_products:
                    break
                    
                product = {
                    "asin": item.get("parent_asin", item.get("asin", "")),
                    "title": self.clean_text(item.get("title", "")),
                    "brand": item.get("brand", "Unknown"),
                    "category": self.normalize_category(item.get("categories", ["Electronics"])),
                    "price": self.parse_price(item.get("price", "")),
                    "description": self.clean_text(item.get("description", "")),
                    "image_url": item.get("images", [""])[0] if item.get("images") else "",
                    "avg_rating": float(item.get("average_rating", 0)),
                    "num_reviews": int(item.get("rating_number", 0)),
                    "created_at": datetime.utcnow()
                }
                
                # Filter out products with insufficient data
                if product["title"] and product["asin"]:
                    products.append(product)
        
        df = pd.DataFrame(products)
        logger.info(f"Processed {len(df)} products")
        return df
    
    def process_reviews(self, raw_data: Dict) -> pd.DataFrame:
        """Process raw review data into normalized DataFrame."""
        reviews = []
        
        if "reviews" in raw_data:
            # Demo data format
            for item in raw_data["reviews"]:
                review = {
                    "review_id": item["review_id"],
                    "asin": item["asin"],
                    "review_text": self.clean_text(item["review_body"]),
                    "rating": int(item["rating"]),
                    "helpful_votes": int(item.get("helpful_vote", 0)),
                    "review_date": datetime.fromtimestamp(item.get("timestamp", 0)),
                    "reviewer_id": item.get("user_id", "")
                }
                reviews.append(review)
        else:
            # HuggingFace dataset format
            for item in raw_data:
                if len(reviews) >= self.max_reviews:
                    break
                    
                review_text = self.clean_text(item.get("text", ""))
                if not review_text or len(review_text) < 20:
                    continue
                
                helpful_votes = int(item.get("helpful_vote", 0))
                if helpful_votes < self.min_helpful_votes:
                    continue
                    
                review = {
                    "review_id": item.get("review_id", f"R{len(reviews)}"),
                    "asin": item.get("parent_asin", item.get("asin", "")),
                    "review_text": review_text,
                    "rating": int(item.get("rating", 3)),
                    "helpful_votes": helpful_votes,
                    "review_date": datetime.fromtimestamp(item.get("timestamp", 0)),
                    "reviewer_id": item.get("user_id", "")
                }
                reviews.append(review)
        
        df = pd.DataFrame(reviews)
        logger.info(f"Processed {len(df)} reviews")
        return df
    
    def get_top_helpful_reviews(self, reviews_df: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
        """Get top helpful reviews per product."""
        top_reviews = (reviews_df
                      .groupby('asin')
                      .apply(lambda x: x.nlargest(top_n, 'helpful_votes'))
                      .reset_index(drop=True))
        
        logger.info(f"Selected {len(top_reviews)} top helpful reviews")
        return top_reviews
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to Parquet format."""
        os.makedirs(settings.parquet_dir, exist_ok=True)
        filepath = os.path.join(settings.parquet_dir, filename)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")
    
    def load_from_parquet(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from Parquet format."""
        filepath = os.path.join(settings.parquet_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        else:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()


def main():
    """Main data loading pipeline."""
    loader = AmazonDataLoader()
    
    # Load raw data
    try:
        raw_dataset = loader.load_raw_dataset()
        raw_data = list(raw_dataset.take(settings.max_reviews))
    except:
        # Use demo data if HF dataset fails
        raw_data = loader._generate_demo_data()
    
    # Process products and reviews
    products_df = loader.process_products(raw_data)
    reviews_df = loader.process_reviews(raw_data)
    
    # Get top helpful reviews
    top_reviews_df = loader.get_top_helpful_reviews(reviews_df)
    
    # Save processed data
    loader.save_to_parquet(products_df, "products.parquet")
    loader.save_to_parquet(top_reviews_df, "reviews.parquet")
    
    print(f"Data loading complete:")
    print(f"- Products: {len(products_df)}")
    print(f"- Reviews: {len(top_reviews_df)}")


if __name__ == "__main__":
    main()