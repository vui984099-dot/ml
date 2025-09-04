"""
Main data ingestion pipeline that orchestrates the entire ETL process.
"""

import os
import logging
from pathlib import Path
from sqlalchemy.orm import Session

from src.config import settings
from src.database import init_database, SessionLocal
from src.etl.data_loader import AmazonDataLoader
from src.etl.text_processor import TextProcessor
from src.models.schemas import Product, Review, TextChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Complete data ingestion and preprocessing pipeline."""
    
    def __init__(self):
        self.loader = AmazonDataLoader()
        self.processor = TextProcessor()
        
    def run_pipeline(self):
        """Execute the complete data ingestion pipeline."""
        logger.info("Starting data ingestion pipeline...")
        
        # Initialize database
        init_database()
        
        # Load and process data
        raw_data = self._load_raw_data()
        products_df = self.loader.process_products(raw_data)
        reviews_df = self.loader.process_reviews(raw_data)
        
        # Get top helpful reviews
        top_reviews_df = self.loader.get_top_helpful_reviews(reviews_df)
        
        # Save to Parquet
        self.loader.save_to_parquet(products_df, "products.parquet")
        self.loader.save_to_parquet(top_reviews_df, "reviews.parquet")
        
        # Process text into chunks
        chunks = self.processor.process_all_text(products_df, top_reviews_df)
        chunks_df = self.processor.save_chunks_metadata(chunks)
        
        # Save to database
        self._save_to_database(products_df, top_reviews_df, chunks)
        
        logger.info("Data ingestion pipeline completed successfully!")
        return {
            "products_count": len(products_df),
            "reviews_count": len(top_reviews_df), 
            "chunks_count": len(chunks)
        }
    
    def _load_raw_data(self):
        """Load raw data with fallback to demo data."""
        try:
            # Try to load from HuggingFace
            dataset = self.loader.load_raw_dataset("raw_review_Electronics")
            raw_data = list(dataset.take(settings.max_reviews))
            logger.info("Successfully loaded data from HuggingFace")
            return raw_data
        except Exception as e:
            logger.warning(f"Failed to load HF dataset: {e}")
            logger.info("Using demo data instead")
            return self.loader._generate_demo_data()
    
    def _save_to_database(self, products_df, reviews_df, chunks):
        """Save processed data to SQLite database."""
        db = SessionLocal()
        
        try:
            # Save products
            logger.info("Saving products to database...")
            for _, row in products_df.iterrows():
                product = Product(
                    asin=row["asin"],
                    title=row["title"],
                    brand=row["brand"],
                    category=row["category"],
                    price=row["price"],
                    description=row["description"],
                    image_url=row["image_url"],
                    avg_rating=row["avg_rating"],
                    num_reviews=row["num_reviews"],
                    created_at=row["created_at"]
                )
                db.merge(product)
            
            # Save reviews
            logger.info("Saving reviews to database...")
            for _, row in reviews_df.iterrows():
                review = Review(
                    review_id=row["review_id"],
                    asin=row["asin"],
                    review_text=row["review_text"],
                    rating=row["rating"],
                    helpful_votes=row["helpful_votes"],
                    review_date=row["review_date"],
                    reviewer_id=row["reviewer_id"]
                )
                db.merge(review)
            
            # Save text chunks
            logger.info("Saving text chunks to database...")
            for chunk in chunks:
                text_chunk = TextChunk(
                    chunk_id=chunk.chunk_id,
                    asin=chunk.asin,
                    chunk_type=chunk.type,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    helpful_votes=chunk.helpful_votes,
                    rating=chunk.rating,
                    faiss_id=chunk.faiss_id
                )
                db.merge(text_chunk)
            
            db.commit()
            logger.info("Successfully saved all data to database")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            db.rollback()
            raise
        finally:
            db.close()


def main():
    """Run the complete data ingestion pipeline."""
    pipeline = DataIngestionPipeline()
    results = pipeline.run_pipeline()
    
    print("\n=== Data Ingestion Complete ===")
    print(f"Products processed: {results['products_count']}")
    print(f"Reviews processed: {results['reviews_count']}")
    print(f"Text chunks created: {results['chunks_count']}")
    print(f"Data saved to: {settings.parquet_dir}")
    print(f"Database: {settings.database_url}")


if __name__ == "__main__":
    main()