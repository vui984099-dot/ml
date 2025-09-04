"""
Text processing and chunking utilities.
"""

import re
import uuid
from typing import List, Dict, Any
from transformers import AutoTokenizer
import pandas as pd
import logging

from src.config import settings
from src.models.schemas import ChunkMetadata

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text chunking and processing for embeddings."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{self.model_name}")
        self.max_tokens = settings.chunk_size_tokens
        self.overlap_tokens = settings.chunk_overlap_tokens
        
    def chunk_text(self, text: str, chunk_type: str, asin: str, 
                   helpful_votes: int = 0, rating: int = None) -> List[ChunkMetadata]:
        """
        Chunk text into overlapping segments for embedding.
        
        Args:
            text: Input text to chunk
            chunk_type: 'review' or 'description'
            asin: Product identifier
            helpful_votes: Number of helpful votes (for reviews)
            rating: Review rating (for reviews)
            
        Returns:
            List of ChunkMetadata objects
        """
        if not text or len(text.strip()) < 20:
            return []
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.max_tokens:
            # Text fits in one chunk
            chunk_id = str(uuid.uuid4())
            return [ChunkMetadata(
                chunk_id=chunk_id,
                asin=asin,
                type=chunk_type,
                text=text.strip(),
                start_char=0,
                end_char=len(text),
                helpful_votes=helpful_votes,
                rating=rating
            )]
        
        chunks = []
        step_size = self.max_tokens - self.overlap_tokens
        
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Calculate character positions (approximate)
            start_char = int((i / len(tokens)) * len(text))
            end_char = int(((i + len(chunk_tokens)) / len(tokens)) * len(text))
            end_char = min(end_char, len(text))
            
            chunk_id = str(uuid.uuid4())
            
            chunk = ChunkMetadata(
                chunk_id=chunk_id,
                asin=asin,
                type=chunk_type,
                text=chunk_text.strip(),
                start_char=start_char,
                end_char=end_char,
                helpful_votes=helpful_votes,
                rating=rating
            )
            chunks.append(chunk)
            
            # Stop if we've reached max chunks per product
            if len(chunks) >= settings.max_chunks_per_product:
                break
        
        return chunks
    
    def process_product_text(self, product_row: Dict[str, Any]) -> List[ChunkMetadata]:
        """Process all text content for a single product."""
        chunks = []
        asin = product_row["asin"]
        
        # Chunk product description
        description = product_row.get("description", "")
        if description:
            desc_chunks = self.chunk_text(
                text=description,
                chunk_type="description", 
                asin=asin,
                helpful_votes=0,
                rating=None
            )
            chunks.extend(desc_chunks)
        
        return chunks
    
    def process_review_text(self, review_row: Dict[str, Any]) -> List[ChunkMetadata]:
        """Process review text into chunks."""
        review_text = review_row.get("review_text", "")
        if not review_text:
            return []
            
        chunks = self.chunk_text(
            text=review_text,
            chunk_type="review",
            asin=review_row["asin"],
            helpful_votes=review_row.get("helpful_votes", 0),
            rating=review_row.get("rating")
        )
        
        return chunks
    
    def process_all_text(self, products_df: pd.DataFrame, 
                        reviews_df: pd.DataFrame) -> List[ChunkMetadata]:
        """Process all products and reviews into chunks."""
        all_chunks = []
        
        # Process product descriptions
        logger.info("Processing product descriptions...")
        for _, product in products_df.iterrows():
            chunks = self.process_product_text(product.to_dict())
            all_chunks.extend(chunks)
        
        # Process reviews
        logger.info("Processing reviews...")
        for _, review in reviews_df.iterrows():
            chunks = self.process_review_text(review.to_dict())
            all_chunks.extend(chunks)
        
        logger.info(f"Generated {len(all_chunks)} total chunks")
        return all_chunks
    
    def save_chunks_metadata(self, chunks: List[ChunkMetadata], filename: str = "chunks_metadata.parquet"):
        """Save chunk metadata to Parquet file."""
        import os
        
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "asin": chunk.asin,
                "type": chunk.type,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "helpful_votes": chunk.helpful_votes,
                "rating": chunk.rating,
                "faiss_id": chunk.faiss_id
            })
        
        chunks_df = pd.DataFrame(chunks_data)
        
        os.makedirs(settings.parquet_dir, exist_ok=True)
        filepath = os.path.join(settings.parquet_dir, filename)
        chunks_df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved {len(chunks_df)} chunks metadata to {filepath}")
        return chunks_df
    
    def load_chunks_metadata(self, filename: str = "chunks_metadata.parquet") -> List[ChunkMetadata]:
        """Load chunk metadata from Parquet file."""
        filepath = os.path.join(settings.parquet_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Chunks metadata file not found: {filepath}")
            return []
        
        chunks_df = pd.read_parquet(filepath)
        
        chunks = []
        for _, row in chunks_df.iterrows():
            chunk = ChunkMetadata(
                chunk_id=row["chunk_id"],
                asin=row["asin"],
                type=row["type"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                helpful_votes=row.get("helpful_votes", 0),
                rating=row.get("rating"),
                faiss_id=row.get("faiss_id")
            )
            chunks.append(chunk)
        
        logger.info(f"Loaded {len(chunks)} chunks metadata from {filepath}")
        return chunks


def main():
    """Test text processing pipeline."""
    processor = TextProcessor()
    
    # Test with sample data
    sample_text = """
    This is a great product that I've been using for months. 
    The build quality is excellent and it performs well under heavy usage.
    Battery life is outstanding - easily lasts a full day of work.
    The only minor issue is that it can get a bit warm during intensive tasks.
    Overall, I would definitely recommend this to anyone looking for a reliable device.
    """
    
    chunks = processor.chunk_text(
        text=sample_text,
        chunk_type="review",
        asin="B123456789",
        helpful_votes=5,
        rating=5
    )
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.text[:100]}...")


if __name__ == "__main__":
    main()