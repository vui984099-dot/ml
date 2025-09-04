"""
Main script to build the complete vector search index.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.etl.text_processor import TextProcessor
from src.indexing.faiss_index import VectorSearchEngine
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_complete_index():
    """Build the complete vector search index pipeline."""
    logger.info("Starting index building pipeline...")
    
    # Load processed chunks
    processor = TextProcessor()
    chunks = processor.load_chunks_metadata()
    
    if not chunks:
        logger.error("No chunks metadata found. Please run data ingestion first:")
        logger.error("python src/etl/ingest_data.py")
        return False
    
    logger.info(f"Loaded {len(chunks)} chunks for indexing")
    
    # Build vector search index
    search_engine = VectorSearchEngine()
    search_engine.build_index_from_chunks(chunks)
    
    # Save the index
    search_engine.save_index()
    
    # Test the index
    test_queries = [
        "battery life",
        "sound quality", 
        "easy setup",
        "good value"
    ]
    
    logger.info("Testing index with sample queries...")
    for query in test_queries:
        try:
            results = search_engine.search_similar_chunks(query, top_k=3)
            logger.info(f"Query '{query}' returned {len(results)} results")
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return False
    
    logger.info("Index building completed successfully!")
    return True


if __name__ == "__main__":
    success = build_complete_index()
    sys.exit(0 if success else 1)