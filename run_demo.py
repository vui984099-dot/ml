"""
Demo runner script for Amazon Product Q&A System.
Runs the complete pipeline with demo data.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import settings
from src.database import init_database
from src.etl.ingest_data import DataIngestionPipeline
from src.indexing.build_index import build_complete_index
from src.models.ctr_model import train_ctr_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_demo_environment():
    """Set up the demo environment with all necessary components."""
    logger.info("🚀 Setting up Amazon Product Q&A Demo")
    
    # Create directories
    directories = ["data/raw", "data/parquet", "data/indexes", "models", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    # Initialize database
    logger.info("📊 Initializing database...")
    init_database()
    logger.info("✅ Database initialized")
    
    return True


def run_complete_pipeline():
    """Run the complete data processing and model building pipeline."""
    logger.info("🔄 Running complete pipeline...")
    
    try:
        # Step 1: Data ingestion
        logger.info("📥 Step 1: Data ingestion and preprocessing")
        pipeline = DataIngestionPipeline()
        results = pipeline.run_pipeline()
        logger.info(f"✅ Processed {results['products_count']} products, {results['reviews_count']} reviews, {results['chunks_count']} chunks")
        
        # Step 2: Build search index
        logger.info("🔍 Step 2: Building vector search index")
        index_success = build_complete_index()
        if index_success:
            logger.info("✅ Vector search index built successfully")
        else:
            logger.error("❌ Failed to build search index")
            return False
        
        # Step 3: Train CTR model
        logger.info("🤖 Step 3: Training CTR model")
        ctr_model = train_ctr_model()
        if ctr_model:
            logger.info("✅ CTR model trained successfully")
        else:
            logger.warning("⚠️ CTR model training failed, continuing without it")
        
        logger.info("🎉 Complete pipeline finished successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False


def test_system_components():
    """Test that all system components are working."""
    logger.info("🧪 Testing system components...")
    
    try:
        # Test Q&A
        from src.qa.qa_engine import QAEngine
        from src.models.schemas import QARequest
        
        qa_engine = QAEngine()
        test_request = QARequest(question="How is the sound quality?", max_chunks=3)
        qa_response = qa_engine.answer_question(test_request)
        logger.info(f"✅ Q&A test: {qa_response.question[:50]}...")
        
        # Test recommendations
        from src.recommendations.recommender import ProductRecommender
        from src.models.schemas import RecommendationRequest
        
        recommender = ProductRecommender()
        rec_request = RecommendationRequest(query="wireless headphones", top_k=3)
        rec_response = recommender.get_recommendations(rec_request)
        logger.info(f"✅ Recommendations test: {len(rec_response.products)} products found")
        
        # Test search
        from src.retrieval.search_engine import RetrievalEngine
        
        search_engine = RetrievalEngine()
        search_results = search_engine.search_and_rerank("battery life", top_k=3)
        logger.info(f"✅ Search test: {len(search_results)} results found")
        
        logger.info("🎉 All components tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component testing failed: {e}")
        return False


def print_startup_instructions():
    """Print instructions for starting the services."""
    print("\n" + "="*60)
    print("🎉 DEMO SETUP COMPLETE!")
    print("="*60)
    print()
    print("🚀 To start the application:")
    print()
    print("Option 1 - Docker (Recommended):")
    print("   docker-compose up")
    print()
    print("Option 2 - Manual:")
    print("   Terminal 1: uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    print("   Terminal 2: streamlit run src/ui/app.py --server.port 8501")
    print()
    print("🌐 Access points:")
    print("   • Streamlit UI: http://localhost:8501")
    print("   • API Backend: http://localhost:8000")
    print("   • API Docs: http://localhost:8000/docs")
    print()
    print("✨ Features to try:")
    print("   • Search: 'wireless headphones', 'gaming laptop'")
    print("   • Q&A: 'How is the battery life?', 'Is the sound quality good?'")
    print("   • Compare: Use ASINs from search results")
    print("   • Recommendations: Query-based or similar products")
    print()
    print("="*60)


def main():
    """Main demo setup function."""
    logger.info("Starting Amazon Product Q&A Demo Setup")
    
    # Setup environment
    if not setup_demo_environment():
        logger.error("Failed to setup environment")
        return False
    
    # Run pipeline
    if not run_complete_pipeline():
        logger.error("Failed to run pipeline")
        return False
    
    # Test components
    if not test_system_components():
        logger.warning("Component testing had issues, but continuing...")
    
    # Print startup instructions
    print_startup_instructions()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)