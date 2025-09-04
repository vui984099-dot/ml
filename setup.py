"""
Setup script for Amazon Product Q&A and Recommendation System.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command: str, description: str):
    """Run a command and handle errors."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed:")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def setup_environment():
    """Set up the development environment."""
    logger.info("🚀 Setting up Amazon Product Q&A System")
    
    # Create necessary directories
    directories = [
        "data/raw", "data/parquet", "data/indexes", 
        "models", "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Download NLTK data (if needed)
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("✅ NLTK data downloaded")
    except Exception as e:
        logger.warning(f"NLTK data download failed: {e}")
    
    # Initialize database
    if not run_command("python -c \"from src.database import init_database; init_database()\"", "Initializing database"):
        logger.warning("Database initialization failed - will be created on first run")
    
    logger.info("🎉 Environment setup completed!")
    return True


def run_data_pipeline():
    """Run the complete data processing pipeline."""
    logger.info("📊 Running data processing pipeline")
    
    steps = [
        ("python src/etl/ingest_data.py", "Data ingestion and preprocessing"),
        ("python src/indexing/build_index.py", "Building vector search index"),
        ("python src/models/ctr_model.py", "Training CTR prediction model")
    ]
    
    for command, description in steps:
        if not run_command(command, description):
            logger.error(f"Pipeline failed at: {description}")
            return False
    
    logger.info("🎉 Data pipeline completed successfully!")
    return True


def run_tests():
    """Run the test suite."""
    logger.info("🧪 Running test suite")
    
    test_commands = [
        ("python -m pytest tests/ -v --tb=short", "Running all tests"),
        ("python -m pytest tests/ -m 'not integration' -v", "Running unit tests"),
    ]
    
    for command, description in test_commands:
        run_command(command, description)  # Don't fail on test failures
    
    logger.info("🧪 Test suite completed")


def start_services():
    """Start the API and UI services."""
    logger.info("🌐 Starting services")
    
    print("\n" + "="*50)
    print("🎉 Setup completed! Ready to start services.")
    print("="*50)
    
    print("\nTo start the system:")
    print("1. Start API backend:")
    print("   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
    print()
    print("2. Start Streamlit UI (in another terminal):")
    print("   streamlit run src/ui/app.py --server.port 8501")
    print()
    print("3. Or use Docker:")
    print("   docker-compose up")
    print()
    print("🌐 Access the application:")
    print("   - API docs: http://localhost:8000/docs")
    print("   - Streamlit UI: http://localhost:8501")
    print("   - Health check: http://localhost:8000/health")


def main():
    """Main setup function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "env":
            setup_environment()
        elif command == "data":
            run_data_pipeline()
        elif command == "test":
            run_tests()
        elif command == "start":
            start_services()
        elif command == "all":
            if setup_environment():
                if run_data_pipeline():
                    run_tests()
                    start_services()
        else:
            print("Usage: python setup.py [env|data|test|start|all]")
            print("  env  - Set up environment and dependencies")
            print("  data - Run data processing pipeline") 
            print("  test - Run test suite")
            print("  start - Show service startup instructions")
            print("  all  - Run complete setup")
    else:
        # Default: run complete setup
        if setup_environment():
            if run_data_pipeline():
                run_tests()
                start_services()


if __name__ == "__main__":
    main()