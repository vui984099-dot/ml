#!/usr/bin/env python3
"""
Quick start script for Amazon Product Q&A System.
Sets up and runs the complete system with demo data.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
import signal
import atexit

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global process tracking
running_processes = []

def cleanup_processes():
    """Clean up running processes on exit."""
    for proc in running_processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            try:
                proc.kill()
            except:
                pass

atexit.register(cleanup_processes)

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\n🛑 Shutting down services...")
    cleanup_processes()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_command(command, description, capture_output=True, timeout=300):
    """Run a command with error handling."""
    logger.info(f"🔄 {description}")
    
    try:
        if capture_output:
            result = subprocess.run(
                command, shell=True, check=True, 
                capture_output=True, text=True, timeout=timeout
            )
            logger.info(f"✅ {description} completed")
            return True, result.stdout
        else:
            # For long-running processes
            proc = subprocess.Popen(command, shell=True)
            running_processes.append(proc)
            return True, proc
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False, None
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} timed out")
        return False, None
    except Exception as e:
        logger.error(f"❌ {description} error: {e}")
        return False, None


def check_dependencies():
    """Check if required dependencies are available."""
    logger.info("🔍 Checking dependencies...")
    
    # Check Python
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    
    # Check pip
    success, _ = run_command("pip --version", "Checking pip")
    if not success:
        logger.error("pip is required but not found")
        return False
    
    logger.info("✅ Dependencies check passed")
    return True


def setup_environment():
    """Set up the development environment."""
    logger.info("🏗️ Setting up environment...")
    
    # Create directories
    directories = [
        "data/raw", "data/parquet", "data/indexes", 
        "models", "logs", "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Install Python packages
    success, output = run_command(
        "pip install -r requirements.txt", 
        "Installing Python dependencies",
        timeout=600
    )
    
    if not success:
        logger.error("Failed to install dependencies")
        return False
    
    logger.info("✅ Environment setup completed")
    return True


def run_data_pipeline():
    """Run the data processing pipeline."""
    logger.info("📊 Running data pipeline...")
    
    # Initialize database and run ingestion
    success, output = run_command(
        "python run_demo.py",
        "Running complete data pipeline",
        timeout=600
    )
    
    if not success:
        logger.warning("Data pipeline had issues, but continuing with demo data...")
        
        # Try individual steps
        steps = [
            ("python -c \"from src.database import init_database; init_database()\"", "Database initialization"),
            ("python -c \"from src.etl.data_loader import AmazonDataLoader; loader = AmazonDataLoader(); demo = loader._generate_demo_data(); print('Demo data ready')\"", "Demo data preparation")
        ]
        
        for command, description in steps:
            run_command(command, description, timeout=60)
    
    logger.info("✅ Data pipeline completed")
    return True


def start_services():
    """Start API and UI services."""
    logger.info("🌐 Starting services...")
    
    # Start API backend
    logger.info("Starting FastAPI backend...")
    api_success, api_proc = run_command(
        "uvicorn src.api.main:app --host 0.0.0.0 --port 8000",
        "FastAPI backend",
        capture_output=False
    )
    
    if not api_success:
        logger.error("Failed to start API backend")
        return False
    
    # Wait for API to start
    time.sleep(10)
    
    # Test API health
    api_ready = False
    for attempt in range(5):
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                api_ready = True
                break
        except:
            time.sleep(2)
    
    if api_ready:
        logger.info("✅ API backend is ready")
    else:
        logger.warning("⚠️ API backend may not be fully ready")
    
    # Start Streamlit UI
    logger.info("Starting Streamlit UI...")
    ui_success, ui_proc = run_command(
        "streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0",
        "Streamlit UI",
        capture_output=False
    )
    
    if not ui_success:
        logger.error("Failed to start UI")
        return False
    
    # Wait for UI to start
    time.sleep(5)
    
    logger.info("✅ Services started successfully")
    return True


def print_access_info():
    """Print access information for the user."""
    print("\n" + "="*70)
    print("🎉 AMAZON PRODUCT Q&A SYSTEM IS READY!")
    print("="*70)
    print()
    print("🌐 Access the application:")
    print("   • Streamlit UI: http://localhost:8501")
    print("   • API Backend: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print()
    print("✨ Try these features:")
    print("   1. 🔍 Search: 'wireless headphones', 'smart speaker'")
    print("   2. ❓ Q&A: 'How is the battery life?', 'Is the sound quality good?'")
    print("   3. 💡 Recommendations: Query-based or similar products")
    print("   4. ⚖️ Compare: Select products from search results")
    print()
    print("🛑 To stop: Press Ctrl+C")
    print("="*70)


def main():
    """Main quickstart function."""
    print("🚀 Amazon Product Q&A System - Quick Start")
    print("==========================================")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Run data pipeline
    if not run_data_pipeline():
        return False
    
    # Start services
    if not start_services():
        return False
    
    # Print access information
    print_access_info()
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        return True


if __name__ == "__main__":
    success = main()
    cleanup_processes()
    sys.exit(0 if success else 1)