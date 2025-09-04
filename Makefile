# Makefile for Amazon Product Q&A and Recommendation System

.PHONY: help install data index test api ui docker clean lint format

# Default target
help:
	@echo "Amazon Product Q&A & Recommendation System"
	@echo "==========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies and setup environment"
	@echo "  make data       - Run data ingestion pipeline"
	@echo "  make index      - Build vector search index"
	@echo "  make test       - Run test suite"
	@echo "  make api        - Start FastAPI backend"
	@echo "  make ui         - Start Streamlit frontend"
	@echo "  make docker     - Build and run with Docker"
	@echo "  make clean      - Clean generated files"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code with black"
	@echo ""
	@echo "Quick start:"
	@echo "  make install && make data && make api"
	@echo ""

# Install dependencies and setup environment
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
	mkdir -p data/raw data/parquet data/indexes models logs
	@echo "✅ Environment setup completed"

# Run data ingestion pipeline
data:
	@echo "Running data ingestion pipeline..."
	python src/etl/ingest_data.py
	@echo "✅ Data ingestion completed"

# Build vector search index
index:
	@echo "Building vector search index..."
	python src/indexing/build_index.py
	@echo "✅ Index building completed"

# Train CTR model
ctr:
	@echo "Training CTR model..."
	python src/models/ctr_model.py
	@echo "✅ CTR model training completed"

# Run complete data pipeline
pipeline: data index ctr

# Run test suite
test:
	@echo "Running test suite..."
	python -m pytest tests/ -v
	@echo "✅ Tests completed"

# Run unit tests only
test-unit:
	@echo "Running unit tests..."
	python -m pytest tests/ -m "not integration" -v

# Run integration tests only
test-integration:
	@echo "Running integration tests..."
	python -m pytest tests/ -m "integration" -v

# Start FastAPI backend
api:
	@echo "Starting FastAPI backend..."
	@echo "API will be available at: http://localhost:8000"
	@echo "API docs will be available at: http://localhost:8000/docs"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit UI
ui:
	@echo "Starting Streamlit UI..."
	@echo "UI will be available at: http://localhost:8501"
	streamlit run src/ui/app.py --server.port 8501

# Build and run with Docker
docker:
	@echo "Building and running with Docker..."
	docker-compose up --build

# Stop Docker services
docker-stop:
	docker-compose down

# Build Docker images only
docker-build:
	docker-compose build

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf data/parquet/* data/indexes/* models/*.pkl
	@echo "✅ Cleanup completed"

# Run code linting
lint:
	@echo "Running code linting..."
	flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

# Format code with black
format:
	@echo "Formatting code..."
	black src/ tests/ --line-length=88
	@echo "✅ Code formatting completed"

# Check code formatting
format-check:
	black src/ tests/ --line-length=88 --check

# Development setup (install + data + index)
dev-setup: install data index
	@echo "🎉 Development setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Start API: make api"
	@echo "2. Start UI: make ui (in another terminal)"
	@echo "3. Visit: http://localhost:8501"

# Production setup with Docker
prod-setup:
	@echo "Setting up for production..."
	docker-compose build
	@echo "✅ Production setup completed"
	@echo ""
	@echo "To start: docker-compose up"

# Show system status
status:
	@echo "System Status:"
	@echo "=============="
	@echo -n "Database: "
	@if [ -f "data/products.db" ]; then echo "✅ Ready"; else echo "❌ Not found"; fi
	@echo -n "Parquet data: "
	@if [ -f "data/parquet/products.parquet" ]; then echo "✅ Ready"; else echo "❌ Not found"; fi
	@echo -n "Vector index: "
	@if [ -f "data/indexes/faiss_index.bin" ]; then echo "✅ Ready"; else echo "❌ Not found"; fi
	@echo -n "CTR model: "
	@if [ -f "models/ctr_model.pkl" ]; then echo "✅ Ready"; else echo "❌ Not found"; fi