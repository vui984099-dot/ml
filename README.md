# Amazon Product Recommendation & Q&A System

A comprehensive retrieval + recommendation + grounded Q&A system built on the McAuley-Lab Amazon reviews dataset.

## 🚀 Features

- **🤖 Grounded Q&A**: Answer product questions with citations from reviews and descriptions
- **💡 Personalized Recommendations**: Similarity-based product recommendations using vector search
- **⚖️ Product Comparisons**: Side-by-side product comparison functionality
- **🌐 Web Interface**: FastAPI backend + Streamlit frontend for easy interaction
- **🔍 Advanced Search**: Multi-stage retrieval with bi-encoder recall and cross-encoder reranking
- **📊 CTR Modeling**: Click-through rate prediction for ranking optimization

## 🏗️ Architecture

### Data Layer
- **Storage**: Parquet files for structured data, SQLite for metadata
- **Vector DB**: FAISS for high-performance similarity search
- **Embeddings**: Sentence-BERT for semantic understanding

### ML Pipeline
- **Bi-encoder**: `all-MiniLM-L6-v2` for fast retrieval (top-50)
- **Cross-encoder**: `ms-marco-MiniLM-L-6-v2` for precise reranking (top-10)
- **LLM Integration**: OpenAI GPT or Google Gemini for answer generation
- **CTR Model**: LightGBM for click prediction and ranking optimization

### API & UI
- **Backend**: FastAPI with async endpoints and automatic documentation
- **Frontend**: Streamlit with interactive components and real-time updates
- **Deployment**: Docker containerization with docker-compose orchestration

## 📁 Project Structure

```
amazon-product-qa/
├── 📊 data/
│   ├── raw/           # Raw HuggingFace dataset
│   ├── parquet/       # Processed data (products, reviews, chunks)
│   └── indexes/       # FAISS vector indexes
├── 🧠 src/
│   ├── etl/           # Data extraction, transformation, loading
│   ├── indexing/      # Vector indexing and FAISS management
│   ├── models/        # ML models (schemas, CTR model)
│   ├── retrieval/     # Search and retrieval engine
│   ├── qa/            # Question answering with RAG
│   ├── recommendations/ # Product recommendation engine
│   ├── comparison/    # Product comparison functionality
│   ├── api/           # FastAPI backend
│   └── ui/            # Streamlit frontend
├── 🧪 tests/          # Comprehensive test suite
├── 🐳 docker/         # Docker configuration
├── 📋 requirements.txt
├── 🔧 Makefile
└── 📖 README.md
```

## 🚀 Quick Start

### Option 1: Local Development

```bash
# 1. Install dependencies
make install

# 2. Run data pipeline (uses demo data if HuggingFace unavailable)
make data

# 3. Build search index
make index

# 4. Start API backend
make api  # http://localhost:8000

# 5. Start UI frontend (in another terminal)
make ui   # http://localhost:8501
```

### Option 2: Docker (Recommended)

```bash
# Build and run everything with Docker
docker-compose up --build

# Access the application
# - UI: http://localhost:8501
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Option 3: Complete Setup

```bash
# Run everything in sequence
make dev-setup

# Then start services
make api  # Terminal 1
make ui   # Terminal 2
```

## 🔧 Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Key configuration options:

```bash
# API Keys (optional - system works with mock LLM)
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_ai_key_here

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Data Configuration
MAX_PRODUCTS=20000
MAX_REVIEWS=100000
TARGET_CATEGORY=Electronics
```

## 📚 API Documentation

### Core Endpoints

- **`POST /search`** - Search products with filters
- **`POST /qa`** - Ask questions with citations
- **`POST /recommendations`** - Get product recommendations
- **`POST /compare`** - Compare two products side-by-side
- **`GET /health`** - System health check

### Example API Usage

```python
import requests

# Search for products
response = requests.post("http://localhost:8000/search", json={
    "query": "wireless headphones",
    "min_rating": 4.0,
    "top_k": 10
})

# Ask a question
response = requests.post("http://localhost:8000/qa", json={
    "question": "How is the battery life?",
    "max_chunks": 5
})

# Get recommendations
response = requests.post("http://localhost:8000/recommendations", json={
    "query": "gaming laptop",
    "top_k": 6
})
```

## 🎯 Usage Examples

### 1. Product Search
```
Query: "wireless bluetooth headphones under $100"
→ Returns ranked products with relevance scores
```

### 2. Question Answering
```
Question: "How is the battery life on this laptop?"
→ Returns answer with citations from actual reviews:
   "Based on customer reviews, the battery typically lasts 6-8 hours [1][2]. 
   However, heavy gaming reduces this to 3-4 hours [3]."
```

### 3. Product Recommendations
```
Input: ASIN "B08N5WRWNW" (Echo Dot)
→ Returns similar smart home devices with similarity scores
```

### 4. Product Comparison
```
Input: Compare "B08N5WRWNW" vs "B08F7PTF53"
→ Returns side-by-side specs, pros/cons, and category winners
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📊 Evaluation Metrics

The system tracks multiple evaluation metrics:

- **Retrieval**: Precision@K, Recall@K, MRR
- **Ranking**: NDCG@10, MAP
- **Q&A**: Groundedness, Helpfulness, Citation accuracy
- **CTR**: AUC, Log-loss for click prediction

## 🔍 System Components

### 1. Data Pipeline (`src/etl/`)
- **`data_loader.py`**: HuggingFace dataset ingestion with fallback demo data
- **`text_processor.py`**: Text cleaning, chunking, and preprocessing
- **`ingest_data.py`**: Complete ETL orchestration

### 2. Vector Search (`src/indexing/`)
- **`embeddings.py`**: Sentence-BERT embedding generation
- **`faiss_index.py`**: FAISS index building and management
- **`build_index.py`**: Index construction pipeline

### 3. Retrieval System (`src/retrieval/`)
- **`search_engine.py`**: Complete retrieval pipeline with reranking

### 4. Q&A System (`src/qa/`)
- **`llm_client.py`**: Multi-provider LLM integration (OpenAI, Google, Mock)
- **`qa_engine.py`**: RAG implementation with citation extraction

### 5. Recommendations (`src/recommendations/`)
- **`recommender.py`**: Multi-strategy recommendation engine

### 6. Comparison (`src/comparison/`)
- **`comparator.py`**: Product comparison with spec extraction

### 7. CTR Modeling (`src/models/`)
- **`ctr_model.py`**: LightGBM-based CTR prediction
- **`schemas.py`**: Data models and API schemas

## 🚀 Deployment

### Local Development
```bash
# Start API
uvicorn src.api.main:app --reload --port 8000

# Start UI  
streamlit run src/ui/app.py --server.port 8501
```

### Docker Production
```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# With custom environment
docker-compose --env-file .env.prod up -d
```

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/health

# Check individual components
make status
```

## 🔧 Development

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Type checking
mypy src/ --ignore-missing-imports
```

### Adding New Features

1. **New API Endpoint**: Add to `src/api/main.py`
2. **New ML Model**: Add to `src/models/`
3. **New UI Component**: Add to `src/ui/components.py`
4. **New Test**: Add to appropriate `tests/test_*.py`

## 📈 Performance

### Benchmarks (on demo data)
- **Search Latency**: ~200ms for top-50 retrieval + reranking
- **Q&A Response**: ~2-5s (depending on LLM provider)
- **Recommendations**: ~100ms for similarity-based
- **Index Size**: ~2MB for 10K products (demo dataset)

### Scaling Considerations
- **FAISS Index**: Supports millions of vectors with HNSW/IVF
- **Database**: SQLite for development, PostgreSQL recommended for production
- **Caching**: Add Redis for production deployments
- **Load Balancing**: Use multiple API instances behind load balancer

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run test suite: `make test`
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: McAuley-Lab Amazon Reviews 2023 dataset
- **Models**: Sentence-Transformers, Hugging Face Transformers
- **Frameworks**: FastAPI, Streamlit, FAISS
- **ML Libraries**: LightGBM, scikit-learn, PyTorch

---

**Built with ❤️ for better product discovery and informed purchasing decisions.**