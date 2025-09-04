"""
FastAPI backend for Amazon product recommendation and Q&A system.
"""

from datetime import datetime
from typing import List, Optional
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.config import settings
from src.database import get_db, init_database
from src.models.schemas import (
    SearchQuery, QARequest, QAResponse, RecommendationRequest, RecommendationResponse,
    ComparisonRequest, ComparisonResponse, HealthResponse, ProductResponse
)
from src.qa.qa_engine import QAEngine
from src.recommendations.recommender import ProductRecommender
from src.comparison.comparator import ProductComparator
from src.retrieval.search_engine import RetrievalEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Amazon Product Q&A and Recommendation API",
    description="Retrieval + recommendation + grounded Q&A system for Amazon products",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
qa_engine = None
recommender = None
comparator = None
retrieval_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize database and load models on startup."""
    global qa_engine, recommender, comparator, retrieval_engine
    
    logger.info("Starting up Amazon Product API...")
    
    try:
        # Initialize database
        init_database()
        logger.info("Database initialized")
        
        # Initialize components
        qa_engine = QAEngine()
        recommender = ProductRecommender()
        comparator = ProductComparator()
        retrieval_engine = RetrievalEngine()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't crash the app, but log the error


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Amazon Product Q&A and Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "qa": "/qa",
            "recommendations": "/recommendations", 
            "compare": "/compare",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Check database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check index status
    index_status = "healthy" if retrieval_engine and hasattr(retrieval_engine.vector_search, '_index_built') and retrieval_engine.vector_search._index_built else "not_loaded"
    
    # Check model status
    model_status = "healthy" if qa_engine else "not_loaded"
    
    overall_status = "healthy" if all([
        db_status == "healthy",
        index_status == "healthy", 
        model_status == "healthy"
    ]) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        database_status=db_status,
        index_status=index_status,
        model_status=model_status
    )


@app.post("/search", response_model=List[ProductResponse])
async def search_products(
    search_query: SearchQuery,
    db: Session = Depends(get_db)
):
    """Search for products using vector similarity."""
    try:
        if not retrieval_engine:
            raise HTTPException(status_code=503, detail="Search engine not initialized")
        
        # Get product ASINs from search
        product_asins = retrieval_engine.search_products_by_query(
            search_query.query, top_k=search_query.top_k
        )
        
        if not product_asins:
            return []
        
        # Get product details from database
        from src.models.schemas import Product
        products_orm = db.query(Product).filter(Product.asin.in_(product_asins)).all()
        
        # Convert to response format
        products = []
        for product in products_orm:
            product_response = ProductResponse(
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
            products.append(product_response)
        
        return products
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/qa", response_model=QAResponse)
async def answer_question(qa_request: QARequest):
    """Answer product questions using RAG."""
    try:
        if not qa_engine:
            raise HTTPException(status_code=503, detail="Q&A engine not initialized")
        
        response = qa_engine.answer_question(qa_request)
        return response
        
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(rec_request: RecommendationRequest):
    """Get product recommendations."""
    try:
        if not recommender:
            raise HTTPException(status_code=503, detail="Recommender not initialized")
        
        response = recommender.get_recommendations(rec_request)
        return response
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


@app.post("/compare", response_model=ComparisonResponse)
async def compare_products(comparison_request: ComparisonRequest):
    """Compare two products side by side."""
    try:
        if not comparator:
            raise HTTPException(status_code=503, detail="Comparator not initialized")
        
        response = comparator.compare_products(comparison_request)
        return response
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/products/{asin}", response_model=ProductResponse)
async def get_product(asin: str, db: Session = Depends(get_db)):
    """Get detailed product information."""
    try:
        from src.models.schemas import Product
        product = db.query(Product).filter(Product.asin == asin).first()
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get product error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get product: {str(e)}")


@app.get("/categories", response_model=List[str])
async def get_categories(db: Session = Depends(get_db)):
    """Get list of available product categories."""
    try:
        from src.models.schemas import Product
        categories = db.query(Product.category).distinct().all()
        return [cat[0] for cat in categories if cat[0]]
        
    except Exception as e:
        logger.error(f"Get categories error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Ensure data directory exists
    os.makedirs(settings.data_dir, exist_ok=True)
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )