"""
Data models and schemas for the Amazon product recommendation system.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# SQLAlchemy Models
class Product(Base):
    __tablename__ = "products"
    
    asin = Column(String, primary_key=True)
    title = Column(Text)
    brand = Column(String)
    category = Column(String)
    price = Column(Float)
    description = Column(Text)
    image_url = Column(String)
    avg_rating = Column(Float)
    num_reviews = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    reviews = relationship("Review", back_populates="product")
    chunks = relationship("TextChunk", back_populates="product")


class Review(Base):
    __tablename__ = "reviews"
    
    review_id = Column(String, primary_key=True)
    asin = Column(String, ForeignKey("products.asin"))
    review_text = Column(Text)
    rating = Column(Integer)
    helpful_votes = Column(Integer)
    review_date = Column(DateTime)
    reviewer_id = Column(String)
    
    product = relationship("Product", back_populates="reviews")


class TextChunk(Base):
    __tablename__ = "text_chunks"
    
    chunk_id = Column(String, primary_key=True)
    asin = Column(String, ForeignKey("products.asin"))
    chunk_type = Column(String)  # 'review' or 'description'
    text = Column(Text)
    start_char = Column(Integer)
    end_char = Column(Integer)
    helpful_votes = Column(Integer)
    rating = Column(Integer)
    faiss_id = Column(Integer, unique=True)
    
    product = relationship("Product", back_populates="chunks")


# Pydantic Models for API
class ProductBase(BaseModel):
    asin: str
    title: str
    brand: Optional[str] = None
    category: str
    price: Optional[float] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    avg_rating: Optional[float] = None
    num_reviews: Optional[int] = 0


class ProductCreate(ProductBase):
    pass


class ProductResponse(ProductBase):
    created_at: datetime
    
    class Config:
        from_attributes = True


class ReviewBase(BaseModel):
    review_id: str
    asin: str
    review_text: str
    rating: int
    helpful_votes: int = 0
    reviewer_id: Optional[str] = None


class ReviewCreate(ReviewBase):
    review_date: Optional[datetime] = None


class ReviewResponse(ReviewBase):
    review_date: datetime
    
    class Config:
        from_attributes = True


class ChunkMetadata(BaseModel):
    chunk_id: str
    asin: str
    type: str  # 'review' or 'description'
    text: str
    start_char: int
    end_char: int
    helpful_votes: int = 0
    rating: Optional[int] = None
    faiss_id: Optional[int] = None


class SearchQuery(BaseModel):
    query: str
    category: Optional[str] = None
    min_rating: Optional[float] = None
    max_price: Optional[float] = None
    top_k: int = Field(default=10, ge=1, le=50)


class RetrievalResult(BaseModel):
    chunk_id: str
    asin: str
    text: str
    score: float
    chunk_type: str
    helpful_votes: int
    rating: Optional[int] = None


class QARequest(BaseModel):
    question: str
    category: Optional[str] = None
    max_chunks: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    asin: str
    review_id: Optional[str] = None
    chunk_id: str
    text_snippet: str
    rating: Optional[int] = None
    helpful_votes: int = 0


class QAResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    confidence_score: float
    suggested_products: List[ProductResponse]


class RecommendationRequest(BaseModel):
    asin: Optional[str] = None  # For similar products
    query: Optional[str] = None  # For query-based recommendations
    user_preferences: Optional[Dict[str, Any]] = None
    exclude_asins: List[str] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=20)


class RecommendationResponse(BaseModel):
    products: List[ProductResponse]
    recommendation_type: str  # 'similar', 'query_based', 'personalized'
    scores: List[float]


class ComparisonRequest(BaseModel):
    asin1: str
    asin2: str


class ComparisonResponse(BaseModel):
    product1: ProductResponse
    product2: ProductResponse
    comparison_summary: str
    key_differences: List[str]
    winner_categories: Dict[str, str]  # category -> asin of winner


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    database_status: str
    index_status: str
    model_status: str