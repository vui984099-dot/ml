"""
FastAPI dependencies for the Amazon product API.
"""

from typing import Optional
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.schemas import Product


async def get_product_or_404(asin: str, db: Session = Depends(get_db)) -> Product:
    """Get product by ASIN or raise 404."""
    product = db.query(Product).filter(Product.asin == asin).first()
    
    if not product:
        raise HTTPException(status_code=404, detail=f"Product not found: {asin}")
    
    return product


async def validate_comparison_request(asin1: str, asin2: str, db: Session = Depends(get_db)):
    """Validate that both products exist for comparison."""
    product1 = db.query(Product).filter(Product.asin == asin1).first()
    product2 = db.query(Product).filter(Product.asin == asin2).first()
    
    if not product1:
        raise HTTPException(status_code=404, detail=f"Product not found: {asin1}")
    
    if not product2:
        raise HTTPException(status_code=404, detail=f"Product not found: {asin2}")
    
    if asin1 == asin2:
        raise HTTPException(status_code=400, detail="Cannot compare product with itself")
    
    return product1, product2


def validate_search_params(
    query: Optional[str] = None,
    category: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
    top_k: int = 10
):
    """Validate search parameters."""
    if not query and not category:
        raise HTTPException(
            status_code=400, 
            detail="Either query or category must be provided"
        )
    
    if top_k < 1 or top_k > 50:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 50"
        )
    
    if min_rating is not None and (min_rating < 1 or min_rating > 5):
        raise HTTPException(
            status_code=400,
            detail="min_rating must be between 1 and 5"
        )
    
    if max_price is not None and max_price < 0:
        raise HTTPException(
            status_code=400,
            detail="max_price must be positive"
        )
    
    return {
        "query": query,
        "category": category,
        "min_rating": min_rating,
        "max_price": max_price,
        "top_k": top_k
    }