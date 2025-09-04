"""
CTR (Click-Through Rate) prediction and conversion modeling.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
from sqlalchemy.orm import Session

from src.config import settings
from src.models.schemas import Product, Review
from src.database import SessionLocal

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generates synthetic interaction logs for CTR modeling."""
    
    def __init__(self):
        pass  # Will get DB session when needed
    
    def generate_interaction_logs(self, num_interactions: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic user interaction logs.
        
        Args:
            num_interactions: Number of synthetic interactions to generate
            
        Returns:
            DataFrame with interaction logs
        """
        logger.info(f"Generating {num_interactions} synthetic interaction logs")
        
        # Get products from database
        db = SessionLocal()
        try:
            products = db.query(Product).all()
        finally:
            db.close()
        
        if not products:
            logger.warning("No products found in database")
            return pd.DataFrame()
        
        interactions = []
        
        for i in range(num_interactions):
            # Randomly select a product
            product = np.random.choice(products)
            
            # Generate synthetic user query
            query = self._generate_synthetic_query(product)
            
            # Calculate features
            features = self._calculate_interaction_features(query, product)
            
            # Simulate click probability based on features
            click_prob = self._simulate_click_probability(features)
            clicked = np.random.random() < click_prob
            
            # Simulate conversion probability (if clicked)
            conversion_prob = self._simulate_conversion_probability(features) if clicked else 0.0
            converted = clicked and (np.random.random() < conversion_prob)
            
            interaction = {
                "interaction_id": f"int_{i:06d}",
                "user_id": f"user_{np.random.randint(1, 1000):04d}",
                "query": query,
                "asin": product.asin,
                "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 365)),
                "clicked": int(clicked),
                "converted": int(converted),
                **features  # Add all calculated features
            }
            
            interactions.append(interaction)
        
        df = pd.DataFrame(interactions)
        logger.info(f"Generated {len(df)} synthetic interactions")
        
        return df
    
    def _generate_synthetic_query(self, product: Product) -> str:
        """Generate a realistic search query for a product."""
        query_templates = [
            f"{product.brand} {product.category.lower()}",
            f"best {product.category.lower()}",
            f"cheap {product.category.lower()}",
            f"high quality {product.category.lower()}",
            f"review {product.brand}",
            product.title.split()[0] if product.title else product.category.lower()
        ]
        
        return np.random.choice(query_templates)
    
    def _calculate_interaction_features(self, query: str, product: Product) -> Dict[str, float]:
        """Calculate features for CTR prediction."""
        features = {}
        
        # Product features
        features["price"] = product.price or 50.0
        features["avg_rating"] = product.avg_rating or 3.0
        features["num_reviews"] = product.num_reviews or 0
        features["log_num_reviews"] = np.log1p(features["num_reviews"])
        
        # Price features
        features["price_log"] = np.log1p(features["price"])
        features["is_expensive"] = 1.0 if features["price"] > 100 else 0.0
        features["is_budget"] = 1.0 if features["price"] < 30 else 0.0
        
        # Rating features
        features["is_highly_rated"] = 1.0 if features["avg_rating"] > 4.0 else 0.0
        features["rating_squared"] = features["avg_rating"] ** 2
        
        # Query-product relevance features
        query_words = set(query.lower().split())
        title_words = set(product.title.lower().split()) if product.title else set()
        brand_words = set(product.brand.lower().split()) if product.brand else set()
        
        features["title_overlap"] = len(query_words & title_words) / max(len(query_words), 1)
        features["brand_match"] = 1.0 if query_words & brand_words else 0.0
        features["query_length"] = len(query.split())
        
        # Category features
        features["category_electronics"] = 1.0 if "electronics" in product.category.lower() else 0.0
        
        # Popularity features
        features["popularity_score"] = (features["avg_rating"] * np.log1p(features["num_reviews"])) / 20.0
        
        return features
    
    def _simulate_click_probability(self, features: Dict[str, float]) -> float:
        """Simulate realistic click probability based on features."""
        # Base probability
        base_prob = 0.1
        
        # Rating boost
        rating_boost = (features["avg_rating"] - 3.0) * 0.1
        
        # Relevance boost
        relevance_boost = features["title_overlap"] * 0.3
        
        # Brand match boost
        brand_boost = features["brand_match"] * 0.2
        
        # Price penalty for very expensive items
        price_penalty = -0.05 if features["is_expensive"] else 0.0
        
        # Review count boost
        review_boost = min(features["log_num_reviews"] * 0.02, 0.2)
        
        click_prob = base_prob + rating_boost + relevance_boost + brand_boost + price_penalty + review_boost
        
        return max(0.01, min(0.8, click_prob))  # Clamp between 1% and 80%
    
    def _simulate_conversion_probability(self, features: Dict[str, float]) -> float:
        """Simulate conversion probability given a click."""
        # Base conversion rate
        base_conv = 0.05
        
        # High rating increases conversion
        rating_boost = (features["avg_rating"] - 3.0) * 0.02
        
        # Many reviews increase trust
        review_boost = min(features["log_num_reviews"] * 0.01, 0.1)
        
        # Price affects conversion (sweet spot around $50-100)
        price = features["price"]
        if 30 <= price <= 100:
            price_boost = 0.03
        elif price > 200:
            price_boost = -0.02
        else:
            price_boost = 0.0
        
        conv_prob = base_conv + rating_boost + review_boost + price_boost
        
        return max(0.01, min(0.3, conv_prob))  # Clamp between 1% and 30%


class CTRModel:
    """CTR prediction model using LightGBM."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    def prepare_features(self, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training."""
        # Define feature columns
        feature_columns = [
            "price", "price_log", "avg_rating", "rating_squared", "num_reviews", "log_num_reviews",
            "is_expensive", "is_budget", "is_highly_rated", "title_overlap", "brand_match",
            "query_length", "category_electronics", "popularity_score"
        ]
        
        # Extract features
        X = interactions_df[feature_columns].fillna(0).values
        y = interactions_df["clicked"].values
        
        self.feature_names = feature_columns
        
        return X, y
    
    def train(self, interactions_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train CTR prediction model.
        
        Args:
            interactions_df: DataFrame with interaction logs
            
        Returns:
            Training metrics
        """
        logger.info("Training CTR model...")
        
        # Prepare data
        X, y = self.prepare_features(interactions_df)
        
        if len(X) == 0:
            raise ValueError("No training data available")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train LightGBM model
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        auc_score = roc_auc_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)
        
        self.is_trained = True
        
        metrics = {
            "auc": auc_score,
            "logloss": logloss,
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        
        logger.info(f"CTR model trained successfully. AUC: {auc_score:.3f}, LogLoss: {logloss:.3f}")
        
        return metrics
    
    def predict_ctr(self, features: Dict[str, float]) -> float:
        """Predict CTR for a single interaction."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert features to array
        feature_array = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        
        # Predict
        ctr_score = self.model.predict(feature_array, num_iteration=self.model.best_iteration)[0]
        
        return float(ctr_score)
    
    def batch_predict_ctr(self, features_list: List[Dict[str, float]]) -> List[float]:
        """Predict CTR for multiple interactions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to feature matrix
        feature_matrix = np.array([
            [features.get(name, 0.0) for name in self.feature_names]
            for features in features_list
        ])
        
        # Predict
        ctr_scores = self.model.predict(feature_matrix, num_iteration=self.model.best_iteration)
        
        return ctr_scores.tolist()
    
    def save_model(self, filepath: str = None) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        filepath = filepath or os.path.join(settings.models_dir, "ctr_model.pkl")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"CTR model saved to {filepath}")
    
    def load_model(self, filepath: str = None) -> None:
        """Load trained model from disk."""
        filepath = filepath or os.path.join(settings.models_dir, "ctr_model.pkl")
        
        if not os.path.exists(filepath):
            logger.warning(f"CTR model file not found: {filepath}")
            return
        
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.is_trained = model_data["is_trained"]
        
        logger.info(f"CTR model loaded from {filepath}")


class RankingOptimizer:
    """Optimizes search ranking using CTR predictions."""
    
    def __init__(self, ctr_model: CTRModel):
        self.ctr_model = ctr_model
    
    def rerank_with_ctr(self, query: str, products: List[ProductResponse], 
                       base_scores: List[float]) -> Tuple[List[ProductResponse], List[float]]:
        """
        Rerank products using CTR predictions.
        
        Args:
            query: Search query
            products: List of candidate products
            base_scores: Base relevance scores
            
        Returns:
            Tuple of (reranked_products, combined_scores)
        """
        if not self.ctr_model.is_trained:
            logger.warning("CTR model not trained, returning original ranking")
            return products, base_scores
        
        # Calculate CTR features for each product
        features_list = []
        for product in products:
            features = self._calculate_ctr_features(query, product)
            features_list.append(features)
        
        # Predict CTR scores
        ctr_scores = self.ctr_model.batch_predict_ctr(features_list)
        
        # Combine base scores with CTR predictions
        combined_scores = []
        for base_score, ctr_score in zip(base_scores, ctr_scores):
            # Weight: 70% relevance, 30% predicted CTR
            combined_score = base_score * 0.7 + ctr_score * 0.3
            combined_scores.append(combined_score)
        
        # Sort by combined scores
        product_score_pairs = list(zip(products, combined_scores))
        product_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_products = [p for p, s in product_score_pairs]
        reranked_scores = [s for p, s in product_score_pairs]
        
        return reranked_products, reranked_scores
    
    def _calculate_ctr_features(self, query: str, product: ProductResponse) -> Dict[str, float]:
        """Calculate CTR features for a query-product pair."""
        features = {}
        
        # Product features
        features["price"] = product.price or 50.0
        features["avg_rating"] = product.avg_rating or 3.0
        features["num_reviews"] = product.num_reviews or 0
        features["log_num_reviews"] = np.log1p(features["num_reviews"])
        
        # Price features
        features["price_log"] = np.log1p(features["price"])
        features["is_expensive"] = 1.0 if features["price"] > 100 else 0.0
        features["is_budget"] = 1.0 if features["price"] < 30 else 0.0
        
        # Rating features
        features["is_highly_rated"] = 1.0 if features["avg_rating"] > 4.0 else 0.0
        features["rating_squared"] = features["avg_rating"] ** 2
        
        # Query-product relevance
        query_words = set(query.lower().split())
        title_words = set(product.title.lower().split()) if product.title else set()
        brand_words = set(product.brand.lower().split()) if product.brand else set()
        
        features["title_overlap"] = len(query_words & title_words) / max(len(query_words), 1)
        features["brand_match"] = 1.0 if query_words & brand_words else 0.0
        features["query_length"] = len(query.split())
        
        # Category features
        features["category_electronics"] = 1.0 if "electronics" in product.category.lower() else 0.0
        
        # Popularity features
        features["popularity_score"] = (features["avg_rating"] * np.log1p(features["num_reviews"])) / 20.0
        
        return features


def train_ctr_model() -> CTRModel:
    """Train and save CTR model with synthetic data."""
    logger.info("Starting CTR model training pipeline...")
    
    # Generate synthetic data
    data_generator = SyntheticDataGenerator()
    interactions_df = data_generator.generate_interaction_logs(num_interactions=10000)
    
    if interactions_df.empty:
        logger.error("No interaction data generated")
        return None
    
    # Train model
    ctr_model = CTRModel()
    metrics = ctr_model.train(interactions_df)
    
    # Save model
    ctr_model.save_model()
    
    # Save training data for analysis
    os.makedirs(settings.data_dir, exist_ok=True)
    interactions_df.to_parquet(os.path.join(settings.data_dir, "synthetic_interactions.parquet"))
    
    logger.info(f"CTR model training complete. Metrics: {metrics}")
    
    return ctr_model


def main():
    """Train and test CTR model."""
    # Train model
    ctr_model = train_ctr_model()
    
    if ctr_model is None:
        print("CTR model training failed")
        return
    
    # Test prediction
    test_features = {
        "price": 75.0,
        "price_log": np.log1p(75.0),
        "avg_rating": 4.2,
        "rating_squared": 4.2 ** 2,
        "num_reviews": 150,
        "log_num_reviews": np.log1p(150),
        "is_expensive": 0.0,
        "is_budget": 0.0,
        "is_highly_rated": 1.0,
        "title_overlap": 0.6,
        "brand_match": 1.0,
        "query_length": 3,
        "category_electronics": 1.0,
        "popularity_score": 0.8
    }
    
    predicted_ctr = ctr_model.predict_ctr(test_features)
    print(f"Predicted CTR: {predicted_ctr:.3f}")
    
    # Test ranking optimizer
    # This would need actual products and scores in a real scenario
    print("CTR model training and testing completed successfully!")


if __name__ == "__main__":
    main()