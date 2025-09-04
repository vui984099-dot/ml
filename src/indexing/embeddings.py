"""
Embedding generation using sentence transformers.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging
from pathlib import Path
import pickle
import os

from src.config import settings
from src.models.schemas import ChunkMetadata

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles embedding generation for text chunks."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Using device: {self.device}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings [num_texts, embedding_dim]
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def encode_chunks(self, chunks: List[ChunkMetadata], 
                     batch_size: int = 32) -> Tuple[np.ndarray, List[ChunkMetadata]]:
        """
        Generate embeddings for text chunks.
        
        Returns:
            Tuple of (embeddings array, chunks list with faiss_ids)
        """
        if not chunks:
            return np.array([]).reshape(0, self.embedding_dim), []
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.encode_texts(texts, batch_size=batch_size)
        
        # Assign FAISS IDs
        for i, chunk in enumerate(chunks):
            chunk.faiss_id = i
        
        return embeddings, chunks
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[ChunkMetadata], 
                       filename: str = "embeddings.pkl"):
        """Save embeddings and metadata to disk."""
        os.makedirs(settings.data_dir, exist_ok=True)
        filepath = os.path.join(settings.data_dir, filename)
        
        data = {
            "embeddings": embeddings,
            "chunks": chunks,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filename: str = "embeddings.pkl") -> Tuple[np.ndarray, List[ChunkMetadata]]:
        """Load embeddings and metadata from disk."""
        filepath = os.path.join(settings.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        embeddings = data["embeddings"]
        chunks = data["chunks"]
        
        logger.info(f"Loaded embeddings shape: {embeddings.shape}")
        logger.info(f"Loaded {len(chunks)} chunks metadata")
        
        return embeddings, chunks


class CrossEncoderReranker:
    """Cross-encoder model for reranking retrieved chunks."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.cross_encoder_model
        logger.info(f"Loading cross-encoder model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
    
    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Score query-text pairs using cross-encoder.
        
        Args:
            query: Search query
            texts: List of candidate texts
            
        Returns:
            List of relevance scores
        """
        if not texts:
            return []
        
        # Create query-text pairs
        pairs = [(query, text) for text in texts]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
    
    def rerank_chunks(self, query: str, chunks: List[ChunkMetadata], 
                     top_k: int = None) -> List[Tuple[ChunkMetadata, float]]:
        """
        Rerank chunks using cross-encoder scores.
        
        Args:
            query: Search query
            chunks: List of candidate chunks
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, score) tuples sorted by score
        """
        if not chunks:
            return []
        
        top_k = top_k or settings.rerank_top_k
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Get cross-encoder scores
        scores = self.score_pairs(query, texts)
        
        # Combine chunks with scores and sort
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return chunk_scores[:top_k]


def main():
    """Test embedding generation pipeline."""
    # Load processed chunks
    processor = TextProcessor()
    chunks = processor.load_chunks_metadata()
    
    if not chunks:
        logger.warning("No chunks found. Run data ingestion first.")
        return
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings, chunks_with_ids = generator.encode_chunks(chunks)
    
    # Save embeddings
    generator.save_embeddings(embeddings, chunks_with_ids)
    
    # Test cross-encoder
    reranker = CrossEncoderReranker()
    sample_query = "good battery life"
    sample_chunks = chunks_with_ids[:5]
    
    reranked = reranker.rerank_chunks(sample_query, sample_chunks)
    
    print(f"\nReranking results for query: '{sample_query}'")
    for i, (chunk, score) in enumerate(reranked):
        print(f"{i+1}. Score: {score:.3f} | {chunk.text[:100]}...")


if __name__ == "__main__":
    main()