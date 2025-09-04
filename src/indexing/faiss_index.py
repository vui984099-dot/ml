"""
FAISS index building and management for vector similarity search.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path

from src.config import settings
from src.models.schemas import ChunkMetadata
from src.indexing.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """Manages FAISS index creation, saving, and searching."""
    
    def __init__(self, embedding_dim: int = None):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunk_mapping = {}  # faiss_id -> chunk_id
        self.chunks_metadata = []
        
    def build_hnsw_index(self, embeddings: np.ndarray, 
                        chunks: List[ChunkMetadata]) -> None:
        """
        Build HNSW index for fast approximate nearest neighbor search.
        
        Args:
            embeddings: Numpy array of embeddings [num_chunks, embedding_dim]
            chunks: List of chunk metadata objects
        """
        if embeddings.size == 0:
            logger.warning("No embeddings provided for index building")
            return
            
        self.embedding_dim = embeddings.shape[1]
        logger.info(f"Building HNSW index with {len(embeddings)} vectors, dim={self.embedding_dim}")
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, settings.faiss_m)
        self.index.hnsw.efConstruction = settings.faiss_ef_construction
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Build chunk mapping
        self.chunks_metadata = chunks
        self.chunk_mapping = {chunk.faiss_id: chunk.chunk_id for chunk in chunks}
        
        logger.info(f"HNSW index built successfully. Total vectors: {self.index.ntotal}")
    
    def build_ivf_index(self, embeddings: np.ndarray, 
                       chunks: List[ChunkMetadata], nlist: int = 100) -> None:
        """
        Build IVF index for larger datasets.
        
        Args:
            embeddings: Numpy array of embeddings
            chunks: List of chunk metadata
            nlist: Number of clusters for IVF
        """
        if embeddings.size == 0:
            return
            
        self.embedding_dim = embeddings.shape[1]
        logger.info(f"Building IVF index with {len(embeddings)} vectors")
        
        # Create quantizer and IVF index
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        
        # Train the index
        self.index.train(embeddings.astype('float32'))
        
        # Add embeddings
        self.index.add(embeddings.astype('float32'))
        
        # Build chunk mapping
        self.chunks_metadata = chunks
        self.chunk_mapping = {chunk.faiss_id: chunk.chunk_id for chunk in chunks}
        
        logger.info(f"IVF index built successfully. Total vectors: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Search for similar chunks using the FAISS index.
        
        Args:
            query_embedding: Query embedding vector [1, embedding_dim]
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Set search parameters for HNSW
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = settings.faiss_ef_search
        
        # Search the index
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx in self.chunk_mapping:
                chunk_id = self.chunk_mapping[idx]
                results.append((chunk_id, float(sim)))
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> ChunkMetadata:
        """Get chunk metadata by chunk ID."""
        for chunk in self.chunks_metadata:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise ValueError(f"Chunk not found: {chunk_id}")
    
    def save_index(self, index_dir: str = None) -> None:
        """Save FAISS index and metadata to disk."""
        index_dir = index_dir or settings.index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(index_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(index_dir, "index_metadata.pkl")
        metadata = {
            "chunk_mapping": self.chunk_mapping,
            "chunks_metadata": self.chunks_metadata,
            "embedding_dim": self.embedding_dim,
            "model_name": getattr(self, 'model_name', settings.embedding_model)
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {index_dir}")
    
    def load_index(self, index_dir: str = None) -> None:
        """Load FAISS index and metadata from disk."""
        index_dir = index_dir or settings.index_dir
        
        # Load FAISS index
        index_path = os.path.join(index_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(index_dir, "index_metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        self.chunk_mapping = metadata["chunk_mapping"]
        self.chunks_metadata = metadata["chunks_metadata"]
        self.embedding_dim = metadata["embedding_dim"]
        
        logger.info(f"Index loaded from {index_dir}")
        logger.info(f"Index contains {self.index.ntotal} vectors")


class VectorSearchEngine:
    """High-level vector search interface combining embeddings and FAISS."""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.index_manager = FAISSIndexManager(self.embedding_generator.embedding_dim)
        self._index_built = False
    
    def build_index_from_chunks(self, chunks: List[ChunkMetadata]) -> None:
        """Build complete search index from text chunks."""
        logger.info("Building vector search index...")
        
        # Generate embeddings
        embeddings, chunks_with_ids = self.embedding_generator.encode_chunks(chunks)
        
        # Build FAISS index
        if len(embeddings) < 1000:
            # Use HNSW for smaller datasets
            self.index_manager.build_hnsw_index(embeddings, chunks_with_ids)
        else:
            # Use IVF for larger datasets
            nlist = min(int(np.sqrt(len(embeddings))), 1000)
            self.index_manager.build_ivf_index(embeddings, chunks_with_ids, nlist=nlist)
        
        self._index_built = True
        logger.info("Vector search index built successfully")
    
    def search_similar_chunks(self, query: str, top_k: int = 50) -> List[ChunkMetadata]:
        """
        Search for chunks similar to the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of ChunkMetadata objects sorted by similarity
        """
        if not self._index_built:
            raise ValueError("Index not built. Call build_index_from_chunks first.")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_texts([query])
        
        # Search index
        results = self.index_manager.search(query_embedding, top_k)
        
        # Convert to ChunkMetadata objects
        similar_chunks = []
        for chunk_id, score in results:
            try:
                chunk = self.index_manager.get_chunk_by_id(chunk_id)
                # Add similarity score as an attribute
                chunk.similarity_score = score
                similar_chunks.append(chunk)
            except ValueError:
                logger.warning(f"Chunk not found: {chunk_id}")
                continue
        
        return similar_chunks
    
    def save_index(self) -> None:
        """Save the complete search index."""
        self.index_manager.save_index()
    
    def load_index(self) -> None:
        """Load the complete search index."""
        self.index_manager.load_index()
        self._index_built = True


def main():
    """Build and test the vector search index."""
    from src.etl.text_processor import TextProcessor
    
    # Load chunks
    processor = TextProcessor()
    chunks = processor.load_chunks_metadata()
    
    if not chunks:
        logger.error("No chunks found. Run data ingestion first.")
        return
    
    # Build search engine
    search_engine = VectorSearchEngine()
    search_engine.build_index_from_chunks(chunks)
    
    # Save index
    search_engine.save_index()
    
    # Test search
    test_queries = [
        "good battery life",
        "sound quality",
        "easy to use",
        "value for money"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search_engine.search_similar_chunks(query, top_k=3)
        
        for i, chunk in enumerate(results):
            score = getattr(chunk, 'similarity_score', 0)
            print(f"{i+1}. Score: {score:.3f} | {chunk.text[:80]}...")


if __name__ == "__main__":
    main()