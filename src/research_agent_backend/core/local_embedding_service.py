"""
Local embedding service implementation using sentence-transformers.

This module provides a concrete implementation of the EmbeddingService
for local sentence-transformers models with caching support.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import os

from sentence_transformers import SentenceTransformer
import numpy as np

from .embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
    EmbeddingDimensionError,
    BatchProcessingError,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models."""
    
    model_name: str = "multi-qa-MiniLM-L6-cos-v1"
    cache_dir: Optional[str] = None
    device: str = "auto"
    normalize_embeddings: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if not self.model_name or self.model_name.strip() == "":
            raise ValueError("Model name cannot be empty")
        
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if self.device not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}, got: {self.device}")


class ModelCacheManager:
    """Singleton manager for caching sentence transformer models."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}  # Instance-level cache, not class-level
        return cls._instance
    
    def get_or_load_model(self, model_name: str, cache_dir: Optional[str] = None) -> SentenceTransformer:
        """Get model from cache or load if not cached."""
        cache_key = f"{model_name}:{cache_dir or 'default'}"
        
        if cache_key not in self._cache:
            try:
                model = SentenceTransformer(model_name, cache_folder=cache_dir)
                self._cache[cache_key] = model
                logger.info(f"Loaded model {model_name} into cache")
            except Exception as e:
                raise ModelNotFoundError(f"Failed to load model {model_name}: {str(e)}")
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Clear all cached models."""
        self._cache.clear()
        logger.info("Cleared model cache")


class LocalEmbeddingService(EmbeddingService):
    """
    Local embedding service using sentence-transformers models.
    
    This service provides embedding generation capabilities using locally
    downloaded sentence-transformers models with caching support.
    """
    
    def __init__(
        self,
        model_name: str = "multi-qa-MiniLM-L6-cos-v1",
        cache_dir: Optional[str] = None,
        config: Optional[EmbeddingModelConfig] = None
    ):
        """
        Initialize the local embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_dir: Directory to cache downloaded models
            config: Configuration object (overrides individual parameters)
        """
        if config:
            self._model_name = config.model_name
            self._cache_dir = config.cache_dir
            self._device = config.device
            self._normalize_embeddings = config.normalize_embeddings
        else:
            self._model_name = model_name
            self._cache_dir = cache_dir
            self._device = "auto"
            self._normalize_embeddings = True
        
        self._cache_manager = ModelCacheManager()
        
        try:
            self._model = self._cache_manager.get_or_load_model(
                self._model_name, 
                self._cache_dir
            )
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model {self._model_name}: {str(e)}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            EmbeddingServiceError: If text is empty or embedding generation fails
            ModelNotFoundError: If the embedding model is not available
        """
        if not text or text.strip() == "":
            raise EmbeddingServiceError("Cannot embed empty text")
        
        if not self.is_model_available():
            raise ModelNotFoundError("Model not available for embedding")
        
        try:
            embedding = self._model.encode(text, convert_to_tensor=False)
            # Convert numpy array to list of floats
            if isinstance(embedding, np.ndarray):
                return embedding.astype(float).tolist()
            else:
                # For non-numpy arrays (like lists or mocks in tests)
                return list(embedding)
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to generate embedding: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text strings.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors, one for each input text
            
        Raises:
            BatchProcessingError: If batch processing fails
            EmbeddingServiceError: If embedding generation fails
            ModelNotFoundError: If the embedding model is not available
        """
        if not texts:
            return []
        
        # Check for empty strings in batch
        empty_texts = [i for i, text in enumerate(texts) if not text or text.strip() == ""]
        if empty_texts:
            raise BatchProcessingError(f"Batch contains empty texts at positions: {empty_texts}")
        
        if not self.is_model_available():
            raise ModelNotFoundError("Model not available for batch embedding")
        
        try:
            embeddings = self._model.encode(texts, convert_to_tensor=False)
            # Convert numpy array to list of lists of floats
            if isinstance(embeddings, np.ndarray):
                return embeddings.astype(float).tolist()
            else:
                # For non-numpy arrays (like nested lists or mocks in tests)
                return [list(embedding) for embedding in embeddings]
        except Exception as e:
            raise BatchProcessingError(f"Failed to generate batch embeddings: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model metadata
            
        Raises:
            EmbeddingServiceError: If model information cannot be retrieved
        """
        if not self.is_model_available():
            raise EmbeddingServiceError("Model not available for info retrieval")
        
        try:
            return {
                "model_name": self._model_name,
                "dimension": self._model.get_sentence_embedding_dimension(),
                "max_seq_length": self._model.get_max_seq_length(),
                "model_type": "local",
                "library": "sentence-transformers"
            }
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to get model info: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            Integer representing the embedding vector dimension
            
        Raises:
            EmbeddingServiceError: If dimension cannot be determined
        """
        if not self.is_model_available():
            raise EmbeddingServiceError("Model not available for dimension retrieval")
        
        try:
            return self._model.get_sentence_embedding_dimension()
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to get embedding dimension: {str(e)}")
    
    def is_model_available(self) -> bool:
        """
        Check if the embedding model is available for use.
        
        Returns:
            True if model is available, False otherwise
        """
        return hasattr(self, '_model') and self._model is not None 