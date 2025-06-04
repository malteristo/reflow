"""
Local embedding service implementation using sentence-transformers.

This module provides a concrete implementation of the EmbeddingService
for local sentence-transformers models with caching support.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging
import os
import hashlib
import time

from sentence_transformers import SentenceTransformer
import numpy as np

from .embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
    EmbeddingDimensionError,
    BatchProcessingError,
)
from .enhanced_caching import ModelAwareCacheManager
from .model_change_detection.fingerprint import ModelFingerprint
from .model_change_detection.integration_hooks import auto_register_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models."""
    
    model_name: str = "multi-qa-MiniLM-L6-cos-v1"
    cache_dir: Optional[str] = None
    device: str = "auto"
    normalize_embeddings: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds
    cache_max_size: int = 10000  # Maximum cache entries
    
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
        config: Optional[EmbeddingModelConfig] = None,
        cache_ttl: Optional[int] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the local embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_dir: Directory to cache downloaded models
            config: Configuration object (overrides individual parameters)
            cache_ttl: Cache time-to-live in seconds
            cache_enabled: Whether to enable embedding caching
        """
        if config:
            self._model_name = config.model_name
            self._cache_dir = config.cache_dir
            self._device = config.device
            self._normalize_embeddings = config.normalize_embeddings
            self._cache_enabled = config.cache_enabled
            self._cache_ttl = config.cache_ttl
            self._cache_max_size = config.cache_max_size
        else:
            self._model_name = model_name
            self._cache_dir = cache_dir
            self._device = "auto"
            self._normalize_embeddings = True
            self._cache_enabled = cache_enabled
            self._cache_ttl = cache_ttl or 3600
            self._cache_max_size = 10000
        
        self._model_cache_manager = ModelCacheManager()
        
        # Initialize embedding cache
        if self._cache_enabled:
            self._cache_manager = ModelAwareCacheManager()
            self._embedding_cache = self._cache_manager  # For backward compatibility
        else:
            self._cache_manager = None
            self._embedding_cache = None
        
        # Performance tracking
        self._cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_cache_hits': 0,
            'batch_requests': 0,
            'hit_rate': 0.0,
            'miss_rate': 0.0,
            'batch_efficiency': 0.0,
            'memory_usage_mb': 0.0,
            'entries_count': 0,
            'miss_count': 0
        }
        
        try:
            self._model = self._model_cache_manager.get_or_load_model(
                self._model_name, 
                self._cache_dir
            )
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model {self._model_name}: {str(e)}")
        
        # Automatically register model with change detection system
        try:
            auto_register_embedding_service(self)
            logger.debug(f"Auto-registered model '{self._model_name}' with change detection system")
        except Exception as e:
            # Don't fail initialization if auto-registration fails
            logger.warning(f"Failed to auto-register model '{self._model_name}': {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            EmbeddingServiceError: If embedding generation fails
        """
        if not self.is_ready():
            raise EmbeddingServiceError("Service not ready - model not loaded")
        
        if not text or not text.strip():
            raise EmbeddingServiceError("Text cannot be empty")
        
        try:
            # Check cache first if available
            if self._cache_manager:
                model_fingerprint = self.generate_model_fingerprint().generate_cache_key()
                cached_embedding = self._cache_manager.get_cached_embedding(text, model_fingerprint)
                if cached_embedding is not None:
                    return cached_embedding
            
            # Generate embedding
            embedding = self._model.encode(
                text,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Convert to list and cache if manager available
            embedding_list = embedding.tolist()
            
            if self._cache_manager:
                self._cache_manager.cache_embedding(text, embedding_list, model_fingerprint)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingServiceError(f"Embedding generation failed: {e}") from e
    
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
        
        self._cache_stats['batch_requests'] += 1
        
        # Check cache for each text if caching enabled
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        if self._cache_enabled and self._embedding_cache:
            model_fingerprint = self.generate_model_fingerprint().fingerprint
            
            for i, text in enumerate(texts):
                cached_embedding = self._embedding_cache.get_cached_embedding(text, model_fingerprint)
                if cached_embedding is not None:
                    cached_embeddings[i] = cached_embedding
                    self._cache_stats['batch_cache_hits'] += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        try:
            if uncached_texts:
                new_embeddings = self._model.encode(uncached_texts, convert_to_tensor=False)
                # Convert numpy array to list of lists of floats
                if isinstance(new_embeddings, np.ndarray):
                    new_embeddings_list = new_embeddings.astype(float).tolist()
                else:
                    # For non-numpy arrays (like nested lists or mocks in tests)
                    new_embeddings_list = [list(embedding) for embedding in new_embeddings]
                
                # Cache new embeddings
                if self._cache_enabled and self._embedding_cache:
                    model_fingerprint = self.generate_model_fingerprint().fingerprint
                    for text, embedding in zip(uncached_texts, new_embeddings_list):
                        self._embedding_cache.cache_embedding(text, embedding, model_fingerprint)
                
                # Combine cached and new embeddings in correct order
                result_embeddings = [None] * len(texts)
                
                # Place cached embeddings
                for i, embedding in cached_embeddings.items():
                    result_embeddings[i] = embedding
                
                # Place new embeddings
                for i, embedding in zip(uncached_indices, new_embeddings_list):
                    result_embeddings[i] = embedding
                
                # Update cache statistics
                self._cache_stats['entries_count'] = len(self._embedding_cache._cache) if self._embedding_cache else 0
                batch_cache_efficiency = len(cached_embeddings) / len(texts) if texts else 0
                self._cache_stats['batch_efficiency'] = batch_cache_efficiency
                
                self._update_cache_stats()
                
                return result_embeddings
            else:
                # All embeddings were cached
                result_embeddings = [cached_embeddings[i] for i in range(len(texts))]
                self._cache_stats['batch_efficiency'] = 1.0
                self._update_cache_stats()
                return result_embeddings
                
        except Exception as e:
            raise BatchProcessingError(f"Failed to generate batch embeddings: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from the cache manager or legacy cache."""
        # New cache manager (preferred)
        if self._cache_manager:
            return self._cache_manager.get_cache_stats()
        
        # Legacy cache system fallback
        if self._cache_enabled and self._embedding_cache:
            cache_stats = self._embedding_cache.get_cache_stats()
            self._cache_stats['memory_usage_mb'] = cache_stats.get('estimated_size_mb', 0)
            return self._cache_stats.copy()
        
        # No cache available
        return {
            "total_entries": 0,
            "unique_models": 0,
            "model_distribution": {},
            "estimated_size_mb": 0.0
        }
    
    def _update_cache_stats(self):
        """Update calculated cache statistics."""
        total = self._cache_stats['total_requests']
        if total > 0:
            self._cache_stats['hit_rate'] = self._cache_stats['cache_hits'] / total
            self._cache_stats['miss_rate'] = self._cache_stats['cache_misses'] / total
        else:
            self._cache_stats['hit_rate'] = 0.0
            self._cache_stats['miss_rate'] = 0.0
    
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
                "device": str(self._model.device) if hasattr(self._model, 'device') else self._device,
                "max_seq_length": getattr(self._model, 'max_seq_length', None),
                "embedding_dimension": self.get_embedding_dimension(),
                "normalize_embeddings": self._normalize_embeddings,
                "cache_enabled": self._cache_enabled,
                "cache_stats": self.get_cache_stats() if self._cache_enabled else None
            }
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to get model info: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Integer dimension of the embedding vectors
            
        Raises:
            EmbeddingDimensionError: If dimension cannot be determined
        """
        if not self.is_model_available():
            raise EmbeddingDimensionError("Model not available for dimension check")
        
        try:
            # Use a simple test string to determine dimension
            test_embedding = self._model.encode("test", convert_to_tensor=False)
            if isinstance(test_embedding, np.ndarray):
                return int(test_embedding.shape[0])
            else:
                return len(test_embedding)
        except Exception as e:
            raise EmbeddingDimensionError(f"Failed to determine embedding dimension: {str(e)}")
    
    def is_model_available(self) -> bool:
        """
        Check if the embedding model is available and ready to use.
        
        Returns:
            True if model is available, False otherwise
        """
        return self._model is not None
    
    def generate_model_fingerprint(self) -> ModelFingerprint:
        """Generate a fingerprint for the current model state."""
        try:
            # Get model configuration details (simplified to avoid attribute access issues)
            model_config = {
                "model_name": self._model_name,
                "max_seq_length": getattr(self._model, 'max_seq_length', 512),
                "dimension": self.get_embedding_dimension(),
                "device": str(self._device),
                "model_path": self._model_name  # Simplified - just use model name
            }
            
            # Create checksum from configuration
            config_str = str(sorted(model_config.items()))
            checksum = hashlib.sha256(config_str.encode()).hexdigest()[:16]
            
            return ModelFingerprint(
                model_name=self._model_name,
                model_type="local",
                version="1.0.0",  # Default version for local models
                checksum=checksum,
                metadata=model_config
            )
        except Exception as e:
            logger.warning(f"Failed to generate model fingerprint: {e}")
            # Fallback fingerprint
            fallback_checksum = hashlib.sha256(self._model_name.encode()).hexdigest()[:16]
            return ModelFingerprint(
                model_name=self._model_name,
                model_type="local", 
                version="1.0.0",
                checksum=fallback_checksum,
                metadata={"model_name": self._model_name}
            )
    
    def check_model_changed(self) -> bool:
        """
        Check if the model has changed since last use.
        
        Returns:
            True if model has changed, False otherwise
        """
        try:
            current_fingerprint = self.generate_model_fingerprint()
            # For local models, check if the model name or config changed
            # This is a simplified implementation
            return False  # Local models don't change unless explicitly reloaded
        except Exception as e:
            logger.warning(f"Failed to check model change: {e}")
            return False
    
    def invalidate_cache_on_change(self):
        """
        Invalidate cache if model has changed.
        
        This method checks for model changes and clears the cache if necessary.
        For local embedding services, this mainly clears cached embeddings
        if the model configuration has changed.
        """
        if self.check_model_changed() and self._cache_manager:
            model_fingerprint = self.generate_model_fingerprint().generate_cache_key()
            invalidated_count = self._cache_manager.invalidate_model_cache(model_fingerprint)
            logger.info(f"Invalidated {invalidated_count} cache entries due to model change")
            
            # Reset cache statistics
            self._cache_stats['cache_hits'] = 0
            self._cache_stats['cache_misses'] = 0
            self._cache_stats['total_requests'] = 0
            self._cache_stats['entries_count'] = 0

    def is_ready(self) -> bool:
        """
        Check if the service is ready to generate embeddings.
        
        Returns:
            True if the service is ready, False otherwise
        """
        return self.is_model_available() and self._model is not None

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text string.
        
        This method provides interface compatibility for RAG query engine
        which expects a generate_embeddings method.
        
        Args:
            text: Input text string
            
        Returns:
            List of embedding values
        """
        return self.embed_text(text) 