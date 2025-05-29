"""
Abstract base class for embedding services.

This module defines the interface that all embedding service implementations
must follow, ensuring consistency across local and API-based services.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""
    pass


class ModelNotFoundError(EmbeddingServiceError):
    """Raised when requested model is not available."""
    pass


class EmbeddingDimensionError(EmbeddingServiceError):
    """Raised when embedding dimensions are inconsistent."""
    pass


class BatchProcessingError(EmbeddingServiceError):
    """Raised when batch processing fails."""
    pass


class EmbeddingService(ABC):
    """
    Abstract base class for embedding services.
    
    This class defines the interface that all embedding service implementations
    must follow, including both local sentence-transformers models and API-based
    providers like OpenAI or Cohere.
    
    All implementing classes must provide:
    - Single text embedding generation
    - Batch text embedding generation
    - Model information and metadata
    - Embedding dimension information
    - Model availability checking
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            EmbeddingServiceError: If embedding generation fails
            ModelNotFoundError: If the embedding model is not available
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model metadata including:
            - model_name: Name/identifier of the model
            - dimension: Embedding vector dimension
            - max_seq_length: Maximum sequence length (optional)
            - model_type: Type of model (local/api)
            
        Raises:
            EmbeddingServiceError: If model information cannot be retrieved
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            Integer representing the embedding vector dimension
            
        Raises:
            EmbeddingServiceError: If dimension cannot be determined
        """
        pass
    
    @abstractmethod
    def is_model_available(self) -> bool:
        """
        Check if the embedding model is available for use.
        
        Returns:
            True if model is available, False otherwise
        """
        pass 