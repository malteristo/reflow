"""
Embedding generation and vector processing integration.

This module provides embedding services for document insertion pipeline
with support for both single and batch embedding generation.
"""

import logging
from typing import Any, List, Optional

from .exceptions import InsertionError


class EmbeddingService:
    """Embedding service for document insertion pipeline."""
    
    def __init__(
        self, 
        embedding_service: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize embedding service.
        
        Args:
            embedding_service: External embedding service (Task 4 dependency)
            logger: Optional logger instance
        """
        self.embedding_service = embedding_service
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            InsertionError: If embedding generation fails
        """
        if self.embedding_service is None:
            # Mock embedding for testing and GREEN PHASE
            # TODO: Replace with actual embedding service in REFACTOR PHASE
            return [0.1, 0.2, 0.3, 0.4, 0.5]
        
        try:
            return self.embedding_service.embed_text(text)
        except Exception as e:
            raise InsertionError(f"Failed to generate embeddings: {e}") from e
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            InsertionError: If batch embedding generation fails
        """
        if self.embedding_service is None:
            # Mock embeddings for testing and GREEN PHASE
            return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]
        
        try:
            return self.embedding_service.embed_batch(texts)
        except Exception as e:
            raise InsertionError(f"Failed to generate batch embeddings: {e}") from e 