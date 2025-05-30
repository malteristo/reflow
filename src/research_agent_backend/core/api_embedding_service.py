"""
API embedding service implementation using external APIs.

This module provides a concrete implementation of the EmbeddingService
abstract base class for API-based embedding providers like OpenAI, Anthropic, etc.

This module now serves as a compatibility layer that imports from the modular
api_embedding_service package. All functionality has been preserved while
improving code organization and maintainability.

For new code, consider importing directly from the package:
    from .api_embedding_service import APIEmbeddingService, APIConfiguration

Example Usage:
    >>> # Direct configuration
    >>> config = APIConfiguration(
    ...     provider="openai",
    ...     api_key="your-api-key",
    ...     model_name="text-embedding-3-small"
    ... )
    >>> service = APIEmbeddingService(config)
    >>> embedding = service.embed_text("Hello world")
    
    >>> # Environment-based configuration
    >>> service = APIEmbeddingService.from_environment("openai")
    >>> embeddings = service.embed_batch(["text1", "text2", "text3"])

Requirements:
    - requests>=2.28.0 for HTTP communication
    - Environment variables for API keys (e.g., OPENAI_API_KEY)
"""

# Import all classes from the modular package for backward compatibility
from .api_embedding_service import (
    APIEmbeddingService,
    APIConfiguration,
    APIError,
    RateLimitError,
    AuthenticationError,
    APIClient,
    BatchProcessor,
    ModelIntegration,
)

# Maintain the same __all__ list for backward compatibility
__all__ = [
    "APIEmbeddingService",
    "APIConfiguration", 
    "APIError",
    "RateLimitError",
    "AuthenticationError",
] 