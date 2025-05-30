"""
API embedding service package.

This package provides a modular implementation of API-based embedding services
with support for multiple providers, comprehensive error handling, and robust
batch processing capabilities.

The package is organized into focused modules:
- exceptions: API-specific exception classes
- config: Configuration management and validation
- client: HTTP client with retry logic and connection pooling
- batch_processor: Batch processing optimization
- model_integration: Model change detection and fingerprinting
- service: Main orchestrating service class

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

Public API:
    - APIEmbeddingService: Main service class
    - APIConfiguration: Configuration dataclass
    - APIError, RateLimitError, AuthenticationError: Exception classes
    - APIClient, BatchProcessor, ModelIntegration: Component classes
"""

from .service import APIEmbeddingService
from .config import APIConfiguration
from .exceptions import APIError, RateLimitError, AuthenticationError, BatchProcessingError
from .client import APIClient
from .batch_processor import BatchProcessor
from .model_integration import ModelIntegration

__all__ = [
    # Main service class
    "APIEmbeddingService",
    
    # Configuration
    "APIConfiguration",
    
    # Exceptions
    "APIError",
    "RateLimitError", 
    "AuthenticationError",
    "BatchProcessingError",
    
    # Component classes (for advanced usage)
    "APIClient",
    "BatchProcessor",
    "ModelIntegration",
]

# For backward compatibility, maintain direct access to the main classes
# This ensures existing imports continue to work:
# from .api_embedding_service import APIEmbeddingService, APIConfiguration 