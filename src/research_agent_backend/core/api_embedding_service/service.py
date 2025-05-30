"""
API embedding service implementation.

This module provides the main APIEmbeddingService class that orchestrates
all the modular components to provide a complete embedding service implementation.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

import requests

from ..embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
    EmbeddingDimensionError,
    BatchProcessingError,
)
from .config import APIConfiguration
from .client import APIClient
from .batch_processor import BatchProcessor
from .model_integration import ModelIntegration
from .exceptions import APIError, AuthenticationError, RateLimitError

logger = logging.getLogger(__name__)


class APIEmbeddingService(EmbeddingService):
    """
    API-based embedding service with comprehensive error handling and retry logic.
    
    This service provides embedding generation capabilities using external HTTP APIs
    with support for batch processing, automatic retries, rate limiting, and robust
    error handling. Implements the EmbeddingService abstract interface.
    
    Features:
        - Single text and batch embedding generation
        - Automatic chunking for large batches
        - Exponential backoff retry logic with configurable parameters
        - Comprehensive error handling for network and API errors
        - HTTP session reuse for performance optimization
        - Model information and dimension detection
        - Provider-specific authentication and endpoint handling
    
    Supported Providers:
        - OpenAI (text-embedding-3-small, text-embedding-3-large, etc.)
        - Anthropic (claude-3-haiku-20240307, etc.)
        - HuggingFace Inference API
        - Any OpenAI-compatible API endpoint
    
    Performance Characteristics:
        - HTTP connection pooling for reduced latency
        - Configurable batch sizes to optimize throughput
        - Memory-efficient streaming for large batches
        - Exponential backoff for resilient error handling
    
    Example:
        >>> config = APIConfiguration(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model_name="text-embedding-3-small"
        ... )
        >>> service = APIEmbeddingService(config)
        >>> 
        >>> # Single embedding
        >>> embedding = service.embed_text("Hello world")
        >>> 
        >>> # Batch processing with automatic chunking
        >>> embeddings = service.embed_batch([
        ...     "First document",
        ...     "Second document", 
        ...     "Third document"
        ... ])
        >>> 
        >>> # Model information
        >>> info = service.get_model_info()
        >>> dimension = service.get_embedding_dimension()
    
    Thread Safety:
        This class is thread-safe for read operations. Multiple threads can safely
        call embed_text() and embed_batch() concurrently. The HTTP session is
        thread-safe according to the requests library documentation.
    """
    
    def __init__(self, config: APIConfiguration) -> None:
        """
        Initialize the API embedding service with configuration.
        
        Creates and configures all modular components including HTTP client,
        batch processor, and model integration handler.
        
        Args:
            config: APIConfiguration instance with validated settings
            
        Raises:
            AuthenticationError: If API configuration is invalid (handled by APIConfiguration)
            
        Note:
            API key validation is performed by APIConfiguration.__post_init__
        """
        self.config = config
        self._cached_dimension: Optional[int] = None
        
        # Initialize modular components
        self.client = APIClient(config)
        self.batch_processor = BatchProcessor(config, self.client)
        self.model_integration = ModelIntegration(config)
    
    @classmethod
    def from_environment(cls, provider: str) -> 'APIEmbeddingService':
        """
        Create service instance from environment variables.
        
        Convenience method that creates an APIConfiguration from environment
        variables and initializes the service.
        
        Args:
            provider: API provider name (e.g., "openai", "anthropic")
            
        Returns:
            APIEmbeddingService instance configured from environment
            
        Raises:
            AuthenticationError: If required environment variables are missing
            
        Example:
            >>> # With OPENAI_API_KEY environment variable set
            >>> service = APIEmbeddingService.from_environment("openai")
        """
        config = APIConfiguration.from_environment(provider)
        return cls(config)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text string.
        
        Sends a request to the configured API endpoint to generate an embedding
        vector for the provided text. Includes comprehensive error handling and
        input validation.
        
        Args:
            text: Input text to embed. Must be non-empty after stripping whitespace.
            
        Returns:
            List of float values representing the embedding vector.
            Length depends on the model (e.g., 1536 for OpenAI text-embedding-3-small).
            
        Raises:
            EmbeddingServiceError: If text is empty or embedding generation fails
            ModelNotFoundError: If the embedding model is not available
            AuthenticationError: If API authentication fails  
            RateLimitError: If API rate limits are exceeded
            APIError: If API returns an error response
            
        Example:
            >>> service = APIEmbeddingService.from_environment("openai")
            >>> embedding = service.embed_text("Machine learning is fascinating")
            >>> print(f"Embedding dimension: {len(embedding)}")
            Embedding dimension: 1536
            
        Performance Notes:
            - Uses HTTP session pooling for efficient connections
            - Implements exponential backoff retry logic
            - Validates input before making API calls to avoid unnecessary requests
        """
        # Input validation with clear error messages
        if not text or text.strip() == "":
            raise EmbeddingServiceError(
                "Cannot embed empty text. Please provide non-empty text content."
            )
        
        # Model availability check
        if not self.is_model_available():
            raise ModelNotFoundError(
                f"Model '{self.config.model_name}' is not available. "
                f"Please check your configuration and API credentials."
            )
        
        # Construct API endpoint URL
        url = urljoin(self.config.base_url, "embeddings")
        
        # Prepare request payload using provider-specific format
        payload = {
            "input": text,
            "model": self.config.model_name
        }
        
        # Add provider-specific parameters if needed
        if self.config.provider == "openai":
            payload["encoding_format"] = "float"  # Ensure float format
        
        try:
            # Make API request with retry logic
            response = self.client.make_request_with_retry("POST", url, json=payload)
            response_data = response.json()
            
            # Extract embedding from response with validation
            if "data" in response_data and len(response_data["data"]) > 0:
                embedding_data = response_data["data"][0]
                if "embedding" in embedding_data:
                    embedding = [float(x) for x in embedding_data["embedding"]]
                    
                    # Log successful embedding generation for debugging
                    logger.debug(
                        f"Generated embedding for text (length: {len(text)}) "
                        f"with dimension {len(embedding)}"
                    )
                    
                    return embedding
                else:
                    raise EmbeddingServiceError("API response missing 'embedding' field")
            else:
                raise EmbeddingServiceError(
                    "Invalid API response: missing or empty 'data' field. "
                    "This may indicate an API format change or server error."
                )
                
        except requests.RequestException as e:
            # Network-level errors (connection, timeout, etc.)
            raise EmbeddingServiceError(f"Network error during embedding request: {str(e)}")
        except (json.JSONDecodeError, KeyError) as e:
            # JSON parsing or structure errors
            raise EmbeddingServiceError(
                f"Failed to parse API response: {str(e)}. "
                f"This may indicate an API format change or server error."
            )
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of text strings.
        
        Efficiently processes multiple texts with automatic chunking, parallel
        processing, and comprehensive error handling. Optimized for throughput
        while respecting API rate limits and batch size constraints.
        
        Args:
            texts: List of input texts to embed. All texts must be non-empty.
            
        Returns:
            List of embedding vectors, one for each input text in the same order.
            Each embedding is a list of float values.
            
        Raises:
            BatchProcessingError: If batch contains empty texts or processing fails
            EmbeddingServiceError: If embedding generation fails
            ModelNotFoundError: If the embedding model is not available
            AuthenticationError: If API authentication fails
            RateLimitError: If API rate limits are exceeded
            
        Example:
            >>> service = APIEmbeddingService.from_environment("openai")
            >>> texts = [
            ...     "First document about AI",
            ...     "Second document about machine learning", 
            ...     "Third document about neural networks"
            ... ]
            >>> embeddings = service.embed_batch(texts)
            >>> print(f"Processed {len(embeddings)} texts")
            Processed 3 texts
            
        Performance Notes:
            - Automatically chunks large batches based on max_batch_size
            - Uses HTTP session pooling for efficient connections
            - Processes chunks sequentially to avoid overwhelming the API
            - Memory-efficient processing for large datasets
        """
        # Model availability check
        if not self.is_model_available():
            raise ModelNotFoundError(
                f"Model '{self.config.model_name}' is not available for batch processing. "
                f"Please check your configuration and API credentials."
            )
        
        # Delegate to batch processor
        return self.batch_processor.process_batch(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model metadata
            
        Raises:
            EmbeddingServiceError: If model information cannot be retrieved
        """
        try:
            # Try to get model info from API
            url = urljoin(self.config.base_url, f"models/{self.config.model_name}")
            
            try:
                response = self.client._session.get(url)
                if response.status_code == 200:
                    api_info = response.json()
                else:
                    api_info = {}
            except:
                api_info = {}
            
            # Try to get embedding dimension, but don't fail if we can't
            try:
                dimension = self.get_embedding_dimension()
            except Exception:
                # If we can't determine dimension (e.g., no explicit config and API call fails),
                # use None as a fallback
                dimension = None
            
            # Build model info dictionary
            info = {
                "model_name": self.config.model_name,
                "model_type": "api",
                "provider": self.config.provider,
                "dimension": dimension,
                "api_info": api_info
            }
            
            return info
            
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
        # Return cached dimension if available
        if self._cached_dimension is not None:
            return self._cached_dimension
        
        # Use explicit dimension from config if provided
        if self.config.embedding_dimension:
            self._cached_dimension = self.config.embedding_dimension
            return self._cached_dimension
        
        # Determine dimension from API by making a test call
        try:
            test_embedding = self.embed_text("test")
            self._cached_dimension = len(test_embedding)
            return self._cached_dimension
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to determine embedding dimension: {str(e)}")
    
    def is_model_available(self) -> bool:
        """
        Check if the embedding model is available for use.
        
        Returns:
            True if model is available, False otherwise
        """
        # Basic availability check based on configuration
        if not self.config.api_key or not self.config.base_url or not self.config.model_name:
            return False
        
        return True
    
    # Model change detection methods - delegate to model integration
    def generate_model_fingerprint(self) -> 'ModelFingerprint':
        """
        Generate a model fingerprint for change detection.
        
        Returns:
            ModelFingerprint object containing model metadata and checksum
        """
        return self.model_integration.generate_model_fingerprint()
    
    def check_model_changed(self) -> bool:
        """
        Check if the model configuration has changed since last check.
        
        Returns:
            True if model configuration has changed, False otherwise
        """
        return self.model_integration.check_model_changed()
    
    def invalidate_cache_on_change(self):
        """
        Invalidate any cached data if model configuration has changed.
        
        For API services, this mainly clears the cached embedding dimension.
        """
        self.model_integration.invalidate_cache_on_change(self) 