"""
API embedding service configuration.

This module provides configuration management for API-based embedding services,
including validation, environment variable loading, and provider-specific defaults.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from .exceptions import AuthenticationError

logger = logging.getLogger(__name__)


@dataclass
class APIConfiguration:
    """
    Configuration for API embedding services with validation and defaults.
    
    This dataclass handles all configuration aspects for API-based embedding services,
    including authentication, endpoints, model selection, and operational parameters.
    Supports both explicit configuration and environment variable loading.
    
    Attributes:
        provider: API provider name (e.g., "openai", "anthropic")
        api_key: Authentication key for the API service
        base_url: Base URL for API endpoints (auto-detected if not provided)
        model_name: Name of the embedding model to use (auto-detected if not provided)
        embedding_dimension: Expected embedding vector dimension (optional)
        max_batch_size: Maximum number of texts to process in a single API call
        max_retries: Maximum number of retry attempts for failed requests
        retry_delay: Base delay in seconds between retry attempts (uses exponential backoff)
        timeout: HTTP request timeout in seconds
    
    Example:
        >>> # Basic configuration
        >>> config = APIConfiguration(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model_name="text-embedding-3-small"
        ... )
        
        >>> # Configuration with custom parameters
        >>> config = APIConfiguration(
        ...     provider="openai",
        ...     api_key="sk-...",
        ...     model_name="text-embedding-3-large",
        ...     embedding_dimension=3072,
        ...     max_batch_size=50,
        ...     max_retries=5,
        ...     timeout=60
        ... )
    
    Raises:
        AuthenticationError: If api_key is None or empty
    """
    
    provider: str
    api_key: Optional[str] = None
    base_url: str = ""
    model_name: str = ""
    embedding_dimension: Optional[int] = None
    max_batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    
    def __post_init__(self) -> None:
        """
        Validate configuration and set provider-specific defaults.
        
        Performs comprehensive validation of all configuration parameters
        and sets appropriate defaults for known providers.
        
        Raises:
            AuthenticationError: If API key is missing or invalid
            ValueError: If other configuration parameters are invalid
        """
        # Validate API key is present - required for all usage
        if not self.api_key:
            raise AuthenticationError("API key is required for API embedding service")
        
        # Validate provider is specified
        if not self.provider:
            raise ValueError("Provider must be specified (e.g., 'openai', 'anthropic')")
        
        # Set default base URLs for known providers
        if not self.base_url:
            if self.provider == "openai":
                self.base_url = "https://api.openai.com/v1"
            elif self.provider == "anthropic":
                self.base_url = "https://api.anthropic.com/v1"
            elif self.provider == "huggingface":
                self.base_url = "https://api-inference.huggingface.co"
            else:
                logger.warning(f"Unknown provider '{self.provider}', base_url must be set explicitly")
        
        # Set default model names for known providers
        if not self.model_name:
            if self.provider == "openai":
                self.model_name = "text-embedding-3-small"
            elif self.provider == "anthropic":
                self.model_name = "claude-3-haiku-20240307"
            else:
                logger.warning(f"Unknown provider '{self.provider}', model_name must be set explicitly")
        
        # Validate numerical parameters
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay <= 0:
            raise ValueError("retry_delay must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.embedding_dimension is not None and self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive if specified")
    
    @classmethod
    def from_environment(cls, provider: str) -> 'APIConfiguration':
        """
        Create configuration from environment variables.
        
        Loads configuration parameters from environment variables using
        standardized naming conventions: {PROVIDER}_API_KEY, {PROVIDER}_BASE_URL, etc.
        
        Args:
            provider: API provider name (will be uppercased for environment variable names)
            
        Returns:
            APIConfiguration instance with values loaded from environment
            
        Raises:
            AuthenticationError: If required API key environment variable is not found
            
        Example:
            >>> # With OPENAI_API_KEY set in environment
            >>> config = APIConfiguration.from_environment("openai")
            >>> assert config.provider == "openai"
            >>> assert config.api_key is not None
        """
        provider_upper = provider.upper()
        
        api_key = os.getenv(f"{provider_upper}_API_KEY")
        if not api_key:
            raise AuthenticationError(
                f"API key not found in environment variable {provider_upper}_API_KEY. "
                f"Please set this environment variable with your {provider} API key."
            )
        
        base_url = os.getenv(f"{provider_upper}_BASE_URL", "")
        model_name = os.getenv(f"{provider_upper}_MODEL_NAME", "")
        
        # Optional numerical parameters
        max_batch_size = int(os.getenv(f"{provider_upper}_MAX_BATCH_SIZE", "100"))
        max_retries = int(os.getenv(f"{provider_upper}_MAX_RETRIES", "3"))
        retry_delay = float(os.getenv(f"{provider_upper}_RETRY_DELAY", "1.0"))
        timeout = int(os.getenv(f"{provider_upper}_TIMEOUT", "30"))
        
        embedding_dimension = None
        if dim_env := os.getenv(f"{provider_upper}_EMBEDDING_DIMENSION"):
            embedding_dimension = int(dim_env)
        
        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            embedding_dimension=embedding_dimension,
            max_batch_size=max_batch_size,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout
        ) 