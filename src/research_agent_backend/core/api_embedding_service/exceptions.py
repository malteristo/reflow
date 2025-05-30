"""
API embedding service exceptions.

This module provides exception classes for API-specific errors that can occur
during embedding service operations. These exceptions provide enhanced error
context and categorization for different failure modes.

Exception Hierarchy:
    APIError
    ├── RateLimitError
    └── AuthenticationError

All exceptions inherit from EmbeddingServiceError (base embedding service exception).
"""

from typing import Optional

from ..embedding_service import EmbeddingServiceError


class APIError(EmbeddingServiceError):
    """
    Exception raised for general API-related errors.
    
    This exception is raised when API requests fail due to server errors,
    invalid responses, or other API-specific issues.
    
    Attributes:
        status_code: HTTP status code if available
        message: Human-readable error description
    """
    
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        """
        Initialize API error with message and optional status code.
        
        Args:
            message: Human-readable error description
            status_code: HTTP status code if available
        """
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """
    Exception raised when API rate limits are exceeded.
    
    This exception includes retry timing information when available
    to help with backoff strategies.
    
    Attributes:
        retry_after: Suggested retry delay in seconds (if provided by API)
        status_code: Always 429 for rate limit errors
    """
    
    def __init__(self, message: str, retry_after: Optional[int] = None) -> None:
        """
        Initialize rate limit error with optional retry timing.
        
        Args:
            message: Human-readable error description
            retry_after: Suggested retry delay in seconds
        """
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """
    Exception raised for API authentication failures.
    
    This typically indicates invalid API keys, expired tokens,
    or insufficient permissions.
    
    Attributes:
        status_code: Always 401 for authentication errors
    """
    
    def __init__(self, message: str) -> None:
        """
        Initialize authentication error.
        
        Args:
            message: Human-readable error description
        """
        super().__init__(message, status_code=401) 