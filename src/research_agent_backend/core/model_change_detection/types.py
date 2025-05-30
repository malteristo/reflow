"""
Type aliases and exception hierarchy for model change detection.

This module provides type definitions and custom exceptions for the 
model change detection system.
"""

from typing import Literal

from ..embedding_service import EmbeddingServiceError

# Type aliases for better code clarity
ModelType = Literal["local", "api"]
ChangeType = Literal["new_model", "version_update", "config_change", "checksum_change"]


# Custom exception hierarchy for model change detection
class ModelChangeError(EmbeddingServiceError):
    """
    Base exception for model change detection operations.
    
    Inherits from EmbeddingServiceError to maintain consistency with
    the broader embedding service error hierarchy.
    """
    pass


class FingerprintMismatchError(ModelChangeError):
    """
    Raised when model fingerprints indicate unexpected changes.
    
    This exception is typically raised when a model checksum doesn't
    match the expected value, indicating potential model corruption
    or unexpected model updates.
    """
    pass


class PersistenceError(ModelChangeError):
    """
    Raised when persistent storage operations fail.
    
    Covers file I/O errors, serialization failures, and storage
    corruption issues during fingerprint persistence operations.
    """
    pass 