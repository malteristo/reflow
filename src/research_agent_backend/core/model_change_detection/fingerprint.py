"""
ModelFingerprint immutable value object for model state representation.

This module provides the core fingerprint data structure for tracking
embedding model states and detecting changes.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict

from .types import ModelType


@dataclass(frozen=True)  # Immutable for safety
class ModelFingerprint:
    """
    Immutable fingerprint representing a specific state of an embedding model.
    
    A ModelFingerprint serves as a cryptographic snapshot of an embedding model's
    state, including its configuration, version, and computed checksum. This enables
    reliable detection of model changes across service restarts and deployments.
    
    The fingerprint is designed to be:
    - Immutable: Prevents accidental modification after creation
    - Serializable: Can be stored and retrieved from persistent storage
    - Comparable: Supports equality operations for change detection
    - Hashable: Can be used as dictionary keys and in sets
    
    Attributes:
        model_name: Unique identifier for the model (e.g., "text-embedding-3-small")
        model_type: Type of model - either "local" or "api"
        version: Model version string (e.g., "1.0.0", "2024-01-15")
        checksum: Cryptographic hash of model configuration and state
        metadata: Additional model-specific metadata (dimensions, provider, etc.)
        created_at: Timestamp when the fingerprint was created
    
    Example:
        >>> fingerprint = ModelFingerprint(
        ...     model_name="multi-qa-MiniLM-L6-cos-v1",
        ...     model_type="local",
        ...     version="1.0.0",
        ...     checksum="a1b2c3d4e5f6",
        ...     metadata={"dimension": 384, "max_seq_length": 512}
        ... )
        >>> print(fingerprint.model_name)
        multi-qa-MiniLM-L6-cos-v1
    
    Raises:
        ValueError: If model_name or checksum are empty/invalid
    """
    
    model_name: str
    model_type: ModelType
    version: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """
        Validate fingerprint data integrity after initialization.
        
        Ensures that critical fields are properly populated and meet
        the requirements for reliable change detection.
        
        Raises:
            ValueError: If validation fails for any required field
        """
        if not self.model_name or not self.model_name.strip():
            raise ValueError("Model name must be non-empty")
        
        if not self.checksum or not self.checksum.strip():
            raise ValueError("Checksum must be non-empty")
        
        # Relaxed checksum length requirement for backward compatibility
        if len(self.checksum) < 3:
            raise ValueError("Checksum must be at least 3 characters")
        
        if self.model_type not in ("local", "api"):
            raise ValueError("Model type must be either 'local' or 'api'")
    
    def __eq__(self, other: object) -> bool:
        """
        Compare fingerprints for equality based on all relevant fields.
        
        Two fingerprints are considered equal if all their identifying
        characteristics match, excluding the creation timestamp which
        is not relevant for change detection.
        
        Args:
            other: Object to compare against
            
        Returns:
            True if fingerprints represent the same model state
        """
        if not isinstance(other, ModelFingerprint):
            return NotImplemented
        
        return (
            self.model_name == other.model_name
            and self.model_type == other.model_type
            and self.version == other.version
            and self.checksum == other.checksum
            and self.metadata == other.metadata
        )
    
    def __hash__(self) -> int:
        """
        Generate hash for use in dictionaries and sets.
        
        The hash is based on all identifying fields except metadata
        and created_at to ensure consistent hashing behavior.
        
        Returns:
            Integer hash value
        """
        return hash((
            self.model_name,
            self.model_type,
            self.version,
            self.checksum
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert fingerprint to dictionary for serialization.
        
        Produces a JSON-serializable dictionary representation that
        can be stored in persistent storage and reconstructed later.
        
        Returns:
            Dictionary representation with ISO timestamp format
        """
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelFingerprint':
        """
        Reconstruct fingerprint from dictionary representation.
        
        Safely deserializes a fingerprint from its dictionary form,
        handling type conversion and validation automatically.
        
        Args:
            data: Dictionary containing fingerprint data
            
        Returns:
            Reconstructed ModelFingerprint instance
            
        Raises:
            ValueError: If data is malformed or invalid
        """
        # Handle datetime deserialization in-place for performance
        if "created_at" in data and isinstance(data["created_at"], str):
            # Create copy only when needed for datetime conversion
            data = {**data, "created_at": datetime.fromisoformat(data["created_at"])}
        
        return cls(**data)
    
    def is_compatible_with(self, other: 'ModelFingerprint') -> bool:
        """
        Check if this fingerprint is compatible with another for caching purposes.
        
        Models are considered compatible if they have the same name and type,
        even if versions or metadata differ. This is useful for determining
        whether cached embeddings can be reused.
        
        Args:
            other: ModelFingerprint to compare compatibility with
            
        Returns:
            True if models are compatible for caching
        """
        return (
            self.model_name == other.model_name
            and self.model_type == other.model_type
        )
    
    def generate_cache_key(self) -> str:
        """
        Generate a unique cache key for this model configuration.
        
        Creates a deterministic cache key that can be used for cache
        invalidation and lookup operations.
        
        Returns:
            Unique string suitable for use as a cache key
        """
        key_data = f"{self.model_name}:{self.model_type}:{self.checksum}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16] 