"""
Model change detection system for embedding services.

This module provides a comprehensive model change detection system that enables
automatic detection of embedding model changes, cache invalidation, and re-indexing
triggers for vector databases. It implements the Observer pattern for event-driven
architecture and provides persistent storage for model fingerprints.

Key Features:
- Model fingerprinting with cryptographic checksums for change detection
- Singleton-based detector with thread-safe operations
- Event-driven re-indexing triggers for vector database updates
- Persistent storage with automatic serialization/deserialization
- Integration with both local and API-based embedding services
- Comprehensive error handling with custom exception hierarchy

Architecture:
- ModelFingerprint: Immutable value object for model metadata
- ModelChangeDetector: Singleton service for centralized change detection
- ModelChangeEvent: Event object for re-indexing notifications
- Custom exceptions for fine-grained error handling

Usage:
    >>> detector = ModelChangeDetector()
    >>> fingerprint = ModelFingerprint(
    ...     model_name="text-embedding-3-small",
    ...     model_type="api",
    ...     version="1.0.0",
    ...     checksum="abc123",
    ...     metadata={"provider": "openai"}
    ... )
    >>> changed = detector.detect_change(fingerprint)
    >>> if changed:
    ...     detector.register_model(fingerprint)

Implementation follows FR-KB-005 (Model Change Detection) from the PRD.
"""

import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from .embedding_service import EmbeddingServiceError

# Module-level logger for consistent logging across all classes
logger = logging.getLogger(__name__)

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


@dataclass
class ModelChangeEvent:
    """
    Event object representing a detected model change requiring action.
    
    ModelChangeEvent serves as a notification mechanism for model changes
    that may require downstream actions such as cache invalidation or
    vector database re-indexing. It follows the Observer pattern to
    enable loose coupling between change detection and response logic.
    
    Attributes:
        model_name: Name of the changed model
        change_type: Type of change that occurred
        old_fingerprint: Previous model state (None for new models)
        new_fingerprint: Current model state
        requires_reindexing: Whether vector database re-indexing is needed
        timestamp: When the change was detected
        metadata: Additional event-specific information
    
    Example:
        >>> event = ModelChangeEvent(
        ...     model_name="text-embedding-3-small",
        ...     change_type="version_update",
        ...     old_fingerprint=old_fp,
        ...     new_fingerprint=new_fp,
        ...     requires_reindexing=True
        ... )
        >>> if event.requires_reindexing:
        ...     trigger_reindexing(event.model_name)
    """
    
    model_name: str
    change_type: ChangeType
    old_fingerprint: Optional[ModelFingerprint]
    new_fingerprint: ModelFingerprint
    requires_reindexing: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary for serialization and logging.
        
        Returns:
            JSON-serializable dictionary representation
        """
        return {
            "model_name": self.model_name,
            "change_type": self.change_type,
            "old_fingerprint": self.old_fingerprint.to_dict() if self.old_fingerprint else None,
            "new_fingerprint": self.new_fingerprint.to_dict(),
            "requires_reindexing": self.requires_reindexing,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def should_invalidate_cache(self) -> bool:
        """
        Determine if this change should trigger cache invalidation.
        
        Returns:
            True if cache invalidation is recommended
        """
        # Cache invalidation is recommended for all change types except metadata-only changes
        return self.change_type in ("new_model", "version_update", "checksum_change")
    
    def get_impact_level(self) -> Literal["low", "medium", "high"]:
        """
        Assess the impact level of this model change.
        
        Returns:
            Impact level classification for prioritization
        """
        if self.change_type == "new_model":
            return "medium"
        elif self.change_type in ("version_update", "checksum_change"):
            return "high"
        else:  # config_change
            return "low"


class ModelChangeDetector:
    """
    Thread-safe singleton service for centralized model change detection.
    
    The ModelChangeDetector provides a centralized registry for tracking
    embedding model states and detecting changes across service restarts.
    It implements the Singleton pattern to ensure global state consistency
    and includes thread-safe operations for concurrent environments.
    
    Key Features:
    - Thread-safe singleton implementation with lazy initialization
    - Persistent storage with automatic serialization/deserialization
    - Event generation for downstream change notifications
    - Comprehensive error handling and logging
    - Performance optimizations for frequent change detection
    
    The detector maintains an in-memory cache of model fingerprints and
    optionally persists them to disk for durability across restarts.
    
    Usage:
        >>> detector = ModelChangeDetector()
        >>> fingerprint = create_model_fingerprint()
        >>> if detector.detect_change(fingerprint):
        ...     detector.register_model(fingerprint)
        ...     handle_model_change(fingerprint)
    
    Thread Safety:
        All public methods are thread-safe through the use of a threading
        lock. This ensures safe concurrent access from multiple threads.
    """
    
    _instance: Optional['ModelChangeDetector'] = None
    _lock = threading.Lock()  # Class-level lock for singleton creation
    
    def __new__(
        cls,
        storage_path: Optional[Union[str, Path]] = None,
        auto_save: bool = False
    ) -> 'ModelChangeDetector':
        """
        Thread-safe singleton instantiation.
        
        Ensures only one instance exists globally while being safe
        for concurrent access from multiple threads.
        
        Args:
            storage_path: Path for persistent fingerprint storage
            auto_save: Whether to automatically save changes to disk
            
        Returns:
            Singleton ModelChangeDetector instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check pattern for thread safety
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        
        return cls._instance
    
    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        auto_save: bool = False
    ) -> None:
        """
        Initialize the detector with storage and configuration options.
        
        Args:
            storage_path: Path for persistent storage (default: "model_fingerprints.json")
            auto_save: Whether to automatically save changes (default: False)
        """
        # Prevent re-initialization of singleton
        if getattr(self, '_initialized', False):
            return
        
        self.storage_path = Path(storage_path) if storage_path else Path("model_fingerprints.json")
        self.auto_save = auto_save
        self._fingerprints: Dict[str, ModelFingerprint] = {}
        self._instance_lock = threading.Lock()  # Instance-level lock for operations
        self._initialized = True
        
        logger.info(f"Initialized ModelChangeDetector with storage: {self.storage_path}")
        
        # Attempt to load existing fingerprints
        self._load_fingerprints_safely()
    
    @property
    def storage_path_str(self) -> str:
        """Get storage path as string for backward compatibility."""
        return str(self.storage_path)
    
    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset singleton instance for testing and development.
        
        Warning:
            This method should only be used in test environments.
            In production, the singleton should persist for the
            application lifetime.
        """
        with cls._lock:
            cls._instance = None
        logger.debug("ModelChangeDetector singleton reset")
    
    def register_model(self, fingerprint: ModelFingerprint) -> None:
        """
        Register a new model fingerprint.
        
        Thread-safe registration of model fingerprints with automatic
        change event generation and optional persistence.
        
        Args:
            fingerprint: ModelFingerprint to register
            
        Raises:
            PersistenceError: If auto_save is enabled and saving fails
        """
        with self._instance_lock:
            old_fingerprint = self._fingerprints.get(fingerprint.model_name)
            self._fingerprints[fingerprint.model_name] = fingerprint
            
            # Determine change type for logging
            if old_fingerprint is None:
                change_type: ChangeType = "new_model"
            elif old_fingerprint.version != fingerprint.version:
                change_type = "version_update"
            elif old_fingerprint.checksum != fingerprint.checksum:
                change_type = "checksum_change"
            else:
                change_type = "config_change"
            
            logger.info(
                f"Registered model '{fingerprint.model_name}' (type: {change_type})"
            )
            
            # Auto-save if enabled (use unsafe version since we already hold the lock)
            if self.auto_save:
                try:
                    self._save_to_disk_unsafe()
                except Exception as e:
                    logger.error(f"Auto-save failed for model '{fingerprint.model_name}': {e}")
                    raise PersistenceError(f"Failed to auto-save model fingerprint: {e}")
    
    def register_model_with_event(self, fingerprint: ModelFingerprint) -> ModelChangeEvent:
        """
        Register a new model fingerprint and return change event.
        
        Extended version that returns the change event for downstream processing.
        
        Args:
            fingerprint: ModelFingerprint to register
            
        Returns:
            ModelChangeEvent representing the registration
            
        Raises:
            PersistenceError: If auto_save is enabled and saving fails
        """
        with self._instance_lock:
            old_fingerprint = self._fingerprints.get(fingerprint.model_name)
            self._fingerprints[fingerprint.model_name] = fingerprint
            
            # Determine change type
            if old_fingerprint is None:
                change_type: ChangeType = "new_model"
            elif old_fingerprint.version != fingerprint.version:
                change_type = "version_update"
            elif old_fingerprint.checksum != fingerprint.checksum:
                change_type = "checksum_change"
            else:
                change_type = "config_change"
            
            # Create change event
            event = ModelChangeEvent(
                model_name=fingerprint.model_name,
                change_type=change_type,
                old_fingerprint=old_fingerprint,
                new_fingerprint=fingerprint,
                requires_reindexing=change_type in ("new_model", "version_update", "checksum_change")
            )
            
            logger.info(
                f"Registered model '{fingerprint.model_name}' "
                f"(type: {change_type}, reindex: {event.requires_reindexing})"
            )
            
            # Auto-save if enabled (use unsafe version since we already hold the lock)
            if self.auto_save:
                try:
                    self._save_to_disk_unsafe()
                except Exception as e:
                    logger.error(f"Auto-save failed for model '{fingerprint.model_name}': {e}")
                    raise PersistenceError(f"Failed to auto-save model fingerprint: {e}")
            
            return event
    
    def get_model_fingerprint(self, model_name: str) -> Optional[ModelFingerprint]:
        """
        Retrieve fingerprint for a specific model.
        
        Thread-safe retrieval of registered model fingerprints.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            ModelFingerprint if found, None otherwise
        """
        with self._instance_lock:
            return self._fingerprints.get(model_name)
    
    def detect_change(self, fingerprint: ModelFingerprint) -> bool:
        """
        Detect if a model has changed compared to registered state.
        
        Performs efficient change detection by comparing the provided
        fingerprint against the currently registered state. Optimized
        for performance by checking checksums first.
        
        Args:
            fingerprint: Current model fingerprint to check
            
        Returns:
            True if change detected, False if no change
        """
        with self._instance_lock:
            existing = self._fingerprints.get(fingerprint.model_name)
            
            if existing is None:
                logger.debug(f"New model detected: {fingerprint.model_name}")
                return True
            
            # Fast path: Check checksum first as it's most likely to differ
            if existing.checksum != fingerprint.checksum:
                logger.debug(
                    f"Model change detected for '{fingerprint.model_name}': "
                    f"checksum {existing.checksum} -> {fingerprint.checksum}"
                )
                return True
            
            # Fallback to full equality check for other differences
            changed = existing != fingerprint
            if changed:
                logger.debug(
                    f"Model change detected for '{fingerprint.model_name}': "
                    f"non-checksum difference (version/metadata)"
                )
            
            return changed
    
    def list_models(self) -> List[str]:
        """
        Get list of all registered model names.
        
        Returns:
            List of registered model names
        """
        with self._instance_lock:
            return list(self._fingerprints.keys())
    
    def get_model_count(self) -> int:
        """
        Get count of registered models.
        
        Returns:
            Number of registered models
        """
        with self._instance_lock:
            return len(self._fingerprints)
    
    def clear_all(self) -> None:
        """
        Clear all registered models.
        
        Warning:
            This operation is irreversible and will remove all
            registered model fingerprints from memory.
        """
        with self._instance_lock:
            count = len(self._fingerprints)
            self._fingerprints.clear()
            logger.info(f"Cleared {count} model fingerprints")
    
    def save_to_disk(self) -> None:
        """
        Persist all fingerprints to disk storage.
        
        Thread-safe serialization of all registered fingerprints
        to the configured storage path.
        
        Raises:
            PersistenceError: If saving fails due to I/O or serialization errors
        """
        with self._instance_lock:
            self._save_to_disk_unsafe()
    
    def _save_to_disk_unsafe(self) -> None:
        """
        Internal method to save fingerprints without acquiring lock.
        
        This method assumes the caller already holds self._instance_lock.
        Used internally by auto-save functionality to avoid deadlock.
        
        Raises:
            PersistenceError: If saving fails due to I/O or serialization errors
        """
        try:
            # Ensure parent directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize fingerprints
            data = {
                name: fingerprint.to_dict()
                for name, fingerprint in self._fingerprints.items()
            }
            
            # Write to temporary file first, then rename for atomicity
            temp_path = self.storage_path.with_suffix('.tmp')
            with temp_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(self.storage_path)
            
            logger.debug(f"Saved {len(data)} fingerprints to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save fingerprints: {e}")
            raise PersistenceError(f"Failed to save fingerprints to {self.storage_path}: {e}")
    
    def load_from_disk(self) -> None:
        """
        Load fingerprints from disk storage.
        
        Thread-safe deserialization of fingerprints from the
        configured storage path.
        
        Raises:
            PersistenceError: If loading fails due to I/O or deserialization errors
        """
        with self._instance_lock:
            try:
                if not self.storage_path.exists():
                    logger.debug(f"Storage file {self.storage_path} does not exist, starting fresh")
                    return
                
                with self.storage_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Deserialize fingerprints with validation
                fingerprints = {}
                for name, fingerprint_data in data.items():
                    try:
                        fingerprints[name] = ModelFingerprint.from_dict(fingerprint_data)
                    except Exception as e:
                        logger.warning(f"Skipping invalid fingerprint for '{name}': {e}")
                
                self._fingerprints = fingerprints
                logger.info(f"Loaded {len(fingerprints)} fingerprints from {self.storage_path}")
                
            except Exception as e:
                logger.error(f"Failed to load fingerprints from {self.storage_path}: {e}")
                raise PersistenceError(f"Failed to load fingerprints from disk: {e}")
    
    def _load_fingerprints_safely(self) -> None:
        """
        Safely attempt to load fingerprints, handling errors gracefully.
        
        This method is used during initialization to load existing
        fingerprints without failing if the storage is corrupted
        or inaccessible.
        """
        try:
            self.load_from_disk()
        except Exception as e:
            logger.warning(f"Could not load existing fingerprints: {e}")
            # Start with empty storage if loading fails
            self._fingerprints = {}
    
    def cleanup_old_fingerprints(self, max_age_days: int = 30) -> int:
        """
        Remove fingerprints older than specified age.
        
        Args:
            max_age_days: Maximum age in days for fingerprints
            
        Returns:
            Number of fingerprints removed
        """
        if max_age_days <= 0:
            raise ValueError("max_age_days must be positive")
        
        with self._instance_lock:
            cutoff = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            old_fingerprints = [
                name for name, fp in self._fingerprints.items()
                if fp.created_at.timestamp() < cutoff
            ]
            
            for name in old_fingerprints:
                del self._fingerprints[name]
            
            if old_fingerprints:
                logger.info(f"Cleaned up {len(old_fingerprints)} old fingerprints")
            
            return len(old_fingerprints)

    def has_models(self) -> bool:
        """
        Check if any models are currently registered.
        
        Fast check for model registry state without acquiring locks
        for read-only operations on atomic operations.
        
        Returns:
            True if at least one model is registered
        """
        return len(self._fingerprints) > 0
    
    def get_models_summary(self) -> Dict[str, Any]:
        """
        Get summary information about registered models.
        
        Useful for integration with document chunking pipeline to
        understand model state without exposing internals.
        
        Returns:
            Dictionary containing model registry summary
        """
        with self._instance_lock:
            return {
                "total_models": len(self._fingerprints),
                "model_names": list(self._fingerprints.keys()),
                "model_types": [fp.model_type for fp in self._fingerprints.values()],
                "storage_path": str(self.storage_path),
                "auto_save_enabled": self.auto_save
            }
    
    def register_models_bulk(self, fingerprints: List[ModelFingerprint]) -> List[ModelChangeEvent]:
        """
        Register multiple models in a single transaction.
        
        Optimized bulk operation for registering multiple models
        with a single lock acquisition and optional auto-save at the end.
        
        Args:
            fingerprints: List of ModelFingerprint objects to register
            
        Returns:
            List of ModelChangeEvent objects for each registration
            
        Raises:
            PersistenceError: If auto_save is enabled and saving fails
        """
        if not fingerprints:
            return []
        
        events = []
        with self._instance_lock:
            for fingerprint in fingerprints:
                old_fingerprint = self._fingerprints.get(fingerprint.model_name)
                self._fingerprints[fingerprint.model_name] = fingerprint
                
                # Determine change type
                if old_fingerprint is None:
                    change_type: ChangeType = "new_model"
                elif old_fingerprint.version != fingerprint.version:
                    change_type = "version_update"
                elif old_fingerprint.checksum != fingerprint.checksum:
                    change_type = "checksum_change"
                else:
                    change_type = "config_change"
                
                # Create change event
                event = ModelChangeEvent(
                    model_name=fingerprint.model_name,
                    change_type=change_type,
                    old_fingerprint=old_fingerprint,
                    new_fingerprint=fingerprint,
                    requires_reindexing=change_type in ("new_model", "version_update", "checksum_change")
                )
                events.append(event)
                
                logger.info(
                    f"Bulk registered model '{fingerprint.model_name}' "
                    f"(type: {change_type}, reindex: {event.requires_reindexing})"
                )
            
            # Single auto-save for all models if enabled
            if self.auto_save:
                try:
                    self._save_to_disk_unsafe()
                except Exception as e:
                    logger.error(f"Bulk auto-save failed for {len(fingerprints)} models: {e}")
                    raise PersistenceError(f"Failed to auto-save bulk model fingerprints: {e}")
        
        return events 