"""
Core change detection algorithms and model registration.

This module provides the main detection logic for identifying model changes
and managing model registration with event generation.
"""

import logging
from typing import List, Optional

from .events import ModelChangeEvent
from .fingerprint import ModelFingerprint
from .singleton import SingletonBase, SingletonMeta
from .types import ChangeType

# Module-level logger for consistent logging across all classes
logger = logging.getLogger(__name__)


class ModelChangeDetector(SingletonBase, metaclass=SingletonMeta):
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
    
    def _post_init(self) -> None:
        """
        Perform additional initialization after base singleton setup.
        
        Attempts to load existing fingerprints from disk storage.
        """
        # Attempt to load existing fingerprints
        self._load_fingerprints_safely()
    
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
                    from .persistence import PersistenceManager
                    persistence = PersistenceManager(self.storage_path)
                    persistence.save_to_disk_unsafe(self._fingerprints)
                except Exception as e:
                    logger.error(f"Auto-save failed for model '{fingerprint.model_name}': {e}")
                    from .types import PersistenceError
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
                    from .persistence import PersistenceManager
                    persistence = PersistenceManager(self.storage_path)
                    persistence.save_to_disk_unsafe(self._fingerprints)
                except Exception as e:
                    logger.error(f"Auto-save failed for model '{fingerprint.model_name}': {e}")
                    from .types import PersistenceError
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
                    from .persistence import PersistenceManager
                    persistence = PersistenceManager(self.storage_path)
                    persistence.save_to_disk_unsafe(self._fingerprints)
                except Exception as e:
                    logger.error(f"Bulk auto-save failed for {len(fingerprints)} models: {e}")
                    from .types import PersistenceError
                    raise PersistenceError(f"Failed to auto-save bulk model fingerprints: {e}")
        
        return events
    
    def _load_fingerprints_safely(self) -> None:
        """
        Safely attempt to load fingerprints, handling errors gracefully.
        
        This method is used during initialization to load existing
        fingerprints without failing if the storage is corrupted
        or inaccessible.
        """
        try:
            from .persistence import PersistenceManager
            persistence = PersistenceManager(self.storage_path)
            self._fingerprints = persistence.load_from_disk()
        except Exception as e:
            logger.warning(f"Could not load existing fingerprints: {e}")
            # Start with empty storage if loading fails
            self._fingerprints = {} 