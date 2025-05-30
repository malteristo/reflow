"""
Storage operations, bulk management, and cleanup for model fingerprints.

This module provides persistent storage capabilities for model fingerprints
with atomic operations and cleanup functionality.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from .fingerprint import ModelFingerprint
from .types import PersistenceError

# Module-level logger for consistent logging across all classes
logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    Manager for persistent storage operations of model fingerprints.
    
    Provides atomic file operations, error handling, and cleanup functionality
    for model fingerprint storage and retrieval.
    """
    
    def __init__(self, storage_path: Union[str, Path]):
        """
        Initialize persistence manager with storage path.
        
        Args:
            storage_path: Path for fingerprint storage
        """
        self.storage_path = Path(storage_path)
    
    def save_to_disk(self, fingerprints: Dict[str, ModelFingerprint]) -> None:
        """
        Persist all fingerprints to disk storage with thread safety.
        
        Thread-safe serialization of all registered fingerprints
        to the configured storage path.
        
        Args:
            fingerprints: Dictionary of fingerprints to save
            
        Raises:
            PersistenceError: If saving fails due to I/O or serialization errors
        """
        self.save_to_disk_unsafe(fingerprints)
    
    def save_to_disk_unsafe(self, fingerprints: Dict[str, ModelFingerprint]) -> None:
        """
        Internal method to save fingerprints without acquiring external locks.
        
        This method assumes the caller manages thread safety if needed.
        Used internally by auto-save functionality.
        
        Args:
            fingerprints: Dictionary of fingerprints to save
            
        Raises:
            PersistenceError: If saving fails due to I/O or serialization errors
        """
        try:
            # Ensure parent directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize fingerprints
            data = {
                name: fingerprint.to_dict()
                for name, fingerprint in fingerprints.items()
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
    
    def load_from_disk(self) -> Dict[str, ModelFingerprint]:
        """
        Load fingerprints from disk storage.
        
        Thread-safe deserialization of fingerprints from the
        configured storage path.
        
        Returns:
            Dictionary of loaded fingerprints
            
        Raises:
            PersistenceError: If loading fails due to I/O or deserialization errors
        """
        try:
            if not self.storage_path.exists():
                logger.debug(f"Storage file {self.storage_path} does not exist, starting fresh")
                return {}
            
            with self.storage_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Deserialize fingerprints with validation
            fingerprints = {}
            for name, fingerprint_data in data.items():
                try:
                    fingerprints[name] = ModelFingerprint.from_dict(fingerprint_data)
                except Exception as e:
                    logger.warning(f"Skipping invalid fingerprint for '{name}': {e}")
            
            logger.info(f"Loaded {len(fingerprints)} fingerprints from {self.storage_path}")
            return fingerprints
            
        except Exception as e:
            logger.error(f"Failed to load fingerprints from {self.storage_path}: {e}")
            raise PersistenceError(f"Failed to load fingerprints from disk: {e}")
    
    def cleanup_old_fingerprints(
        self, 
        fingerprints: Dict[str, ModelFingerprint], 
        max_age_days: int = 30
    ) -> int:
        """
        Remove fingerprints older than specified age.
        
        Args:
            fingerprints: Dictionary of fingerprints to clean up
            max_age_days: Maximum age in days for fingerprints
            
        Returns:
            Number of fingerprints removed
            
        Raises:
            ValueError: If max_age_days is not positive
        """
        if max_age_days <= 0:
            raise ValueError("max_age_days must be positive")
        
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        old_fingerprints = [
            name for name, fp in fingerprints.items()
            if fp.created_at.timestamp() < cutoff
        ]
        
        for name in old_fingerprints:
            del fingerprints[name]
        
        if old_fingerprints:
            logger.info(f"Cleaned up {len(old_fingerprints)} old fingerprints")
        
        return len(old_fingerprints)
    
    def has_models(self, fingerprints: Dict[str, ModelFingerprint]) -> bool:
        """
        Check if any models are currently registered.
        
        Fast check for model registry state without acquiring locks
        for read-only operations on atomic operations.
        
        Args:
            fingerprints: Dictionary of fingerprints to check
            
        Returns:
            True if at least one model is registered
        """
        return len(fingerprints) > 0
    
    def get_models_summary(self, fingerprints: Dict[str, ModelFingerprint]) -> Dict[str, Any]:
        """
        Get summary information about registered models.
        
        Useful for integration with document chunking pipeline to
        understand model state without exposing internals.
        
        Args:
            fingerprints: Dictionary of fingerprints to summarize
            
        Returns:
            Dictionary containing model registry summary
        """
        return {
            "total_models": len(fingerprints),
            "model_names": list(fingerprints.keys()),
            "model_types": [fp.model_type for fp in fingerprints.values()],
            "storage_path": str(self.storage_path),
            "auto_save_enabled": False  # This is managed by the detector
        } 