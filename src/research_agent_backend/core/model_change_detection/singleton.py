"""
Thread-safe singleton pattern implementation for ModelChangeDetector.

This module provides singleton infrastructure to ensure global state
consistency for model change detection.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Union

from .fingerprint import ModelFingerprint

# Module-level logger for consistent logging across all classes
logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass for ModelChangeDetector.
    
    This metaclass implements the thread-safe singleton pattern using
    the double-check locking approach for optimal performance.
    """
    
    _instances: Dict[type, object] = {}
    _lock = threading.Lock()  # Class-level lock for singleton creation
    
    def __call__(
        cls, 
        storage_path: Optional[Union[str, Path]] = None,
        auto_save: bool = False
    ):
        """
        Thread-safe singleton instantiation.
        
        Ensures only one instance exists globally while being safe
        for concurrent access from multiple threads.
        
        Args:
            storage_path: Path for persistent fingerprint storage
            auto_save: Whether to automatically save changes to disk
            
        Returns:
            Singleton instance
        """
        if cls not in cls._instances:
            with cls._lock:
                # Double-check pattern for thread safety
                if cls not in cls._instances:
                    instance = super().__call__(storage_path, auto_save)
                    cls._instances[cls] = instance
        
        return cls._instances[cls]
    
    @classmethod
    def reset_singleton(cls, target_class: type) -> None:
        """
        Reset singleton instance for testing and development.
        
        Warning:
            This method should only be used in test environments.
            In production, the singleton should persist for the
            application lifetime.
            
        Args:
            target_class: The class to reset the singleton for
        """
        with cls._lock:
            if target_class in cls._instances:
                del cls._instances[target_class]
        logger.debug(f"{target_class.__name__} singleton reset")


class SingletonBase:
    """
    Base class providing singleton functionality and instance management.
    
    This class provides common functionality for singleton objects including
    initialization tracking and thread-safe operations.
    """
    
    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        auto_save: bool = False
    ) -> None:
        """
        Initialize the singleton with storage and configuration options.
        
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
        
        logger.info(f"Initialized singleton with storage: {self.storage_path}")
        
        # Subclasses can override this to perform additional initialization
        self._post_init()
    
    def _post_init(self) -> None:
        """
        Hook for subclasses to perform additional initialization.
        
        This method is called after the base initialization is complete
        and can be overridden by subclasses to add specific setup logic.
        """
        pass
    
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
        """
        if hasattr(cls, '__class__') and hasattr(cls.__class__, 'reset_singleton'):
            cls.__class__.reset_singleton(cls)
        else:
            logger.warning(f"Cannot reset singleton for {cls.__name__} - no metaclass support") 