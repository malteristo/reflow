"""
Model change detection system for embedding services.

This module provides backwards compatibility for the model change detection system
by importing from the modular model_change_detection package.

Implementation follows FR-KB-005 (Model Change Detection) from the PRD.
"""

# Import all public API from the modular package
from .model_change_detection import (
    # Main classes
    ModelChangeDetector,
    ModelFingerprint,
    ModelChangeEvent,
    
    # Type aliases
    ModelType,
    ChangeType,
    
    # Exception classes
    ModelChangeError,
    FingerprintMismatchError,
    PersistenceError,
    
    # Service components
    PersistenceManager,
    SingletonMeta,
    SingletonBase
)

# Module-level logger for consistency
import logging
logger = logging.getLogger(__name__)

# For compatibility, also provide the original module exports
__all__ = [
    'ModelChangeDetector',
    'ModelFingerprint', 
    'ModelChangeEvent',
    'ModelType',
    'ChangeType',
    'ModelChangeError',
    'FingerprintMismatchError',
    'PersistenceError',
    'PersistenceManager',
    'SingletonMeta',
    'SingletonBase',
    'logger'
] 