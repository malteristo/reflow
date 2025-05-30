"""
Model Change Detection Package for Research Agent.

This package provides comprehensive model change detection capabilities with
modular architecture for maintainability and testability.

Public API:
- ModelChangeDetector: Main singleton detector class
- ModelFingerprint: Immutable model state representation  
- ModelChangeEvent: Change notification object
- Exception classes: ModelChangeError, FingerprintMismatchError, PersistenceError
- Type aliases: ModelType, ChangeType
"""

# Main detector class
from .detector import ModelChangeDetector

# Core data structures
from .fingerprint import ModelFingerprint
from .events import ModelChangeEvent

# Exception classes and types
from .types import (
    ModelType,
    ChangeType,
    ModelChangeError,
    FingerprintMismatchError,
    PersistenceError
)

# Service components for advanced usage
from .persistence import PersistenceManager
from .singleton import SingletonMeta, SingletonBase

# For backward compatibility, ensure all original module exports are available
__all__ = [
    # Main classes
    'ModelChangeDetector',
    'ModelFingerprint',
    'ModelChangeEvent',
    
    # Type aliases
    'ModelType',
    'ChangeType',
    
    # Exception classes
    'ModelChangeError',
    'FingerprintMismatchError',
    'PersistenceError',
    
    # Service components
    'PersistenceManager',
    'SingletonMeta',
    'SingletonBase'
] 