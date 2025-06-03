"""
Model Change Detection Package for Research Agent.

This package provides comprehensive model change detection capabilities with
modular architecture for maintainability and testability.

Public API:
- ModelChangeDetector: Main singleton detector class
- ModelFingerprint: Immutable model state representation  
- ModelChangeEvent: Change notification object
- ConfigurationIntegrationHooks: Configuration system integration
- ModelCompatibilityValidator: Query compatibility validation
- Exception classes: ModelChangeError, FingerprintMismatchError, PersistenceError, QueryValidationError
- Type aliases: ModelType, ChangeType
- Integration utilities: auto_register_embedding_service, add_config_change_callback, trigger_config_change
- Query validation utilities: validate_query_compatibility, set_compatibility_strict_mode
"""

# Main detector class
from .detector import ModelChangeDetector

# Core data structures
from .fingerprint import ModelFingerprint
from .events import ModelChangeEvent

# Configuration system integration
from .integration_hooks import (
    ConfigurationIntegrationHooks,
    get_integration_hooks,
    auto_register_embedding_service,
    add_config_change_callback,
    trigger_config_change
)

# Query validation integration
from .query_validation import (
    ModelCompatibilityValidator,
    QueryValidationError,
    get_compatibility_validator,
    validate_query_compatibility,
    set_compatibility_strict_mode
)

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
    
    # Configuration integration
    'ConfigurationIntegrationHooks',
    'get_integration_hooks',
    'auto_register_embedding_service',
    'add_config_change_callback',
    'trigger_config_change',
    
    # Query validation
    'ModelCompatibilityValidator',
    'QueryValidationError',
    'get_compatibility_validator',
    'validate_query_compatibility',
    'set_compatibility_strict_mode',
    
    # Type aliases
    'ModelType',
    'ChangeType',
    
    # Exception classes
    'ModelChangeError',
    'FingerprintMismatchError',
    'PersistenceError',
    'QueryValidationError',
    
    # Service components
    'PersistenceManager',
    'SingletonMeta',
    'SingletonBase'
] 