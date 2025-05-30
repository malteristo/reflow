"""Configuration management compatibility layer.

This module provides backward compatibility for the refactored configuration system.
All functionality has been moved to the config package with better organization.

For new code, prefer importing directly from the config package:
    from research_agent_backend.utils.config import ConfigManager
    
This compatibility layer maintains existing imports.
"""

# Import all functionality from the modular config package
from .config.manager import ConfigManager
from .config.paths import ConfigPaths
from .config.file_operations import FileOperations
from .config.schema_validation import SchemaValidator
from .config.inheritance import InheritanceResolver
from .config.environment import EnvironmentHandler

# Export the same public interface for backward compatibility
__all__ = [
    'ConfigManager',
    'ConfigPaths',
    'FileOperations', 
    'SchemaValidator',
    'InheritanceResolver',
    'EnvironmentHandler'
]

# Legacy functions for backward compatibility
def get_config(*args, **kwargs):
    """Legacy function for backward compatibility."""
    return ConfigManager().get(*args, **kwargs)

def load_config(*args, **kwargs):
    """Legacy function for backward compatibility."""
    return ConfigManager().load_config(*args, **kwargs) 