"""Configuration management package.

This package provides a modular configuration system with support for:
- JSON schema validation
- Environment variable overrides
- Configuration inheritance
- Default value resolution
- Path management and file operations

Usage:
    from research_agent_backend.utils.config import ConfigManager
    
    config = ConfigManager()
    value = config.get("embedding.model", "default-model")
"""

from .manager import ConfigManager
from .paths import ConfigPaths
from .file_operations import FileOperations
from .schema_validation import SchemaValidator
from .inheritance import ConfigInheritance
from .environment import EnvironmentHandler

# Backward compatibility: expose ConfigManager as the main interface
__all__ = [
    'ConfigManager',
    'ConfigPaths', 
    'FileOperations',
    'SchemaValidator',
    'ConfigInheritance',
    'EnvironmentHandler'
]

# Legacy import support for existing code
def get_config(*args, **kwargs):
    """Legacy function for backward compatibility."""
    return ConfigManager().get(*args, **kwargs)

def load_config(*args, **kwargs):
    """Legacy function for backward compatibility."""
    return ConfigManager().load_config(*args, **kwargs) 