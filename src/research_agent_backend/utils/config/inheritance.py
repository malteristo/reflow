"""
Configuration inheritance and merging for Research Agent.

This module provides configuration inheritance resolution, merging capabilities,
and default configuration handling for the Research Agent configuration system.

Implements FR-CF-001: Configuration-driven behavior with inheritance.
"""

import logging
from typing import Any, Dict, Optional, List, Set
from copy import deepcopy

from ...exceptions.config_exceptions import (
    ConfigurationMergeError,
)
from .file_operations import FileOperations
from .paths import ConfigPaths


logger = logging.getLogger(__name__)


class ConfigInheritance:
    """
    Configuration inheritance and merging for Research Agent.
    
    Handles configuration inheritance using 'extends' fields, deep merging,
    and default configuration integration.
    """
    
    def __init__(self, file_ops: FileOperations, paths: ConfigPaths) -> None:
        """
        Initialize configuration inheritance handler.
        
        Args:
            file_ops: File operations instance
            paths: Configuration paths instance
        """
        self.file_ops = file_ops
        self.paths = paths
        self.logger = logger
    
    def resolve_config_inheritance(
        self, 
        config: Dict[str, Any], 
        visited_files: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Resolve configuration inheritance using the 'extends' field.
        
        Args:
            config: Configuration dictionary that may contain 'extends' field
            visited_files: Set of already visited files to prevent circular references
            
        Returns:
            Merged configuration with inheritance resolved
            
        Raises:
            ConfigurationMergeError: If circular reference is detected or merging fails
        """
        if visited_files is None:
            visited_files = set()
        
        # Check if this config extends another
        if 'extends' not in config:
            return config
        
        extends_path = config['extends']
        resolved_extends_path = str(self.file_ops.resolve_path(extends_path))
        
        # Check for circular references
        if resolved_extends_path in visited_files:
            raise ConfigurationMergeError(
                f"Circular reference detected in configuration inheritance: {resolved_extends_path}",
                list(visited_files) + [resolved_extends_path],
                []
            )
        
        # Add current file to visited set
        visited_files.add(resolved_extends_path)
        
        try:
            # Load the parent configuration
            parent_config = self.file_ops.load_json_file(resolved_extends_path)
            
            # Recursively resolve parent's inheritance
            resolved_parent = self.resolve_config_inheritance(parent_config, visited_files.copy())
            
            # Remove the 'extends' field from current config before merging
            current_config = {k: v for k, v in config.items() if k != 'extends'}
            
            # Merge parent and current configurations
            merged_config = self.deep_merge_dicts(resolved_parent, current_config)
            
            return merged_config
            
        except Exception as e:
            if "not found" in str(e).lower():
                raise ConfigurationMergeError(
                    f"Failed to resolve configuration inheritance: {e}",
                    [resolved_extends_path],
                    []
                ) from e
            else:
                raise ConfigurationMergeError(
                    f"Unexpected error during configuration merging: {e}",
                    [resolved_extends_path],
                    []
                ) from e
    
    def merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with default values.
        
        Args:
            config: User configuration
            
        Returns:
            Configuration merged with defaults
        """
        try:
            # Try to load default configuration
            default_config_path = self.file_ops.resolve_path(
                f"{self.paths.DEFAULT_CONFIG_DIR}/default_config.json"
            )
            
            if default_config_path.exists():
                default_config = self.file_ops.load_json_file(default_config_path)
                # Resolve defaults inheritance first
                resolved_defaults = self.resolve_config_inheritance(default_config)
                # Merge user config over defaults
                return self.deep_merge_dicts(resolved_defaults, config)
            else:
                return config
                
        except Exception:
            # If defaults can't be loaded, return user config as-is
            return config
    
    def deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override values taking precedence.
        
        Args:
            base: Base dictionary
            override: Override dictionary (values take precedence)
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self.deep_merge_dicts(result[key], value)
            else:
                # Override value (or add new key)
                result[key] = deepcopy(value)
        
        return result
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Args:
            *configs: Configuration dictionaries to merge (later configs override earlier ones)
            
        Returns:
            Merged configuration dictionary
            
        Raises:
            ConfigurationMergeError: If merging fails
        """
        if not configs:
            return {}
        
        try:
            result = deepcopy(configs[0])
            for config in configs[1:]:
                result = self.deep_merge_dicts(result, config)
            return result
        except Exception as e:
            raise ConfigurationMergeError(
                f"Failed to merge configurations: {e}",
                [],
                []
            ) from e 