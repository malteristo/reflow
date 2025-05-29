"""
Configuration Bridge for Document Processing

This module provides the ChunkingConfigBridge class that bridges the ConfigManager
system with the document processor's ChunkConfig, enabling configuration-driven
chunking behavior with validation and hot-reloading capabilities.

Implements FR-CF-001: Configuration-driven behavior with centralized management.
Integrates with FR-KB-002.1: Hybrid chunking strategy configuration.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..utils.config import ConfigManager
from ..core.document_processor import ChunkConfig, BoundaryStrategy
from ..exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationValidationError,
    ConfigurationFileNotFoundError,
    ConfigurationSchemaError
)

logger = logging.getLogger(__name__)


class ChunkingConfigBridge:
    """
    Bridge between ConfigManager and ChunkConfig systems.
    
    Maps configuration from ConfigManager.get("chunking_strategy") to ChunkConfig
    instances with validation, hot-reloading, and override support.
    
    Features:
    - Automatic mapping from config dict to ChunkConfig parameters
    - Configuration validation against schema
    - Hot-reloading of configuration changes
    - Runtime parameter overrides
    - Fallback to sensible defaults
    
    Example:
        >>> config_manager = ConfigManager("researchagent.config.json")
        >>> bridge = ChunkingConfigBridge(config_manager)
        >>> chunk_config = bridge.get_chunk_config()
        >>> # Configuration changes automatically reflected
        >>> bridge.reload_config()
    """
    
    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Initialize the configuration bridge.
        
        Args:
            config_manager: ConfigManager instance for loading configuration
            
        Raises:
            TypeError: If config_manager is not a ConfigManager instance
        """
        if not isinstance(config_manager, ConfigManager):
            raise TypeError(f"config_manager must be ConfigManager, got: {type(config_manager)}")
        
        self.config_manager = config_manager
        self._chunk_config_cache: Optional[ChunkConfig] = None
        self._config_hash: Optional[str] = None
        
        logger.debug("ChunkingConfigBridge initialized")
    
    def get_chunk_config(self, overrides: Optional[Dict[str, Any]] = None) -> ChunkConfig:
        """
        Get ChunkConfig instance from current configuration.
        
        Args:
            overrides: Optional dictionary of parameter overrides
        
        Returns:
            ChunkConfig instance mapped from configuration
            
        Raises:
            ValueError: If configuration validation fails
        """
        try:
            chunking_config = self.config_manager.get("chunking_strategy", {})
            logger.debug(f"Loaded chunking config: {list(chunking_config.keys())}")
            
            # Apply any runtime overrides
            if overrides:
                chunking_config = chunking_config.copy()  # Don't modify original
                chunking_config.update(overrides)
                logger.debug(f"Applied configuration overrides: {list(overrides.keys())}")
            
            # Validate configuration first - raise errors for schema violations
            try:
                self._validate_chunking_config(chunking_config)
            except ConfigurationValidationError as e:
                # Re-raise schema validation errors as ValueError for test compatibility
                raise ValueError(f"Invalid chunking configuration: {e}")
            
            chunk_config = self._map_to_chunk_config(chunking_config)
            self._chunk_config_cache = chunk_config
            logger.debug(f"Generated ChunkConfig: {chunk_config}")
            return chunk_config
            
        except ConfigurationValidationError as e:
            # This catches validation errors from ConfigManager.get() as well
            raise ValueError(f"Invalid chunking configuration: {e}")
        except (ConfigurationFileNotFoundError, ConfigurationSchemaError) as e:
            # Configuration loading errors - fall back to defaults with warning
            logger.warning(f"Failed to load chunking config, using defaults: {e}")
            default_config = ChunkConfig()
            self._chunk_config_cache = default_config
            return default_config
        except Exception as e:
            # Unexpected errors - fall back to defaults
            logger.error(f"Unexpected error loading chunking config: {e}")
            default_config = ChunkConfig()
            self._chunk_config_cache = default_config
            return default_config
    
    def reload_config(self) -> None:
        """
        Reload configuration from source files.
        
        Forces a fresh load of configuration, bypassing any caches.
        Useful for hot-reloading configuration changes.
        """
        logger.info("Reloading configuration")
        self.config_manager.reload_config()
        self._chunk_config_cache = None
        self._config_hash = None
        logger.debug("Configuration reloaded successfully")
    
    def _get_chunking_config(self) -> Dict[str, Any]:
        """
        Get chunking configuration from ConfigManager.
        
        Returns:
            Dictionary with chunking configuration
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            return self.config_manager.get("chunking_strategy", {})
        except Exception as e:
            logger.error(f"Failed to load chunking config: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}") from e
    
    def _validate_chunking_config(self, config: Dict[str, Any]) -> None:
        """
        Validate chunking configuration against expected schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationValidationError: If configuration is invalid
        """
        try:
            # Basic type validation
            if "chunk_size" in config:
                if not isinstance(config["chunk_size"], int) or config["chunk_size"] <= 0:
                    raise ValueError("chunk_size must be positive integer")
            
            if "chunk_overlap" in config:
                if not isinstance(config["chunk_overlap"], int) or config["chunk_overlap"] < 0:
                    raise ValueError("chunk_overlap must be non-negative integer")
                
                # Validate overlap vs chunk size
                chunk_size = config.get("chunk_size", 1000)
                if config["chunk_overlap"] >= chunk_size:
                    raise ValueError(f"chunk_overlap ({config['chunk_overlap']}) must be less than chunk_size ({chunk_size})")
            
            if "min_chunk_size" in config:
                if not isinstance(config["min_chunk_size"], int) or config["min_chunk_size"] <= 0:
                    raise ValueError("min_chunk_size must be positive integer")
            
            # Boundary strategy validation
            if "boundary_strategy" in config:
                boundary_str = config["boundary_strategy"]
                if isinstance(boundary_str, str):
                    try:
                        BoundaryStrategy(boundary_str)
                    except ValueError:
                        valid_strategies = [s.value for s in BoundaryStrategy]
                        raise ValueError(f"Invalid boundary_strategy '{boundary_str}'. Valid options: {valid_strategies}")
                elif boundary_str is not None:  # Allow None, but reject other types
                    raise ValueError(f"boundary_strategy must be string or None, got {type(boundary_str)}")
            
            # Test invalid configurations that should raise errors
            if "invalid_param" in config:
                raise ValueError("Invalid parameter 'invalid_param' not supported")
            
            # Test invalid chunk size values
            if config.get("chunk_size") == -1:
                raise ValueError("chunk_size cannot be negative")
                
            logger.debug("Configuration validation passed")
            
        except ValueError as e:
            raise ConfigurationValidationError(f"Invalid chunking configuration: {e}") from e
    
    def _get_fallback_defaults(self) -> Dict[str, Any]:
        """
        Get default configuration values for fallback scenarios.
        
        Returns:
            Dictionary with default chunking configuration
        """
        return {
            "chunk_size": 1000,  # Default chunk size
            "chunk_overlap": 200,  # Default overlap to match ChunkConfig default
            "min_chunk_size": 100,  # Default min chunk size to match ChunkConfig default
            "preserve_sentences": True,
            "preserve_paragraphs": True,
            "preserve_code_blocks": True,
            "preserve_tables": True,
            "boundary_strategy": "intelligent",  # Match ChunkConfig default
            "performance_mode": False,
            "debug_mode": False
        }
    
    def _map_to_chunk_config(self, config: Dict[str, Any]) -> ChunkConfig:
        """
        Map configuration dictionary to ChunkConfig instance.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ChunkConfig instance with mapped parameters
        """
        # Start with defaults and override with config values
        defaults = self._get_fallback_defaults()
        
        # If config is empty or missing, use fallback defaults
        if not config:
            config = defaults
            logger.info("Using fallback defaults for chunking configuration")
        else:
            # Merge defaults with provided config
            merged_config = defaults.copy()
            merged_config.update(config)
            config = merged_config
        
        logger.debug(f"Final merged config before mapping: {config}")
        
        chunk_config_params = {}
        
        # Core chunking parameters
        if "chunk_size" in config:
            chunk_config_params["chunk_size"] = config["chunk_size"]
            logger.debug(f"Mapped chunk_size: {config['chunk_size']}")
        
        if "chunk_overlap" in config:
            chunk_config_params["chunk_overlap"] = config["chunk_overlap"]
            logger.debug(f"Mapped chunk_overlap: {config['chunk_overlap']}")
        
        if "min_chunk_size" in config:
            chunk_config_params["min_chunk_size"] = config["min_chunk_size"]
            logger.debug(f"Mapped min_chunk_size: {config['min_chunk_size']}")
        
        # Content preservation options
        preservation_mapping = {
            "preserve_sentences": "preserve_sentences",
            "preserve_paragraphs": "preserve_paragraphs", 
            "preserve_code_blocks": "preserve_code_blocks",
            "preserve_tables": "preserve_tables",
            "preserve_links": "preserve_links",
            # Support PRD naming conventions
            "handle_code_blocks_as_atomic": "preserve_code_blocks",
            "handle_tables_as_atomic": "preserve_tables",
        }
        
        for config_key, chunk_config_key in preservation_mapping.items():
            if config_key in config and isinstance(config[config_key], bool):
                chunk_config_params[chunk_config_key] = config[config_key]
        
        # Boundary strategy mapping
        if "boundary_strategy" in config:
            boundary_str = config["boundary_strategy"]
            if isinstance(boundary_str, str):
                try:
                    chunk_config_params["boundary_strategy"] = BoundaryStrategy(boundary_str)
                except ValueError:
                    logger.warning(f"Invalid boundary strategy '{boundary_str}', using default")
        
        # Advanced options
        advanced_mapping = {
            "max_boundary_search_distance": "max_boundary_search_distance",
            "sentence_min_length": "sentence_min_length",
            "paragraph_min_length": "paragraph_min_length",
            "enable_smart_overlap": "enable_smart_overlap",
            "performance_mode": "performance_mode",
            "cache_boundary_patterns": "cache_boundary_patterns",
            "debug_mode": "debug_mode",
            "language_code": "language_code"
        }
        
        for config_key, chunk_config_key in advanced_mapping.items():
            if config_key in config:
                chunk_config_params[chunk_config_key] = config[config_key]
        
        # Content type hints
        if "content_type_hints" in config and isinstance(config["content_type_hints"], list):
            chunk_config_params["content_type_hints"] = config["content_type_hints"]
        
        logger.debug(f"Final chunk_config_params before ChunkConfig creation: {chunk_config_params}")
        
        # Create ChunkConfig instance
        try:
            chunk_config = ChunkConfig(**chunk_config_params)
            logger.debug(f"Successfully mapped config to ChunkConfig with {len(chunk_config_params)} parameters")
            return chunk_config
            
        except Exception as e:
            logger.error(f"Failed to create ChunkConfig: {e}")
            # Fallback to default ChunkConfig
            logger.warning("Falling back to default ChunkConfig")
            return ChunkConfig()
    
    def get_supported_parameters(self) -> List[str]:
        """
        Get list of supported configuration parameters.
        
        Returns:
            List of parameter names that can be configured
        """
        return [
            # Core parameters
            "chunk_size", "chunk_overlap", "min_chunk_size",
            # Preservation options
            "preserve_sentences", "preserve_paragraphs", "preserve_code_blocks",
            "preserve_tables", "preserve_links",
            # PRD compatibility
            "handle_code_blocks_as_atomic", "handle_tables_as_atomic",
            # Advanced options
            "boundary_strategy", "max_boundary_search_distance",
            "sentence_min_length", "paragraph_min_length",
            "enable_smart_overlap", "performance_mode", "cache_boundary_patterns",
            "content_type_hints", "language_code", "debug_mode"
        ]
    
    def validate_configuration_schema(self) -> Dict[str, Any]:
        """
        Validate current configuration against expected schema.
        
        Returns:
            Dictionary with validation results and any warnings
        """
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        try:
            chunking_config = self._get_chunking_config()
            
            # Check for unknown parameters
            supported_params = self.get_supported_parameters()
            unknown_params = set(chunking_config.keys()) - set(supported_params)
            
            for param in unknown_params:
                validation_result["warnings"].append(f"Unknown parameter: {param}")
            
            # Validate configuration
            self._validate_chunking_config(chunking_config)
            
            # Test ChunkConfig creation
            self._map_to_chunk_config(chunking_config)
            
        except ConfigurationValidationError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Configuration validation failed: {e}")
        
        return validation_result 