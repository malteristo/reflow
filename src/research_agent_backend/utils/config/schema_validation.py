"""
Schema validation for configuration management.

This module provides JSON schema loading, validation, and error handling
capabilities for the Research Agent configuration system.

Implements FR-CF-001: Configuration-driven behavior with validation.
"""

import logging
from typing import Any, Dict, Optional, List

import jsonschema

from ...exceptions.config_exceptions import (
    ConfigurationSchemaError,
    ConfigurationValidationError,
)
from .file_operations import FileOperations
from .paths import ConfigPaths


logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Schema validation for configuration management.
    
    Handles JSON schema loading, validation, and comprehensive error reporting.
    """
    
    def __init__(self, file_ops: FileOperations, paths: ConfigPaths) -> None:
        """
        Initialize schema validator.
        
        Args:
            file_ops: File operations instance
            paths: Configuration paths instance
        """
        self.file_ops = file_ops
        self.paths = paths
        self.logger = logger
    
    def load_schema(self, schema_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load JSON schema for configuration validation.
        
        Args:
            schema_file: Path to schema file (default: config_schema.json)
            
        Returns:
            Loaded JSON schema
            
        Raises:
            ConfigurationSchemaError: If schema loading fails
        """
        if not schema_file:
            schema_file = self.file_ops.resolve_path(
                f"{self.paths.SCHEMA_DIR}/{self.paths.DEFAULT_CONFIG_SCHEMA}"
            )
        else:
            schema_file = self.file_ops.resolve_path(schema_file)
        
        try:
            return self.file_ops.load_json_file(schema_file)
        except Exception as e:
            if "not found" in str(e).lower():
                raise ConfigurationSchemaError(
                    f"Configuration schema file not found: {schema_file}",
                    str(schema_file)
                ) from e
            else:
                raise ConfigurationSchemaError(
                    f"Invalid configuration schema: {e}",
                    str(schema_file)
                ) from e
    
    def validate_config_against_schema(
        self, 
        config: Dict[str, Any], 
        schema: Optional[Dict[str, Any]] = None,
        config_file: str = "unknown"
    ) -> None:
        """
        Validate configuration against JSON schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: JSON schema (loads default if not provided)
            config_file: Configuration file name for error reporting
            
        Raises:
            ConfigurationValidationError: If validation fails
        """
        if schema is None:
            try:
                schema = self.load_schema()
            except ConfigurationSchemaError:
                # If schema is not available, skip validation
                return
        
        try:
            jsonschema.validate(config, schema)
        except jsonschema.ValidationError as e:
            # Extract validation errors for user-friendly messages
            validation_errors = [e.message]
            invalid_fields = []
            
            if e.absolute_path:
                field_path = ".".join(str(p) for p in e.absolute_path)
                invalid_fields.append(field_path)
            
            # Collect additional validation errors if available
            context = getattr(e, 'context', [])
            for ctx_error in context:
                validation_errors.append(ctx_error.message)
                if ctx_error.absolute_path:
                    field_path = ".".join(str(p) for p in ctx_error.absolute_path)
                    invalid_fields.append(field_path)
            
            raise ConfigurationValidationError(
                f"Configuration validation failed: {e.message}",
                config_file,
                validation_errors,
                invalid_fields
            ) from e
        except jsonschema.SchemaError as e:
            raise ConfigurationSchemaError(
                f"Invalid JSON schema: {e.message}",
                schema_errors=[e.message]
            ) from e
    
    def validate_config(
        self, 
        config: Dict[str, Any],
        schema_file: Optional[str] = None,
        config_file: str = "unknown"
    ) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema_file: Path to schema file (uses default if None)
            config_file: Configuration file name for error reporting
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigurationValidationError: If validation fails
            ConfigurationSchemaError: If schema is invalid
        """
        schema = None
        if schema_file:
            schema = self.load_schema(schema_file)
        
        self.validate_config_against_schema(config, schema, config_file)
        return True 