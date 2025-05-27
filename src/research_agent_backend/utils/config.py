"""
Configuration management for Research Agent.

This module provides the ConfigManager class for loading, validating, and managing
configuration settings from JSON files and environment variables.

Implements FR-CF-001: Configuration-driven behavior with validation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from copy import deepcopy

import jsonschema
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from ..exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
    ConfigurationSchemaError,
    ConfigurationMergeError,
    EnvironmentVariableError,
)


@dataclass
class ConfigPaths:
    """Configuration file paths and constants."""
    
    DEFAULT_CONFIG_FILE: str = "researchagent.config.json"
    DEFAULT_CONFIG_DIR: str = "./config/defaults"
    SCHEMA_DIR: str = "./config/schema"
    ENV_FILE: str = ".env"
    DEFAULT_CONFIG_SCHEMA: str = "config_schema.json"


class ConfigManager:
    """
    Configuration manager for Research Agent.
    
    Handles loading, validation, and merging of configuration from multiple sources:
    - Default configuration files
    - User configuration files
    - Environment variables
    
    Implements FR-CF-001: Configuration-driven behavior with centralized management.
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        project_root: Optional[Union[str, Path]] = None,
        load_env: bool = True,
    ) -> None:
        """
        Initialize the ConfigManager.
        
        Args:
            config_file: Path to main configuration file (default: researchagent.config.json)
            project_root: Project root directory (default: current working directory)
            load_env: Whether to load environment variables from .env file
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.config_file = config_file or ConfigPaths.DEFAULT_CONFIG_FILE
        self.paths = ConfigPaths()
        
        # Configuration data
        self._config: Dict[str, Any] = {}
        self._schema: Dict[str, Any] = {}
        self._loaded = False
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables if requested
        if load_env:
            self._load_environment_variables()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration. Loads if not already loaded."""
        if not self._loaded:
            self.load_config()
        return deepcopy(self._config)
    
    @property
    def is_loaded(self) -> bool:
        """Check if configuration has been loaded."""
        return self._loaded
    
    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path relative to the project root.
        
        Args:
            path: Path to resolve (can be relative or absolute)
            
        Returns:
            Resolved absolute path
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return (self.project_root / path).resolve()
    
    def _load_environment_variables(self) -> None:
        """
        Load environment variables from .env file if it exists.
        
        Does not raise errors if .env file is missing.
        """
        env_file_path = self._resolve_path(self.paths.ENV_FILE)
        if env_file_path.exists():
            try:
                self.logger.debug(f"Loading environment variables from {env_file_path}")
                load_dotenv(env_file_path)
                self.logger.info(f"Successfully loaded environment variables from {env_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load environment variables from {env_file_path}: {e}")
        else:
            self.logger.debug(f"Environment file not found at {env_file_path}, skipping")
    
    def _load_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and parse a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON data
            
        Raises:
            ConfigurationFileNotFoundError: If file doesn't exist
            ConfigurationError: If JSON parsing fails
        """
        resolved_path = self._resolve_path(str(file_path))
        
        self.logger.debug(f"Attempting to load configuration file: {resolved_path}")
        
        if not resolved_path.exists():
            error_msg = f"Configuration file not found: {resolved_path}"
            self.logger.error(error_msg)
            raise ConfigurationFileNotFoundError(error_msg)
        
        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.logger.info(f"Successfully loaded configuration from {resolved_path}")
                return config_data
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file {resolved_path}: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
        except PermissionError as e:
            error_msg = f"Permission denied reading configuration file {resolved_path}: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
        except Exception as e:
            error_msg = f"Error reading configuration file {resolved_path}: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def load_config(
        self, 
        force_reload: bool = False,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from all sources.
        
        Args:
            force_reload: Force reloading even if already loaded
            validate: Whether to validate configuration against schema
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigurationError: If loading fails
            ConfigurationValidationError: If validation fails
        """
        if self._loaded and not force_reload:
            self.logger.debug("Configuration already loaded, returning cached version")
            return deepcopy(self._config)
        
        self.logger.info(f"Loading configuration from {self.config_file}")
        
        try:
            # Load main configuration file
            main_config_path = self._resolve_path(self.config_file)
            self.logger.debug(f"Loading main configuration file: {main_config_path}")
            raw_config = self._load_json_file(main_config_path)
            
            # Resolve configuration inheritance (extends)
            self.logger.debug("Resolving configuration inheritance")
            inherited_config = self._resolve_config_inheritance(raw_config)
            
            # Merge with defaults if available
            self.logger.debug("Merging with default configuration")
            merged_config = self._merge_with_defaults(inherited_config)
            
            # Apply environment variable overrides
            self.logger.debug("Applying environment variable overrides")
            self._config = self._apply_environment_overrides(merged_config)
            
            # Validate configuration if requested
            if validate:
                self.logger.debug("Validating configuration against schema")
                self._validate_config_against_schema(self._config)
            
            # Mark as loaded
            self._loaded = True
            self.logger.info("Configuration loaded successfully")
            
            return deepcopy(self._config)
            
        except (ConfigurationValidationError, ConfigurationSchemaError, ConfigurationMergeError) as e:
            # Re-raise validation and merge errors as-is with logging
            self.logger.error(f"Configuration loading failed: {e}")
            self._loaded = False
            raise
        except PermissionError as e:
            error_msg = f"Permission denied accessing configuration files: {e}"
            self.logger.error(error_msg)
            self._loaded = False
            raise ConfigurationError(error_msg) from e
        except FileNotFoundError as e:
            error_msg = f"Configuration file not found: {e}"
            self.logger.error(error_msg)
            self._loaded = False
            raise ConfigurationFileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading configuration: {e}"
            self.logger.error(error_msg, exc_info=True)
            self._loaded = False
            raise ConfigurationError(error_msg) from e
    
    def reload_config(self) -> Dict[str, Any]:
        """
        Force reload configuration from all sources.
        
        Returns:
            Reloaded configuration dictionary
        """
        return self.load_config(force_reload=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'embedding_model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.config
        keys = key.split('.')
        
        try:
            for k in keys:
                config = config[k]
            return config
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            
        Note:
            This modifies the in-memory configuration only.
            Use save_config() to persist changes.
        """
        if not self._loaded:
            self.load_config()
        
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def has(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get(key, object()) is not object()
    
    def reset(self) -> None:
        """Reset configuration state, forcing reload on next access."""
        self._config = {}
        self._schema = {}
        self._loaded = False
    
    def _load_schema(self, schema_file: Optional[str] = None) -> Dict[str, Any]:
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
            schema_file = self._resolve_path(
                f"{self.paths.SCHEMA_DIR}/{self.paths.DEFAULT_CONFIG_SCHEMA}"
            )
        else:
            schema_file = self._resolve_path(schema_file)
        
        try:
            return self._load_json_file(schema_file)
        except ConfigurationFileNotFoundError as e:
            raise ConfigurationSchemaError(
                f"Configuration schema file not found: {schema_file}",
                str(schema_file)
            ) from e
        except ConfigurationError as e:
            raise ConfigurationSchemaError(
                f"Invalid configuration schema: {e}",
                str(schema_file)
            ) from e
    
    def _validate_config_against_schema(
        self, 
        config: Dict[str, Any], 
        schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Validate configuration against JSON schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: JSON schema (loads default if not provided)
            
        Raises:
            ConfigurationValidationError: If validation fails
        """
        if schema is None:
            try:
                schema = self._load_schema()
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
                self.config_file,
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
        config: Optional[Dict[str, Any]] = None,
        schema_file: Optional[str] = None
    ) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate (uses loaded config if None)
            schema_file: Path to schema file (uses default if None)
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigurationValidationError: If validation fails
            ConfigurationSchemaError: If schema is invalid
        """
        if config is None:
            config = self.config
        
        schema = None
        if schema_file:
            schema = self._load_schema(schema_file)
        
        self._validate_config_against_schema(config, schema)
        return True
    
    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                # Override value (or add new key)
                result[key] = deepcopy(value)
        
        return result
    
    def diagnose_config_issues(self) -> Dict[str, Any]:
        """
        Diagnose common configuration issues.
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            "config_file_exists": False,
            "config_file_readable": False,
            "config_file_valid_json": False,
            "schema_file_exists": False,
            "env_file_exists": False,
            "missing_env_vars": [],
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check main config file
            config_path = self._resolve_path(self.config_file)
            diagnostics["config_file_exists"] = config_path.exists()
            
            if diagnostics["config_file_exists"]:
                try:
                    with open(config_path, 'r') as f:
                        diagnostics["config_file_readable"] = True
                        json.load(f)
                        diagnostics["config_file_valid_json"] = True
                except PermissionError:
                    diagnostics["errors"].append("Permission denied reading config file")
                except json.JSONDecodeError as e:
                    diagnostics["errors"].append(f"Invalid JSON in config file: {e}")
            else:
                diagnostics["errors"].append("Configuration file does not exist")
            
            # Check schema file
            schema_path = self._resolve_path(f"{self.paths.SCHEMA_DIR}/{self.paths.DEFAULT_CONFIG_SCHEMA}")
            diagnostics["schema_file_exists"] = schema_path.exists()
            if not diagnostics["schema_file_exists"]:
                diagnostics["warnings"].append("Schema file not found - validation will be skipped")
            
            # Check environment file
            env_path = self._resolve_path(self.paths.ENV_FILE)
            diagnostics["env_file_exists"] = env_path.exists()
            if not diagnostics["env_file_exists"]:
                diagnostics["warnings"].append("Environment file (.env) not found")
            
            # Check for common missing environment variables
            common_env_vars = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'PERPLEXITY_API_KEY']
            for var in common_env_vars:
                if not os.getenv(var):
                    diagnostics["missing_env_vars"].append(var)
            
        except Exception as e:
            diagnostics["errors"].append(f"Error during diagnostics: {e}")
        
        return diagnostics
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration state.
        
        Returns:
            Dictionary with configuration summary
        """
        summary = {
            "loaded": self._loaded,
            "config_file": self.config_file,
            "project_root": str(self.project_root),
            "config_keys": [],
            "environment_overrides": [],
            "total_config_size": 0
        }
        
        if self._loaded:
            try:
                config = self.config
                summary["config_keys"] = list(self._get_all_keys(config))
                summary["total_config_size"] = len(json.dumps(config))
                
                # Check which environment variables are active
                env_mapping = self._get_env_mapping()
                for env_var, config_key in env_mapping.items():
                    if os.getenv(env_var):
                        summary["environment_overrides"].append({
                            "env_var": env_var,
                            "config_key": config_key,
                            "value_set": True
                        })
            except Exception as e:
                summary["error"] = str(e)
        
        return summary
    
    def _get_all_keys(self, config: Dict[str, Any], prefix: str = "") -> List[str]:
        """
        Get all configuration keys using dot notation.
        
        Args:
            config: Configuration dictionary
            prefix: Key prefix for nested keys
            
        Returns:
            List of all configuration keys
        """
        keys = []
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value, full_key))
        return keys
    
    def _resolve_config_inheritance(self, config: Dict[str, Any], visited_files: Optional[set] = None) -> Dict[str, Any]:
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
        resolved_extends_path = str(self._resolve_path(extends_path))
        
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
            parent_config = self._load_json_file(resolved_extends_path)
            
            # Recursively resolve parent's inheritance
            resolved_parent = self._resolve_config_inheritance(parent_config, visited_files.copy())
            
            # Remove the 'extends' field from current config before merging
            current_config = {k: v for k, v in config.items() if k != 'extends'}
            
            # Merge parent and current configurations
            merged_config = self._deep_merge_dicts(resolved_parent, current_config)
            
            return merged_config
            
        except (ConfigurationFileNotFoundError, ConfigurationError) as e:
            raise ConfigurationMergeError(
                f"Failed to resolve configuration inheritance: {e}",
                [resolved_extends_path],
                []
            ) from e
        except Exception as e:
            raise ConfigurationMergeError(
                f"Unexpected error during configuration merging: {e}",
                [resolved_extends_path],
                []
            ) from e
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with default values.
        
        Args:
            config: User configuration
            
        Returns:
            Configuration merged with defaults
        """
        try:
            # Try to load default configuration
            default_config_path = self._resolve_path(
                f"{self.paths.DEFAULT_CONFIG_DIR}/default_config.json"
            )
            
            if default_config_path.exists():
                default_config = self._load_json_file(default_config_path)
                # Resolve defaults inheritance first
                resolved_defaults = self._resolve_config_inheritance(default_config)
                # Merge user config over defaults
                return self._deep_merge_dicts(resolved_defaults, config)
            else:
                return config
                
        except Exception:
            # If defaults can't be loaded, return user config as-is
            return config
    
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
                result = self._deep_merge_dicts(result, config)
            return result
        except Exception as e:
            raise ConfigurationMergeError(
                f"Failed to merge configurations: {e}",
                [],
                []
            ) from e
    
    def _get_env_mapping(self) -> Dict[str, str]:
        """
        Get mapping of environment variable names to configuration keys.
        
        Returns:
            Dictionary mapping env var names to config keys
        """
        return {
            # API Keys for external services
            'ANTHROPIC_API_KEY': 'api_keys.anthropic',
            'OPENAI_API_KEY': 'api_keys.openai',
            'PERPLEXITY_API_KEY': 'api_keys.perplexity',
            'GOOGLE_API_KEY': 'api_keys.google',
            'MISTRAL_API_KEY': 'api_keys.mistral',
            'AZURE_OPENAI_API_KEY': 'api_keys.azure_openai',
            'AZURE_OPENAI_ENDPOINT': 'api_endpoints.azure_openai',
            'OPENROUTER_API_KEY': 'api_keys.openrouter',
            'XAI_API_KEY': 'api_keys.xai',
            'OLLAMA_API_KEY': 'api_keys.ollama',
            'OLLAMA_BASE_URL': 'api_endpoints.ollama',
            
            # Database and storage credentials
            'DATABASE_URL': 'database.url',
            'REDIS_URL': 'cache.redis_url',
            
            # Security and encryption
            'SECRET_KEY': 'security.secret_key',
            'ENCRYPTION_KEY': 'security.encryption_key',
            
            # Environment overrides
            'RESEARCH_AGENT_LOG_LEVEL': 'logging.level',
            'RESEARCH_AGENT_DEBUG': 'debug.enabled',
            'RESEARCH_AGENT_CACHE_DIR': 'performance.cache_directory',
        }
    
    def _convert_env_value(self, value: str, target_type: str = 'string') -> Any:
        """
        Convert environment variable string to appropriate Python type.
        
        Args:
            value: Environment variable value (always string)
            target_type: Target type ('string', 'boolean', 'integer', 'float', 'json')
            
        Returns:
            Converted value
            
        Raises:
            EnvironmentVariableError: If conversion fails
        """
        if not value:
            return None
        
        try:
            if target_type == 'boolean':
                return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            elif target_type == 'integer':
                return int(value)
            elif target_type == 'float':
                return float(value)
            elif target_type == 'json':
                import json
                return json.loads(value)
            else:  # string
                return value
        except (ValueError, json.JSONDecodeError) as e:
            raise EnvironmentVariableError(
                f"Failed to convert environment variable value '{value}' to {target_type}: {e}",
                value
            ) from e
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        env_mapping = self._get_env_mapping()
        result = deepcopy(config)
        
        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Determine target type based on config key
                target_type = 'string'
                if 'debug' in config_key.lower() or 'enabled' in config_key.lower():
                    target_type = 'boolean'
                elif config_key.endswith('.level') and env_var.endswith('_LOG_LEVEL'):
                    target_type = 'string'  # Log levels are strings
                
                try:
                    converted_value = self._convert_env_value(env_value, target_type)
                    self._set_nested_value(result, config_key, converted_value)
                    self.logger.debug(f"Applied environment override: {env_var} -> {config_key}")
                except EnvironmentVariableError as e:
                    # Log error but continue (don't fail config loading)
                    self.logger.warning(f"Failed to apply environment variable {env_var}: {e}")
        
        return result
    
    def _set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        Set a nested value in configuration using dot notation.
        
        Args:
            config: Configuration dictionary to modify
            key_path: Dot-separated key path (e.g., 'api_keys.openai')
            value: Value to set
        """
        keys = key_path.split('.')
        current = config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    def get_env_var(self, var_name: str, default: Any = None, required: bool = False) -> Any:
        """
        Get environment variable with optional type conversion and validation.
        
        Args:
            var_name: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value or default
            
        Raises:
            EnvironmentVariableError: If required variable is missing
        """
        value = os.getenv(var_name, default)
        
        if required and value is None:
            raise EnvironmentVariableError(
                f"Required environment variable '{var_name}' is not set",
                var_name
            )
        
        return value
    
    def validate_required_env_vars(self, required_vars: List[str]) -> Dict[str, str]:
        """
        Validate that all required environment variables are set.
        
        Args:
            required_vars: List of required environment variable names
            
        Returns:
            Dictionary of variable names and values
            
        Raises:
            EnvironmentVariableError: If any required variables are missing
        """
        missing_vars = []
        result = {}
        
        for var_name in required_vars:
            value = os.getenv(var_name)
            if value is None:
                missing_vars.append(var_name)
            else:
                result[var_name] = value
        
        if missing_vars:
            raise EnvironmentVariableError(
                f"Missing required environment variables: {', '.join(missing_vars)}",
                None,
                missing_vars
            )
        
        return result 