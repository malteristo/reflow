"""
Main configuration manager for Research Agent.

This module provides the ConfigManager class that orchestrates all configuration
management functionality including loading, validation, inheritance, and environment
variable handling.

Implements FR-CF-001: Configuration-driven behavior with centralized management.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from copy import deepcopy

from ...exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
    ConfigurationSchemaError,
    ConfigurationMergeError,
    ConfigurationMigrationError,
    EnvironmentVariableError,
)
from .paths import ConfigPaths
from .file_operations import FileOperations
from .schema_validation import SchemaValidator
from .inheritance import ConfigInheritance
from .environment import EnvironmentHandler
from .migration import ConfigurationMigrator, ConfigurationValidator, MigrationResult


logger = logging.getLogger(__name__)


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
        auto_migrate: bool = True,
    ) -> None:
        """
        Initialize the ConfigManager.
        
        Args:
            config_file: Path to main configuration file (default: researchagent.config.json)
            project_root: Project root directory (default: current working directory)
            load_env: Whether to load environment variables from .env file
            auto_migrate: Whether to automatically migrate old configuration versions
        """
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.config_file = config_file or ConfigPaths.DEFAULT_CONFIG_FILE
        self.paths = ConfigPaths()
        self.auto_migrate = auto_migrate
        
        # Configuration data
        self._config: Dict[str, Any] = {}
        self._loaded = False
        self._migration_warnings: List[str] = []
        
        # Set up logging
        self.logger = logger
        
        # Initialize components
        self.file_ops = FileOperations(self.project_root, self.paths.ENV_FILE)
        self.schema_validator = SchemaValidator(self.file_ops, self.paths)
        self.inheritance = ConfigInheritance(self.file_ops, self.paths)
        self.env_handler = EnvironmentHandler()
        
        # Initialize migration components
        self.migrator = ConfigurationMigrator(self.file_ops, self.paths)
        self.validator = ConfigurationValidator(self.file_ops, self.paths, self.migrator)
        
        # Load environment variables if requested
        if load_env:
            self.file_ops.load_environment_variables()
    
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
    
    def load_config(
        self, 
        force_reload: bool = False,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from all sources with automatic migration support.
        
        Args:
            force_reload: Force reloading even if already loaded
            validate: Whether to validate configuration against schema
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigurationError: If loading fails
            ConfigurationValidationError: If validation fails
            ConfigurationMigrationError: If migration fails
        """
        if self._loaded and not force_reload:
            self.logger.debug("Configuration already loaded, returning cached version")
            return deepcopy(self._config)
        
        self.logger.info(f"Loading configuration from {self.config_file}")
        
        try:
            # Reset migration warnings
            self._migration_warnings = []
            
            # Load main configuration file
            main_config_path = self.file_ops.resolve_path(self.config_file)
            self.logger.debug(f"Loading main configuration file: {main_config_path}")
            raw_config = self.file_ops.load_json_file(main_config_path)
            
            # Resolve configuration inheritance (extends)
            self.logger.debug("Resolving configuration inheritance")
            inherited_config = self.inheritance.resolve_config_inheritance(raw_config)
            
            # Merge with defaults if available
            self.logger.debug("Merging with default configuration")
            merged_config = self.inheritance.merge_with_defaults(inherited_config)
            
            # Apply environment variable overrides
            self.logger.debug("Applying environment variable overrides")
            pre_validation_config = self.env_handler.apply_environment_overrides(merged_config)
            
            # Validate and migrate configuration if needed
            if validate:
                self.logger.debug("Validating and migrating configuration")
                validated_config, warnings = self.validator.validate_and_migrate(
                    pre_validation_config,
                    config_file=self.config_file,
                    auto_migrate=self.auto_migrate
                )
                self._config = validated_config
                self._migration_warnings = warnings
                
                # Log migration warnings
                for warning in warnings:
                    self.logger.warning(f"Configuration: {warning}")
            else:
                self._config = pre_validation_config
            
            # Mark as loaded
            self._loaded = True
            self.logger.info("Configuration loaded successfully")
            
            return deepcopy(self._config)
            
        except (ConfigurationValidationError, ConfigurationSchemaError, ConfigurationMergeError, ConfigurationMigrationError) as e:
            # Re-raise configuration errors as-is with logging
            self.logger.error(f"Configuration loading failed: {e}")
            self._loaded = False
            raise
        except ConfigurationFileNotFoundError as e:
            # Re-raise file not found errors as-is
            self.logger.error(f"Configuration file not found: {e}")
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
        
        # Store old value for change detection
        old_value = self.get(key, None)
        
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
        
        # Trigger model change detection for embedding model changes
        if key.startswith('embedding_model') and old_value != value:
            self._trigger_embedding_model_change(key, old_value, value)
    
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
        self._loaded = False
    
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
        
        return self.schema_validator.validate_config(
            config, 
            schema_file, 
            self.config_file
        )
    
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
        return self.inheritance.merge_configs(*configs)
    
    def diagnose_config_issues(self) -> Dict[str, Any]:
        """
        Diagnose common configuration issues.
        
        Returns:
            Dictionary with diagnostic information
        """
        # Get file diagnostics
        file_diagnostics = self.file_ops.diagnose_file_issues(
            self.config_file,
            self.paths.SCHEMA_DIR,
            self.paths.DEFAULT_CONFIG_SCHEMA
        )
        
        # Get environment diagnostics
        env_diagnostics = self.env_handler.diagnose_env_issues()
        
        # Merge diagnostics
        diagnostics = {
            **file_diagnostics,
            **env_diagnostics
        }
        
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
                env_mapping = self.env_handler.get_env_mapping()
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
        return self.env_handler.get_env_var(var_name, default, required)
    
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
        return self.env_handler.validate_required_env_vars(required_vars)
    
    # Configuration Migration Methods
    
    @property
    def migration_warnings(self) -> List[str]:
        """Get warnings from the last configuration migration."""
        return self._migration_warnings.copy()
    
    def get_config_version(self) -> str:
        """
        Get the version of the currently loaded configuration.
        
        Returns:
            Configuration version string
        """
        config = self.config
        return self.migrator.detect_config_version(config)
    
    def needs_migration(self) -> bool:
        """
        Check if the current configuration needs migration.
        
        Returns:
            True if migration is needed
        """
        config = self.config
        return self.migrator.needs_migration(config)
    
    def migrate_config_file(
        self,
        target_version: Optional[str] = None,
        create_backup: bool = True
    ) -> MigrationResult:
        """
        Migrate the configuration file to target version.
        
        Args:
            target_version: Target version (defaults to current)
            create_backup: Whether to create backup before migration
            
        Returns:
            Migration result with details
            
        Raises:
            ConfigurationMigrationError: If migration fails
        """
        config_path = self.file_ops.resolve_path(self.config_file)
        result = self.migrator.migrate_config_file(
            config_path,
            target_version,
            create_backup
        )
        
        # Reload configuration after migration
        if result.success:
            self.reload_config()
        
        return result
    
    def create_config_backup(self, backup_suffix: Optional[str] = None) -> str:
        """
        Create a backup of the current configuration file.
        
        Args:
            backup_suffix: Optional suffix for backup filename
            
        Returns:
            Path to created backup file
            
        Raises:
            ConfigurationError: If backup creation fails
        """
        config_path = self.file_ops.resolve_path(self.config_file)
        return self.migrator.create_backup(config_path, backup_suffix)
    
    def restore_config_backup(self, backup_path: Union[str, Path]) -> None:
        """
        Restore configuration from backup.
        
        Args:
            backup_path: Path to backup file
            
        Raises:
            ConfigurationError: If restore fails
        """
        config_path = self.file_ops.resolve_path(self.config_file)
        self.migrator.restore_backup(backup_path, config_path)
        
        # Reload configuration after restore
        self.reload_config()
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path to save the configuration file
                        (uses default config file if None)
        """
        if not self._loaded:
            raise ConfigurationError("Cannot save unloaded configuration")
        
        # Use provided path or default config file
        save_path = Path(config_path) if config_path else Path(self.config_file)
        
        # Save configuration
        self.file_ops.save_config(self._config, save_path)
        
        # Update config_file if a new path was provided
        if config_path:
            self.config_file = str(save_path)
    
    def _trigger_embedding_model_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """
        Trigger model change detection when embedding model configuration changes.
        
        Args:
            key: Configuration key that changed
            old_value: Previous value
            new_value: New value
        """
        try:
            from ...core.model_change_detection.integration_hooks import trigger_config_change
            
            # Trigger configuration change detection
            trigger_config_change('embedding_model', {
                'config_key': key,
                'old_value': old_value,
                'new_value': new_value,
                'full_config': self.get('embedding_model', {})
            })
            
            logger.debug(f"Triggered model change detection for config key: {key}")
            
        except ImportError:
            # Model change detection not available
            logger.debug("Model change detection not available, skipping callback")
        except Exception as e:
            # Don't fail configuration changes if model change detection fails
            logger.warning(f"Failed to trigger model change detection: {e}") 