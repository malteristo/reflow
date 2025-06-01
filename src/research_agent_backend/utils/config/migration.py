"""
Configuration migration and versioning system for Research Agent.

This module provides schema version detection, automatic configuration migration,
backup/restore functionality, and rollback capabilities for configuration files.

Implements FR-CF-001: Configuration validation and migration with version management.
"""

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from ...exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationValidationError,
    ConfigurationMigrationError,
)
from .file_operations import FileOperations
from .paths import ConfigPaths


logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Result of a configuration migration operation."""
    success: bool
    from_version: str
    to_version: str
    backup_path: Optional[str] = None
    migration_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ConfigVersion:
    """Configuration version information."""
    version: str
    schema_path: Optional[str] = None
    migration_rules: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=lambda: ["version"])
    deprecated_fields: List[str] = field(default_factory=list)


class ConfigurationMigrator:
    """
    Configuration migration system for Research Agent.
    
    Handles version detection, automatic migration, backup/restore, and rollback
    functionality for configuration files with schema evolution support.
    """
    
    # Supported configuration versions and their migration rules
    SUPPORTED_VERSIONS = {
        "1.0.0": ConfigVersion(
            version="1.0.0",
            required_fields=["version", "embedding_model", "vector_store"],
            deprecated_fields=[],
        ),
        "1.1.0": ConfigVersion(
            version="1.1.0", 
            required_fields=["version", "embedding_model", "vector_store", "rag_pipeline"],
            deprecated_fields=["old_chunking_config"],
            migration_rules={
                "chunking_strategy": {
                    "type": "hybrid",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "markdown_aware": True
                },
                "rag_pipeline": {
                    "vector_search_top_k": 50,
                    "rerank_top_k": 10,
                    "enable_reranking": True
                }
            }
        ),
        "1.2.0": ConfigVersion(
            version="1.2.0",
            required_fields=["version", "embedding_model", "vector_store", "rag_pipeline", "collections"],
            deprecated_fields=["old_chunking_config", "legacy_search_config"],
            migration_rules={
                "collections": {
                    "default_type": "research",
                    "auto_create": True,
                    "metadata_fields": ["source", "created_at", "user_id"]
                },
                "performance": {
                    "enable_caching": True,
                    "cache_directory": "./cache",
                    "embedding_cache_size": 1000,
                    "query_cache_size": 100
                }
            }
        )
    }
    
    CURRENT_VERSION = "1.2.0"
    
    def __init__(
        self,
        file_ops: FileOperations,
        paths: ConfigPaths,
        backup_directory: Optional[str] = None
    ) -> None:
        """
        Initialize configuration migrator.
        
        Args:
            file_ops: File operations instance
            paths: Configuration paths instance  
            backup_directory: Directory for configuration backups
        """
        self.file_ops = file_ops
        self.paths = paths
        self.backup_directory = Path(backup_directory or "./config/backups")
        self.logger = logger
        
        # Ensure backup directory exists
        self.backup_directory.mkdir(parents=True, exist_ok=True)
    
    def detect_config_version(self, config: Dict[str, Any]) -> str:
        """
        Detect the version of a configuration dictionary.
        
        Args:
            config: Configuration dictionary to analyze
            
        Returns:
            Detected version string
            
        Raises:
            ConfigurationError: If version cannot be determined
        """
        # Check for explicit version field
        if "version" in config:
            version = str(config["version"]).strip()
            if version in self.SUPPORTED_VERSIONS:
                return version
            else:
                # Unknown version - assume latest for forward compatibility
                self.logger.warning(f"Unknown configuration version '{version}', treating as current")
                return self.CURRENT_VERSION
        
        # Infer version from configuration structure
        if "rag_pipeline" in config and "collections" in config:
            return "1.2.0"
        elif "rag_pipeline" in config:
            return "1.1.0"
        elif "embedding_model" in config and "vector_store" in config:
            return "1.0.0"
        else:
            # Very old or incomplete config - start with 1.0.0
            self.logger.warning("Cannot determine config version, assuming 1.0.0")
            return "1.0.0"
    
    def needs_migration(self, config: Dict[str, Any]) -> bool:
        """
        Check if configuration needs migration to current version.
        
        Args:
            config: Configuration to check
            
        Returns:
            True if migration is needed
        """
        current_version = self.detect_config_version(config)
        return current_version != self.CURRENT_VERSION
    
    def create_backup(
        self,
        config_path: Union[str, Path],
        backup_suffix: Optional[str] = None
    ) -> str:
        """
        Create a backup of the configuration file.
        
        Args:
            config_path: Path to configuration file to backup
            backup_suffix: Optional suffix for backup filename
            
        Returns:
            Path to created backup file
            
        Raises:
            ConfigurationError: If backup creation fails
        """
        try:
            config_path = Path(config_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if backup_suffix:
                backup_name = f"{config_path.stem}_{backup_suffix}_{timestamp}.backup.json"
            else:
                backup_name = f"{config_path.stem}_{timestamp}.backup.json"
            
            backup_path = self.backup_directory / backup_name
            
            # Copy configuration file to backup
            shutil.copy2(config_path, backup_path)
            
            self.logger.info(f"Configuration backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            error_msg = f"Failed to create configuration backup: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def restore_backup(
        self,
        backup_path: Union[str, Path],
        target_path: Union[str, Path]
    ) -> None:
        """
        Restore configuration from backup.
        
        Args:
            backup_path: Path to backup file
            target_path: Path where to restore configuration
            
        Raises:
            ConfigurationError: If restore fails
        """
        try:
            backup_path = Path(backup_path)
            target_path = Path(target_path)
            
            if not backup_path.exists():
                raise ConfigurationError(f"Backup file not found: {backup_path}")
            
            # Copy backup to target location
            shutil.copy2(backup_path, target_path)
            
            self.logger.info(f"Configuration restored from backup: {backup_path} -> {target_path}")
            
        except Exception as e:
            error_msg = f"Failed to restore configuration backup: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def migrate_config(
        self,
        config: Dict[str, Any],
        target_version: Optional[str] = None
    ) -> Tuple[Dict[str, Any], MigrationResult]:
        """
        Migrate configuration to target version.
        
        Args:
            config: Configuration to migrate
            target_version: Target version (defaults to current)
            
        Returns:
            Tuple of (migrated_config, migration_result)
            
        Raises:
            ConfigurationMigrationError: If migration fails
        """
        start_time = time.time()
        target_version = target_version or self.CURRENT_VERSION
        
        try:
            from_version = self.detect_config_version(config)
            
            if from_version == target_version:
                return config, MigrationResult(
                    success=True,
                    from_version=from_version,
                    to_version=target_version,
                    migration_time=time.time() - start_time
                )
            
            self.logger.info(f"Migrating configuration from {from_version} to {target_version}")
            
            migrated_config = config.copy()
            result = MigrationResult(
                success=False,
                from_version=from_version,
                to_version=target_version
            )
            
            # Apply migration rules step by step
            current_version = from_version
            while current_version != target_version:
                next_version = self._get_next_version(current_version, target_version)
                if not next_version:
                    raise ConfigurationMigrationError(
                        f"No migration path from {current_version} to {target_version}"
                    )
                
                migrated_config = self._apply_migration_step(
                    migrated_config, current_version, next_version, result
                )
                current_version = next_version
            
            # Update version in migrated config
            migrated_config["version"] = target_version
            
            # Remove deprecated fields
            if target_version in self.SUPPORTED_VERSIONS:
                deprecated = self.SUPPORTED_VERSIONS[target_version].deprecated_fields
                for field in deprecated:
                    if field in migrated_config:
                        del migrated_config[field]
                        result.warnings.append(f"Removed deprecated field: {field}")
            
            result.success = True
            result.migration_time = time.time() - start_time
            
            self.logger.info(f"Configuration migration completed successfully in {result.migration_time:.2f}s")
            return migrated_config, result
            
        except Exception as e:
            error_msg = f"Configuration migration failed: {e}"
            self.logger.error(error_msg)
            raise ConfigurationMigrationError(error_msg) from e
    
    def migrate_config_file(
        self,
        config_path: Union[str, Path],
        target_version: Optional[str] = None,
        create_backup: bool = True
    ) -> MigrationResult:
        """
        Migrate a configuration file to target version.
        
        Args:
            config_path: Path to configuration file
            target_version: Target version (defaults to current)
            create_backup: Whether to create backup before migration
            
        Returns:
            Migration result
            
        Raises:
            ConfigurationMigrationError: If migration fails
        """
        config_path = Path(config_path)
        target_version = target_version or self.CURRENT_VERSION
        
        try:
            # Load current configuration
            config = self.file_ops.load_json_file(str(config_path))
            
            # Check if migration is needed
            if not self.needs_migration(config):
                return MigrationResult(
                    success=True,
                    from_version=self.detect_config_version(config),
                    to_version=target_version,
                    migration_time=0.0
                )
            
            # Create backup if requested
            backup_path = None
            if create_backup:
                backup_path = self.create_backup(config_path, "pre_migration")
            
            # Perform migration
            migrated_config, result = self.migrate_config(config, target_version)
            result.backup_path = backup_path
            
            # Write migrated configuration back to file
            with open(config_path, 'w') as f:
                json.dump(migrated_config, f, indent=2)
            
            self.logger.info(f"Configuration file migrated: {config_path}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to migrate configuration file {config_path}: {e}"
            self.logger.error(error_msg)
            raise ConfigurationMigrationError(error_msg) from e
    
    def _get_next_version(self, current: str, target: str) -> Optional[str]:
        """Get the next version in migration path."""
        versions = list(self.SUPPORTED_VERSIONS.keys())
        try:
            current_idx = versions.index(current)
            target_idx = versions.index(target)
            
            if current_idx < target_idx:
                return versions[current_idx + 1] if current_idx + 1 < len(versions) else None
            else:
                return versions[current_idx - 1] if current_idx > 0 else None
                
        except ValueError:
            return None
    
    def _apply_migration_step(
        self,
        config: Dict[str, Any],
        from_version: str,
        to_version: str,
        result: MigrationResult
    ) -> Dict[str, Any]:
        """Apply a single migration step between versions."""
        if to_version not in self.SUPPORTED_VERSIONS:
            raise ConfigurationMigrationError(f"Unknown target version: {to_version}")
        
        version_config = self.SUPPORTED_VERSIONS[to_version]
        migrated = config.copy()
        
        # Apply migration rules
        for field, default_value in version_config.migration_rules.items():
            if field not in migrated:
                migrated[field] = default_value
                result.warnings.append(f"Added missing field with default: {field}")
        
        # Validate required fields
        for field in version_config.required_fields:
            if field not in migrated:
                result.errors.append(f"Missing required field after migration: {field}")
        
        return migrated


class ConfigurationValidator:
    """
    Enhanced configuration validator with migration support.
    
    Extends the basic schema validation with version-aware validation,
    default value injection, and comprehensive field validation.
    """
    
    def __init__(
        self,
        file_ops: FileOperations,
        paths: ConfigPaths,
        migrator: Optional[ConfigurationMigrator] = None
    ) -> None:
        """
        Initialize enhanced configuration validator.
        
        Args:
            file_ops: File operations instance
            paths: Configuration paths instance
            migrator: Configuration migrator instance
        """
        self.file_ops = file_ops
        self.paths = paths
        self.migrator = migrator or ConfigurationMigrator(file_ops, paths)
        self.logger = logger
    
    def validate_and_migrate(
        self,
        config: Dict[str, Any],
        config_file: str = "unknown",
        auto_migrate: bool = True
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate configuration and automatically migrate if needed.
        
        Args:
            config: Configuration to validate
            config_file: Configuration file name for error reporting
            auto_migrate: Whether to automatically migrate old versions
            
        Returns:
            Tuple of (validated_config, warnings)
            
        Raises:
            ConfigurationValidationError: If validation fails
            ConfigurationMigrationError: If migration fails
        """
        warnings = []
        validated_config = config.copy()
        
        try:
            # Check if migration is needed
            if self.migrator.needs_migration(config):
                if auto_migrate:
                    self.logger.info(f"Auto-migrating configuration: {config_file}")
                    validated_config, migration_result = self.migrator.migrate_config(config)
                    
                    if not migration_result.success:
                        raise ConfigurationMigrationError(
                            f"Automatic migration failed: {migration_result.errors}"
                        )
                    
                    warnings.extend(migration_result.warnings)
                    warnings.append(
                        f"Configuration auto-migrated from {migration_result.from_version} "
                        f"to {migration_result.to_version}"
                    )
                else:
                    current_version = self.migrator.detect_config_version(config)
                    warnings.append(
                        f"Configuration version {current_version} is outdated. "
                        f"Consider migrating to {self.migrator.CURRENT_VERSION}"
                    )
            
            # Inject default values for missing optional fields
            validated_config = self._inject_default_values(validated_config)
            
            # Perform enhanced field validation
            validation_warnings = self._validate_fields(validated_config)
            warnings.extend(validation_warnings)
            
            return validated_config, warnings
            
        except Exception as e:
            error_msg = f"Configuration validation and migration failed: {e}"
            self.logger.error(error_msg)
            raise ConfigurationValidationError(error_msg, config_file) from e
    
    def _inject_default_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Inject default values for missing optional configuration fields."""
        defaults = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "performance": {
                "enable_caching": True,
                "cache_directory": "./cache",
                "embedding_cache_size": 1000,
                "query_cache_size": 100,
                "cache_ttl_hours": 24
            },
            "api": {
                "timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "security": {
                "max_query_length": 5000,
                "max_document_size_mb": 100,
                "allowed_file_types": [".md", ".txt", ".pdf", ".doc", ".docx"],
                "enable_content_filtering": False
            }
        }
        
        enhanced_config = config.copy()
        
        for section, section_defaults in defaults.items():
            if section not in enhanced_config:
                enhanced_config[section] = section_defaults
            else:
                # Merge with existing section
                for key, default_value in section_defaults.items():
                    if key not in enhanced_config[section]:
                        enhanced_config[section][key] = default_value
        
        return enhanced_config
    
    def _validate_fields(self, config: Dict[str, Any]) -> List[str]:
        """Perform enhanced field validation and return warnings."""
        warnings = []
        
        # Validate embedding model configuration
        if "embedding_model" in config:
            model_config = config["embedding_model"]
            if model_config.get("type") == "local":
                if "max_seq_length" not in model_config:
                    warnings.append("Local embedding model missing max_seq_length, using default")
            
        # Validate chunking strategy
        if "chunking_strategy" in config:
            chunking = config["chunking_strategy"]
            chunk_size = chunking.get("chunk_size", 512)
            chunk_overlap = chunking.get("chunk_overlap", 50)
            
            if chunk_overlap >= chunk_size:
                warnings.append(
                    f"Chunk overlap ({chunk_overlap}) should be less than chunk size ({chunk_size})"
                )
        
        # Validate vector store configuration
        if "vector_store" in config:
            vs_config = config["vector_store"]
            persist_dir = vs_config.get("persist_directory")
            if persist_dir and not Path(persist_dir).parent.exists():
                warnings.append(
                    f"Vector store persist directory parent does not exist: {persist_dir}"
                )
        
        # Validate performance settings
        if "performance" in config:
            perf = config["performance"]
            cache_dir = perf.get("cache_directory")
            if cache_dir and not Path(cache_dir).parent.exists():
                warnings.append(
                    f"Cache directory parent does not exist: {cache_dir}"
                )
        
        return warnings 