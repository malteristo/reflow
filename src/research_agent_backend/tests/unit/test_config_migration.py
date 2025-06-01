"""
Unit tests for configuration migration system.

Tests cover version detection, migration rules, backup/restore functionality,
and enhanced validation with migration support.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from research_agent_backend.utils.config.migration import (
    ConfigurationMigrator,
    ConfigurationValidator,
    MigrationResult,
    ConfigVersion,
)
from research_agent_backend.utils.config import ConfigManager
from research_agent_backend.utils.config.file_operations import FileOperations
from research_agent_backend.utils.config.paths import ConfigPaths
from research_agent_backend.exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationMigrationError,
    ConfigurationValidationError,
)


class TestConfigurationMigrator:
    """Test ConfigurationMigrator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.paths = ConfigPaths()
        self.file_ops = FileOperations(self.temp_dir, self.paths.ENV_FILE)
        self.migrator = ConfigurationMigrator(self.file_ops, self.paths)
        
        # Test configurations for different versions
        self.v1_0_config = {
            "version": "1.0.0",
            "embedding_model": {
                "name": "test-model",
                "type": "local"
            },
            "vector_store": {
                "type": "chromadb",
                "persist_directory": "./data"
            }
        }
        
        self.v1_1_config = {
            "version": "1.1.0",
            "embedding_model": {
                "name": "test-model",
                "type": "local"
            },
            "vector_store": {
                "type": "chromadb",
                "persist_directory": "./data"
            },
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
        
        self.v1_2_config = {
            "version": "1.2.0",
            "embedding_model": {
                "name": "test-model",
                "type": "local"
            },
            "vector_store": {
                "type": "chromadb",
                "persist_directory": "./data"
            },
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
            },
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
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detect_config_version_explicit(self):
        """Test version detection with explicit version field."""
        assert self.migrator.detect_config_version(self.v1_0_config) == "1.0.0"
        assert self.migrator.detect_config_version(self.v1_1_config) == "1.1.0"
        assert self.migrator.detect_config_version(self.v1_2_config) == "1.2.0"
    
    def test_detect_config_version_inferred(self):
        """Test version detection by inferring from structure."""
        # Config without version but with 1.2.0 structure
        config_no_version = {**self.v1_2_config}
        del config_no_version["version"]
        assert self.migrator.detect_config_version(config_no_version) == "1.2.0"
        
        # Config with only 1.1.0 structure
        config_v11_structure = {
            "embedding_model": {"name": "test", "type": "local"},
            "vector_store": {"type": "chromadb"},
            "rag_pipeline": {"enable_reranking": True}
        }
        assert self.migrator.detect_config_version(config_v11_structure) == "1.1.0"
        
        # Config with only 1.0.0 structure
        config_v10_structure = {
            "embedding_model": {"name": "test", "type": "local"},
            "vector_store": {"type": "chromadb"}
        }
        assert self.migrator.detect_config_version(config_v10_structure) == "1.0.0"
    
    def test_detect_config_version_unknown(self):
        """Test version detection with unknown version."""
        unknown_config = {"version": "2.0.0", "future_field": "value"}
        # Should treat unknown versions as current for forward compatibility
        assert self.migrator.detect_config_version(unknown_config) == "1.2.0"
    
    def test_needs_migration(self):
        """Test migration requirement detection."""
        assert self.migrator.needs_migration(self.v1_0_config) == True
        assert self.migrator.needs_migration(self.v1_1_config) == True
        assert self.migrator.needs_migration(self.v1_2_config) == False
    
    def test_migrate_config_v1_0_to_v1_2(self):
        """Test migration from version 1.0.0 to 1.2.0."""
        migrated_config, result = self.migrator.migrate_config(self.v1_0_config)
        
        assert result.success == True
        assert result.from_version == "1.0.0"
        assert result.to_version == "1.2.0"
        assert len(result.warnings) > 0  # Should have warnings about added fields
        
        # Check that required fields were added
        assert "chunking_strategy" in migrated_config
        assert "rag_pipeline" in migrated_config
        assert "collections" in migrated_config
        assert "performance" in migrated_config
        assert migrated_config["version"] == "1.2.0"
    
    def test_migrate_config_v1_1_to_v1_2(self):
        """Test migration from version 1.1.0 to 1.2.0."""
        migrated_config, result = self.migrator.migrate_config(self.v1_1_config)
        
        assert result.success == True
        assert result.from_version == "1.1.0"
        assert result.to_version == "1.2.0"
        
        # Check that missing fields were added
        assert "collections" in migrated_config
        assert "performance" in migrated_config
        assert migrated_config["version"] == "1.2.0"
    
    def test_migrate_config_no_migration_needed(self):
        """Test migration when no migration is needed."""
        migrated_config, result = self.migrator.migrate_config(self.v1_2_config)
        
        assert result.success == True
        assert result.from_version == "1.2.0"
        assert result.to_version == "1.2.0"
        assert result.migration_time >= 0
        assert migrated_config == self.v1_2_config
    
    def test_create_backup(self):
        """Test configuration backup creation."""
        # Create test config file
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.v1_0_config, f)
        
        backup_path = self.migrator.create_backup(config_file, "test")
        
        assert Path(backup_path).exists()
        assert "test" in backup_path
        assert backup_path.endswith(".backup.json")
        
        # Verify backup content
        with open(backup_path) as f:
            backup_content = json.load(f)
        assert backup_content == self.v1_0_config
    
    def test_restore_backup(self):
        """Test configuration backup restoration."""
        # Create test files
        config_file = Path(self.temp_dir) / "test_config.json"
        backup_file = Path(self.temp_dir) / "backup.json"
        
        with open(config_file, 'w') as f:
            json.dump(self.v1_2_config, f)
        
        with open(backup_file, 'w') as f:
            json.dump(self.v1_0_config, f)
        
        # Restore from backup
        self.migrator.restore_backup(backup_file, config_file)
        
        # Verify restoration
        with open(config_file) as f:
            restored_content = json.load(f)
        assert restored_content == self.v1_0_config
    
    def test_migrate_config_file(self):
        """Test full configuration file migration."""
        # Create test config file
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.v1_0_config, f)
        
        result = self.migrator.migrate_config_file(config_file, create_backup=True)
        
        assert result.success == True
        assert result.backup_path is not None
        assert Path(result.backup_path).exists()
        
        # Verify migrated file content
        with open(config_file) as f:
            migrated_content = json.load(f)
        assert migrated_content["version"] == "1.2.0"
        assert "collections" in migrated_content


class TestConfigurationValidator:
    """Test ConfigurationValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.paths = ConfigPaths()
        self.file_ops = FileOperations(self.temp_dir, self.paths.ENV_FILE)
        self.migrator = ConfigurationMigrator(self.file_ops, self.paths)
        self.validator = ConfigurationValidator(self.file_ops, self.paths, self.migrator)
        
        self.old_config = {
            "version": "1.0.0",
            "embedding_model": {
                "name": "test-model",
                "type": "local"
            },
            "vector_store": {
                "type": "chromadb",
                "persist_directory": "./data"
            }
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_and_migrate_with_auto_migration(self):
        """Test validation and migration with auto-migration enabled."""
        validated_config, warnings = self.validator.validate_and_migrate(
            self.old_config,
            auto_migrate=True
        )
        
        assert validated_config["version"] == "1.2.0"
        assert len(warnings) > 0
        assert any("auto-migrated" in warning.lower() for warning in warnings)
        
        # Check that default values were injected
        assert "logging" in validated_config
        assert "performance" in validated_config
        assert "api" in validated_config
        assert "security" in validated_config
    
    def test_validate_and_migrate_without_auto_migration(self):
        """Test validation and migration with auto-migration disabled."""
        validated_config, warnings = self.validator.validate_and_migrate(
            self.old_config,
            auto_migrate=False
        )
        
        assert validated_config["version"] == "1.0.0"  # Should not migrate
        assert len(warnings) > 0
        assert any("outdated" in warning.lower() for warning in warnings)
    
    def test_inject_default_values(self):
        """Test default value injection."""
        minimal_config = {
            "version": "1.2.0",
            "embedding_model": {"name": "test", "type": "local"},
            "vector_store": {"type": "chromadb"}
        }
        
        enhanced_config = self.validator._inject_default_values(minimal_config)
        
        assert "logging" in enhanced_config
        assert enhanced_config["logging"]["level"] == "INFO"
        assert "performance" in enhanced_config
        assert enhanced_config["performance"]["enable_caching"] == True
        assert "api" in enhanced_config
        assert enhanced_config["api"]["timeout"] == 30
        assert "security" in enhanced_config
    
    def test_validate_fields_warnings(self):
        """Test field validation warnings."""
        config_with_issues = {
            "version": "1.2.0",
            "embedding_model": {
                "name": "test",
                "type": "local"
                # Missing max_seq_length
            },
            "chunking_strategy": {
                "type": "hybrid",
                "chunk_size": 100,
                "chunk_overlap": 150  # Overlap >= chunk_size (should warn)
            },
            "vector_store": {
                "type": "chromadb",
                "persist_directory": "/nonexistent/path/data"  # Parent doesn't exist
            },
            "performance": {
                "cache_directory": "/nonexistent/cache"  # Parent doesn't exist
            }
        }
        
        warnings = self.validator._validate_fields(config_with_issues)
        
        assert len(warnings) >= 3
        assert any("max_seq_length" in warning for warning in warnings)
        assert any("overlap" in warning and "chunk size" in warning for warning in warnings)


class TestConfigManagerMigrationIntegration:
    """Test ConfigManager integration with migration system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create old version config file
        self.old_config = {
            "version": "1.0.0",
            "embedding_model": {
                "name": "test-model",
                "type": "local"
            },
            "vector_store": {
                "type": "chromadb",
                "persist_directory": "./data"
            }
        }
        
        self.config_file = Path(self.temp_dir) / "test_config.json"
        with open(self.config_file, 'w') as f:
            json.dump(self.old_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_auto_migration(self):
        """Test ConfigManager automatic migration on load."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False,
            auto_migrate=True
        )
        
        # Load config should trigger migration
        config = config_manager.load_config()
        
        assert config["version"] == "1.2.0"
        assert len(config_manager.migration_warnings) > 0
        assert config_manager.get_config_version() == "1.2.0"
        assert not config_manager.needs_migration()
    
    def test_config_manager_no_auto_migration(self):
        """Test ConfigManager without automatic migration."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False,
            auto_migrate=False
        )
        
        config = config_manager.load_config()
        
        assert config["version"] == "1.0.0"  # Should not migrate
        assert config_manager.get_config_version() == "1.0.0"
        assert config_manager.needs_migration() == True
    
    def test_config_manager_manual_migration(self):
        """Test ConfigManager manual migration."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False,
            auto_migrate=False
        )
        
        config_manager.load_config()
        assert config_manager.needs_migration() == True
        
        # Perform manual migration
        result = config_manager.migrate_config_file()
        
        assert result.success == True
        assert result.backup_path is not None
        assert config_manager.get_config_version() == "1.2.0"
        assert not config_manager.needs_migration()
    
    def test_config_manager_backup_restore(self):
        """Test ConfigManager backup and restore functionality."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False,
            auto_migrate=False  # Disable auto-migration for this test
        )
        
        # Create backup
        backup_path = config_manager.create_config_backup("manual")
        assert Path(backup_path).exists()
        
        # Load and manually migrate (modifies the file)
        config_manager.load_config()
        result = config_manager.migrate_config_file()
        assert result.success == True
        assert config_manager.get_config_version() == "1.2.0"
        
        # Restore from backup
        config_manager.restore_config_backup(backup_path)
        assert config_manager.get_config_version() == "1.0.0"
    
    def test_config_manager_save_config(self):
        """Test ConfigManager save functionality."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        config_manager.load_config()
        config_manager.set("test_key", "test_value")
        
        # Save the modified config
        config_manager.save_config()
        
        # Reload to verify save
        config_manager.reload_config()
        assert config_manager.get("test_key") == "test_value"


class TestMigrationErrorHandling:
    """Test error handling in migration system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.paths = ConfigPaths()
        self.file_ops = FileOperations(self.temp_dir, self.paths.ENV_FILE)
        self.migrator = ConfigurationMigrator(self.file_ops, self.paths)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backup_nonexistent_file(self):
        """Test backup creation for nonexistent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.json"
        
        with pytest.raises(ConfigurationError):
            self.migrator.create_backup(nonexistent_file)
    
    def test_restore_nonexistent_backup(self):
        """Test restore from nonexistent backup."""
        nonexistent_backup = Path(self.temp_dir) / "nonexistent_backup.json"
        target_file = Path(self.temp_dir) / "target.json"
        
        with pytest.raises(ConfigurationError):
            self.migrator.restore_backup(nonexistent_backup, target_file)
    
    def test_migrate_invalid_config(self):
        """Test migration with invalid configuration."""
        # Create a config that will fail migration due to missing required fields after migration
        invalid_config = {
            "version": "1.0.0",
            # Missing embedding_model and vector_store which are required
        }
        
        # This should fail because required fields are missing
        migrated_config, result = self.migrator.migrate_config(invalid_config)
        
        # Migration should succeed but have errors about missing required fields
        assert result.success == True
        assert len(result.errors) > 0
        assert any("Missing required field" in error for error in result.errors) 