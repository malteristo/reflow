"""
Unit tests for ConfigManager class.

Tests cover configuration loading, validation, merging, environment variables,
and error handling scenarios.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from research_agent_backend.utils.config import ConfigManager, ConfigPaths
from research_agent_backend.exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
    ConfigurationSchemaError,
    ConfigurationMergeError,
    EnvironmentVariableError,
)


class TestConfigPaths:
    """Test ConfigPaths dataclass."""
    
    def test_default_paths(self):
        """Test default configuration paths."""
        paths = ConfigPaths()
        assert paths.DEFAULT_CONFIG_FILE == "researchagent.config.json"
        assert paths.DEFAULT_CONFIG_DIR == "./config/defaults"
        assert paths.SCHEMA_DIR == "./config/schema"
        assert paths.ENV_FILE == ".env"
        assert paths.DEFAULT_CONFIG_SCHEMA == "config_schema.json"


class TestConfigManagerInitialization:
    """Test ConfigManager initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        config_manager = ConfigManager(load_env=False)
        assert config_manager.config_file == "researchagent.config.json"
        assert isinstance(config_manager.project_root, Path)
        assert not config_manager.is_loaded
    
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ConfigManager(
                config_file="custom.json",
                project_root=temp_dir,
                load_env=False
            )
            assert config_manager.config_file == "custom.json"
            assert config_manager.project_root == Path(temp_dir).resolve()
    
    @patch('research_agent_backend.utils.config.file_operations.load_dotenv')
    def test_init_with_env_loading(self, mock_load_dotenv):
        """Test initialization with environment variable loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("TEST_VAR=test_value")
            
            ConfigManager(project_root=temp_dir, load_env=True)
            mock_load_dotenv.assert_called_once()


class TestConfigManagerBasicOperations:
    """Test basic ConfigManager operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_data = {
            "version": "1.0.0",
            "embedding_model": {
                "name": "test-model",
                "type": "local"
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        # Create test config file
        self.config_file = Path(self.temp_dir) / "test_config.json"
        with open(self.config_file, 'w') as f:
            json.dump(self.config_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config_basic(self):
        """Test basic configuration loading."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        loaded_config = config_manager.load_config(validate=False)
        assert loaded_config == self.config_data
        assert config_manager.is_loaded
    
    def test_config_property_lazy_loading(self):
        """Test lazy loading via config property."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        assert not config_manager.is_loaded
        config = config_manager.config
        assert config_manager.is_loaded
        
        # After auto-migration from 1.0.0 to 1.2.0, expect the expanded config
        # This includes additional fields added by the migration process
        expected_config = {
            "version": "1.2.0",  # Auto-migrated from 1.0.0
            "embedding_model": {
                "name": "test-model",
                "type": "local"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Added by migration
            },
            # Additional sections added by auto-migration
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
        assert config == expected_config
    
    def test_get_method_dot_notation(self):
        """Test getting configuration values with dot notation."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        assert config_manager.get("version") == "1.2.0"
        assert config_manager.get("embedding_model.name") == "test-model"
        assert config_manager.get("embedding_model.type") == "local"
        assert config_manager.get("nonexistent", "default") == "default"
    
    def test_set_method_dot_notation(self):
        """Test setting configuration values with dot notation."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        config_manager.set("new_value", "test")
        config_manager.set("nested.value", "nested_test")
        
        assert config_manager.get("new_value") == "test"
        assert config_manager.get("nested.value") == "nested_test"
    
    def test_has_method(self):
        """Test checking if configuration keys exist."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        assert config_manager.has("version")
        assert config_manager.has("embedding_model.name")
        assert not config_manager.has("nonexistent")
    
    def test_reset_method(self):
        """Test resetting configuration state."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        # Load config
        config_manager.load_config(validate=False)
        assert config_manager.is_loaded
        
        # Reset
        config_manager.reset()
        assert not config_manager.is_loaded
    
    def test_reload_config(self):
        """Test force reloading configuration."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        # Load initial config
        config1 = config_manager.load_config(validate=False)
        
        # Modify config file
        modified_data = {**self.config_data, "version": "2.0.0"}
        with open(self.config_file, 'w') as f:
            json.dump(modified_data, f)
        
        # Reload
        config2 = config_manager.reload_config()
        assert config2["version"] == "2.0.0"


class TestConfigManagerMerging:
    """Test configuration merging functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create base config
        self.base_config = {
            "version": "1.0.0",
            "embedding_model": {
                "name": "base-model",
                "type": "local",
                "batch_size": 32
            }
        }
        
        # Create override config
        self.override_config = {
            "embedding_model": {
                "name": "override-model",
                "custom_field": "test"
            },
            "new_section": {
                "value": "new"
            }
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deep_merge_dicts(self):
        """Test deep merging of dictionaries."""
        config_manager = ConfigManager(project_root=self.temp_dir, load_env=False)
        
        merged = config_manager._deep_merge_dicts(self.base_config, self.override_config)
        
        # Check that override took precedence
        assert merged["embedding_model"]["name"] == "override-model"
        # Check that base values were preserved
        assert merged["embedding_model"]["type"] == "local"
        assert merged["embedding_model"]["batch_size"] == 32
        # Check that new values were added
        assert merged["embedding_model"]["custom_field"] == "test"
        assert merged["new_section"]["value"] == "new"
    
    def test_merge_configs_public_method(self):
        """Test public merge_configs method."""
        config_manager = ConfigManager(project_root=self.temp_dir, load_env=False)
        
        merged = config_manager.merge_configs(self.base_config, self.override_config)
        assert merged["embedding_model"]["name"] == "override-model"
        assert merged["embedding_model"]["type"] == "local"
    
    def test_config_inheritance_basic(self):
        """Test configuration inheritance via extends field."""
        # Create parent config
        parent_config = {"parent_value": "test", "shared": "parent"}
        parent_file = Path(self.temp_dir) / "parent.json"
        with open(parent_file, 'w') as f:
            json.dump(parent_config, f)
        
        # Create child config
        child_config = {"extends": "./parent.json", "child_value": "test", "shared": "child"}
        child_file = Path(self.temp_dir) / "child.json"
        with open(child_file, 'w') as f:
            json.dump(child_config, f)
        
        config_manager = ConfigManager(
            config_file=str(child_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        merged = config_manager._resolve_config_inheritance(child_config)
        
        assert merged["parent_value"] == "test"
        assert merged["child_value"] == "test"
        assert merged["shared"] == "child"  # Child overrides parent
        assert "extends" not in merged  # extends field should be removed
    
    def test_circular_inheritance_detection(self):
        """Test detection of circular inheritance."""
        # Create circular configs
        config_a = {"extends": "./config_b.json", "value": "a"}
        config_b = {"extends": "./config_a.json", "value": "b"}
        
        file_a = Path(self.temp_dir) / "config_a.json"
        file_b = Path(self.temp_dir) / "config_b.json"
        
        with open(file_a, 'w') as f:
            json.dump(config_a, f)
        with open(file_b, 'w') as f:
            json.dump(config_b, f)
        
        config_manager = ConfigManager(
            config_file=str(file_a),
            project_root=self.temp_dir,
            load_env=False
        )
        
        with pytest.raises(ConfigurationMergeError, match="Circular reference"):
            config_manager._resolve_config_inheritance(config_a)


class TestConfigManagerEnvironmentVariables:
    """Test environment variable integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.base_config = {
            "api_keys": {},
            "logging": {"level": "INFO"}
        }
        
        self.config_file = Path(self.temp_dir) / "config.json"
        with open(self.config_file, 'w') as f:
            json.dump(self.base_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key", "RESEARCH_AGENT_LOG_LEVEL": "DEBUG"})
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        config = config_manager.load_config(validate=False)
        
        assert config["api_keys"]["anthropic"] == "test_key"
        assert config["logging"]["level"] == "DEBUG"
    
    def test_convert_env_value_types(self):
        """Test environment variable type conversion."""
        config_manager = ConfigManager(project_root=self.temp_dir, load_env=False)
        
        # Test boolean conversion
        assert config_manager._convert_env_value("true", "boolean") is True
        assert config_manager._convert_env_value("false", "boolean") is False
        assert config_manager._convert_env_value("1", "boolean") is True
        assert config_manager._convert_env_value("0", "boolean") is False
        
        # Test integer conversion
        assert config_manager._convert_env_value("42", "integer") == 42
        
        # Test float conversion
        assert config_manager._convert_env_value("3.14", "float") == 3.14
        
        # Test JSON conversion
        assert config_manager._convert_env_value('{"key": "value"}', "json") == {"key": "value"}
        
        # Test string (default)
        assert config_manager._convert_env_value("test", "string") == "test"
    
    def test_convert_env_value_errors(self):
        """Test environment variable conversion errors."""
        config_manager = ConfigManager(project_root=self.temp_dir, load_env=False)
        
        with pytest.raises(EnvironmentVariableError):
            config_manager._convert_env_value("not_a_number", "integer")
        
        with pytest.raises(EnvironmentVariableError):
            config_manager._convert_env_value("invalid_json", "json")
    
    def test_get_env_var(self):
        """Test environment variable getter."""
        config_manager = ConfigManager(project_root=self.temp_dir, load_env=False)
        
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            assert config_manager.get_env_var("TEST_VAR") == "test_value"
            assert config_manager.get_env_var("NONEXISTENT", "default") == "default"
        
        with pytest.raises(EnvironmentVariableError):
            config_manager.get_env_var("REQUIRED_VAR", required=True)
    
    def test_validate_required_env_vars(self):
        """Test validation of required environment variables."""
        config_manager = ConfigManager(project_root=self.temp_dir, load_env=False)
        
        with patch.dict(os.environ, {"VAR1": "value1", "VAR2": "value2"}):
            result = config_manager.validate_required_env_vars(["VAR1", "VAR2"])
            assert result == {"VAR1": "value1", "VAR2": "value2"}
        
        with pytest.raises(EnvironmentVariableError, match="Missing required"):
            config_manager.validate_required_env_vars(["MISSING_VAR"])


class TestConfigManagerValidation:
    """Test configuration validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create valid config
        self.valid_config = {
            "version": "1.0.0",
            "embedding_model": {
                "name": "test-model",
                "type": "local"
            }
        }
        
        # Create simple schema
        self.schema = {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "embedding_model": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"enum": ["local", "api"]}
                    },
                    "required": ["name", "type"]
                }
            },
            "required": ["version"]
        }
        
        # Create schema file
        schema_dir = Path(self.temp_dir) / "config" / "schema"
        schema_dir.mkdir(parents=True)
        schema_file = schema_dir / "config_schema.json"
        with open(schema_file, 'w') as f:
            json.dump(self.schema, f)
        
        # Create config file
        self.config_file = Path(self.temp_dir) / "config.json"
        with open(self.config_file, 'w') as f:
            json.dump(self.valid_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_schema_loading(self):
        """Test loading JSON schema."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        loaded_schema = config_manager._load_schema()
        assert loaded_schema == self.schema
    
    def test_valid_config_validation(self):
        """Test validation of valid configuration."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        # Should not raise exception
        assert config_manager.validate_config(self.valid_config, None) is True
    
    def test_invalid_config_validation(self):
        """Test validation of invalid configuration."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        invalid_config = {
            "embedding_model": {
                "name": "test",
                "type": "invalid_type"  # Should be "local" or "api"
            }
        }
        
        with pytest.raises(ConfigurationValidationError):
            config_manager._validate_config_against_schema(invalid_config, self.schema)
    
    def test_validation_during_load(self):
        """Test automatic validation during config loading."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        # Should load successfully with validation
        config = config_manager.load_config(validate=True)
        assert config == self.valid_config
        
        # Test with invalid config
        invalid_config = {"invalid": "structure"}
        with open(self.config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        config_manager.reset()
        with pytest.raises(ConfigurationValidationError):
            config_manager.load_config(validate=True)


class TestConfigManagerErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        config_manager = ConfigManager(
            config_file="nonexistent.json",
            project_root=self.temp_dir,
            load_env=False
        )
        
        with pytest.raises(ConfigurationFileNotFoundError):
            config_manager.load_config()
    
    def test_invalid_json_file(self):
        """Test handling of invalid JSON file."""
        invalid_file = Path(self.temp_dir) / "invalid.json"
        invalid_file.write_text("{ invalid json")
        
        config_manager = ConfigManager(
            config_file=str(invalid_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        with pytest.raises(ConfigurationError, match="Invalid JSON"):
            config_manager.load_config()
    
    def test_permission_error(self):
        """Test handling of permission errors."""
        config_file = Path(self.temp_dir) / "config.json"
        config_file.write_text('{"test": "value"}')
        
        config_manager = ConfigManager(
            config_file=str(config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        # Mock permission error
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(ConfigurationError, match="Permission denied"):
                config_manager.load_config()
    
    def test_diagnose_config_issues(self):
        """Test configuration diagnostics."""
        config_manager = ConfigManager(
            config_file="nonexistent.json",
            project_root=self.temp_dir,
            load_env=False
        )
        
        diagnostics = config_manager.diagnose_config_issues()
        
        assert diagnostics["config_file_exists"] is False
        assert "Configuration file does not exist" in diagnostics["errors"]
        assert diagnostics["schema_file_exists"] is False
    
    def test_get_config_summary(self):
        """Test configuration summary."""
        config_data = {"test": "value"}
        config_file = Path(self.temp_dir) / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config_manager = ConfigManager(
            config_file=str(config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        # Before loading
        summary = config_manager.get_config_summary()
        assert summary["loaded"] is False
        
        # After loading
        config_manager.load_config(validate=False)
        summary = config_manager.get_config_summary()
        assert summary["loaded"] is True
        assert "test" in summary["config_keys"]


class TestConfigManagerIntegration:
    """Integration tests combining multiple features."""
    
    def setup_method(self):
        """Set up complex test scenario."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create default config
        default_dir = Path(self.temp_dir) / "config" / "defaults"
        default_dir.mkdir(parents=True)
        default_config = {
            "version": "1.0.0",
            "embedding_model": {
                "name": "default-model",
                "type": "local",
                "batch_size": 32
            },
            "logging": {
                "level": "INFO",
                "format": "default"
            }
        }
        with open(default_dir / "default_config.json", 'w') as f:
            json.dump(default_config, f)
        
        # Create main config that extends defaults
        main_config = {
            "extends": "./config/defaults/default_config.json",
            "project": {
                "name": "Test Project"
            },
            "embedding_model": {
                "name": "custom-model"  # Override default
            }
        }
        self.config_file = Path(self.temp_dir) / "config.json"
        with open(self.config_file, 'w') as f:
            json.dump(main_config, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env_key", "RESEARCH_AGENT_LOG_LEVEL": "DEBUG"})
    def test_full_config_pipeline(self):
        """Test complete configuration loading pipeline."""
        config_manager = ConfigManager(
            config_file=str(self.config_file),
            project_root=self.temp_dir,
            load_env=False
        )
        
        config = config_manager.load_config(validate=False)
        
        # Check inheritance worked
        assert config["version"] == "1.0.0"  # From default
        assert config["embedding_model"]["batch_size"] == 32  # From default
        assert config["embedding_model"]["name"] == "custom-model"  # Overridden
        assert config["project"]["name"] == "Test Project"  # From main
        
        # Check environment overrides
        assert config["api_keys"]["anthropic"] == "env_key"
        assert config["logging"]["level"] == "DEBUG"  # Overridden by env var
        
        # Check that extends field was removed
        assert "extends" not in config 