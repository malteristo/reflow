"""
Integration tests for ConfigManager with real project configuration files.

Tests the configuration system using the actual config files and schema
from the project.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from ...utils.config import ConfigManager
from ...exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
)


class TestConfigManagerProjectIntegration:
    """Test ConfigManager with real project configuration."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        # Navigate from test file to project root
        test_dir = Path(__file__).parent
        return test_dir.parent.parent.parent.parent
    
    def test_load_real_project_config(self, project_root):
        """Test loading the actual project configuration."""
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        # Check if config file exists
        config_path = project_root / "researchagent.config.json"
        if not config_path.exists():
            pytest.skip("Project configuration file not found")
        
        # Load configuration without validation first
        config = config_manager.load_config(validate=False)
        
        # Check basic structure expectations
        assert isinstance(config, dict)
        assert len(config) > 0
        
        # Log success
        print(f"Successfully loaded config with {len(config)} top-level keys")
    
    def test_validate_real_project_config(self, project_root):
        """Test validating the actual project configuration against schema."""
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        # Check if both config and schema files exist
        config_path = project_root / "researchagent.config.json"
        schema_path = project_root / "config" / "schema" / "config_schema.json"
        
        if not config_path.exists():
            pytest.skip("Project configuration file not found")
        if not schema_path.exists():
            pytest.skip("Project schema file not found")
        
        # Load and validate configuration
        try:
            config = config_manager.load_config(validate=True)
            assert isinstance(config, dict)
            print("Configuration validation passed")
        except ConfigurationValidationError as e:
            pytest.fail(f"Configuration validation failed: {e}")
    
    def test_config_with_environment_variables(self, project_root):
        """Test configuration loading with environment variables."""
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        config_path = project_root / "researchagent.config.json"
        if not config_path.exists():
            pytest.skip("Project configuration file not found")
        
        # Test with environment variables
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test_anthropic_key",
            "OPENAI_API_KEY": "test_openai_key",
            "RESEARCH_AGENT_LOG_LEVEL": "DEBUG"
        }):
            config = config_manager.load_config(validate=False)
            
            # Check that environment variables were applied
            assert config.get("api_keys", {}).get("anthropic") == "test_anthropic_key"
            assert config.get("api_keys", {}).get("openai") == "test_openai_key"
            assert config.get("logging", {}).get("level") == "DEBUG"
    
    def test_config_diagnostics_on_real_project(self, project_root):
        """Test configuration diagnostics with real project."""
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        diagnostics = config_manager.diagnose_config_issues()
        
        # Check diagnostic results
        assert isinstance(diagnostics, dict)
        assert "config_file_exists" in diagnostics
        assert "schema_file_exists" in diagnostics
        assert "env_file_exists" in diagnostics
        assert "warnings" in diagnostics
        assert "errors" in diagnostics
        
        print(f"Diagnostics: {diagnostics}")
        
        # If config file exists, it should be readable and valid JSON
        if diagnostics["config_file_exists"]:
            assert diagnostics["config_file_readable"]
            assert diagnostics["config_file_valid_json"]
    
    def test_config_summary_on_real_project(self, project_root):
        """Test configuration summary with real project."""
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        config_path = project_root / "researchagent.config.json"
        if not config_path.exists():
            pytest.skip("Project configuration file not found")
        
        # Get summary before loading
        summary_before = config_manager.get_config_summary()
        assert summary_before["loaded"] is False
        
        # Load config and get summary
        config_manager.load_config(validate=False)
        summary_after = config_manager.get_config_summary()
        
        assert summary_after["loaded"] is True
        assert len(summary_after["config_keys"]) > 0
        assert summary_after["total_config_size"] > 0
        
        print(f"Config summary: {summary_after}")


class TestConfigManagerErrorScenarios:
    """Test error handling scenarios in realistic conditions."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        test_dir = Path(__file__).parent
        return test_dir.parent.parent.parent.parent
    
    def test_missing_config_file_error(self, project_root):
        """Test error handling when config file is missing."""
        config_manager = ConfigManager(
            config_file="nonexistent_config.json",
            project_root=project_root,
            load_env=False
        )
        
        with pytest.raises(ConfigurationFileNotFoundError):
            config_manager.load_config()
    
    def test_schema_validation_with_invalid_config(self, project_root):
        """Test schema validation error handling."""
        import tempfile
        import json
        
        schema_path = project_root / "config" / "schema" / "config_schema.json"
        if not schema_path.exists():
            pytest.skip("Project schema file not found")
        
        # Create temporary invalid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            invalid_config = {"completely": "invalid", "structure": True}
            json.dump(invalid_config, f)
            temp_config_path = f.name
        
        try:
            config_manager = ConfigManager(
                config_file=temp_config_path,
                project_root=project_root,
                load_env=False
            )
            
            with pytest.raises(ConfigurationValidationError):
                config_manager.load_config(validate=True)
        finally:
            os.unlink(temp_config_path)
    
    def test_environment_variable_error_handling(self, project_root):
        """Test environment variable error handling."""
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        # Test with invalid environment variable that can't be converted
        with patch.dict(os.environ, {"RESEARCH_AGENT_LOG_LEVEL": "INVALID_LEVEL"}):
            # This should not fail, just log a warning
            try:
                result = config_manager._apply_environment_overrides({"logging": {"level": "INFO"}})
                # Should still work, just not apply the invalid env var
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"Environment variable error handling failed: {e}")


class TestConfigManagerPerformance:
    """Test performance characteristics of configuration loading."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        test_dir = Path(__file__).parent
        return test_dir.parent.parent.parent.parent
    
    def test_config_loading_performance(self, project_root):
        """Test that configuration loading is reasonably fast."""
        import time
        
        config_path = project_root / "researchagent.config.json"
        if not config_path.exists():
            pytest.skip("Project configuration file not found")
        
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        # Measure loading time
        start_time = time.time()
        config = config_manager.load_config(validate=False)
        end_time = time.time()
        
        loading_time = end_time - start_time
        
        # Should load in under 1 second for reasonable config sizes
        assert loading_time < 1.0, f"Config loading took {loading_time:.3f} seconds"
        
        print(f"Configuration loaded in {loading_time:.3f} seconds")
    
    def test_cached_config_access(self, project_root):
        """Test that cached configuration access is fast."""
        import time
        
        config_path = project_root / "researchagent.config.json"
        if not config_path.exists():
            pytest.skip("Project configuration file not found")
        
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        # Load config once
        config_manager.load_config(validate=False)
        
        # Measure cached access time
        start_time = time.time()
        for _ in range(100):
            config_manager.get("version", "default")
        end_time = time.time()
        
        access_time = end_time - start_time
        
        # 100 cached accesses should be very fast
        assert access_time < 0.01, f"100 cached accesses took {access_time:.3f} seconds"
        
        print(f"100 cached config accesses took {access_time:.3f} seconds") 