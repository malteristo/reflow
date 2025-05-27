#!/usr/bin/env python3
"""
Simple test runner for the configuration system.

This script provides a quick way to test the configuration system
without running the full test suite.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from research_agent_backend.utils.config import ConfigManager
from research_agent_backend.exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
)


def test_basic_config_loading():
    """Test basic configuration loading."""
    print("Testing basic configuration loading...")
    
    try:
        config_manager = ConfigManager(
            project_root=project_root,
            load_env=False
        )
        
        # Test diagnostics first
        print("\nRunning configuration diagnostics...")
        diagnostics = config_manager.diagnose_config_issues()
        
        print(f"Config file exists: {diagnostics['config_file_exists']}")
        print(f"Schema file exists: {diagnostics['schema_file_exists']}")
        print(f"Env file exists: {diagnostics['env_file_exists']}")
        
        if diagnostics['errors']:
            print(f"Errors: {diagnostics['errors']}")
        if diagnostics['warnings']:
            print(f"Warnings: {diagnostics['warnings']}")
        
        # Try to load configuration
        if diagnostics['config_file_exists']:
            print("\nLoading configuration...")
            config = config_manager.load_config(validate=False)
            print(f"Successfully loaded configuration with {len(config)} top-level keys")
            
            # Test some basic operations
            print(f"Version: {config_manager.get('version', 'Not set')}")
            
            # Test environment variable integration
            print("\nTesting environment variable integration...")
            os.environ["ANTHROPIC_API_KEY"] = "test_anthropic_key"
            
            config_with_env = config_manager._apply_environment_overrides(config)
            api_key = config_with_env.get("api_keys", {}).get("anthropic")
            if api_key:
                print(f"Environment variable override worked: {api_key}")
            else:
                print("Environment variable override not applied")
            
            # Clean up
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            
            # Test configuration summary
            print("\nConfiguration summary:")
            summary = config_manager.get_config_summary()
            print(f"Loaded: {summary['loaded']}")
            print(f"Config keys: {len(summary['config_keys'])}")
            print(f"Config size: {summary['total_config_size']} bytes")
            
        else:
            print("Configuration file not found, skipping load test")
        
        print("\n✅ Basic configuration loading test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic configuration loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variable_conversion():
    """Test environment variable type conversion."""
    print("\nTesting environment variable type conversion...")
    
    try:
        config_manager = ConfigManager(project_root=project_root, load_env=False)
        
        # Test various type conversions
        test_cases = [
            ("true", "boolean", True),
            ("false", "boolean", False),
            ("1", "boolean", True),
            ("0", "boolean", False),
            ("42", "integer", 42),
            ("3.14", "float", 3.14),
            ('{"key": "value"}', "json", {"key": "value"}),
            ("test_string", "string", "test_string"),
        ]
        
        for value, type_name, expected in test_cases:
            result = config_manager._convert_env_value(value, type_name)
            assert result == expected, f"Expected {expected}, got {result} for {value} -> {type_name}"
            print(f"✓ {value} -> {type_name} = {result}")
        
        print("✅ Environment variable conversion test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Environment variable conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_merging():
    """Test configuration merging functionality."""
    print("\nTesting configuration merging...")
    
    try:
        config_manager = ConfigManager(project_root=project_root, load_env=False)
        
        base_config = {
            "version": "1.0.0",
            "features": {
                "feature1": {"enabled": True, "config": "base"},
                "feature2": {"enabled": False}
            }
        }
        
        override_config = {
            "features": {
                "feature1": {"config": "override"},
                "feature3": {"enabled": True}
            },
            "new_section": {
                "value": "new"
            }
        }
        
        merged = config_manager._deep_merge_dicts(base_config, override_config)
        
        # Verify merge results
        assert merged["version"] == "1.0.0"  # From base
        assert merged["features"]["feature1"]["enabled"] is True  # From base (preserved)
        assert merged["features"]["feature1"]["config"] == "override"  # From override
        assert merged["features"]["feature2"]["enabled"] is False  # From base (preserved)
        assert merged["features"]["feature3"]["enabled"] is True  # From override (new)
        assert merged["new_section"]["value"] == "new"  # From override (new)
        
        print("✓ Deep merge preserves base values")
        print("✓ Deep merge applies overrides")
        print("✓ Deep merge adds new values")
        
        print("✅ Configuration merging test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration merging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all configuration tests."""
    print("=" * 60)
    print("Research Agent Configuration System Test Runner")
    print("=" * 60)
    
    tests = [
        test_basic_config_loading,
        test_environment_variable_conversion,
        test_configuration_merging,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All configuration tests passed!")
        return 0
    else:
        print("💥 Some configuration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 