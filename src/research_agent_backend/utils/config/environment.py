"""
Environment variable handling for configuration management.

This module provides environment variable handling, type conversion, and
validation capabilities for the Research Agent configuration system.

Implements FR-CF-001: Configuration-driven behavior with environment overrides.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from copy import deepcopy

from ...exceptions.config_exceptions import (
    EnvironmentVariableError,
)


logger = logging.getLogger(__name__)


class EnvironmentHandler:
    """
    Environment variable handling for configuration management.
    
    Handles environment variable overrides, type conversion, and validation.
    """
    
    def __init__(self) -> None:
        """Initialize environment handler."""
        self.logger = logger
    
    def get_env_mapping(self) -> Dict[str, str]:
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
    
    def convert_env_value(self, value: str, target_type: str = 'string') -> Any:
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
                return json.loads(value)
            else:  # string
                return value
        except (ValueError, json.JSONDecodeError) as e:
            raise EnvironmentVariableError(
                f"Failed to convert environment variable value '{value}' to {target_type}: {e}",
                value
            ) from e
    
    def apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        env_mapping = self.get_env_mapping()
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
                    converted_value = self.convert_env_value(env_value, target_type)
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
    
    def diagnose_env_issues(self) -> Dict[str, Any]:
        """
        Diagnose environment variable related issues.
        
        Returns:
            Dictionary with environment diagnostic information
        """
        diagnostics = {
            "missing_env_vars": [],
            "warnings": []
        }
        
        # Check for common missing environment variables
        common_env_vars = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'PERPLEXITY_API_KEY']
        for var in common_env_vars:
            if not os.getenv(var):
                diagnostics["missing_env_vars"].append(var)
        
        if diagnostics["missing_env_vars"]:
            diagnostics["warnings"].append("Some API keys are not set in environment variables")
        
        return diagnostics 