"""
File operations for configuration management.

This module provides file loading, path resolution, and environment loading
capabilities for the Research Agent configuration system.

Implements FR-CF-001: Configuration-driven behavior with file handling.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Union
from dotenv import load_dotenv

from ...exceptions.config_exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
)


logger = logging.getLogger(__name__)


class FileOperations:
    """
    File operations for configuration management.
    
    Handles file loading, path resolution, and environment variable loading.
    """
    
    def __init__(self, project_root: Path, env_file: str) -> None:
        """
        Initialize file operations.
        
        Args:
            project_root: Project root directory
            env_file: Environment file name
        """
        self.project_root = project_root
        self.env_file = env_file
        self.logger = logger
    
    def resolve_path(self, path: str) -> Path:
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
    
    def load_environment_variables(self) -> None:
        """
        Load environment variables from .env file if it exists.
        
        Does not raise errors if .env file is missing.
        """
        env_file_path = self.resolve_path(self.env_file)
        if env_file_path.exists():
            try:
                self.logger.debug(f"Loading environment variables from {env_file_path}")
                load_dotenv(env_file_path)
                self.logger.info(f"Successfully loaded environment variables from {env_file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load environment variables from {env_file_path}: {e}")
        else:
            self.logger.debug(f"Environment file not found at {env_file_path}, skipping")
    
    def load_json_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
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
        resolved_path = self.resolve_path(str(file_path))
        
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
    
    def diagnose_file_issues(self, config_file: str, schema_dir: str, default_schema: str) -> Dict[str, Any]:
        """
        Diagnose common file-related configuration issues.
        
        Args:
            config_file: Configuration file path
            schema_dir: Schema directory path
            default_schema: Default schema filename
            
        Returns:
            Dictionary with file diagnostic information
        """
        diagnostics = {
            "config_file_exists": False,
            "config_file_readable": False,
            "config_file_valid_json": False,
            "schema_file_exists": False,
            "env_file_exists": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check main config file
            config_path = self.resolve_path(config_file)
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
            schema_path = self.resolve_path(f"{schema_dir}/{default_schema}")
            diagnostics["schema_file_exists"] = schema_path.exists()
            if not diagnostics["schema_file_exists"]:
                diagnostics["warnings"].append("Schema file not found - validation will be skipped")
            
            # Check environment file
            env_path = self.resolve_path(self.env_file)
            diagnostics["env_file_exists"] = env_path.exists()
            if not diagnostics["env_file_exists"]:
                diagnostics["warnings"].append("Environment file (.env) not found")
            
        except Exception as e:
            diagnostics["errors"].append(f"Error during file diagnostics: {e}")
        
        return diagnostics 