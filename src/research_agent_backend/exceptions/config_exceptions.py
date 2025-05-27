"""
Configuration-related exceptions for Research Agent.

Custom exception classes for handling configuration loading, validation,
and environment variable errors with user-friendly messages.

Implements FR-CF-001: Configuration error handling with detailed feedback.
"""

from typing import Any, Dict, List, Optional, Union


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ) -> None:
        """
        Initialize configuration error.
        
        Args:
            message: Error description
            config_file: Configuration file path that caused the error
            suggestions: List of suggested fixes
        """
        super().__init__(message)
        self.config_file = config_file
        self.suggestions = suggestions or []
    
    def __str__(self) -> str:
        """Return formatted error message with suggestions."""
        msg = super().__str__()
        
        if self.config_file:
            msg = f"{msg}\nConfig file: {self.config_file}"
        
        if self.suggestions:
            msg += "\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"\n  {i}. {suggestion}"
        
        return msg


class ConfigurationFileNotFoundError(ConfigurationError):
    """Exception raised when a configuration file is not found."""
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        searched_paths: Optional[List[str]] = None
    ) -> None:
        """
        Initialize file not found error.
        
        Args:
            message: Error description
            config_file: Configuration file that was not found
            searched_paths: Paths that were searched for the file
        """
        suggestions = [
            "Check if the configuration file exists at the specified path",
            "Verify file permissions allow reading",
            "Use absolute path if relative path is not working",
        ]
        
        if searched_paths:
            suggestions.append(f"Searched in: {', '.join(searched_paths)}")
        
        super().__init__(message, config_file, suggestions)
        self.searched_paths = searched_paths or []


class ConfigurationValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""
    
    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        invalid_fields: Optional[List[str]] = None
    ) -> None:
        """
        Initialize validation error.
        
        Args:
            message: Error description
            config_file: Configuration file with validation errors
            validation_errors: List of specific validation error messages
            invalid_fields: List of field names that failed validation
        """
        suggestions = [
            "Check the configuration schema documentation",
            "Validate JSON syntax using a JSON validator",
            "Compare with the example configuration file",
        ]
        
        if invalid_fields:
            suggestions.append(f"Fix these fields: {', '.join(invalid_fields)}")
        
        super().__init__(message, config_file, suggestions)
        self.validation_errors = validation_errors or []
        self.invalid_fields = invalid_fields or []
    
    def __str__(self) -> str:
        """Return formatted validation error with details."""
        msg = super().__str__()
        
        if self.validation_errors:
            msg += "\n\nValidation errors:"
            for i, error in enumerate(self.validation_errors, 1):
                msg += f"\n  {i}. {error}"
        
        return msg


class EnvironmentVariableError(ConfigurationError):
    """Exception raised when environment variable handling fails."""
    
    def __init__(
        self,
        message: str,
        variable_name: Optional[str] = None,
        required_variables: Optional[List[str]] = None
    ) -> None:
        """
        Initialize environment variable error.
        
        Args:
            message: Error description
            variable_name: Name of the problematic environment variable
            required_variables: List of required environment variables
        """
        suggestions = [
            "Check if .env file exists in the project root",
            "Verify environment variable names are correct",
            "Ensure no extra spaces in variable definitions",
        ]
        
        if variable_name:
            suggestions.append(f"Set {variable_name} in .env file or environment")
        
        if required_variables:
            suggestions.append(f"Required variables: {', '.join(required_variables)}")
        
        super().__init__(message, None, suggestions)
        self.variable_name = variable_name
        self.required_variables = required_variables or []


class ConfigurationSchemaError(ConfigurationError):
    """Exception raised when configuration schema is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        schema_file: Optional[str] = None,
        schema_errors: Optional[List[str]] = None
    ) -> None:
        """
        Initialize schema error.
        
        Args:
            message: Error description
            schema_file: Schema file that caused the error
            schema_errors: List of schema-specific errors
        """
        suggestions = [
            "Ensure the schema file exists and is valid JSON",
            "Check schema syntax against JSON Schema specification",
            "Verify schema file permissions",
        ]
        
        super().__init__(message, schema_file, suggestions)
        self.schema_errors = schema_errors or []


class ConfigurationMergeError(ConfigurationError):
    """Exception raised when configuration merging fails."""
    
    def __init__(
        self,
        message: str,
        source_files: Optional[List[str]] = None,
        conflicting_keys: Optional[List[str]] = None
    ) -> None:
        """
        Initialize merge error.
        
        Args:
            message: Error description
            source_files: List of configuration files being merged
            conflicting_keys: Keys that have merge conflicts
        """
        suggestions = [
            "Check for conflicting configuration values",
            "Ensure all configuration files have valid JSON syntax",
            "Review merge precedence rules",
        ]
        
        if conflicting_keys:
            suggestions.append(f"Resolve conflicts in keys: {', '.join(conflicting_keys)}")
        
        super().__init__(message, None, suggestions)
        self.source_files = source_files or []
        self.conflicting_keys = conflicting_keys or [] 