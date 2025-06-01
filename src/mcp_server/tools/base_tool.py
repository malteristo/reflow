"""
Base MCP Tool for Research Agent.

Provides common functionality and interface for all MCP tools that map
CLI commands to MCP protocol operations.

Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolValidationError:
    """Represents a parameter validation error."""
    parameter: str
    message: str
    value: Any = None


class BaseMCPTool(ABC):
    """
    Base class for all MCP tools.
    
    Provides common functionality for parameter validation, error handling,
    and response formatting for MCP tools that interface with CLI commands.
    """
    
    def __init__(self):
        """Initialize the base MCP tool."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def get_tool_name(self) -> str:
        """
        Get the name of this MCP tool.
        
        Returns:
            str: The tool name used for MCP registration
        """
        pass
    
    @abstractmethod
    def get_tool_description(self) -> str:
        """
        Get the description of this MCP tool.
        
        Returns:
            str: Human-readable description of what this tool does
        """
        pass
    
    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        
        Args:
            parameters: Dictionary of parameters for the tool
            
        Returns:
            Dict containing the tool execution result
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """
        Validate tool parameters.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Get required parameters for this tool
        required_params = self.get_required_parameters()
        
        # Check for missing required parameters
        for param in required_params:
            if param not in parameters:
                errors.append(ToolValidationError(
                    parameter=param,
                    message=f"Required parameter '{param}' is missing"
                ))
            elif parameters[param] is None or parameters[param] == "":
                errors.append(ToolValidationError(
                    parameter=param,
                    message=f"Required parameter '{param}' cannot be empty",
                    value=parameters[param]
                ))
        
        # Perform tool-specific validation
        tool_errors = self.validate_tool_parameters(parameters)
        errors.extend(tool_errors)
        
        return errors
    
    def get_required_parameters(self) -> List[str]:
        """
        Get the list of required parameters for this tool.
        
        Returns:
            List of required parameter names
        """
        return []
    
    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """
        Perform tool-specific parameter validation.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            List of validation errors specific to this tool
        """
        return []
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for this tool's parameters.
        
        Returns:
            Dict containing the JSON schema for parameters
        """
        return {
            "type": "object",
            "properties": {},
            "required": self.get_required_parameters()
        }
    
    def format_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a successful tool response.
        
        Args:
            data: The response data
            
        Returns:
            Formatted success response
        """
        return {
            "status": "success",
            "tool": self.get_tool_name(),
            **data
        }
    
    def format_error(self, message: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format an error response.
        
        Args:
            message: Error message
            parameters: Optional parameters that caused the error
            
        Returns:
            Formatted error response
        """
        error_response = {
            "status": "error",
            "tool": self.get_tool_name(),
            "error": {
                "message": message,
                "type": "tool_error"
            }
        }
        
        if parameters:
            error_response["error"]["parameters"] = parameters
        
        return error_response
    
    def format_validation_error(self, errors: List[ToolValidationError]) -> Dict[str, Any]:
        """
        Format a validation error response.
        
        Args:
            errors: List of validation errors
            
        Returns:
            Formatted validation error response
        """
        error_details = []
        for error in errors:
            detail = {
                "parameter": error.parameter,
                "message": error.message
            }
            if error.value is not None:
                detail["value"] = error.value
            error_details.append(detail)
        
        return {
            "status": "error",
            "tool": self.get_tool_name(),
            "error": {
                "message": "Parameter validation failed",
                "type": "validation_error",
                "details": error_details
            }
        }
    
    def safe_execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely execute the tool with error handling and validation.
        
        Args:
            parameters: Dictionary of parameters for the tool
            
        Returns:
            Dict containing the tool execution result or error
        """
        try:
            # Validate parameters first
            validation_errors = self.validate_parameters(parameters)
            if validation_errors:
                return self.format_validation_error(validation_errors)
            
            # Execute the tool
            result = self.execute(parameters)
            
            # Ensure result has proper format
            if not isinstance(result, dict):
                return self.format_error("Tool returned invalid response format")
            
            if "status" not in result:
                result["status"] = "success"
                result["tool"] = self.get_tool_name()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}", exc_info=True)
            return self.format_error(f"Tool execution failed: {str(e)}", parameters)
    
    def sanitize_path(self, path: str) -> str:
        """
        Sanitize a file path parameter.
        
        Args:
            path: File path to sanitize
            
        Returns:
            Sanitized path
        """
        if not path:
            return ""
        
        # Basic path sanitization
        # Remove any null bytes
        path = path.replace('\x00', '')
        
        # Strip whitespace
        path = path.strip()
        
        return path
    
    def sanitize_collection_name(self, name: str) -> str:
        """
        Sanitize a collection name parameter.
        
        Args:
            name: Collection name to sanitize
            
        Returns:
            Sanitized collection name
        """
        if not name:
            return ""
        
        # Basic collection name sanitization
        # Remove special characters that might cause issues
        import re
        name = re.sub(r'[^\w\-_.]', '', name)
        
        return name.strip()
    
    def parse_collections_list(self, collections: Optional[str]) -> List[str]:
        """
        Parse a comma-separated list of collections.
        
        Args:
            collections: Comma-separated collection names
            
        Returns:
            List of collection names
        """
        if not collections:
            return []
        
        # Split by comma and clean up
        collection_list = [
            self.sanitize_collection_name(c.strip()) 
            for c in collections.split(',')
        ]
        
        # Filter out empty names
        return [c for c in collection_list if c] 