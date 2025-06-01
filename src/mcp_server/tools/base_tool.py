"""
Base MCP Tool for Research Agent.

Provides common functionality and interface for all MCP tools that map
CLI commands to MCP protocol operations.

Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
Enhanced with comprehensive parameter validation (subtask 15.4).
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import validation components
try:
    from ..validation.json_schema_validator import JSONSchemaValidator
    from ..validation.security_validator import SecurityValidator
    from ..validation.business_validator import BusinessValidator
    from ..validation.validation_registry import ValidationRegistry
except ImportError:
    # Graceful fallback if validation components aren't available
    JSONSchemaValidator = None
    SecurityValidator = None
    BusinessValidator = None
    ValidationRegistry = None

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
    
    Enhanced with comprehensive validation including JSON schema validation,
    security validation, and business logic validation.
    """
    
    def __init__(self):
        """Initialize the base MCP tool."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Initialized {self.__class__.__name__}")
        
        # Initialize validation components if available
        self._init_validation_components()
    
    def _init_validation_components(self) -> None:
        """Initialize validation components."""
        try:
            self.json_validator = JSONSchemaValidator() if JSONSchemaValidator else None
            self.security_validator = SecurityValidator() if SecurityValidator else None
            self.business_validator = BusinessValidator() if BusinessValidator else None
            self.validation_registry = ValidationRegistry() if ValidationRegistry else None
            
            if all([self.json_validator, self.security_validator, self.business_validator, self.validation_registry]):
                self.enhanced_validation_available = True
                self.logger.debug("Enhanced validation components initialized successfully")
            else:
                self.enhanced_validation_available = False
                self.logger.warning("Some validation components not available, falling back to basic validation")
        except Exception as e:
            self.enhanced_validation_available = False
            self.logger.warning(f"Failed to initialize enhanced validation: {e}")
    
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
        Validate tool parameters using basic validation.
        
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
    
    def validate_parameters_enhanced(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters using enhanced validation with all validators.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            Dict containing comprehensive validation result:
            {
                "valid": bool,
                "errors": List[Dict] with detailed error information,
                "warnings": List[str] with warning messages,
                "sanitized_params": Dict with cleaned parameters
            }
        """
        if not self.enhanced_validation_available:
            # Fall back to basic validation
            basic_errors = self.validate_parameters(parameters)
            return {
                "valid": len(basic_errors) == 0,
                "errors": [{"field": err.parameter, "message": err.message, "category": "basic"} for err in basic_errors],
                "warnings": [],
                "sanitized_params": parameters
            }
        
        all_errors = []
        all_warnings = []
        sanitized_params = parameters.copy()
        
        try:
            # 1. JSON Schema Validation
            schema = self.get_parameter_schema()
            if schema and self.json_validator:
                schema_result = self.json_validator.validate(parameters, schema)
                if not schema_result["valid"]:
                    all_errors.extend(schema_result["errors"])
                else:
                    sanitized_params = schema_result["sanitized_params"]
            
            # 2. Security Validation
            if self.security_validator:
                security_errors, security_warnings = self._validate_security(parameters)
                all_errors.extend(security_errors)
                all_warnings.extend(security_warnings)
            
            # 3. Business Logic Validation
            if self.business_validator:
                business_errors, business_warnings = self._validate_business_logic(parameters)
                all_errors.extend(business_errors)
                all_warnings.extend(business_warnings)
            
            # 4. Tool-specific enhanced validation
            tool_result = self.validate_tool_parameters_enhanced(parameters)
            if tool_result.get("errors"):
                all_errors.extend(tool_result["errors"])
            if tool_result.get("warnings"):
                all_warnings.extend(tool_result["warnings"])
            
            return {
                "valid": len(all_errors) == 0,
                "errors": all_errors,
                "warnings": all_warnings,
                "sanitized_params": sanitized_params
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced validation failed: {e}", exc_info=True)
            return {
                "valid": False,
                "errors": [{"field": "validation", "message": f"Validation system error: {str(e)}", "category": "system"}],
                "warnings": [],
                "sanitized_params": parameters
            }
    
    def _validate_security(self, parameters: Dict[str, Any]) -> tuple[List[Dict], List[str]]:
        """Apply security validation to parameters."""
        errors = []
        warnings = []
        
        for key, value in parameters.items():
            if isinstance(value, str):
                # Check for path parameters
                if 'path' in key.lower() or 'file' in key.lower():
                    result = self.security_validator.validate_file_path(value)
                    if not result["valid"]:
                        errors.extend([{"field": key, "message": error, "category": "security"} for error in result["errors"]])
                
                # Check for collection names
                elif 'collection' in key.lower():
                    result = self.security_validator.validate_collection_name(value)
                    if not result["valid"]:
                        errors.extend([{"field": key, "message": error, "category": "security"} for error in result["errors"]])
                
                # Check for project names
                elif 'project' in key.lower():
                    result = self.security_validator.validate_project_name(value)
                    if not result["valid"]:
                        errors.extend([{"field": key, "message": error, "category": "security"} for error in result["errors"]])
                
                # General text input sanitization
                else:
                    result = self.security_validator.sanitize_text_input(value)
                    if not result["safe"]:
                        errors.extend([{"field": key, "message": error, "category": "security"} for error in result["errors"]])
        
        return errors, warnings
    
    def _validate_business_logic(self, parameters: Dict[str, Any]) -> tuple[List[Dict], List[str]]:
        """Apply business logic validation to parameters."""
        errors = []
        warnings = []
        
        # Validate query parameters
        if "query" in parameters:
            result = self.business_validator.validate_query_content(parameters["query"])
            if not result["valid"]:
                errors.extend([{"field": "query", "message": error, "category": "business_logic"} for error in result["errors"]])
            if result.get("warnings"):
                warnings.extend(result["warnings"])
        
        # Validate top_k parameter
        if "top_k" in parameters:
            result = self.business_validator.validate_top_k_parameter(parameters["top_k"])
            if not result["valid"]:
                errors.extend([{"field": "top_k", "message": error, "category": "business_logic"} for error in result["errors"]])
        
        # Validate collections parameter
        if "collections" in parameters and parameters["collections"] is not None:
            result = self.business_validator.validate_collections_parameter(parameters["collections"])
            if not result["valid"]:
                errors.extend([{"field": "collections", "message": error, "category": "business_logic"} for error in result["errors"]])
            if result.get("warnings"):
                warnings.extend(result["warnings"])
        
        # Validate collection_name parameter
        if "collection_name" in parameters:
            result = self.business_validator.validate_collection_name_format(parameters["collection_name"])
            if not result["valid"]:
                errors.extend([{"field": "collection_name", "message": error, "category": "business_logic"} for error in result["errors"]])
            if result.get("warnings"):
                warnings.extend(result["warnings"])
        
        # Validate project_name parameter
        if "project_name" in parameters:
            result = self.business_validator.validate_project_name_format(parameters["project_name"])
            if not result["valid"]:
                errors.extend([{"field": "project_name", "message": error, "category": "business_logic"} for error in result["errors"]])
            if result.get("warnings"):
                warnings.extend(result["warnings"])
        
        # Validate collection_type parameter
        if "collection_type" in parameters:
            result = self.business_validator.validate_collection_type(parameters["collection_type"])
            if not result["valid"]:
                errors.extend([{"field": "collection_type", "message": error, "category": "business_logic"} for error in result["errors"]])
        
        return errors, warnings
    
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
    
    def validate_tool_parameters_enhanced(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform enhanced tool-specific parameter validation.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            Dict containing enhanced validation result
        """
        # Default implementation converts basic validation to enhanced format
        basic_errors = self.validate_tool_parameters(parameters)
        return {
            "errors": [{"field": err.parameter, "message": err.message, "category": "tool_specific"} for err in basic_errors],
            "warnings": []
        }
    
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
    
    def get_validation_config(self) -> Dict[str, Any]:
        """
        Get validation configuration for this tool.
        
        Returns:
            Dict containing validation settings
        """
        return {
            "enhanced_validation_enabled": self.enhanced_validation_available,
            "validation_categories": ["security", "business_logic", "schema_validation"],
            "strict_mode": False
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
    
    def format_enhanced_validation_error(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format an enhanced validation error response.
        
        Args:
            validation_result: Result from validate_parameters_enhanced
            
        Returns:
            Formatted validation error response
        """
        return {
            "status": "error",
            "tool": self.get_tool_name(),
            "error": {
                "message": "Parameter validation failed",
                "type": "enhanced_validation_error",
                "errors": validation_result["errors"],
                "warnings": validation_result["warnings"]
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
            # Use enhanced validation if available
            if self.enhanced_validation_available:
                validation_result = self.validate_parameters_enhanced(parameters)
                if not validation_result["valid"]:
                    return self.format_enhanced_validation_error(validation_result)
                
                # Use sanitized parameters for execution
                parameters = validation_result["sanitized_params"]
            else:
                # Fall back to basic validation
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
        
        # Use enhanced security validator if available
        if self.enhanced_validation_available and self.security_validator:
            result = self.security_validator.validate_file_path(path)
            return result.get("sanitized_path", "")
        
        # Basic path sanitization fallback
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