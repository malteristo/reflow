"""
Base MCP Tool for Research Agent.

Provides common functionality and interface for all MCP tools that map
CLI commands to MCP protocol operations.

Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
Enhanced with comprehensive parameter validation (subtask 15.4).
Updated with standardized response formatting (Task 32).
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import standardized response formatter
try:
    from ..utils.response_formatter import ResponseFormatter
except ImportError:
    ResponseFormatter = None

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
    and standardized response formatting for MCP tools that interface with CLI commands.
    
    Enhanced with:
    - Comprehensive validation (JSON schema, security, business logic)
    - Standardized response formatting (Task 32)
    - Parameter sanitization and security checks
    """
    
    def __init__(self):
        """Initialize the base MCP tool."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Initialized {self.__class__.__name__}")
        
        # Initialize standardized response formatter
        self._init_response_formatter()
        
        # Initialize validation components if available
        self._init_validation_components()
    
    def _init_response_formatter(self) -> None:
        """Initialize the standardized response formatter."""
        if ResponseFormatter:
            self.response_formatter = ResponseFormatter(self.get_tool_name())
            self.standardized_responses = True
            self.logger.debug("Standardized response formatter initialized")
        else:
            self.standardized_responses = False
            self.logger.warning("ResponseFormatter not available, using legacy responses")
    
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
            Dict containing the tool execution result in standardized format
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
        
        except Exception as e:
            all_errors.append({
                "field": "validation_system",
                "message": f"Validation system error: {str(e)}",
                "category": "system"
            })
        
        return {
            "valid": len(all_errors) == 0,
            "errors": all_errors,
            "warnings": all_warnings,
            "sanitized_params": sanitized_params
        }
    
    def _validate_security(self, parameters: Dict[str, Any]) -> tuple[List[Dict], List[str]]:
        """
        Validate parameters for security issues.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        try:
            if self.security_validator:
                security_result = self.security_validator.validate_parameters(parameters)
                if security_result.get("path_injection_detected"):
                    errors.append({
                        "field": "path_parameters",
                        "message": "Potential path injection attack detected",
                        "category": "security"
                    })
                
                if security_result.get("suspicious_patterns"):
                    for pattern in security_result["suspicious_patterns"]:
                        warnings.append(f"Suspicious pattern detected: {pattern}")
        
        except Exception as e:
            warnings.append(f"Security validation failed: {str(e)}")
        
        return errors, warnings
    
    def _validate_business_logic(self, parameters: Dict[str, Any]) -> tuple[List[Dict], List[str]]:
        """
        Validate parameters against business logic rules.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        try:
            if self.business_validator:
                # Validate collection names
                if "collection" in parameters:
                    collection_result = self.business_validator.validate_collection_name(parameters["collection"])
                    if not collection_result["valid"]:
                        errors.append({
                            "field": "collection",
                            "message": collection_result["message"],
                            "category": "business_logic"
                        })
                
                # Validate file paths
                if "path" in parameters:
                    path_result = self.business_validator.validate_file_path(parameters["path"])
                    if not path_result["valid"]:
                        errors.append({
                            "field": "path",
                            "message": path_result["message"],
                            "category": "business_logic"
                        })
        
        except Exception as e:
            warnings.append(f"Business logic validation failed: {str(e)}")
        
        return errors, warnings
    
    # Required methods for subclasses to implement
    def get_required_parameters(self) -> List[str]:
        """
        Get list of required parameters for this tool.
        
        Returns:
            List of required parameter names
        """
        return []
    
    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """
        Validate tool-specific parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            List of validation errors
        """
        return []
    
    def validate_tool_parameters_enhanced(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced tool-specific parameter validation.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Dict with validation results including errors and warnings
        """
        return {"errors": [], "warnings": []}
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for parameter validation.
        
        Returns:
            JSON schema dictionary or None if not implemented
        """
        return {}
    
    def get_validation_config(self) -> Dict[str, Any]:
        """
        Get validation configuration for this tool.
        
        Returns:
            Dict with validation settings and rules
        """
        return {
            "strict_mode": False,
            "security_level": "medium",
            "sanitize_inputs": True
        }

    # UPDATED: Standardized Response Methods
    def format_success_response(
        self, 
        data: Any, 
        message: str = None,
        operation: str = None,
        warnings: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized success response.
        
        Args:
            data: Response data
            message: Optional success message
            operation: Operation name for tracking
            warnings: Optional warnings to include
            
        Returns:
            Standardized success response
        """
        if self.standardized_responses:
            if operation:
                self.response_formatter.start_operation(operation)
            return self.response_formatter.success(
                data=data,
                message=message or "Operation completed successfully",
                warnings=warnings
            )
        else:
            # Legacy format for backward compatibility
            return {
                "status": "success",
                "data": data,
                "message": message or "Operation completed successfully"
            }
    
    def format_error(
        self, 
        message: str, 
        operation: str = None,
        errors: List[Dict[str, str]] = None,
        data: Any = None
    ) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            message: Error message
            operation: Operation name for tracking
            errors: List of field-specific errors
            data: Optional partial data
            
        Returns:
            Standardized error response
        """
        if self.standardized_responses:
            if operation:
                self.response_formatter.start_operation(operation)
            return self.response_formatter.error(
                message=message,
                errors=errors,
                data=data
            )
        else:
            # Legacy format for backward compatibility
            return {
                "status": "error",
                "message": message,
                "errors": errors or []
            }
    
    def format_validation_error(self, errors: List[ToolValidationError]) -> Dict[str, Any]:
        """
        Create a standardized validation error response.
        
        Args:
            errors: List of validation errors
            
        Returns:
            Standardized validation error response
        """
        if self.standardized_responses:
            validation_errors = [
                {"field": err.parameter, "message": err.message}
                for err in errors
            ]
            return self.response_formatter.validation_error(validation_errors)
        else:
            # Legacy format for backward compatibility
            return {
                "status": "error",
                "message": "Parameter validation failed",
                "errors": [
                    {"parameter": err.parameter, "message": err.message, "value": err.value}
                    for err in errors
                ]
            }
    
    def format_enhanced_validation_error(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized enhanced validation error response.
        
        Args:
            validation_result: Enhanced validation result with errors and warnings
            
        Returns:
            Standardized validation error response
        """
        if self.standardized_responses:
            return self.response_formatter.validation_error(
                validation_errors=validation_result.get("errors", []),
                message="Enhanced parameter validation failed"
            )
        else:
            # Legacy format for backward compatibility
            return {
                "status": "error",
                "message": "Enhanced parameter validation failed",
                "errors": validation_result.get("errors", []),
                "warnings": validation_result.get("warnings", [])
            }
    
    def format_not_found(self, resource_type: str, identifier: str) -> Dict[str, Any]:
        """
        Create a standardized not found error response.
        
        Args:
            resource_type: Type of resource not found
            identifier: Resource identifier
            
        Returns:
            Standardized not found response
        """
        if self.standardized_responses:
            return self.response_formatter.not_found(resource_type, identifier)
        else:
            # Legacy format for backward compatibility
            return {
                "status": "error",
                "message": f"{resource_type.title()} not found: {identifier}"
            }
    
    def safe_execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely execute the tool with comprehensive validation and error handling.
        
        Args:
            parameters: Tool parameters
            
        Returns:
            Standardized response with execution result or error
        """
        try:
            self.logger.debug(f"Executing {self.get_tool_name()} with parameters: {parameters}")
            
            # Enhanced parameter validation if available
            if self.enhanced_validation_available:
                validation_result = self.validate_parameters_enhanced(parameters)
                if not validation_result["valid"]:
                    self.logger.warning(f"Enhanced validation failed for {self.get_tool_name()}: {validation_result['errors']}")
                    return self.format_enhanced_validation_error(validation_result)
                
                # Use sanitized parameters
                parameters = validation_result["sanitized_params"]
                warnings = validation_result.get("warnings", [])
            else:
                # Basic validation fallback
                validation_errors = self.validate_parameters(parameters)
                if validation_errors:
                    self.logger.warning(f"Validation failed for {self.get_tool_name()}: {validation_errors}")
                    return self.format_validation_error(validation_errors)
                warnings = []
            
            # Execute the tool
            result = self.execute(parameters)
            
            # Add warnings to successful results if any
            if warnings and isinstance(result, dict) and result.get("success"):
                if "warnings" not in result:
                    result["warnings"] = warnings
                else:
                    result["warnings"].extend(warnings)
            
            self.logger.debug(f"Successfully executed {self.get_tool_name()}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing {self.get_tool_name()}: {e}", exc_info=True)
            return self.format_error(
                message=f"Tool execution failed: {str(e)}",
                operation="safe_execute"
            )
    
    # Utility methods for parameter sanitization
    def sanitize_path(self, path: str) -> str:
        """
        Sanitize file path to prevent path traversal attacks.
        
        Args:
            path: File path to sanitize
            
        Returns:
            Sanitized path
        """
        import os
        import posixpath
        
        # Normalize path and remove dangerous components
        normalized = posixpath.normpath(path)
        
        # Remove leading slashes and path traversal components
        parts = []
        for part in normalized.split('/'):
            if part and part != '.' and part != '..':
                parts.append(part)
        
        return '/'.join(parts)
    
    def sanitize_collection_name(self, name: str) -> str:
        """
        Sanitize collection name to ensure it follows naming conventions.
        
        Args:
            name: Collection name to sanitize
            
        Returns:
            Sanitized collection name
        """
        import re
        
        # Remove non-alphanumeric characters except hyphens and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', name)
        
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = '_' + sanitized
        
        # Limit length
        return sanitized[:50] if len(sanitized) > 50 else sanitized
    
    def parse_collections_list(self, collections: Optional[str]) -> List[str]:
        """
        Parse collections parameter that can be a single collection or comma-separated list.
        
        Args:
            collections: Collections string to parse
            
        Returns:
            List of collection names
        """
        if not collections:
            return []
        
        if isinstance(collections, list):
            return collections
        
        # Split by comma and clean up
        return [self.sanitize_collection_name(name.strip()) for name in collections.split(',') if name.strip()] 