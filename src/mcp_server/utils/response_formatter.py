"""
Standardized Response Formatter for MCP Tools.

Provides consistent response formatting across all MCP tools to ensure
uniform client experience and proper error handling.

Addresses Task 32 Critical Issue #2: Missing Response Standardization.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class ResponseStatus(Enum):
    """Standardized response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


@dataclass
class ResponseMetadata:
    """Metadata included with every MCP tool response."""
    timestamp: str
    operation: str
    execution_time: float
    tool_name: str
    version: str = "1.0.0"
    request_id: Optional[str] = None


@dataclass
class MCPResponse:
    """Standardized MCP tool response structure."""
    success: bool
    status: ResponseStatus
    data: Any
    message: str
    metadata: ResponseMetadata
    errors: Optional[List[Dict[str, str]]] = None
    warnings: Optional[List[str]] = None


class ResponseFormatter:
    """
    Utility class for creating standardized MCP tool responses.
    
    Ensures all MCP tools return consistent response format:
    {
      "success": bool,
      "status": "success" | "error" | "warning" | "partial",
      "data": Any,
      "message": str,
      "metadata": {
        "timestamp": str,
        "operation": str,
        "execution_time": float,
        "tool_name": str,
        "version": str,
        "request_id": str
      },
      "errors": [{"field": str, "message": str}],  # Optional
      "warnings": [str]  # Optional
    }
    """
    
    def __init__(self, tool_name: str, version: str = "1.0.0"):
        """
        Initialize response formatter for specific tool.
        
        Args:
            tool_name: Name of the MCP tool
            version: Tool version
        """
        self.tool_name = tool_name
        self.version = version
        self._start_time = None
        self._operation = None
        self._request_id = None
    
    def start_operation(self, operation: str, request_id: Optional[str] = None) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation being performed
            request_id: Optional request ID for tracking
        """
        self._start_time = time.time()
        self._operation = operation
        self._request_id = request_id
    
    def _get_execution_time(self) -> float:
        """Get execution time since start_operation was called."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def _create_metadata(self) -> ResponseMetadata:
        """Create response metadata."""
        return ResponseMetadata(
            timestamp=datetime.utcnow().isoformat() + "Z",
            operation=self._operation or "unknown",
            execution_time=self._get_execution_time(),
            tool_name=self.tool_name,
            version=self.version,
            request_id=self._request_id
        )
    
    def success(
        self,
        data: Any,
        message: str = "Operation completed successfully",
        warnings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a success response.
        
        Args:
            data: Response data
            message: Success message
            warnings: Optional warnings to include
            
        Returns:
            Standardized success response dictionary
        """
        response = MCPResponse(
            success=True,
            status=ResponseStatus.WARNING if warnings else ResponseStatus.SUCCESS,
            data=data,
            message=message,
            metadata=self._create_metadata(),
            warnings=warnings
        )
        return asdict(response)
    
    def error(
        self,
        message: str,
        errors: Optional[List[Dict[str, str]]] = None,
        data: Any = None
    ) -> Dict[str, Any]:
        """
        Create an error response.
        
        Args:
            message: Error message
            errors: List of field-specific errors
            data: Optional partial data
            
        Returns:
            Standardized error response dictionary
        """
        response = MCPResponse(
            success=False,
            status=ResponseStatus.ERROR,
            data=data,
            message=message,
            metadata=self._create_metadata(),
            errors=errors
        )
        return asdict(response)
    
    def partial(
        self,
        data: Any,
        message: str = "Operation partially completed",
        warnings: Optional[List[str]] = None,
        errors: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create a partial success response.
        
        Args:
            data: Partial response data
            message: Partial completion message
            warnings: Optional warnings
            errors: Optional errors that occurred
            
        Returns:
            Standardized partial response dictionary
        """
        response = MCPResponse(
            success=True,  # Still considered success with partial results
            status=ResponseStatus.PARTIAL,
            data=data,
            message=message,
            metadata=self._create_metadata(),
            errors=errors,
            warnings=warnings
        )
        return asdict(response)
    
    def validation_error(
        self,
        validation_errors: List[Dict[str, str]],
        message: str = "Validation failed"
    ) -> Dict[str, Any]:
        """
        Create a validation error response.
        
        Args:
            validation_errors: List of validation errors with field and message
            message: Overall validation error message
            
        Returns:
            Standardized validation error response
        """
        return self.error(
            message=message,
            errors=validation_errors
        )
    
    def not_found(
        self,
        resource_type: str,
        identifier: str
    ) -> Dict[str, Any]:
        """
        Create a not found error response.
        
        Args:
            resource_type: Type of resource not found (e.g., "document", "collection")
            identifier: Resource identifier that was not found
            
        Returns:
            Standardized not found error response
        """
        return self.error(
            message=f"{resource_type.title()} not found: {identifier}",
            errors=[{
                "field": "identifier",
                "message": f"{resource_type} '{identifier}' does not exist"
            }]
        )
    
    def permission_denied(
        self,
        operation: str,
        resource: str = ""
    ) -> Dict[str, Any]:
        """
        Create a permission denied error response.
        
        Args:
            operation: Operation that was denied
            resource: Resource that access was denied to
            
        Returns:
            Standardized permission denied error response
        """
        message = f"Permission denied for operation: {operation}"
        if resource:
            message += f" on resource: {resource}"
        
        return self.error(
            message=message,
            errors=[{
                "field": "authorization",
                "message": "Insufficient permissions for this operation"
            }]
        )


def create_response_formatter(tool_name: str) -> ResponseFormatter:
    """
    Factory function to create a response formatter.
    
    Args:
        tool_name: Name of the MCP tool
        
    Returns:
        Configured ResponseFormatter instance
    """
    return ResponseFormatter(tool_name)


# Convenience functions for common response patterns
def success_response(tool_name: str, operation: str, data: Any, message: str = None) -> Dict[str, Any]:
    """Quick success response creation."""
    formatter = create_response_formatter(tool_name)
    formatter.start_operation(operation)
    return formatter.success(data, message or "Operation completed successfully")


def error_response(tool_name: str, operation: str, message: str, errors: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Quick error response creation."""
    formatter = create_response_formatter(tool_name)
    formatter.start_operation(operation)
    return formatter.error(message, errors)


def validation_error_response(tool_name: str, operation: str, validation_errors: List[Dict[str, str]]) -> Dict[str, Any]:
    """Quick validation error response creation."""
    formatter = create_response_formatter(tool_name)
    formatter.start_operation(operation)
    return formatter.validation_error(validation_errors) 