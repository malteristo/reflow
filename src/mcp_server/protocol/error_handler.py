"""
Error Handler for Research Agent MCP Server.

Manages error codes, error responses, and error categorization
according to the MCP protocol specification.

Implements subtask 15.2: STDIO Communication Layer.
"""

import logging
from typing import Dict, Any, Optional
from enum import IntEnum

logger = logging.getLogger(__name__)


class MCPErrorCode(IntEnum):
    """MCP protocol error codes as defined in the specification."""
    
    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP-specific errors (as defined in protocol_spec.md)
    SYSTEM_ERROR = -32000
    CONFIGURATION_ERROR = -32001
    COLLECTION_ERROR = -32002
    DOCUMENT_ERROR = -32003
    QUERY_ERROR = -32004


class ErrorHandler:
    """
    Handles error management for MCP protocol.
    
    Provides error code validation, categorization, and response formatting
    according to the MCP protocol specification.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_categories = {
            MCPErrorCode.CONFIGURATION_ERROR: "Configuration Errors",
            MCPErrorCode.COLLECTION_ERROR: "Collection Errors",
            MCPErrorCode.DOCUMENT_ERROR: "Document Errors",
            MCPErrorCode.QUERY_ERROR: "Query Errors",
            MCPErrorCode.SYSTEM_ERROR: "System Errors",
            MCPErrorCode.PARSE_ERROR: "Parse Errors",
            MCPErrorCode.INVALID_REQUEST: "Invalid Request Errors",
            MCPErrorCode.METHOD_NOT_FOUND: "Method Not Found Errors",
            MCPErrorCode.INVALID_PARAMS: "Invalid Parameters Errors",
            MCPErrorCode.INTERNAL_ERROR: "Internal Errors"
        }
        logger.debug("ErrorHandler initialized")
    
    def is_valid_error_code(self, code: int) -> bool:
        """
        Check if an error code is valid according to MCP specification.
        
        Args:
            code: Error code to validate
            
        Returns:
            True if valid error code, False otherwise
        """
        try:
            MCPErrorCode(code)
            return True
        except ValueError:
            return False
    
    def get_error_category(self, code: int) -> Optional[str]:
        """
        Get the category name for an error code.
        
        Args:
            code: Error code
            
        Returns:
            Category name or None if invalid code
        """
        try:
            error_code = MCPErrorCode(code)
            return self.error_categories.get(error_code)
        except ValueError:
            return None
    
    def create_configuration_error(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a configuration error response.
        
        Args:
            message: Error message
            details: Optional error details
            
        Returns:
            Error object for JSON-RPC response
        """
        return self._create_error_object(
            MCPErrorCode.CONFIGURATION_ERROR,
            message,
            details
        )
    
    def create_collection_error(
        self, 
        message: str, 
        collection_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a collection error response.
        
        Args:
            message: Error message
            collection_name: Name of the collection that caused the error
            details: Optional error details
            
        Returns:
            Error object for JSON-RPC response
        """
        error_details = details or {}
        if collection_name:
            error_details["collection"] = collection_name
        
        return self._create_error_object(
            MCPErrorCode.COLLECTION_ERROR,
            message,
            error_details
        )
    
    def create_document_error(
        self, 
        message: str, 
        document_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a document error response.
        
        Args:
            message: Error message
            document_path: Path of the document that caused the error
            details: Optional error details
            
        Returns:
            Error object for JSON-RPC response
        """
        error_details = details or {}
        if document_path:
            error_details["document_path"] = document_path
        
        return self._create_error_object(
            MCPErrorCode.DOCUMENT_ERROR,
            message,
            error_details
        )
    
    def create_query_error(
        self, 
        message: str, 
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a query error response.
        
        Args:
            message: Error message
            query: Query that caused the error
            details: Optional error details
            
        Returns:
            Error object for JSON-RPC response
        """
        error_details = details or {}
        if query:
            error_details["query"] = query
        
        return self._create_error_object(
            MCPErrorCode.QUERY_ERROR,
            message,
            error_details
        )
    
    def create_system_error(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a system error response.
        
        Args:
            message: Error message
            details: Optional error details
            
        Returns:
            Error object for JSON-RPC response
        """
        return self._create_error_object(
            MCPErrorCode.SYSTEM_ERROR,
            message,
            details
        )
    
    def create_invalid_params_error(
        self, 
        message: str, 
        invalid_params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create an invalid parameters error response.
        
        Args:
            message: Error message
            invalid_params: Dictionary of parameter names to error messages
            
        Returns:
            Error object for JSON-RPC response
        """
        details = {}
        if invalid_params:
            details["invalid_parameters"] = invalid_params
        
        return self._create_error_object(
            MCPErrorCode.INVALID_PARAMS,
            message,
            details
        )
    
    def create_method_not_found_error(
        self, 
        method_name: str
    ) -> Dict[str, Any]:
        """
        Create a method not found error response.
        
        Args:
            method_name: Name of the method that was not found
            
        Returns:
            Error object for JSON-RPC response
        """
        return self._create_error_object(
            MCPErrorCode.METHOD_NOT_FOUND,
            f"Method '{method_name}' not found",
            {"method": method_name}
        )
    
    def create_parse_error(self, details: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a parse error response.
        
        Args:
            details: Optional details about the parse error
            
        Returns:
            Error object for JSON-RPC response
        """
        error_details = {}
        if details:
            error_details["details"] = details
        
        return self._create_error_object(
            MCPErrorCode.PARSE_ERROR,
            "Parse error",
            error_details
        )
    
    def create_internal_error(
        self, 
        message: str = "Internal error",
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an internal error response.
        
        Args:
            message: Error message
            details: Optional error details
            
        Returns:
            Error object for JSON-RPC response
        """
        return self._create_error_object(
            MCPErrorCode.INTERNAL_ERROR,
            message,
            details
        )
    
    def _create_error_object(
        self, 
        code: MCPErrorCode, 
        message: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized error object.
        
        Args:
            code: Error code
            message: Error message
            data: Optional error data
            
        Returns:
            Error object for JSON-RPC response
        """
        error_obj = {
            "code": int(code),
            "message": message
        }
        
        if data:
            error_obj["data"] = data
        
        logger.error(f"Created error: {code} - {message}")
        return error_obj
    
    def log_error(
        self, 
        code: int, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error with appropriate context.
        
        Args:
            code: Error code
            message: Error message
            context: Optional context information
        """
        category = self.get_error_category(code)
        log_message = f"[{category or 'Unknown'}] {message}"
        
        if context:
            log_message += f" | Context: {context}"
        
        logger.error(log_message) 