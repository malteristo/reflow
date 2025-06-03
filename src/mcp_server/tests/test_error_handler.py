"""
Comprehensive tests for the ErrorHandler class to improve coverage.
"""

import pytest
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

from src.mcp_server.protocol.error_handler import ErrorHandler, MCPErrorCode


class TestErrorHandler:
    """Test basic ErrorHandler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ErrorHandler()
    
    def test_initialization(self):
        """Test ErrorHandler initialization."""
        assert self.handler is not None
        assert hasattr(self.handler, 'error_categories')
        assert len(self.handler.error_categories) > 0
    
    def test_is_valid_error_code_valid_codes(self):
        """Test is_valid_error_code with valid codes."""
        # Test standard JSON-RPC errors
        assert self.handler.is_valid_error_code(-32700) == True  # PARSE_ERROR
        assert self.handler.is_valid_error_code(-32600) == True  # INVALID_REQUEST
        assert self.handler.is_valid_error_code(-32601) == True  # METHOD_NOT_FOUND
        assert self.handler.is_valid_error_code(-32602) == True  # INVALID_PARAMS
        assert self.handler.is_valid_error_code(-32603) == True  # INTERNAL_ERROR
        
        # Test MCP-specific errors
        assert self.handler.is_valid_error_code(-32000) == True  # SYSTEM_ERROR
        assert self.handler.is_valid_error_code(-32001) == True  # CONFIGURATION_ERROR
        assert self.handler.is_valid_error_code(-32002) == True  # COLLECTION_ERROR
        assert self.handler.is_valid_error_code(-32003) == True  # DOCUMENT_ERROR
        assert self.handler.is_valid_error_code(-32004) == True  # QUERY_ERROR
    
    def test_is_valid_error_code_invalid_codes(self):
        """Test is_valid_error_code with invalid codes."""
        assert self.handler.is_valid_error_code(12345) == False
        assert self.handler.is_valid_error_code(-99999) == False
        assert self.handler.is_valid_error_code(0) == False
    
    def test_get_error_category_valid_codes(self):
        """Test get_error_category with valid codes."""
        category = self.handler.get_error_category(-32700)  # PARSE_ERROR
        assert category is not None
        
        category = self.handler.get_error_category(-32001)  # CONFIGURATION_ERROR
        assert category is not None
    
    def test_get_error_category_invalid_codes(self):
        """Test get_error_category with invalid codes."""
        assert self.handler.get_error_category(12345) is None
        assert self.handler.get_error_category(-99999) is None
    
    def test_create_configuration_error(self):
        """Test creating configuration error."""
        message = "Invalid configuration parameter"
        details = {"parameter": "model_name", "value": "invalid_model"}
        
        error = self.handler.create_configuration_error(message, details)
        
        assert error["code"] == MCPErrorCode.CONFIGURATION_ERROR
        assert error["message"] == message
        assert error["data"] == details
    
    def test_create_configuration_error_no_details(self):
        """Test creating configuration error without details."""
        message = "Invalid configuration"
        
        error = self.handler.create_configuration_error(message)
        
        assert error["code"] == MCPErrorCode.CONFIGURATION_ERROR
        assert error["message"] == message
        assert "data" not in error  # data key is only present if data is provided
    
    def test_create_collection_error(self):
        """Test creating collection error."""
        message = "Collection not found"
        collection_name = "test_collection"
        details = {"operation": "query"}
        
        error = self.handler.create_collection_error(message, collection_name, details)
        
        assert error["code"] == MCPErrorCode.COLLECTION_ERROR
        assert error["message"] == message
        assert error["data"]["collection"] == collection_name
        assert error["data"]["operation"] == "query"
    
    def test_create_collection_error_no_collection_name(self):
        """Test creating collection error without collection name."""
        message = "Collection operation failed"
        
        error = self.handler.create_collection_error(message)
        
        assert error["code"] == MCPErrorCode.COLLECTION_ERROR
        assert error["message"] == message
        assert "data" not in error  # data key is only present if data is provided
    
    def test_create_document_error(self):
        """Test creating document error."""
        message = "Document processing failed"
        document_path = "/path/to/document.md"
        details = {"line": 42, "column": 15}
        
        error = self.handler.create_document_error(message, document_path, details)
        
        assert error["code"] == MCPErrorCode.DOCUMENT_ERROR
        assert error["message"] == message
        assert error["data"]["document_path"] == document_path
        assert error["data"]["line"] == 42
        assert error["data"]["column"] == 15
    
    def test_create_document_error_no_path(self):
        """Test creating document error without document path."""
        message = "Document processing failed"
        
        error = self.handler.create_document_error(message)
        
        assert error["code"] == MCPErrorCode.DOCUMENT_ERROR
        assert error["message"] == message
        assert "data" not in error  # data key is only present if data is provided
    
    def test_create_query_error(self):
        """Test creating query error."""
        message = "Query execution failed"
        query = "search for relevant documents"
        details = {"timeout": True, "duration": 30.5}
        
        error = self.handler.create_query_error(message, query, details)
        
        assert error["code"] == MCPErrorCode.QUERY_ERROR
        assert error["message"] == message
        assert error["data"]["query"] == query
        assert error["data"]["timeout"] == True
        assert error["data"]["duration"] == 30.5
    
    def test_create_query_error_no_query(self):
        """Test creating query error without query."""
        message = "Query failed"
        
        error = self.handler.create_query_error(message)
        
        assert error["code"] == MCPErrorCode.QUERY_ERROR
        assert error["message"] == message
        assert "data" not in error  # data key is only present if data is provided
    
    def test_create_system_error(self):
        """Test creating system error."""
        message = "System resource unavailable"
        details = {"resource": "memory", "usage": "95%"}
        
        error = self.handler.create_system_error(message, details)
        
        assert error["code"] == MCPErrorCode.SYSTEM_ERROR
        assert error["message"] == message
        assert error["data"]["resource"] == "memory"
        assert error["data"]["usage"] == "95%"
    
    def test_create_system_error_no_details(self):
        """Test creating system error without details."""
        message = "System error occurred"
        
        error = self.handler.create_system_error(message)
        
        assert error["code"] == MCPErrorCode.SYSTEM_ERROR
        assert error["message"] == message
        assert "data" not in error  # data key is only present if data is provided
    
    def test_create_invalid_params_error(self):
        """Test creating invalid parameters error."""
        message = "Invalid parameters provided"
        invalid_params = {
            "collection": "must be a string",
            "limit": "must be a positive integer"
        }
        
        error = self.handler.create_invalid_params_error(message, invalid_params)
        
        assert error["code"] == MCPErrorCode.INVALID_PARAMS
        assert error["message"] == message
        assert error["data"]["invalid_parameters"] == invalid_params
    
    def test_create_invalid_params_error_no_params(self):
        """Test creating invalid parameters error without invalid params."""
        message = "Invalid parameters"
        
        error = self.handler.create_invalid_params_error(message)
        
        assert error["code"] == MCPErrorCode.INVALID_PARAMS
        assert error["message"] == message
        assert "data" not in error  # data key is only present if data is provided
    
    def test_create_method_not_found_error(self):
        """Test creating method not found error."""
        method_name = "non_existent_method"
        
        error = self.handler.create_method_not_found_error(method_name)
        
        assert error["code"] == MCPErrorCode.METHOD_NOT_FOUND
        assert error["message"] == f"Method '{method_name}' not found"
        assert error["data"]["method"] == method_name
    
    def test_create_parse_error(self):
        """Test creating parse error."""
        details = "Invalid JSON format at line 5"
        
        error = self.handler.create_parse_error(details)
        
        assert error["code"] == MCPErrorCode.PARSE_ERROR
        assert "parse error" in error["message"].lower()
        assert error["data"]["details"] == details
    
    def test_create_parse_error_no_details(self):
        """Test creating parse error without details."""
        error = self.handler.create_parse_error()
        
        assert error["code"] == MCPErrorCode.PARSE_ERROR
        assert "parse error" in error["message"].lower()
        assert "data" not in error  # data key is only present if data is provided
    
    def test_create_internal_error(self):
        """Test creating internal error."""
        message = "Unexpected internal failure"
        details = {"stack_trace": "...", "error_id": "ERR_001"}
        
        error = self.handler.create_internal_error(message, details)
        
        assert error["code"] == MCPErrorCode.INTERNAL_ERROR
        assert error["message"] == message
        assert error["data"]["stack_trace"] == "..."
        assert error["data"]["error_id"] == "ERR_001"
    
    def test_create_internal_error_default_message(self):
        """Test creating internal error with default message."""
        error = self.handler.create_internal_error()
        
        assert error["code"] == MCPErrorCode.INTERNAL_ERROR
        assert error["message"] == "Internal error"
        assert "data" not in error  # data key is only present if data is provided
    
    @patch('src.mcp_server.protocol.error_handler.logger')
    def test_log_error(self, mock_logger):
        """Test error logging."""
        code = -32700
        message = "Parse error occurred"
        context = {"request_id": "req_123", "timestamp": "2024-01-01T00:00:00Z"}
        
        self.handler.log_error(code, message, context)
        
        # Verify logger was called
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        # Check for category-based format: "[Parse Errors] message | Context: ..."
        assert "Parse Errors" in call_args or "Parse" in call_args
        assert message in call_args
        assert "Context:" in call_args
    
    @patch('src.mcp_server.protocol.error_handler.logger')
    def test_log_error_no_context(self, mock_logger):
        """Test error logging without context."""
        code = -32001
        message = "Configuration error"
        
        self.handler.log_error(code, message)
        
        # Verify logger was called
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        # Check for category-based format: "[Configuration Errors] message"
        assert "Configuration" in call_args
        assert message in call_args


class TestMCPErrorCode:
    """Test MCPErrorCode enumeration."""
    
    def test_standard_json_rpc_codes(self):
        """Test standard JSON-RPC error codes."""
        assert MCPErrorCode.PARSE_ERROR == -32700
        assert MCPErrorCode.INVALID_REQUEST == -32600
        assert MCPErrorCode.METHOD_NOT_FOUND == -32601
        assert MCPErrorCode.INVALID_PARAMS == -32602
        assert MCPErrorCode.INTERNAL_ERROR == -32603
    
    def test_mcp_specific_codes(self):
        """Test MCP-specific error codes."""
        assert MCPErrorCode.SYSTEM_ERROR == -32000
        assert MCPErrorCode.CONFIGURATION_ERROR == -32001
        assert MCPErrorCode.COLLECTION_ERROR == -32002
        assert MCPErrorCode.DOCUMENT_ERROR == -32003
        assert MCPErrorCode.QUERY_ERROR == -32004
    
    def test_error_code_uniqueness(self):
        """Test that all error codes are unique."""
        codes = [code.value for code in MCPErrorCode]
        assert len(codes) == len(set(codes)) 