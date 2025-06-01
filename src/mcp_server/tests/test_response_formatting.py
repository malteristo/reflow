"""
Test Response Formatting for Research Agent MCP Server.

Comprehensive test suite for response formatting functionality.
Implements TDD RED phase for subtask 15.5: Design and Implement Response Formatting.
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Optional

# Import the response formatter (will be created)
from src.mcp_server.protocol.response_formatter import (
    ResponseFormatter,
    MCPResponse,
    MCPSuccessResponse,
    MCPErrorResponse,
    QueryResponse,
    CollectionResponse,
    IngestResponse,
    ProjectResponse,
    AugmentResponse,
    ResponseFormattingError
)


class TestResponseFormatter:
    """Test the ResponseFormatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
    
    def test_formatter_initialization(self):
        """Test ResponseFormatter initialization."""
        assert self.formatter is not None
        assert hasattr(self.formatter, 'format_success_response')
        assert hasattr(self.formatter, 'format_error_response')
        assert hasattr(self.formatter, 'format_tool_response')
    
    def test_format_success_response_basic(self):
        """Test basic success response formatting."""
        request_id = "test-123"
        content = "Test response content"
        
        response = self.formatter.format_success_response(
            request_id=request_id,
            content=content
        )
        
        # Should return MCPSuccessResponse
        assert isinstance(response, MCPSuccessResponse)
        assert response.jsonrpc == "2.0"
        assert response.id == request_id
        assert response.result is not None
        assert "content" in response.result
    
    def test_format_error_response_basic(self):
        """Test basic error response formatting."""
        request_id = "test-123"
        error_code = -32000
        error_message = "Test error"
        
        response = self.formatter.format_error_response(
            request_id=request_id,
            error_code=error_code,
            error_message=error_message
        )
        
        # Should return MCPErrorResponse
        assert isinstance(response, MCPErrorResponse)
        assert response.jsonrpc == "2.0"
        assert response.id == request_id
        assert response.error is not None
        assert response.error["code"] == error_code
        assert response.error["message"] == error_message
    
    def test_format_query_response_success(self):
        """Test query tool response formatting."""
        request_id = "query-123"
        query_data = {
            "status": "success",
            "results": [
                {
                    "content": "Test content chunk",
                    "relevance_score": 0.95,
                    "relevance_label": "Highly Relevant",
                    "source_document": "test.md",
                    "header_path": "Section > Subsection",
                    "metadata": {
                        "document_title": "Test Document",
                        "content_type": "prose",
                        "chunk_sequence_id": 1
                    }
                }
            ],
            "query_refinement": {
                "status": "optimal",
                "suggestions": ["keyword1", "keyword2"],
                "message": "Query executed successfully"
            }
        }
        
        response = self.formatter.format_tool_response(
            tool_name="query_knowledge_base",
            request_id=request_id,
            data=query_data
        )
        
        assert isinstance(response, QueryResponse)
        assert response.jsonrpc == "2.0"
        assert response.id == request_id
        assert response.status == "success"
        assert len(response.results) == 1
        assert response.results[0]["relevance_score"] == 0.95
    
    def test_format_collection_response_list(self):
        """Test collection management response formatting."""
        request_id = "collection-123"
        collection_data = {
            "status": "success",
            "action": "list",
            "collections": [
                {
                    "name": "general",
                    "type": "fundamental",
                    "document_count": 42,
                    "created_date": "2024-01-01"
                },
                {
                    "name": "project_alpha",
                    "type": "project-specific",
                    "document_count": 15,
                    "created_date": "2024-01-15"
                }
            ]
        }
        
        response = self.formatter.format_tool_response(
            tool_name="manage_collections",
            request_id=request_id,
            data=collection_data
        )
        
        assert isinstance(response, CollectionResponse)
        assert response.status == "success"
        assert response.action == "list"
        assert len(response.collections) == 2
        assert response.collections[0]["name"] == "general"
    
    def test_format_ingest_response_success(self):
        """Test document ingestion response formatting."""
        request_id = "ingest-123"
        ingest_data = {
            "status": "success",
            "processed_files": [
                {
                    "path": "/path/to/doc1.md",
                    "chunks_created": 5,
                    "status": "success"
                },
                {
                    "path": "/path/to/doc2.md", 
                    "chunks_created": 3,
                    "status": "success"
                }
            ],
            "total_chunks": 8,
            "collection": "test_collection",
            "processing_time": 2.34
        }
        
        response = self.formatter.format_tool_response(
            tool_name="ingest_documents",
            request_id=request_id,
            data=ingest_data
        )
        
        assert isinstance(response, IngestResponse)
        assert response.status == "success"
        assert len(response.processed_files) == 2
        assert response.total_chunks == 8
        assert response.collection == "test_collection"


class TestMCPResponseClasses:
    """Test MCP response data classes."""
    
    def test_mcp_success_response_creation(self):
        """Test MCPSuccessResponse creation and serialization."""
        response = MCPSuccessResponse(
            id="test-123",
            result={
                "content": [
                    {
                        "type": "text",
                        "text": "Test response"
                    }
                ]
            }
        )
        
        assert response.jsonrpc == "2.0"
        assert response.id == "test-123"
        assert response.result["content"][0]["text"] == "Test response"
        
        # Test JSON serialization
        json_data = response.to_dict()
        assert json_data["jsonrpc"] == "2.0"
        assert json_data["id"] == "test-123"
        assert "result" in json_data
    
    def test_mcp_error_response_creation(self):
        """Test MCPErrorResponse creation and serialization."""
        response = MCPErrorResponse(
            id="test-123",
            error={
                "code": -32000,
                "message": "Test error",
                "data": {
                    "details": "Specific error details"
                }
            }
        )
        
        assert response.jsonrpc == "2.0"
        assert response.id == "test-123"
        assert response.error["code"] == -32000
        assert response.error["message"] == "Test error"
        
        # Test JSON serialization
        json_data = response.to_dict()
        assert json_data["jsonrpc"] == "2.0"
        assert json_data["error"]["data"]["details"] == "Specific error details"
    
    def test_query_response_creation(self):
        """Test QueryResponse creation."""
        response = QueryResponse(
            id="query-123",
            status="success",
            results=[
                {
                    "content": "Test content",
                    "relevance_score": 0.9,
                    "relevance_label": "Highly Relevant",
                    "source_document": "test.md"
                }
            ],
            query_refinement={
                "status": "optimal",
                "suggestions": [],
                "message": "Query executed successfully"
            }
        )
        
        assert response.id == "query-123"
        assert response.status == "success"
        assert len(response.results) == 1
        assert response.query_refinement["status"] == "optimal"


class TestErrorResponseFormatting:
    """Test error response formatting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
    
    def test_format_configuration_error(self):
        """Test configuration error response formatting."""
        response = self.formatter.format_error_response(
            request_id="test-123",
            error_code=-32001,
            error_message="Configuration error",
            error_data={
                "details": "Missing configuration file",
                "guidance": "Please check config.json exists"
            }
        )
        
        assert response.error["code"] == -32001
        assert "Configuration error" in response.error["message"]
        assert response.error["data"]["guidance"] == "Please check config.json exists"
    
    def test_format_collection_error(self):
        """Test collection error response formatting."""
        response = self.formatter.format_error_response(
            request_id="test-123",
            error_code=-32002,
            error_message="Collection not found",
            error_data={
                "collection_name": "invalid_collection",
                "available_collections": ["general", "project_alpha"]
            }
        )
        
        assert response.error["code"] == -32002
        assert "Collection not found" in response.error["message"]
        assert "invalid_collection" in response.error["data"]["collection_name"]
    
    def test_format_document_error(self):
        """Test document error response formatting."""
        response = self.formatter.format_error_response(
            request_id="test-123",
            error_code=-32003,
            error_message="Document parsing failed",
            error_data={
                "file_path": "/path/to/invalid.pdf",
                "error_details": "Unsupported file format"
            }
        )
        
        assert response.error["code"] == -32003
        assert "Document parsing failed" in response.error["message"]
        assert "/path/to/invalid.pdf" in response.error["data"]["file_path"]
    
    def test_format_query_error(self):
        """Test query error response formatting."""
        response = self.formatter.format_error_response(
            request_id="test-123",
            error_code=-32004,
            error_message="Query processing failed",
            error_data={
                "query": "invalid query with special chars",
                "error_type": "embedding_failure"
            }
        )
        
        assert response.error["code"] == -32004
        assert "Query processing failed" in response.error["message"]
        assert "embedding_failure" in response.error["data"]["error_type"]


class TestProgressAndStatusFormatting:
    """Test progress and status update formatting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
    
    def test_format_progress_update(self):
        """Test progress update formatting for long-running operations."""
        progress_data = {
            "operation": "ingest_documents",
            "progress": 0.65,
            "current_file": "document_42.md",
            "files_processed": 42,
            "total_files": 65,
            "estimated_time_remaining": 23.5
        }
        
        response = self.formatter.format_progress_response(
            request_id="ingest-123",
            progress_data=progress_data
        )
        
        assert response.operation == "ingest_documents"
        assert response.progress == 0.65
        assert response.files_processed == 42
        assert response.total_files == 65
    
    def test_format_status_update(self):
        """Test status update formatting."""
        status_data = {
            "operation": "query_knowledge_base",
            "status": "processing",
            "stage": "embedding_generation",
            "message": "Generating embeddings for query"
        }
        
        response = self.formatter.format_status_response(
            request_id="query-123",
            status_data=status_data
        )
        
        assert response.operation == "query_knowledge_base"
        assert response.status == "processing"
        assert response.stage == "embedding_generation"


class TestJSONRPCCompliance:
    """Test JSON-RPC 2.0 protocol compliance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
    
    def test_jsonrpc_version_always_present(self):
        """Test that jsonrpc version is always set to 2.0."""
        success_response = self.formatter.format_success_response("123", "test")
        error_response = self.formatter.format_error_response("123", -32000, "error")
        
        assert success_response.jsonrpc == "2.0"
        assert error_response.jsonrpc == "2.0"
    
    def test_request_id_preserved(self):
        """Test that request ID is preserved in responses."""
        request_id = "unique-request-id-12345"
        
        success_response = self.formatter.format_success_response(request_id, "test")
        error_response = self.formatter.format_error_response(request_id, -32000, "error")
        
        assert success_response.id == request_id
        assert error_response.id == request_id
    
    def test_success_response_structure(self):
        """Test success response follows JSON-RPC 2.0 structure."""
        response = self.formatter.format_success_response("123", "test content")
        response_dict = response.to_dict()
        
        # Must have exactly these fields for success
        required_fields = {"jsonrpc", "result", "id"}
        assert set(response_dict.keys()) == required_fields
        
        # jsonrpc must be exactly "2.0"
        assert response_dict["jsonrpc"] == "2.0"
        
        # result must be present and not null
        assert response_dict["result"] is not None
    
    def test_error_response_structure(self):
        """Test error response follows JSON-RPC 2.0 structure."""
        response = self.formatter.format_error_response("123", -32000, "test error")
        response_dict = response.to_dict()
        
        # Must have exactly these fields for error
        required_fields = {"jsonrpc", "error", "id"}
        assert set(response_dict.keys()) == required_fields
        
        # error must have code and message
        error_fields = {"code", "message"}
        assert error_fields.issubset(set(response_dict["error"].keys()))
        
        # error code must be integer
        assert isinstance(response_dict["error"]["code"], int)


class TestToolSpecificFormatting:
    """Test tool-specific response formatting requirements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
    
    def test_query_tool_response_format(self):
        """Test query_knowledge_base tool response format matches specification."""
        query_data = {
            "status": "success",
            "results": [
                {
                    "content": "Test chunk content here",
                    "relevance_score": 0.92,
                    "relevance_label": "Highly Relevant", 
                    "source_document": "test_document.md",
                    "header_path": "Introduction > Overview",
                    "metadata": {
                        "document_title": "Test Document Title",
                        "content_type": "prose",
                        "chunk_sequence_id": 3
                    }
                }
            ],
            "query_refinement": {
                "status": "optimal",
                "suggestions": ["python", "async"],
                "message": "Query executed successfully with good specificity"
            }
        }
        
        response = self.formatter.format_tool_response(
            tool_name="query_knowledge_base",
            request_id="query-456",
            data=query_data
        )
        
        # Check QueryResponse specific requirements
        assert hasattr(response, 'status')
        assert hasattr(response, 'results')
        assert hasattr(response, 'query_refinement')
        
        # Check result structure matches specification
        result = response.results[0]
        required_result_fields = {
            "content", "relevance_score", "relevance_label", 
            "source_document", "header_path", "metadata"
        }
        assert required_result_fields.issubset(set(result.keys()))
        
        # Check query_refinement structure
        refinement = response.query_refinement
        required_refinement_fields = {"status", "suggestions", "message"}
        assert required_refinement_fields.issubset(set(refinement.keys()))
    
    def test_manage_collections_response_format(self):
        """Test manage_collections tool response format."""
        collection_data = {
            "status": "success",
            "action": "info",
            "collection_info": {
                "name": "project_alpha",
                "type": "project-specific",
                "document_count": 25,
                "created_date": "2024-01-15T10:30:00Z",
                "last_updated": "2024-01-20T14:22:00Z",
                "total_chunks": 150,
                "storage_size": "2.1MB"
            }
        }
        
        response = self.formatter.format_tool_response(
            tool_name="manage_collections",
            request_id="collection-789",
            data=collection_data
        )
        
        assert hasattr(response, 'status')
        assert hasattr(response, 'action')
        assert response.action == "info"
        assert "collection_info" in response.__dict__ or hasattr(response, 'collection_info')


class TestContentTypeHandling:
    """Test different content type handling in responses."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResponseFormatter()
    
    def test_markdown_content_formatting(self):
        """Test markdown content is properly formatted."""
        markdown_content = """
        # Test Document
        
        This is a **test** document with *formatting*.
        
        - List item 1
        - List item 2
        
        ```python
        def example():
            return "code"
        ```
        """
        
        response = self.formatter.format_success_response(
            request_id="md-123",
            content=markdown_content,
            content_type="markdown"
        )
        
        # Should preserve markdown formatting
        result_content = response.result["content"][0]["text"]
        assert "**test**" in result_content
        assert "```python" in result_content
    
    def test_json_content_formatting(self):
        """Test JSON content is properly formatted."""
        json_content = {
            "collections": ["general", "project_alpha"],
            "total_documents": 67,
            "status": "active"
        }
        
        response = self.formatter.format_success_response(
            request_id="json-123",
            content=json_content,
            content_type="json"
        )
        
        # JSON should be serialized properly
        result_content = response.result["content"][0]["text"]
        parsed_content = json.loads(result_content)
        assert parsed_content["total_documents"] == 67
        assert "general" in parsed_content["collections"]
    
    def test_plain_text_formatting(self):
        """Test plain text content formatting."""
        text_content = "This is plain text content without special formatting."
        
        response = self.formatter.format_success_response(
            request_id="text-123",
            content=text_content,
            content_type="text"
        )
        
        result_content = response.result["content"][0]["text"]
        assert result_content == text_content
        assert response.result["content"][0]["type"] == "text"


# This file tests functionality that doesn't exist yet (RED phase)
# All tests should fail until implementation is created
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 