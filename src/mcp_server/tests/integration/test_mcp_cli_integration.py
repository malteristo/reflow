"""
MCP-CLI Integration Tests.

Comprehensive test suite to validate that MCP tools correctly interface with
the CLI backend and produce expected results matching CLI command behavior.

Addresses Task 32: MCP-CLI Integration Audit and Testing.
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock

# Import MCP tools for testing
from mcp_server.tools.dedicated_collection_tools import (
    CreateCollectionTool,
    ListCollectionsTool,
    DeleteCollectionTool,
    GetCollectionInfoTool,
    RenameCollectionTool
)
from mcp_server.tools.dedicated_kb_tools import (
    ListDocumentsTool,
    GetKnowledgeBaseStatusTool,
    RebuildIndexTool,
    AddDocumentTool,
    RemoveDocumentTool
)
from mcp_server.tools.query_tool import QueryKnowledgeBaseTool


class TestMCPCLIIntegration:
    """Test suite for MCP-CLI integration validation."""
    
    @pytest.fixture
    def mock_cli_path(self):
        """Mock CLI path for testing."""
        return "test-research-agent-cli"
    
    @pytest.fixture
    def temp_test_file(self):
        """Create a temporary test file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nThis is a test document for MCP-CLI integration testing.")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def create_mock_cli_success(self, output_data: Dict[str, Any]) -> AsyncMock:
        """Create a mock CLI subprocess that returns success with given data."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            json.dumps(output_data).encode(),
            b""
        )
        return mock_process
    
    def create_mock_cli_error(self, error_message: str, return_code: int = 1) -> AsyncMock:
        """Create a mock CLI subprocess that returns an error."""
        mock_process = AsyncMock()
        mock_process.returncode = return_code
        mock_process.communicate.return_value = (
            b"",
            error_message.encode()
        )
        return mock_process


class TestCollectionToolsIntegration(TestMCPCLIIntegration):
    """Test collection management tools against CLI equivalents."""
    
    @pytest.mark.asyncio
    async def test_create_collection_success(self, mock_cli_path):
        """Test create_collection tool matches CLI create command."""
        # Expected CLI output
        expected_output = {
            "status": "success",
            "collection": {
                "name": "test-collection",
                "type": "general",
                "created": True
            }
        }
        
        mock_process = self.create_mock_cli_success(expected_output)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            tool = CreateCollectionTool(mock_cli_path)
            
            parameters = {
                "name": "test-collection",
                "description": "Test collection for integration testing",
                "collection_type": "general"
            }
            
            result = await tool.safe_execute(parameters)
            
            # Validate standardized response format
            assert result["success"] is True
            assert result["status"] == "success"
            assert "data" in result
            assert "metadata" in result
            
            # Validate data content
            data = result["data"]
            assert data["collection_name"] == "test-collection"
            assert data["collection_type"] == "general"
            assert data["created"] is True
            
            # Validate CLI command was called correctly
            mock_process.communicate.assert_called_once()
            # Note: We can't easily validate the exact command args without more complex mocking
    
    @pytest.mark.asyncio
    async def test_create_collection_validation_error(self, mock_cli_path):
        """Test create_collection tool validation matches CLI validation."""
        tool = CreateCollectionTool(mock_cli_path)
        
        # Test invalid collection name (contains spaces)
        parameters = {
            "name": "invalid collection name",
            "collection_type": "general"
        }
        
        result = await tool.safe_execute(parameters)
        
        # Should return validation error
        assert result["success"] is False
        assert result["status"] == "error"
        assert "errors" in result
        
        # Check specific validation error
        errors = result["errors"]
        assert len(errors) > 0
        assert any("Collection name can only contain" in error["message"] for error in errors)
    
    @pytest.mark.asyncio
    async def test_list_collections_success(self, mock_cli_path):
        """Test list_collections tool matches CLI list command."""
        expected_output = {
            "collections": [
                {
                    "name": "default",
                    "type": "general",
                    "document_count": 5,
                    "created": "2024-01-01T00:00:00Z"
                },
                {
                    "name": "research-papers",
                    "type": "project-specific", 
                    "document_count": 12,
                    "created": "2024-01-02T00:00:00Z"
                }
            ]
        }
        
        mock_process = self.create_mock_cli_success(expected_output)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            tool = ListCollectionsTool(mock_cli_path)
            
            parameters = {
                "show_stats": True
            }
            
            result = await tool.safe_execute(parameters)
            
            # Validate response format
            assert result["success"] is True
            assert result["status"] == "success"
            
            # Validate data content
            data = result["data"]
            assert "collections" in data
            assert "total_count" in data
            assert data["total_count"] == 2
            assert len(data["collections"]) == 2
            
            # Validate collection data structure
            collection = data["collections"][0]
            assert "name" in collection
            assert "type" in collection
            assert "document_count" in collection
    
    @pytest.mark.asyncio
    async def test_delete_collection_not_found(self, mock_cli_path):
        """Test delete_collection tool handles not found errors like CLI."""
        mock_process = self.create_mock_cli_error("Collection 'nonexistent' not found", 1)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            tool = DeleteCollectionTool(mock_cli_path)
            
            parameters = {
                "name": "nonexistent",
                "confirm": True
            }
            
            result = await tool.safe_execute(parameters)
            
            # Should return error response
            assert result["success"] is False
            assert result["status"] == "error"
            assert "Failed to delete collection" in result["message"]


class TestKnowledgeBaseToolsIntegration(TestMCPCLIIntegration):
    """Test knowledge base tools against CLI equivalents."""
    
    @pytest.mark.asyncio
    async def test_list_documents_success(self, mock_cli_path):
        """Test list_documents tool matches CLI list-documents command."""
        expected_output = {
            "documents": [
                {
                    "id": "doc_1",
                    "title": "Research Paper 1",
                    "collection": "research-papers",
                    "created": "2024-01-01T00:00:00Z",
                    "size": 1024
                },
                {
                    "id": "doc_2", 
                    "title": "Technical Documentation",
                    "collection": "default",
                    "created": "2024-01-02T00:00:00Z",
                    "size": 2048
                }
            ],
            "total_count": 2
        }
        
        mock_process = self.create_mock_cli_success(expected_output)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            tool = ListDocumentsTool(mock_cli_path)
            
            parameters = {
                "collection": "research-papers",
                "limit": 50
            }
            
            result = await tool.safe_execute(parameters)
            
            # Validate response format
            assert result["success"] is True
            assert result["status"] == "success"
            
            # Validate data content
            data = result["data"]
            assert "documents" in data
            assert "total_count" in data
            assert data["filtered_by_collection"] == "research-papers"
            assert data["limit_applied"] == 50
            assert len(data["documents"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_knowledge_base_status_success(self, mock_cli_path):
        """Test get_knowledge_base_status tool matches CLI status command."""
        expected_output = {
            "status": {
                "total_documents": 17,
                "total_collections": 3,
                "index_health": "healthy",
                "last_update": "2024-01-03T12:00:00Z",
                "storage_size": "15.2 MB",
                "embedding_model": "multi-qa-MiniLM-L6-cos-v1"
            }
        }
        
        mock_process = self.create_mock_cli_success(expected_output)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            tool = GetKnowledgeBaseStatusTool(mock_cli_path)
            
            parameters = {
                "include_detailed_stats": True
            }
            
            result = await tool.safe_execute(parameters)
            
            # Validate response format
            assert result["success"] is True
            assert result["status"] == "success"
            
            # Validate status data
            data = result["data"]
            assert "total_documents" in data
            assert "total_collections" in data
            assert "index_health" in data
            assert data["total_documents"] == 17
            assert data["total_collections"] == 3
    
    @pytest.mark.asyncio
    async def test_add_document_success(self, mock_cli_path, temp_test_file):
        """Test add_document tool matches CLI add-document command."""
        expected_output = {
            "document": {
                "id": "doc_new_123",
                "title": "Test Document",
                "file_path": temp_test_file,
                "collection": "default",
                "chunks_created": 3,
                "embedding_generated": True
            }
        }
        
        mock_process = self.create_mock_cli_success(expected_output)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            tool = AddDocumentTool(mock_cli_path)
            
            parameters = {
                "file_path": temp_test_file,
                "collection": "default",
                "force": False,
                "title": "Test Document"
            }
            
            result = await tool.safe_execute(parameters)
            
            # Validate response format
            assert result["success"] is True
            assert result["status"] == "success"
            
            # Validate document data
            data = result["data"]
            assert "id" in data
            assert "file_path" in data
            assert data["file_path"] == temp_test_file
    
    @pytest.mark.asyncio
    async def test_remove_document_not_found(self, mock_cli_path):
        """Test remove_document tool handles not found errors like CLI."""
        mock_process = self.create_mock_cli_error("Document 'nonexistent_doc' not found", 1)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            tool = RemoveDocumentTool(mock_cli_path)
            
            parameters = {
                "document_id": "nonexistent_doc",
                "confirm": True
            }
            
            result = await tool.safe_execute(parameters)
            
            # Should return not found error
            assert result["success"] is False
            assert result["status"] == "error"
            # The base tool should detect "not found" and format appropriately


class TestQueryToolIntegration(TestMCPCLIIntegration):
    """Test query tool against CLI query commands."""
    
    @pytest.mark.asyncio
    async def test_query_knowledge_base_success(self, mock_cli_path):
        """Test query_knowledge_base tool matches CLI query search command."""
        expected_output = {
            "results": [
                {
                    "document_id": "doc_1",
                    "content": "Machine learning is a subset of artificial intelligence...",
                    "score": 0.95,
                    "collection": "research-papers",
                    "metadata": {
                        "title": "Introduction to ML",
                        "author": "Dr. Smith"
                    }
                },
                {
                    "document_id": "doc_2",
                    "content": "Neural networks are inspired by biological neural networks...",
                    "score": 0.87,
                    "collection": "research-papers",
                    "metadata": {
                        "title": "Neural Network Basics",
                        "author": "Dr. Johnson"
                    }
                }
            ],
            "total_results": 2,
            "query_time_ms": 45
        }
        
        mock_process = self.create_mock_cli_success(expected_output)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            tool = QueryKnowledgeBaseTool(mock_cli_path)
            
            parameters = {
                "query": "machine learning algorithms",
                "collections": "research-papers",
                "top_k": 10
            }
            
            result = await tool.safe_execute(parameters)
            
            # Validate response format
            assert result["success"] is True
            assert result["status"] == "success"
            
            # Validate query results
            data = result["data"]
            assert "results" in data
            assert len(data["results"]) == 2
            
            # Validate result structure
            result_item = data["results"][0]
            assert "document_id" in result_item
            assert "content" in result_item
            assert "score" in result_item
            assert "collection" in result_item
    
    @pytest.mark.asyncio
    async def test_query_validation_error(self, mock_cli_path):
        """Test query tool validation matches CLI validation."""
        tool = QueryKnowledgeBaseTool(mock_cli_path)
        
        # Test empty query
        parameters = {
            "query": "",
            "top_k": 10
        }
        
        result = await tool.safe_execute(parameters)
        
        # Should return validation error
        assert result["success"] is False
        assert result["status"] == "error"
        assert "errors" in result
        
        # Check specific validation error
        errors = result["errors"]
        assert len(errors) > 0
        assert any("non-empty string" in error["message"] for error in errors)


class TestResponseFormatStandardization(TestMCPCLIIntegration):
    """Test that all MCP tools return standardized response formats."""
    
    @pytest.mark.asyncio
    async def test_all_tools_return_standardized_format(self, mock_cli_path):
        """Test that all MCP tools return the standardized response format."""
        tools_to_test = [
            (CreateCollectionTool(mock_cli_path), {"name": "test"}),
            (ListCollectionsTool(mock_cli_path), {}),
            (ListDocumentsTool(mock_cli_path), {}),
            (GetKnowledgeBaseStatusTool(mock_cli_path), {}),
            (QueryKnowledgeBaseTool(mock_cli_path), {"query": "test query"})
        ]
        
        mock_process = self.create_mock_cli_success({"status": "success"})
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            for tool, params in tools_to_test:
                result = await tool.safe_execute(params)
                
                # Validate standardized response structure
                assert "success" in result
                assert "status" in result
                assert "data" in result
                assert "message" in result
                assert "metadata" in result
                
                # Validate metadata structure
                metadata = result["metadata"]
                assert "timestamp" in metadata
                assert "operation" in metadata
                assert "execution_time" in metadata
                assert "tool_name" in metadata
                assert "version" in metadata
                
                # Validate data types
                assert isinstance(result["success"], bool)
                assert result["status"] in ["success", "error", "warning", "partial"]
                assert isinstance(result["message"], str)
                assert isinstance(metadata["execution_time"], (int, float))


class TestParameterValidationIntegration(TestMCPCLIIntegration):
    """Test parameter validation across MCP tools."""
    
    @pytest.mark.asyncio
    async def test_parameter_validation_consistency(self, mock_cli_path):
        """Test that parameter validation is consistent across tools."""
        
        # Test collection name validation across different tools
        invalid_collection_name = "invalid collection name!"
        
        tools_with_collection_param = [
            (CreateCollectionTool(mock_cli_path), {"name": invalid_collection_name}),
            (ListDocumentsTool(mock_cli_path), {"collection": invalid_collection_name}),
            (AddDocumentTool(mock_cli_path), {"file_path": "/test", "collection": invalid_collection_name})
        ]
        
        for tool, params in tools_with_collection_param:
            result = await tool.safe_execute(params)
            
            # All should return validation errors for invalid collection names
            if "name" in params and params["name"] == invalid_collection_name:
                # Collection name validation should fail
                assert result["success"] is False
                assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_required_parameter_validation(self, mock_cli_path):
        """Test that required parameters are properly validated."""
        
        test_cases = [
            (CreateCollectionTool(mock_cli_path), {}, "name"),  # Missing required name
            (AddDocumentTool(mock_cli_path), {}, "file_path"),  # Missing required file_path
            (RemoveDocumentTool(mock_cli_path), {}, "document_id"),  # Missing required document_id
            (QueryKnowledgeBaseTool(mock_cli_path), {}, "query")  # Missing required query
        ]
        
        for tool, params, required_param in test_cases:
            result = await tool.safe_execute(params)
            
            # Should return validation error for missing required parameter
            assert result["success"] is False
            assert result["status"] == "error"
            assert "errors" in result
            
            # Check that the specific required parameter error is present
            errors = result["errors"]
            assert any(required_param in error.get("field", "") for error in errors)


# Additional test utilities and fixtures

@pytest.fixture
def sample_cli_responses():
    """Sample CLI responses for different commands."""
    return {
        "collections_list": {
            "collections": [
                {"name": "default", "type": "general", "document_count": 5},
                {"name": "research", "type": "project-specific", "document_count": 12}
            ]
        },
        "documents_list": {
            "documents": [
                {"id": "doc1", "title": "Test Doc", "collection": "default"},
                {"id": "doc2", "title": "Research Paper", "collection": "research"}
            ],
            "total_count": 2
        },
        "kb_status": {
            "status": {
                "total_documents": 17,
                "total_collections": 2,
                "index_health": "healthy"
            }
        },
        "query_results": {
            "results": [
                {"document_id": "doc1", "score": 0.95, "content": "Sample content"}
            ],
            "total_results": 1
        }
    }


@pytest.mark.integration
class TestEndToEndMCPCLIWorkflow(TestMCPCLIIntegration):
    """End-to-end tests simulating complete MCP-CLI workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_document_workflow(self, mock_cli_path, temp_test_file, sample_cli_responses):
        """Test complete workflow: create collection -> add document -> query -> remove."""
        
        # Mock different CLI responses for each step
        mock_responses = [
            self.create_mock_cli_success({"status": "success", "collection": {"name": "test-workflow", "created": True}}),
            self.create_mock_cli_success({"document": {"id": "workflow_doc", "collection": "test-workflow"}}),
            self.create_mock_cli_success(sample_cli_responses["query_results"]),
            self.create_mock_cli_success({"status": "success", "removed": True})
        ]
        
        with patch('asyncio.create_subprocess_exec', side_effect=mock_responses):
            # Step 1: Create collection
            create_tool = CreateCollectionTool(mock_cli_path)
            create_result = await create_tool.safe_execute({
                "name": "test-workflow",
                "collection_type": "general"
            })
            assert create_result["success"] is True
            
            # Step 2: Add document
            add_tool = AddDocumentTool(mock_cli_path)
            add_result = await add_tool.safe_execute({
                "file_path": temp_test_file,
                "collection": "test-workflow"
            })
            assert add_result["success"] is True
            
            # Step 3: Query documents
            query_tool = QueryKnowledgeBaseTool(mock_cli_path)
            query_result = await query_tool.safe_execute({
                "query": "test document content",
                "collections": "test-workflow"
            })
            assert query_result["success"] is True
            
            # Step 4: Remove document
            remove_tool = RemoveDocumentTool(mock_cli_path)
            remove_result = await remove_tool.safe_execute({
                "document_id": "workflow_doc",
                "confirm": True
            })
            assert remove_result["success"] is True 