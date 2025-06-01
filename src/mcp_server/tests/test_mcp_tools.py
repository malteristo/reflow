"""
Test suite for MCP Tools.

Tests the Research Agent MCP tools that map CLI commands to MCP protocol.
Follows TDD methodology for subtask 15.3: Map CLI Tools to MCP Resources and Actions.
"""

import pytest
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# These imports will fail initially - this is expected in TDD RED phase
try:
    from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
    from src.mcp_server.tools.collections_tool import ManageCollectionsTool
    from src.mcp_server.tools.documents_tool import IngestDocumentsTool
    from src.mcp_server.tools.projects_tool import ManageProjectsTool
    from src.mcp_server.tools.augment_tool import AugmentKnowledgeTool
    from src.mcp_server.tools.base_tool import BaseMCPTool
except ImportError:
    # Expected during RED phase
    pass


class TestBaseMCPTool:
    """Test the base MCP tool functionality."""
    
    def test_base_tool_exists(self):
        """Test that BaseMCPTool class exists and can be imported."""
        from src.mcp_server.tools.base_tool import BaseMCPTool
        assert BaseMCPTool is not None
    
    def test_base_tool_initialization(self):
        """Test BaseMCPTool can be initialized through concrete implementation."""
        from src.mcp_server.tools.base_tool import BaseMCPTool
        
        class TestTool(BaseMCPTool):
            def get_tool_name(self):
                return "test_tool"
            def get_tool_description(self):
                return "Test tool description"
            def execute(self, parameters):
                return {"status": "success", "message": "test"}
        
        tool = TestTool()
        assert tool is not None
        assert tool.get_tool_name() == "test_tool"
    
    def test_base_tool_has_required_methods(self):
        """Test BaseMCPTool has required abstract methods."""
        from src.mcp_server.tools.base_tool import BaseMCPTool
        
        class TestTool(BaseMCPTool):
            def get_tool_name(self):
                return "test_tool"
            def get_tool_description(self):
                return "Test tool description"
            def execute(self, parameters):
                return {"status": "success"}
        
        tool = TestTool()
        
        # Should have these methods
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'validate_parameters')
        assert hasattr(tool, 'get_tool_name')
        assert hasattr(tool, 'get_tool_description')
        assert callable(tool.execute)
        assert callable(tool.validate_parameters)
    
    def test_base_tool_parameter_validation(self):
        """Test parameter validation functionality."""
        from src.mcp_server.tools.base_tool import BaseMCPTool
        
        class TestTool(BaseMCPTool):
            def get_tool_name(self):
                return "test_tool"
            def get_tool_description(self):
                return "Test tool description"
            def execute(self, parameters):
                return {"status": "success"}
            def get_required_parameters(self):
                return ["query"]
        
        tool = TestTool()
        
        # Test with valid parameters
        valid_params = {"query": "test query", "collections": "test-collection"}
        result = tool.validate_parameters(valid_params)
        assert isinstance(result, list)
        assert len(result) == 0  # No errors
        
        # Test with missing required parameter
        invalid_params = {"collections": "test-collection"}  # Missing query
        result = tool.validate_parameters(invalid_params)
        assert len(result) > 0  # Should have errors
    
    def test_base_tool_error_handling(self):
        """Test error handling in base tool."""
        from src.mcp_server.tools.base_tool import BaseMCPTool
        
        class TestTool(BaseMCPTool):
            def get_tool_name(self):
                return "test_tool"
            def get_tool_description(self):
                return "Test tool description"
            def execute(self, parameters):
                return {"status": "success"}
        
        tool = TestTool()
        
        # Test error formatting
        error = tool.format_error("Test error", {"param": "value"})
        assert isinstance(error, dict)
        assert error["status"] == "error"
        assert "error" in error
        assert "message" in error["error"]
        assert error["error"]["message"] == "Test error"


class TestQueryKnowledgeBaseTool:
    """Test the query knowledge base MCP tool."""
    
    def test_query_tool_exists(self):
        """Test that QueryKnowledgeBaseTool exists."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        assert QueryKnowledgeBaseTool is not None
    
    def test_query_tool_initialization(self):
        """Test QueryKnowledgeBaseTool initialization."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        tool = QueryKnowledgeBaseTool()
        assert tool is not None
        assert tool.get_tool_name() == "query_knowledge_base"
    
    def test_query_tool_parameter_validation(self):
        """Test parameter validation for query tool."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        tool = QueryKnowledgeBaseTool()
        
        # Valid parameters
        valid_params = {
            "query": "What is machine learning?",
            "collections": "ml-papers",
            "top_k": 10
        }
        errors = tool.validate_parameters(valid_params)
        assert len(errors) == 0
        
        # Invalid parameters
        invalid_params = {
            "query": "",  # Empty query
            "top_k": -1   # Invalid top_k
        }
        errors = tool.validate_parameters(invalid_params)
        assert len(errors) > 0
    
    @patch('src.research_agent_backend.cli.query.search')
    def test_query_tool_execution(self, mock_search):
        """Test query tool execution with CLI integration."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        
        # Mock CLI response
        mock_search.return_value = {
            "results": [
                {
                    "score": 0.95,
                    "document_id": "doc1",
                    "content": "Machine learning is...",
                    "collection": "ml-papers"
                }
            ],
            "total_results": 1
        }
        
        tool = QueryKnowledgeBaseTool()
        params = {
            "query": "What is machine learning?",
            "collections": "ml-papers",
            "top_k": 10
        }
        
        result = tool.execute(params)
        assert result["status"] == "success"
        assert "results" in result
        assert len(result["results"]) > 0
    
    def test_query_tool_error_handling(self):
        """Test query tool error handling."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        tool = QueryKnowledgeBaseTool()
        
        # Test with missing required parameter
        invalid_params = {"top_k": 10}  # Missing query
        result = tool.execute(invalid_params)
        assert result["status"] == "error"
        assert "error" in result


class TestManageCollectionsTool:
    """Test the manage collections MCP tool."""
    
    def test_collections_tool_exists(self):
        """Test that ManageCollectionsTool exists."""
        from src.mcp_server.tools.collections_tool import ManageCollectionsTool
        assert ManageCollectionsTool is not None
    
    def test_collections_tool_initialization(self):
        """Test ManageCollectionsTool initialization."""
        from src.mcp_server.tools.collections_tool import ManageCollectionsTool
        tool = ManageCollectionsTool()
        assert tool is not None
        assert tool.get_tool_name() == "manage_collections"
    
    def test_collections_tool_actions(self):
        """Test collections tool supports required actions."""
        from src.mcp_server.tools.collections_tool import ManageCollectionsTool
        tool = ManageCollectionsTool()
        
        supported_actions = tool.get_supported_actions()
        expected_actions = ["create", "list", "delete", "info", "rename"]
        
        for action in expected_actions:
            assert action in supported_actions
    
    @patch('src.research_agent_backend.cli.collections.create_collection')
    def test_collections_create_action(self, mock_create):
        """Test collections tool create action."""
        from src.mcp_server.tools.collections_tool import ManageCollectionsTool
        
        mock_create.return_value = {"status": "success", "collection": "test-collection"}
        
        tool = ManageCollectionsTool()
        params = {
            "action": "create",
            "collection_name": "test-collection",
            "description": "Test collection"
        }
        
        result = tool.execute(params)
        assert result["status"] == "success"
        mock_create.assert_called_once()
    
    @patch('src.research_agent_backend.cli.collections.list_collections')
    def test_collections_list_action(self, mock_list):
        """Test collections tool list action."""
        from src.mcp_server.tools.collections_tool import ManageCollectionsTool
        
        mock_list.return_value = {
            "collections": [
                {"name": "collection1", "type": "general", "count": 10},
                {"name": "collection2", "type": "project", "count": 5}
            ]
        }
        
        tool = ManageCollectionsTool()
        params = {"action": "list"}
        
        result = tool.execute(params)
        assert result["status"] == "success"
        assert "collections" in result
        assert len(result["collections"]) == 2


class TestIngestDocumentsTool:
    """Test the ingest documents MCP tool."""
    
    def test_ingest_tool_exists(self):
        """Test that IngestDocumentsTool exists."""
        from src.mcp_server.tools.documents_tool import IngestDocumentsTool
        assert IngestDocumentsTool is not None
    
    def test_ingest_tool_initialization(self):
        """Test IngestDocumentsTool initialization."""
        from src.mcp_server.tools.documents_tool import IngestDocumentsTool
        tool = IngestDocumentsTool()
        assert tool is not None
        assert tool.get_tool_name() == "ingest_documents"
    
    def test_ingest_tool_parameter_validation(self):
        """Test parameter validation for ingest tool."""
        from src.mcp_server.tools.documents_tool import IngestDocumentsTool
        tool = IngestDocumentsTool()
        
        # Valid parameters
        valid_params = {
            "path": "/path/to/document.md",
            "collection": "test-collection"
        }
        errors = tool.validate_parameters(valid_params)
        assert len(errors) == 0
        
        # Invalid parameters
        invalid_params = {
            "path": "",  # Empty path
            "collection": "invalid@collection"  # Invalid collection name
        }
        errors = tool.validate_parameters(invalid_params)
        assert len(errors) > 0
    
    @patch('src.research_agent_backend.cli.knowledge_base.add_document')
    def test_ingest_single_document(self, mock_add):
        """Test ingesting a single document."""
        from src.mcp_server.tools.documents_tool import IngestDocumentsTool
        
        mock_add.return_value = {
            "status": "success",
            "document_id": "doc123",
            "chunks_created": 5
        }
        
        tool = IngestDocumentsTool()
        params = {
            "path": "/path/to/document.md",
            "collection": "test-collection"
        }
        
        result = tool.execute(params)
        assert result["status"] == "success"
        assert "document_id" in result
        mock_add.assert_called_once()
    
    @patch('src.research_agent_backend.cli.knowledge_base.ingest_folder')
    def test_ingest_folder(self, mock_ingest):
        """Test ingesting a folder of documents."""
        from src.mcp_server.tools.documents_tool import IngestDocumentsTool
        
        mock_ingest.return_value = {
            "status": "success",
            "documents_processed": 10,
            "total_chunks": 50
        }
        
        tool = IngestDocumentsTool()
        params = {
            "path": "/path/to/folder",
            "collection": "test-collection",
            "recursive": True,
            "pattern": "*.md"
        }
        
        result = tool.execute(params)
        assert result["status"] == "success"
        assert "documents_processed" in result
        mock_ingest.assert_called_once()


class TestManageProjectsTool:
    """Test the manage projects MCP tool."""
    
    def test_projects_tool_exists(self):
        """Test that ManageProjectsTool exists."""
        from src.mcp_server.tools.projects_tool import ManageProjectsTool
        assert ManageProjectsTool is not None
    
    def test_projects_tool_initialization(self):
        """Test ManageProjectsTool initialization."""
        from src.mcp_server.tools.projects_tool import ManageProjectsTool
        tool = ManageProjectsTool()
        assert tool is not None
        assert tool.get_tool_name() == "manage_projects"
    
    def test_projects_tool_actions(self):
        """Test projects tool supports required actions."""
        from src.mcp_server.tools.projects_tool import ManageProjectsTool
        tool = ManageProjectsTool()
        
        supported_actions = tool.get_supported_actions()
        expected_actions = ["create", "list", "info", "activate", "archive", "delete"]
        
        for action in expected_actions:
            assert action in supported_actions
    
    @patch('src.research_agent_backend.cli.projects.create_project')
    def test_projects_create_action(self, mock_create):
        """Test projects tool create action."""
        from src.mcp_server.tools.projects_tool import ManageProjectsTool
        
        mock_create.return_value = {"status": "success", "project": "test-project"}
        
        tool = ManageProjectsTool()
        params = {
            "action": "create",
            "project_name": "test-project",
            "description": "Test project"
        }
        
        result = tool.execute(params)
        assert result["status"] == "success"
        mock_create.assert_called_once()


class TestAugmentKnowledgeTool:
    """Test the augment knowledge MCP tool."""
    
    def test_augment_tool_exists(self):
        """Test that AugmentKnowledgeTool exists."""
        from src.mcp_server.tools.augment_tool import AugmentKnowledgeTool
        assert AugmentKnowledgeTool is not None
    
    def test_augment_tool_initialization(self):
        """Test AugmentKnowledgeTool initialization."""
        from src.mcp_server.tools.augment_tool import AugmentKnowledgeTool
        tool = AugmentKnowledgeTool()
        assert tool is not None
        assert tool.get_tool_name() == "augment_knowledge"
    
    def test_augment_tool_actions(self):
        """Test augment tool supports required actions."""
        from src.mcp_server.tools.augment_tool import AugmentKnowledgeTool
        tool = AugmentKnowledgeTool()
        
        supported_actions = tool.get_supported_actions()
        expected_actions = ["add_external_result", "add_research_report", "feedback"]
        
        for action in expected_actions:
            assert action in supported_actions
    
    @patch('src.research_agent_backend.cli.augmentation.add_external_result')
    def test_augment_external_result(self, mock_add):
        """Test augment tool add external result action."""
        from src.mcp_server.tools.augment_tool import AugmentKnowledgeTool
        
        mock_add.return_value = {"status": "success", "result_id": "result123"}
        
        tool = AugmentKnowledgeTool()
        params = {
            "action": "add_external_result",
            "query": "test query",
            "result_data": {"title": "Test Result", "content": "Test content"}
        }
        
        result = tool.execute(params)
        assert result["status"] == "success"
        mock_add.assert_called_once()


class TestMCPToolIntegration:
    """Test MCP tool integration with FastMCP server."""
    
    def test_all_tools_can_be_registered(self):
        """Test that all tools can be registered with FastMCP."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        from src.mcp_server.tools.collections_tool import ManageCollectionsTool
        from src.mcp_server.tools.documents_tool import IngestDocumentsTool
        from src.mcp_server.tools.projects_tool import ManageProjectsTool
        from src.mcp_server.tools.augment_tool import AugmentKnowledgeTool
        
        tools = [
            QueryKnowledgeBaseTool(),
            ManageCollectionsTool(),
            IngestDocumentsTool(),
            ManageProjectsTool(),
            AugmentKnowledgeTool()
        ]
        
        # Each tool should have required methods for FastMCP registration
        for tool in tools:
            assert hasattr(tool, 'get_tool_name')
            assert hasattr(tool, 'get_tool_description')
            assert hasattr(tool, 'execute')
            assert callable(tool.execute)
    
    def test_tool_parameter_schemas(self):
        """Test that tools provide proper parameter schemas."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        
        tool = QueryKnowledgeBaseTool()
        schema = tool.get_parameter_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema
        assert "required" in schema
    
    def test_tool_response_format(self):
        """Test that tools return properly formatted responses."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        
        tool = QueryKnowledgeBaseTool()
        
        # Test success response format
        with patch('src.research_agent_backend.cli.query.search') as mock_search:
            mock_search.return_value = {"results": [], "total_results": 0}
            
            result = tool.execute({"query": "test"})
            assert isinstance(result, dict)
            assert "status" in result
            assert result["status"] in ["success", "error"]
    
    def test_error_response_consistency(self):
        """Test that all tools return consistent error responses."""
        from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
        from src.mcp_server.tools.collections_tool import ManageCollectionsTool
        
        tools = [QueryKnowledgeBaseTool(), ManageCollectionsTool()]
        
        for tool in tools:
            # Test with invalid parameters
            result = tool.execute({})  # Empty parameters
            assert isinstance(result, dict)
            assert result["status"] == "error"
            assert "error" in result
            assert "message" in result["error"] 