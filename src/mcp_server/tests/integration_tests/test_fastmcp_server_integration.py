"""
Comprehensive integration tests for FastMCP server implementation.

Tests end-to-end functionality including STDIO communication, protocol compliance,
CLI tool integration, error handling, and feedback systems.
"""

import asyncio
import json
import pytest
import subprocess
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import io
import sys
from typing import Dict, Any, List, Optional
from io import StringIO

# Import MCP server components
from src.mcp_server.server import MCPServer
from src.mcp_server.communication.stdio_handler import StdioHandler
from src.mcp_server.communication.message_processor import MessageProcessor
from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
from src.mcp_server.tools.collections_tool import ManageCollectionsTool
from src.mcp_server.tools.documents_tool import IngestDocumentsTool
from src.mcp_server.tools.projects_tool import ManageProjectsTool
from src.mcp_server.tools.augment_tool import AugmentKnowledgeTool


class TestFastMCPServerIntegration:
    """Integration tests for complete FastMCP server functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_stdin = Mock()
        self.mock_stdout = Mock()
        self.mock_stderr = Mock()
        
    def teardown_method(self):
        """Clean up test environment after each test."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_mcp_server_initialization_and_startup(self):
        """Test that the FastMCP server properly initializes and starts up."""
        # Initialize the server
        server = MCPServer("test-server", "0.1.0")
        
        # Verify initial state
        assert not server.is_initialized
        assert not server.is_running
        assert server.name == "test-server"
        assert server.version == "0.1.0"
        
        # Initialize the server
        await server.initialize()
        
        # Verify initialization
        assert server.is_initialized
        assert not server.is_running  # Not started yet
        
        # Start STDIO communication
        await server.start_stdio()
        
        # Verify running state
        assert server.is_running
        
        # Test tool registration
        tool_names = server.get_tool_names()
        expected_tools = [
            "ping", "server_info", 
            "query_knowledge_base", "manage_collections", 
            "ingest_documents", "manage_projects", "augment_knowledge"
        ]
        for tool in expected_tools:
            assert tool in tool_names
        
        # Test supported transports
        transports = server.get_supported_transports()
        assert "stdio" in transports
        
        # Shutdown the server
        await server.shutdown()
        
        # Verify shutdown state
        assert not server.is_initialized
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_stdio_protocol_compliance(self):
        """Test that STDIO communication follows MCP protocol standards."""
        # Create server with custom streams for testing
        input_stream = StringIO()
        output_stream = StringIO()
        error_stream = StringIO()
        
        # Create STDIO handler with custom streams
        handler = StdioHandler(
            timeout=30.0,
            input_stream=input_stream,
            output_stream=output_stream,
            error_stream=error_stream
        )
        
        # Test valid JSON-RPC 2.0 request
        valid_request = {
            "jsonrpc": "2.0",
            "method": "ping",
            "id": 1,
            "params": {}
        }
        
        # Process the message
        response = await handler.process_message(json.dumps(valid_request))
        
        # Verify response structure
        assert response["jsonrpc"] == "2.0"
        assert "id" in response
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["message"] == "pong"
        
        # Test invalid JSON-RPC request (missing version)
        invalid_request = {
            "method": "ping",
            "id": 2
        }
        
        error_response = await handler.process_message(json.dumps(invalid_request))
        
        # Verify error response structure
        assert error_response["jsonrpc"] == "2.0"
        assert "error" in error_response
        assert error_response["error"]["code"] == -32600  # Invalid Request
        
        # Test message framing
        test_message = {"test": "data"}
        framed = handler.frame_message(test_message)
        assert framed.endswith("\n")
        assert json.loads(framed.strip()) == test_message
        
        # Test protocol validation with MessageProcessor
        processor = MessageProcessor()
        
        # Test valid response validation
        valid_response = {
            "jsonrpc": "2.0",
            "result": {"message": "test"},
            "id": 1
        }
        is_valid = await processor.validate_mcp_response(valid_response)
        assert is_valid
        
        # Test invalid response validation (both result and error)
        invalid_response = {
            "jsonrpc": "2.0",
            "result": {"message": "test"},
            "error": {"code": -1, "message": "error"},
            "id": 1
        }
        is_valid = await processor.validate_mcp_response(invalid_response)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_query_knowledge_base_tool_integration(self):
        """Test end-to-end query knowledge base tool functionality."""
        # Initialize tool with mock CLI backend
        tool = QueryKnowledgeBaseTool()
        
        # Test parameters
        params = {
            "query": "What is the capital of France?",
            "collections": ["general"],
            "top_k": 10
        }
        
        # Mock CLI execution to return expected results
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # Create mock process
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(
                json.dumps({
                    "status": "success",
                    "results": [
                        {
                            "content": "Paris is the capital of France",
                            "metadata": {"source": "geography.md", "score": 0.95}
                        }
                    ]
                }).encode(),
                b""
            ))
            mock_exec.return_value = mock_process
            
            # Execute tool
            result = await tool.execute(params)
            
            # Validate result structure
            assert result["status"] == "success"
            assert "results" in result
            assert len(result["results"]) > 0
            
            # Validate CLI command was called correctly
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[0]
            assert "research-agent-cli" in call_args
            assert "query" in call_args
            assert "What is the capital of France?" in call_args
        
        # Remove the failing assertion at the end
        # Test should now pass if implementation is correct

    @pytest.mark.asyncio
    async def test_collections_management_tool_integration(self):
        """Test end-to-end collections management tool functionality."""
        # Initialize tool with mock CLI backend
        tool = ManageCollectionsTool()
        
        # Test parameters for creating a collection
        params = {
            "action": "create",
            "collection_name": "test_collection",
            "description": "A test collection",
            "collection_type": "general"
        }
        
        # Mock CLI execution to return expected results
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # Create mock process
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(
                json.dumps({
                    "status": "success",
                    "collection_name": "test_collection",
                    "message": "Collection created successfully"
                }).encode(),
                b""
            ))
            mock_exec.return_value = mock_process
            
            # Execute tool
            result = await tool.execute(params)
            
            # Validate result structure
            assert result["status"] == "success"
            assert "action" in result
            assert result["action"] == "create"
            assert "collection_name" in result
            
            # Validate CLI command was called correctly
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[0]
            assert "research-agent-cli" in call_args
            assert "collections" in call_args
            assert "create" in call_args
            assert "test_collection" in call_args

    @pytest.mark.asyncio
    async def test_document_ingestion_tool_integration(self):
        """Test end-to-end document ingestion tool functionality."""
        # Initialize tool with mock CLI backend
        tool = IngestDocumentsTool()
        
        # Test parameters for document ingestion
        params = {
            "action": "add_document",
            "path": "/path/to/document.md",
            "collection": "test_collection"
        }
        
        # Mock CLI execution to return expected results
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # Create mock process
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(
                json.dumps({
                    "status": "success",
                    "document_id": "doc_123",
                    "path": "/path/to/document.md",
                    "collection": "test_collection",
                    "chunks_created": 5,
                    "message": "Document added successfully"
                }).encode(),
                b""
            ))
            mock_exec.return_value = mock_process
            
            # Execute tool
            result = await tool.execute(params)
            
            # Validate result structure
            assert result["status"] == "success"
            assert "action" in result
            assert result["action"] == "add_document"
            assert "document_id" in result
            assert "chunks_created" in result
            
            # Validate CLI command was called correctly
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[0]
            assert "research-agent-cli" in call_args
            assert "kb" in call_args
            assert "add-document" in call_args
            assert "/path/to/document.md" in call_args

    @pytest.mark.asyncio
    async def test_projects_management_tool_integration(self):
        """Test end-to-end projects management tool functionality."""
        tool = ManageProjectsTool()
        
        # Test create project
        params = {
            "action": "create",
            "project_name": "test-project",
            "description": "Integration test project"
        }
        
        # Mock CLI execution to return expected results
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # Create mock process
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(
                json.dumps({
                    "status": "success",
                    "project_name": "test-project",
                    "message": "Project 'test-project' created successfully"
                }).encode(),
                b""
            ))
            mock_exec.return_value = mock_process
            
            # Execute tool
            result = await tool.execute(params)
            
            # Validate result structure
            assert result["status"] == "success"
            assert "action" in result
            assert result["action"] == "create"
            assert "project_name" in result
            
            # Validate CLI command was called correctly
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[0]
            assert "research-agent-cli" in call_args
            assert "projects" in call_args
            assert "create" in call_args
            assert "test-project" in call_args

    @pytest.mark.asyncio
    async def test_knowledge_augmentation_tool_integration(self):
        """Test end-to-end knowledge augmentation tool functionality."""
        tool = AugmentKnowledgeTool()
        
        # Test add external result
        params = {
            "action": "add-external-result",
            "query": "Machine learning algorithms",
            "result": {
                "content": "Neural networks are a type of machine learning algorithm",
                "source": "external research"
            },
            "collection": "research"
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({
                "status": "success",
                "message": "External result added successfully"
            })
            mock_run.return_value = mock_result
            
            result = await tool.execute(params)
            assert result["status"] == "success"
        
        # This will fail until proper tool integration is implemented
        assert False, "Knowledge augmentation tool integration not implemented"

    @pytest.mark.asyncio
    async def test_parameter_validation_integration(self):
        """Test parameter validation works correctly across all tools."""
        tool = QueryKnowledgeBaseTool()
        
        # Test invalid parameters trigger validation errors
        invalid_params = {
            "query": "",  # Empty query should fail
            "top_k": 1000  # Too large top_k should fail
        }
        
        with pytest.raises(Exception) as exc_info:
            await tool.execute(invalid_params)
        
        # Should get proper validation error
        assert "validation" in str(exc_info.value).lower()
        
        # This will fail until proper validation integration is implemented
        assert False, "Parameter validation integration not implemented"

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling works correctly across the entire system."""
        tool = QueryKnowledgeBaseTool()
        
        # Test CLI command failure handling
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1  # Command failed
            mock_result.stderr = "Command execution failed"
            mock_run.return_value = mock_result
            
            result = await tool.execute({"query": "test", "collections": ["test"]})
            
            # Should get properly formatted error response
            assert result["status"] == "error"
            assert "error" in result
            assert result["error"]["code"] in [-32001, -32002, -32003, -32004]
        
        # This will fail until proper error handling integration is implemented
        assert False, "Error handling integration not implemented"

    @pytest.mark.asyncio
    async def test_response_formatting_integration(self):
        """Test response formatting works correctly for all tool types."""
        tool = QueryKnowledgeBaseTool()
        
        # Mock successful CLI execution
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({
                "status": "success",
                "results": [{"content": "Test result", "score": 0.9}]
            })
            mock_run.return_value = mock_result
            
            result = await tool.execute({"query": "test", "collections": ["test"]})
            
            # Validate response format follows MCP protocol
            assert "content" in result
            assert isinstance(result["content"], list)
            
            for content_item in result["content"]:
                assert "type" in content_item
                assert "text" in content_item
                assert content_item["type"] in ["text", "markdown", "json"]
        
        # This will fail until proper response formatting integration is implemented
        assert False, "Response formatting integration not implemented"

    @pytest.mark.asyncio
    async def test_feedback_system_integration(self):
        """Test structured feedback and progress reporting integration."""
        tool = IngestDocumentsTool()  # Use ingestion for progress testing
        
        progress_events = []
        
        def progress_callback(event):
            progress_events.append(event)
        
        # Test folder ingestion with progress reporting
        params = {
            "action": "ingest-folder",
            "folder_path": self.test_dir,
            "collection": "test",
            "progress_callback": progress_callback
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({
                "status": "success",
                "message": "Folder ingested successfully",
                "files_processed": 5
            })
            mock_run.return_value = mock_result
            
            result = await tool.execute(params)
            
            # Should have progress events
            assert len(progress_events) > 0
            
            # Validate progress event structure
            for event in progress_events:
                assert "progress" in event
                assert "stage" in event
                assert "message" in event
        
        # This will fail until proper feedback system integration is implemented
        assert False, "Feedback system integration not implemented"

    @pytest.mark.asyncio
    async def test_concurrent_operations_integration(self):
        """Test server can handle multiple concurrent operations correctly."""
        server = MCPServer()
        await server.initialize()
        
        # Create multiple concurrent requests
        queries = [
            {"query": f"test query {i}", "collections": ["test"]}
            for i in range(5)
        ]
        
        # Execute queries concurrently
        tasks = []
        for query in queries:
            tool = QueryKnowledgeBaseTool()
            task = asyncio.create_task(tool.execute(query))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert result["status"] == "success"
        
        # This will fail until proper concurrent operation support is implemented
        assert False, "Concurrent operations integration not implemented"

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_integration(self):
        """Test complete end-to-end workflow from client request to response."""
        # Simulate complete workflow:
        # 1. Create collection
        # 2. Ingest documents
        # 3. Query knowledge base
        # 4. Get contextual feedback
        
        # Step 1: Create collection
        collections_tool = ManageCollectionsTool()
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success"})
            mock_run.return_value = mock_result
            
            collection_result = await collections_tool.execute({
                "action": "create",
                "name": "integration-test",
                "type": "general"
            })
            assert collection_result["status"] == "success"
        
        # Step 2: Ingest documents
        documents_tool = IngestDocumentsTool()
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success", "chunks_created": 10})
            mock_run.return_value = mock_result
            
            ingest_result = await documents_tool.execute({
                "action": "ingest-folder",
                "folder_path": self.test_dir,
                "collection": "integration-test"
            })
            assert ingest_result["status"] == "success"
        
        # Step 3: Query knowledge base
        query_tool = QueryKnowledgeBaseTool()
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({
                "status": "success",
                "results": [{"content": "Integration test result", "score": 0.9}],
                "feedback": {
                    "suggestions": ["Try broader search terms"],
                    "confidence": "high"
                }
            })
            mock_run.return_value = mock_result
            
            query_result = await query_tool.execute({
                "query": "integration testing",
                "collections": ["integration-test"]
            })
            assert query_result["status"] == "success"
            assert "feedback" in query_result
        
        # This will fail until proper end-to-end workflow integration is implemented
        assert False, "End-to-end workflow integration not implemented"

    @pytest.mark.asyncio
    async def test_security_validation_integration(self):
        """Test security validation works across the entire request pipeline."""
        tool = IngestDocumentsTool()
        
        # Test path traversal attack prevention
        malicious_params = {
            "action": "add-document",
            "file_path": "../../etc/passwd",  # Path traversal attempt
            "collection": "test"
        }
        
        with pytest.raises(Exception) as exc_info:
            await tool.execute(malicious_params)
        
        # Should get security validation error
        assert "security" in str(exc_info.value).lower() or "path" in str(exc_info.value).lower()
        
        # This will fail until proper security validation integration is implemented
        assert False, "Security validation integration not implemented"

    @pytest.mark.asyncio
    async def test_performance_under_load_integration(self):
        """Test system performance under load conditions."""
        server = MCPServer()
        await server.initialize()
        
        # Create high load scenario
        num_concurrent_requests = 50
        
        async def make_request():
            tool = QueryKnowledgeBaseTool()
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps({"status": "success", "results": []})
                mock_run.return_value = mock_result
                
                return await tool.execute({"query": "load test", "collections": ["test"]})
        
        # Measure performance
        import time
        start_time = time.time()
        
        tasks = [make_request() for _ in range(num_concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Validate performance criteria
        assert duration < 30.0  # Should complete within 30 seconds
        assert len([r for r in results if not isinstance(r, Exception)]) >= num_concurrent_requests * 0.95  # 95% success rate
        
        # This will fail until proper performance optimization is implemented
        assert False, "Performance under load integration not implemented"


class TestMCPProtocolCompliance:
    """Specific tests for MCP protocol compliance validation."""
    
    @pytest.mark.asyncio
    async def test_json_rpc_message_format_compliance(self):
        """Test all messages follow JSON-RPC 2.0 format specifications."""
        processor = MessageProcessor()
        
        # Test valid request format
        valid_request = {
            "jsonrpc": "2.0",
            "id": "req-001",
            "method": "tools/list",
            "params": {}
        }
        
        # Should parse without errors
        parsed = await processor.parse_request(json.dumps(valid_request))
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == "req-001"
        
        # Test invalid request format
        invalid_request = {
            "id": "req-002",
            "method": "tools/list"
            # Missing jsonrpc field
        }
        
        with pytest.raises(Exception):
            await processor.parse_request(json.dumps(invalid_request))
        
        # This will fail until proper protocol compliance is implemented
        assert False, "JSON-RPC protocol compliance not implemented"

    @pytest.mark.asyncio
    async def test_mcp_specific_message_compliance(self):
        """Test MCP-specific message format requirements."""
        processor = MessageProcessor()
        
        # Test MCP tool response format
        tool_response = {
            "jsonrpc": "2.0",
            "id": "req-001",
            "result": {
                "content": [
                    {"type": "text", "text": "Response content"},
                    {"type": "markdown", "text": "# Markdown content"}
                ]
            }
        }
        
        # Should validate as proper MCP response
        is_valid = await processor.validate_mcp_response(tool_response)
        assert is_valid
        
        # This will fail until proper MCP compliance is implemented
        assert False, "MCP message compliance not implemented"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 