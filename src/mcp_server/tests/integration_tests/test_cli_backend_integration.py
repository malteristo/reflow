"""
Integration tests for CLI backend communication and command execution.

Tests the integration between MCP tools and the research-agent-cli backend,
including command formatting, parameter passing, and response handling.
"""

import json
import pytest
import subprocess
import tempfile
from unittest.mock import Mock, patch, call
from pathlib import Path

from src.mcp_server.tools.query_tool import QueryKnowledgeBaseTool
from src.mcp_server.tools.collections_tool import ManageCollectionsTool
from src.mcp_server.tools.documents_tool import IngestDocumentsTool
from src.mcp_server.tools.projects_tool import ManageProjectsTool
from src.mcp_server.tools.augment_tool import AugmentKnowledgeTool


class TestCLIBackendIntegration:
    """Tests for MCP tool integration with CLI backend commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_query_tool_cli_command_generation(self):
        """Test query tool generates correct CLI commands."""
        tool = QueryKnowledgeBaseTool()
        
        # Test basic query command
        params = {
            "query": "machine learning basics",
            "collections": ["research", "general"],
            "top_k": 15
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success", "results": []})
            mock_run.return_value = mock_result
            
            await tool.execute(params)
            
            # Verify CLI command structure
            call_args = mock_run.call_args[0][0]
            assert "research-agent-cli" in call_args
            assert "query" in call_args
            assert "search" in call_args
            assert "--query" in call_args
            assert "machine learning basics" in call_args
            assert "--collections" in call_args
            assert "research,general" in call_args
            assert "--top-k" in call_args
            assert "15" in call_args
        
        # This will fail until proper CLI command generation is implemented
        assert False, "CLI command generation not implemented"

    @pytest.mark.asyncio
    async def test_collections_tool_cli_command_generation(self):
        """Test collections tool generates correct CLI commands for different actions."""
        tool = ManageCollectionsTool()
        
        # Test create collection command
        create_params = {
            "action": "create",
            "name": "test-collection",
            "type": "research",
            "description": "Test collection"
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success"})
            mock_run.return_value = mock_result
            
            await tool.execute(create_params)
            
            call_args = mock_run.call_args[0][0]
            assert "research-agent-cli" in call_args
            assert "collections" in call_args
            assert "create" in call_args
            assert "--name" in call_args
            assert "test-collection" in call_args
            assert "--type" in call_args
            assert "research" in call_args
        
        # Test list collections command
        list_params = {"action": "list"}
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success", "collections": []})
            mock_run.return_value = mock_result
            
            await tool.execute(list_params)
            
            call_args = mock_run.call_args[0][0]
            assert "research-agent-cli" in call_args
            assert "collections" in call_args
            assert "list" in call_args
        
        # This will fail until proper CLI command generation is implemented
        assert False, "Collections CLI command generation not implemented"

    @pytest.mark.asyncio
    async def test_documents_tool_cli_command_generation(self):
        """Test documents tool generates correct CLI commands."""
        tool = IngestDocumentsTool()
        
        # Create test file
        test_file = Path(self.test_dir) / "test.md"
        test_file.write_text("# Test\nContent")
        
        # Test add document command
        params = {
            "action": "add-document",
            "file_path": str(test_file),
            "collection": "test-collection"
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success", "chunks_created": 2})
            mock_run.return_value = mock_result
            
            await tool.execute(params)
            
            call_args = mock_run.call_args[0][0]
            assert "research-agent-cli" in call_args
            assert "kb" in call_args
            assert "add-document" in call_args
            assert "--file" in call_args
            assert str(test_file) in call_args
            assert "--collection" in call_args
            assert "test-collection" in call_args
        
        # This will fail until proper CLI command generation is implemented
        assert False, "Documents CLI command generation not implemented"

    @pytest.mark.asyncio
    async def test_projects_tool_cli_command_generation(self):
        """Test projects tool generates correct CLI commands."""
        tool = ManageProjectsTool()
        
        # Test create project command
        params = {
            "action": "create",
            "name": "test-project",
            "description": "Test project description"
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success"})
            mock_run.return_value = mock_result
            
            await tool.execute(params)
            
            call_args = mock_run.call_args[0][0]
            assert "research-agent-cli" in call_args
            assert "projects" in call_args
            assert "create" in call_args
            assert "--name" in call_args
            assert "test-project" in call_args
        
        # This will fail until proper CLI command generation is implemented
        assert False, "Projects CLI command generation not implemented"

    @pytest.mark.asyncio
    async def test_augment_tool_cli_command_generation(self):
        """Test knowledge augmentation tool generates correct CLI commands."""
        tool = AugmentKnowledgeTool()
        
        # Test add external result command
        params = {
            "action": "add-external-result",
            "query": "test query",
            "result": {
                "content": "Test content",
                "source": "external"
            },
            "collection": "research"
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success"})
            mock_run.return_value = mock_result
            
            await tool.execute(params)
            
            call_args = mock_run.call_args[0][0]
            assert "research-agent-cli" in call_args
            assert "kb" in call_args
            assert "add-external-result" in call_args
        
        # This will fail until proper CLI command generation is implemented
        assert False, "Augment tool CLI command generation not implemented"

    @pytest.mark.asyncio
    async def test_cli_error_handling_integration(self):
        """Test proper handling of CLI command errors."""
        tool = QueryKnowledgeBaseTool()
        
        # Test CLI command failure
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stderr = "CLI command failed: Invalid collection name"
            mock_run.return_value = mock_result
            
            result = await tool.execute({
                "query": "test",
                "collections": ["invalid-collection"]
            })
            
            # Should return proper error response
            assert result["status"] == "error"
            assert "error" in result
            assert result["error"]["code"] == -32002  # Tool execution error
            assert "CLI command failed" in result["error"]["message"]
        
        # This will fail until proper error handling is implemented
        assert False, "CLI error handling integration not implemented"

    @pytest.mark.asyncio
    async def test_cli_timeout_handling(self):
        """Test handling of CLI command timeouts."""
        tool = IngestDocumentsTool()  # Use ingestion for timeout testing
        
        # Mock a timeout scenario
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("research-agent-cli", 60)
            
            result = await tool.execute({
                "action": "ingest-folder",
                "folder_path": self.test_dir,
                "collection": "test"
            })
            
            # Should return timeout error
            assert result["status"] == "error"
            assert result["error"]["code"] == -32003  # Timeout error
            assert "timeout" in result["error"]["message"].lower()
        
        # This will fail until proper timeout handling is implemented
        assert False, "CLI timeout handling not implemented"

    @pytest.mark.asyncio
    async def test_cli_json_response_parsing(self):
        """Test parsing of JSON responses from CLI commands."""
        tool = QueryKnowledgeBaseTool()
        
        # Test valid JSON response
        mock_response = {
            "status": "success",
            "results": [
                {
                    "content": "Test result content",
                    "metadata": {
                        "source": "test.md",
                        "score": 0.85,
                        "chunk_id": "chunk_001"
                    }
                }
            ],
            "query_stats": {
                "total_results": 1,
                "processing_time": 0.15
            }
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(mock_response)
            mock_run.return_value = mock_result
            
            result = await tool.execute({
                "query": "test query",
                "collections": ["test"]
            })
            
            # Should parse and return structured data
            assert result["status"] == "success"
            assert "results" in result
            assert "query_stats" in result
            assert result["results"][0]["metadata"]["score"] == 0.85
        
        # This will fail until proper JSON parsing is implemented
        assert False, "CLI JSON response parsing not implemented"

    @pytest.mark.asyncio
    async def test_cli_parameter_sanitization(self):
        """Test CLI parameters are properly sanitized for security."""
        tool = IngestDocumentsTool()
        
        # Test potentially dangerous parameters
        dangerous_params = {
            "action": "add-document",
            "file_path": "test.md; rm -rf /",  # Command injection attempt
            "collection": "test"
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps({"status": "success"})
            mock_run.return_value = mock_result
            
            await tool.execute(dangerous_params)
            
            # Verify command was sanitized
            call_args = mock_run.call_args[0][0]
            command_string = " ".join(call_args)
            assert "; rm -rf /" not in command_string
            assert "test.md" in command_string  # But legitimate part should remain
        
        # This will fail until proper parameter sanitization is implemented
        assert False, "CLI parameter sanitization not implemented"

    @pytest.mark.asyncio
    async def test_cli_concurrent_execution(self):
        """Test CLI can handle concurrent command execution."""
        import asyncio
        
        tool = QueryKnowledgeBaseTool()
        
        # Create multiple concurrent CLI calls
        async def execute_query(query_id):
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps({
                    "status": "success",
                    "results": [{"content": f"Result for query {query_id}"}]
                })
                mock_run.return_value = mock_result
                
                return await tool.execute({
                    "query": f"test query {query_id}",
                    "collections": ["test"]
                })
        
        # Execute multiple queries concurrently
        tasks = [execute_query(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        for i, result in enumerate(results):
            assert result["status"] == "success"
            assert f"query {i}" in str(result)
        
        # This will fail until proper concurrent execution is implemented
        assert False, "CLI concurrent execution not implemented"


class TestCLIResponseHandling:
    """Tests for handling different types of CLI responses."""
    
    @pytest.mark.asyncio
    async def test_success_response_handling(self):
        """Test handling of successful CLI responses."""
        tool = QueryKnowledgeBaseTool()
        
        success_response = {
            "status": "success",
            "results": [{"content": "Success result"}],
            "metadata": {"total": 1}
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = json.dumps(success_response)
            mock_run.return_value = mock_result
            
            result = await tool.execute({"query": "test", "collections": ["test"]})
            
            assert result["status"] == "success"
            assert "results" in result
            assert "metadata" in result
        
        # This will fail until proper response handling is implemented
        assert False, "Success response handling not implemented"

    @pytest.mark.asyncio
    async def test_error_response_handling(self):
        """Test handling of error responses from CLI."""
        tool = QueryKnowledgeBaseTool()
        
        error_response = {
            "status": "error",
            "error": {
                "code": "COLLECTION_NOT_FOUND",
                "message": "Collection 'invalid' does not exist"
            }
        }
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = json.dumps(error_response)
            mock_run.return_value = mock_result
            
            result = await tool.execute({"query": "test", "collections": ["invalid"]})
            
            assert result["status"] == "error"
            assert "error" in result
            assert result["error"]["code"] == -32002  # Tool execution error
            assert "Collection 'invalid' does not exist" in result["error"]["message"]
        
        # This will fail until proper error response handling is implemented
        assert False, "Error response handling not implemented"

    @pytest.mark.asyncio
    async def test_malformed_json_response_handling(self):
        """Test handling of malformed JSON responses from CLI."""
        tool = QueryKnowledgeBaseTool()
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Invalid JSON response {malformed"
            mock_run.return_value = mock_result
            
            result = await tool.execute({"query": "test", "collections": ["test"]})
            
            assert result["status"] == "error"
            assert result["error"]["code"] == -32001  # Parse error
            assert "json" in result["error"]["message"].lower()
        
        # This will fail until proper malformed JSON handling is implemented
        assert False, "Malformed JSON response handling not implemented"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 