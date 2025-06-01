"""
Test suite for STDIO Communication Layer.

Tests the Research Agent MCP server STDIO communication implementation
according to the protocol specification.

Implements TDD for subtask 15.2: Implement STDIO Communication Layer.
"""

import asyncio
import json
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from io import StringIO
import sys

# These imports will fail initially - this is expected in TDD RED phase
try:
    from src.mcp_server.communication.stdio_handler import StdioHandler
    from src.mcp_server.communication.message_processor import MessageProcessor
    from src.mcp_server.server import MCPServer
    from src.mcp_server.protocol.message_handler import MessageHandler
    from src.mcp_server.protocol.error_handler import ErrorHandler
except ImportError:
    # Expected during RED phase - modules don't exist yet
    StdioHandler = None
    MessageProcessor = None
    MCPServer = None
    MessageHandler = None
    ErrorHandler = None


class TestStdioHandler:
    """Test suite for STDIO communication handler."""
    
    def test_stdio_handler_exists(self):
        """Test that StdioHandler class exists."""
        assert StdioHandler is not None, "StdioHandler class must be implemented"
    
    def test_stdio_handler_initialization(self):
        """Test StdioHandler can be initialized with proper configuration."""
        # This will fail initially - expected in RED phase
        handler = StdioHandler()
        
        assert hasattr(handler, 'stdin'), "Handler must have stdin attribute"
        assert hasattr(handler, 'stdout'), "Handler must have stdout attribute"
        assert hasattr(handler, 'stderr'), "Handler must have stderr attribute"
        assert hasattr(handler, 'message_processor'), "Handler must have message processor"
    
    def test_stdio_read_capability(self):
        """Test that handler can read from stdin."""
        # Mock stdin input
        test_input = '{"jsonrpc": "2.0", "method": "ping", "id": 1}\n'
        
        with patch('sys.stdin', StringIO(test_input)):
            handler = StdioHandler()
            message = handler.read_message()
            
            assert message is not None, "Handler must read message from stdin"
            assert isinstance(message, dict), "Message must be parsed as dictionary"
            assert message.get("jsonrpc") == "2.0", "Message must be valid JSON-RPC 2.0"
    
    def test_stdio_write_capability(self):
        """Test that handler can write to stdout."""
        test_response = {
            "jsonrpc": "2.0",
            "result": {"message": "pong"},
            "id": 1
        }
        
        with patch('sys.stdout', StringIO()) as mock_stdout:
            handler = StdioHandler()
            handler.write_message(test_response)
            
            output = mock_stdout.getvalue()
            assert len(output) > 0, "Handler must write to stdout"
            
            # Parse the output back to verify it's valid JSON
            parsed = json.loads(output.strip())
            assert parsed["jsonrpc"] == "2.0"
            assert parsed["id"] == 1
    
    def test_error_writing_capability(self):
        """Test that handler can write errors to stderr."""
        test_error = "Test error message"
        
        with patch('sys.stderr', StringIO()) as mock_stderr:
            handler = StdioHandler()
            handler.write_error(test_error)
            
            output = mock_stderr.getvalue()
            assert test_error in output, "Error must be written to stderr"
    
    def test_message_framing(self):
        """Test proper message framing for STDIO communication."""
        handler = StdioHandler()
        
        test_message = {"jsonrpc": "2.0", "method": "test", "id": 1}
        framed_message = handler.frame_message(test_message)
        
        # Check that message is properly framed (typically with newlines)
        assert framed_message.endswith('\n'), "Message must be newline-terminated"
        
        # Verify the framed message can be parsed back
        parsed = json.loads(framed_message.strip())
        assert parsed == test_message, "Framed message must be parseable"


class TestMessageProcessor:
    """Test suite for message processing in STDIO context."""
    
    def test_message_processor_exists(self):
        """Test that MessageProcessor class exists."""
        assert MessageProcessor is not None, "MessageProcessor class must be implemented"
    
    def test_json_rpc_parsing(self):
        """Test JSON-RPC message parsing."""
        processor = MessageProcessor()
        
        valid_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_base",
                "arguments": {"query": "test"}
            },
            "id": "test-1"
        }
        
        parsed = processor.parse_request(valid_request)
        
        assert parsed.jsonrpc == "2.0"
        assert parsed.method == "tools/call"
        assert parsed.id == "test-1"
        assert "name" in parsed.params
    
    def test_invalid_json_rpc_handling(self):
        """Test handling of invalid JSON-RPC messages."""
        processor = MessageProcessor()
        
        invalid_requests = [
            {},  # Empty object
            {"jsonrpc": "1.0"},  # Wrong version
            {"jsonrpc": "2.0"},  # Missing method
            {"jsonrpc": "2.0", "method": "test"},  # Missing id
        ]
        
        for invalid_request in invalid_requests:
            with pytest.raises(ValueError):
                processor.parse_request(invalid_request)
    
    def test_response_formatting(self):
        """Test JSON-RPC response formatting."""
        processor = MessageProcessor()
        
        result_data = {"status": "success", "data": "test"}
        request_id = "test-1"
        
        response = processor.format_response(result_data, request_id)
        
        assert response["jsonrpc"] == "2.0"
        assert response["result"] == result_data
        assert response["id"] == request_id
        assert "error" not in response
    
    def test_error_response_formatting(self):
        """Test JSON-RPC error response formatting."""
        processor = MessageProcessor()
        
        error_code = -32000
        error_message = "Test error"
        error_data = {"details": "Test error details"}
        request_id = "test-1"
        
        response = processor.format_error_response(
            error_code, error_message, error_data, request_id
        )
        
        assert response["jsonrpc"] == "2.0"
        assert response["error"]["code"] == error_code
        assert response["error"]["message"] == error_message
        assert response["error"]["data"] == error_data
        assert response["id"] == request_id
        assert "result" not in response


class TestAsyncStdioOperations:
    """Test suite for asynchronous STDIO operations."""
    
    @pytest.mark.asyncio
    async def test_async_message_reading(self):
        """Test asynchronous message reading from stdin."""
        # This will fail initially
        handler = StdioHandler()
        
        # Mock async stdin reading
        test_messages = [
            '{"jsonrpc": "2.0", "method": "ping", "id": 1}\n',
            '{"jsonrpc": "2.0", "method": "pong", "id": 2}\n'
        ]
        
        async def mock_read_line():
            for message in test_messages:
                yield message
        
        with patch.object(handler, 'read_line_async', mock_read_line()):
            messages = []
            async for message in handler.read_messages_async():
                messages.append(message)
                if len(messages) >= 2:
                    break
            
            assert len(messages) == 2
            assert all(msg.get("jsonrpc") == "2.0" for msg in messages)
    
    @pytest.mark.asyncio
    async def test_async_message_writing(self):
        """Test asynchronous message writing to stdout."""
        handler = StdioHandler()
        
        test_response = {
            "jsonrpc": "2.0",
            "result": {"message": "async_pong"},
            "id": 1
        }
        
        with patch('sys.stdout', StringIO()) as mock_stdout:
            await handler.write_message_async(test_response)
            
            output = mock_stdout.getvalue()
            assert len(output) > 0
            
            parsed = json.loads(output.strip())
            assert parsed["result"]["message"] == "async_pong"
    
    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self):
        """Test handling multiple concurrent messages."""
        handler = StdioHandler()
        processor = MessageProcessor()
        
        # Simulate multiple concurrent requests
        requests = [
            {"jsonrpc": "2.0", "method": "test1", "id": 1},
            {"jsonrpc": "2.0", "method": "test2", "id": 2},
            {"jsonrpc": "2.0", "method": "test3", "id": 3}
        ]
        
        # Process requests concurrently
        tasks = []
        for request in requests:
            task = asyncio.create_task(
                processor.process_message_async(request)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        assert len(results) == 3
        # All should either succeed or fail gracefully
        for result in results:
            assert not isinstance(result, Exception) or \
                   isinstance(result, (ValueError, NotImplementedError))


class TestStdioIntegrationWithFastMCP:
    """Test suite for STDIO integration with FastMCP framework."""
    
    def test_fastmcp_stdio_server_creation(self):
        """Test creating FastMCP server with STDIO transport."""
        # This will fail initially
        from fastmcp import FastMCP
        
        # Create server instance
        server = MCPServer()
        
        assert hasattr(server, 'mcp'), "Server must have FastMCP instance"
        assert server.mcp is not None, "FastMCP instance must be initialized"
    
    def test_fastmcp_stdio_mode_support(self):
        """Test that FastMCP server supports STDIO mode."""
        server = MCPServer()
        
        # Check if STDIO mode is supported
        supported_transports = server.get_supported_transports()
        assert "stdio" in supported_transports, "STDIO transport must be supported"
    
    @pytest.mark.asyncio
    async def test_fastmcp_stdio_server_run(self):
        """Test running FastMCP server in STDIO mode."""
        server = MCPServer()
        
        # Mock the STDIO streams to avoid blocking
        with patch('sys.stdin'), patch('sys.stdout'), patch('sys.stderr'):
            # This should not raise an exception
            assert hasattr(server, 'run_stdio'), "Server must have run_stdio method"
            
            # Test that the method exists and is callable
            assert callable(server.run_stdio), "run_stdio must be callable"


class TestStdioErrorHandling:
    """Test suite for STDIO error handling."""
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON input."""
        handler = StdioHandler()
        
        malformed_inputs = [
            '{"invalid": json}',  # Invalid JSON syntax
            'not json at all',    # Not JSON
            '',                   # Empty string
            '\n',                 # Just newline
        ]
        
        for malformed_input in malformed_inputs:
            with patch('sys.stdin', StringIO(malformed_input)):
                try:
                    message = handler.read_message()
                    # Should either return None or raise appropriate exception
                    assert message is None or isinstance(message, dict)
                except (json.JSONDecodeError, ValueError):
                    # Expected for malformed input
                    pass
    
    def test_connection_loss_handling(self):
        """Test handling of connection loss scenarios."""
        handler = StdioHandler()
        
        # Simulate EOF on stdin
        with patch('sys.stdin', StringIO('')):
            message = handler.read_message()
            assert message is None, "EOF should return None"
    
    def test_timeout_handling(self):
        """Test timeout handling for STDIO operations."""
        handler = StdioHandler()
        
        # Test that handler supports timeout configuration
        assert hasattr(handler, 'timeout'), "Handler should support timeout configuration"
        
        # Test default timeout value
        assert handler.timeout > 0, "Default timeout should be positive"


class TestStdioSessionManagement:
    """Test suite for STDIO session management."""
    
    def test_session_initialization(self):
        """Test STDIO session initialization."""
        handler = StdioHandler()
        
        # Session should be properly initialized
        assert hasattr(handler, 'session_id'), "Handler should have session_id"
        assert handler.session_id is not None, "Session ID should be set"
    
    def test_session_cleanup(self):
        """Test proper session cleanup."""
        handler = StdioHandler()
        initial_session_id = handler.session_id
        
        # Cleanup should reset session state
        handler.cleanup()
        
        # Session should be properly cleaned up
        assert hasattr(handler, 'is_active'), "Handler should track active state"
        assert not handler.is_active, "Handler should be inactive after cleanup"
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown of STDIO handler."""
        handler = StdioHandler()
        
        # Should support graceful shutdown
        assert hasattr(handler, 'shutdown'), "Handler should support graceful shutdown"
        
        # Shutdown should be idempotent
        handler.shutdown()
        handler.shutdown()  # Should not raise exception


if __name__ == "__main__":
    # Run the tests to see them fail (RED phase)
    pytest.main([__file__, "-v"]) 