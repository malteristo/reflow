"""
Research Agent MCP Server.

Main server implementation using FastMCP framework for STDIO communication
with Cursor IDE and other MCP clients.

Implements subtask 15.2: STDIO Communication Layer.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP

from .communication.stdio_handler import StdioHandler
from .communication.message_processor import MessageProcessor
from .protocol.message_handler import MessageHandler
from .protocol.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Research Agent MCP Server.
    
    Provides FastMCP-based server implementation with STDIO communication
    support for integration with Cursor IDE and other MCP clients.
    """
    
    def __init__(self, name: str = "research-agent", version: str = "0.1.0"):
        """
        Initialize the MCP server.
        
        Args:
            name: Server name
            version: Server version
        """
        self.name = name
        self.version = version
        
        # Initialize FastMCP instance
        self.mcp = FastMCP(name)
        
        # Initialize handlers
        self.stdio_handler = StdioHandler()
        self.message_processor = MessageProcessor()
        self.message_handler = MessageHandler()
        self.error_handler = ErrorHandler()
        
        # Server configuration
        self.timeout_config = {
            'default': 30,
            'query': 60,
            'ingestion': 300
        }
        
        # Track registered tools
        self._tools = {}
        
        logger.info(f"MCPServer '{name}' v{version} initialized")
    
    def get_tool_names(self) -> List[str]:
        """
        Get list of registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_supported_transports(self) -> List[str]:
        """
        Get list of supported transport methods.
        
        Returns:
            List of supported transports
        """
        return ["stdio"]
    
    def register_tool(self, name: str, handler: callable) -> None:
        """
        Register a tool with the server.
        
        Args:
            name: Tool name
            handler: Tool handler function
        """
        self._tools[name] = handler
        logger.debug(f"Registered tool: {name}")
    
    async def run_stdio(self) -> None:
        """
        Run the server in STDIO mode.
        
        This method starts the server and handles STDIO communication
        until the connection is closed or an error occurs.
        """
        logger.info("Starting MCP server in STDIO mode")
        
        try:
            # Start the FastMCP server in STDIO mode
            await self.mcp.run_stdio()
            
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the server.
        """
        logger.info("Shutting down MCP server")
        
        try:
            # Cleanup handlers
            if hasattr(self.stdio_handler, 'shutdown'):
                self.stdio_handler.shutdown()
            
            # Additional cleanup can be added here
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("MCP server shutdown complete")
    
    def _setup_default_tools(self) -> None:
        """
        Set up default tools for the server.
        
        This is a placeholder for tool registration.
        Actual tools will be implemented in subsequent subtasks.
        """
        # Placeholder tool registrations
        # These will be replaced with actual implementations
        
        @self.mcp.tool()
        def ping() -> str:
            """Simple ping tool for testing."""
            return "pong"
        
        @self.mcp.tool()
        def server_info() -> Dict[str, Any]:
            """Get server information."""
            return {
                "name": self.name,
                "version": self.version,
                "supported_transports": self.get_supported_transports(),
                "tools": self.get_tool_names()
            }
        
        # Register tools in internal tracking
        self._tools["ping"] = ping
        self._tools["server_info"] = server_info
        
        logger.debug("Default tools registered")


def create_server() -> MCPServer:
    """
    Factory function to create and configure an MCP server.
    
    Returns:
        Configured MCPServer instance
    """
    server = MCPServer()
    server._setup_default_tools()
    return server


async def main():
    """
    Main entry point for running the MCP server.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run server
    server = create_server()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main()) 