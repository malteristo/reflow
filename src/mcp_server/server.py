"""
Research Agent MCP Server.

Main server implementation using FastMCP framework for STDIO communication
with Cursor IDE and other MCP clients.

Implements subtask 15.2: STDIO Communication Layer.
Enhanced with model change detection tools for Task 35.
"""

import asyncio
import logging
import sys
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP

from .communication.stdio_handler import StdioHandler
from .communication.message_processor import MessageProcessor
from .protocol.message_handler import MessageHandler
from .protocol.error_handler import ErrorHandler
from .tools.query_tool import QueryKnowledgeBaseTool
from .tools.collections_tool import ManageCollectionsTool
from .tools.documents_tool import IngestDocumentsTool
from .tools.projects_tool import ManageProjectsTool
from .tools.augment_tool import AugmentKnowledgeTool, SubmitFeedbackTool
from .tools.model_management_tools import (
    DetectModelChangesTool,
    ReindexCollectionTool,
    GetReindexStatusTool
)

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Research Agent MCP Server.
    
    Provides FastMCP-based server implementation with STDIO communication
    support for integration with Cursor IDE and other MCP clients.
    Enhanced with model change detection capabilities.
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
        
        # Server state
        self._is_initialized = False
        self._is_running = False
        
        # Initialize FastMCP instance
        self.mcp = FastMCP(name)
        
        # Initialize handlers
        self.stdio_handler = StdioHandler()
        self.message_processor = MessageProcessor()
        self.message_handler = MessageHandler()
        self.error_handler = ErrorHandler()
        
        # Initialize MCP tools
        self.query_tool = QueryKnowledgeBaseTool()
        self.collections_tool = ManageCollectionsTool()
        self.documents_tool = IngestDocumentsTool()
        self.projects_tool = ManageProjectsTool()
        self.augment_tool = AugmentKnowledgeTool()
        self.feedback_tool = SubmitFeedbackTool()
        
        # Initialize model management tools
        self.detect_model_changes_tool = DetectModelChangesTool()
        self.reindex_collection_tool = ReindexCollectionTool()
        self.get_reindex_status_tool = GetReindexStatusTool()
        
        # Server configuration
        self.timeout_config = {
            'default': 30,
            'query': 60,
            'ingestion': 300,
            'reindexing': 3600  # Extended timeout for reindexing operations
        }
        
        # Track registered tools
        self._tools = {}
        
        logger.info(f"MCPServer '{name}' v{version} initialized with model management")
    
    @property
    def is_initialized(self) -> bool:
        """Check if server is initialized."""
        return self._is_initialized
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._is_running
    
    async def initialize(self) -> None:
        """
        Initialize the server and all its components.
        """
        logger.info("Initializing MCP server")
        
        try:
            # Initialize all tools
            await self._initialize_tools()
            
            # Setup tool registrations
            self._setup_default_tools()
            
            # Mark as initialized
            self._is_initialized = True
            
            logger.info("MCP server initialization complete")
            
        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            raise
    
    async def _initialize_tools(self) -> None:
        """Initialize all MCP tools."""
        logger.debug("Initializing MCP tools")
        
        # Initialize each tool if it has an initialize method
        tools = [
            self.query_tool,
            self.collections_tool,
            self.documents_tool,
            self.projects_tool,
            self.augment_tool,
            self.feedback_tool,
            # Model management tools
            self.detect_model_changes_tool,
            self.reindex_collection_tool,
            self.get_reindex_status_tool
        ]
        
        for tool in tools:
            if hasattr(tool, 'initialize'):
                await tool.initialize()
        
        logger.debug("MCP tools initialized")
    
    async def start_stdio(self) -> None:
        """
        Start the server in STDIO mode.
        """
        if not self._is_initialized:
            await self.initialize()
        
        logger.info("Starting STDIO communication")
        
        try:
            self._is_running = True
            
            # In a real implementation, this would start the STDIO loop
            # For integration testing, we'll simulate this
            logger.info("STDIO communication started")
            
        except Exception as e:
            logger.error(f"Failed to start STDIO communication: {e}")
            self._is_running = False
            raise
    
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
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool is not found
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        try:
            tool_handler = self._tools[tool_name]
            
            # Execute tool based on type
            if hasattr(tool_handler, 'execute'):
                return await tool_handler.execute(params)
            else:
                # For simple functions
                return {"status": "success", "result": tool_handler()}
                
        except Exception as e:
            logger.error(f"Tool execution failed for '{tool_name}': {e}")
            return {
                "status": "error",
                "error": {
                    "code": -32002,
                    "message": f"Tool execution failed: {str(e)}"
                }
            }
    
    async def run_stdio(self) -> None:
        """
        Run the server in STDIO mode.
        
        This method starts the server and handles STDIO communication
        until the connection is closed or an error occurs.
        """
        logger.info("Starting MCP server in STDIO mode")
        
        try:
            # Initialize if not already done
            if not self._is_initialized:
                await self.initialize()
            
            # Start STDIO communication
            await self.start_stdio()
            
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
            # Mark as not running
            self._is_running = False
            
            # Cleanup handlers
            if hasattr(self.stdio_handler, 'shutdown'):
                self.stdio_handler.shutdown()
            
            # Cleanup tools
            tools = [
                self.query_tool,
                self.collections_tool,
                self.documents_tool,
                self.projects_tool,
                self.augment_tool,
                self.feedback_tool,
                # Model management tools
                self.detect_model_changes_tool,
                self.reindex_collection_tool,
                self.get_reindex_status_tool
            ]
            
            for tool in tools:
                if hasattr(tool, 'shutdown'):
                    await tool.shutdown()
            
            # Mark as not initialized
            self._is_initialized = False
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("MCP server shutdown complete")
    
    def _setup_default_tools(self) -> None:
        """
        Set up default tools and MCP tools for the server.
        """
        logger.debug("Setting up tools")
        
        # Setup basic server tools
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
        
        # Setup MCP tools
        @self.mcp.tool()
        async def query_knowledge_base(
            query: str,
            collections: Optional[List[str]] = None,
            top_k: int = 10
        ) -> Dict[str, Any]:
            """Query the knowledge base."""
            params = {
                "query": query,
                "collections": collections or [],
                "top_k": top_k
            }
            return await self.query_tool.execute(params)
        
        @self.mcp.tool()
        async def manage_collections(
            action: str,
            name: Optional[str] = None,
            type: Optional[str] = None,
            description: Optional[str] = None
        ) -> Dict[str, Any]:
            """Manage collections."""
            params = {
                "action": action,
                "name": name,
                "type": type,
                "description": description
            }
            return await self.collections_tool.execute(params)
        
        @self.mcp.tool()
        async def ingest_documents(
            action: str,
            path: Optional[str] = None,
            collection: Optional[str] = None,
            document_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """Ingest documents into the knowledge base."""
            params = {
                "action": action,
                "path": path,
                "collection": collection,
                "document_id": document_id
            }
            return await self.documents_tool.execute(params)
        
        @self.mcp.tool()
        async def manage_projects(
            action: str,
            name: Optional[str] = None,
            description: Optional[str] = None
        ) -> Dict[str, Any]:
            """Manage research projects."""
            params = {
                "action": action,
                "name": name,
                "description": description
            }
            return await self.projects_tool.execute(params)
        
        @self.mcp.tool()
        async def augment_knowledge(
            action: str,
            content: Optional[str] = None,
            source: Optional[str] = None,
            collection: Optional[str] = None
        ) -> Dict[str, Any]:
            """Augment knowledge base with external information."""
            params = {
                "action": action,
                "content": content,
                "source": source,
                "collection": collection
            }
            return await self.augment_tool.execute(params)
        
        @self.mcp.tool()
        async def submit_feedback(
            chunk_id: str,
            rating: str,
            reason: str,
            comment: Optional[str] = None,
            user_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """Submit user feedback (thumbs up/down) for a knowledge base chunk to improve content quality."""
            params = {
                "chunk_id": chunk_id,
                "rating": rating,
                "reason": reason,
                "comment": comment,
                "user_id": user_id
            }
            return await self.feedback_tool.execute(params)
        
        # Setup model management tools
        @self.mcp.tool()
        async def detect_model_changes(
            auto_register: bool = True,
            show_collections: bool = True,
            check_compatibility: bool = True
        ) -> Dict[str, Any]:
            """Detect embedding model changes and identify collections requiring re-indexing."""
            params = {
                "auto_register": auto_register,
                "show_collections": show_collections,
                "check_compatibility": check_compatibility
            }
            return await self.detect_model_changes_tool.execute(params)
        
        @self.mcp.tool()
        async def reindex_collection(
            collections: Optional[str] = None,
            parallel: bool = True,
            workers: Optional[int] = None,
            batch_size: int = 50,
            force: bool = False,
            track_progress: bool = True
        ) -> Dict[str, Any]:
            """Re-index collections with the current embedding model."""
            params = {
                "collections": collections,
                "parallel": parallel,
                "workers": workers,
                "batch_size": batch_size,
                "force": force,
                "track_progress": track_progress
            }
            return await self.reindex_collection_tool.execute(params)
        
        @self.mcp.tool()
        async def get_reindex_status(
            collection: Optional[str] = None,
            show_completed: bool = True,
            show_progress_details: bool = True,
            include_history: bool = False
        ) -> Dict[str, Any]:
            """Get re-indexing status and progress information."""
            params = {
                "collection": collection,
                "show_completed": show_completed,
                "show_progress_details": show_progress_details,
                "include_history": include_history
            }
            return await self.get_reindex_status_tool.execute(params)
        
        # Register tools in internal tracking
        self._tools["ping"] = ping
        self._tools["server_info"] = server_info
        self._tools["query_knowledge_base"] = self.query_tool
        self._tools["manage_collections"] = self.collections_tool
        self._tools["ingest_documents"] = self.documents_tool
        self._tools["manage_projects"] = self.projects_tool
        self._tools["augment_knowledge"] = self.augment_tool
        self._tools["submit_feedback"] = self.feedback_tool
        # Model management tools
        self._tools["detect_model_changes"] = self.detect_model_changes_tool
        self._tools["reindex_collection"] = self.reindex_collection_tool
        self._tools["get_reindex_status"] = self.get_reindex_status_tool
        
        logger.debug("Tools setup complete")


def create_server() -> MCPServer:
    """
    Factory function to create and configure an MCP server.
    
    Returns:
        Configured MCPServer instance
    """
    server = MCPServer()
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