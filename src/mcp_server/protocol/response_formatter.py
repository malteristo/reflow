"""
Response Formatter for Research Agent MCP Server.

Formats server responses according to MCP protocol, ensuring consistency 
and clarity for clients.

Implements subtask 15.5: Design and Implement Response Formatting.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, asdict, field
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseFormattingError(Exception):
    """Exception raised for response formatting errors."""
    pass


@dataclass
class MCPResponse:
    """Base class for MCP responses."""
    jsonrpc: str = "2.0"
    id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class MCPSuccessResponse(MCPResponse):
    """MCP success response following JSON-RPC 2.0 format."""
    result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPErrorResponse(MCPResponse):
    """MCP error response following JSON-RPC 2.0 format."""
    error: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse(MCPResponse):
    """Query tool specific response."""
    status: str = ""
    results: List[Dict[str, Any]] = field(default_factory=list)
    query_refinement: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionResponse(MCPResponse):
    """Collection management response."""
    status: str = ""
    action: str = ""
    collections: List[Dict[str, Any]] = field(default_factory=list)
    collection_info: Optional[Dict[str, Any]] = None


@dataclass
class IngestResponse(MCPResponse):
    """Document ingestion response."""
    status: str = ""
    processed_files: List[Dict[str, Any]] = field(default_factory=list)
    total_chunks: int = 0
    collection: str = ""
    processing_time: float = 0.0


@dataclass
class ProjectResponse(MCPResponse):
    """Project management response."""
    status: str = ""
    action: str = ""
    projects: List[Dict[str, Any]] = field(default_factory=list)
    project_info: Optional[Dict[str, Any]] = None


@dataclass
class AugmentResponse(MCPResponse):
    """Knowledge augmentation response."""
    status: str = ""
    content_added: bool = False
    collection: str = ""
    source: str = ""


@dataclass
class ProgressResponse(MCPResponse):
    """Progress update response for long-running operations."""
    operation: str = ""
    progress: float = 0.0
    current_file: str = ""
    files_processed: int = 0
    total_files: int = 0
    estimated_time_remaining: float = 0.0


@dataclass
class StatusResponse(MCPResponse):
    """Status update response."""
    operation: str = ""
    status: str = ""
    stage: str = ""
    message: str = ""


class ContentFormatter:
    """Handles content formatting for different content types."""
    
    @staticmethod
    def format_content(content: Union[str, Dict, List], content_type: str) -> str:
        """
        Format content based on its type.
        
        Args:
            content: Content to format
            content_type: Type of content (text, markdown, json)
            
        Returns:
            Formatted content as string
        """
        try:
            if content_type == "json":
                return ContentFormatter._format_json_content(content)
            elif content_type == "markdown":
                return ContentFormatter._format_markdown_content(content)
            else:  # text or default
                return ContentFormatter._format_text_content(content)
                
        except Exception as e:
            logger.error(f"Error formatting content: {e}")
            # Return string representation as fallback
            return str(content)
    
    @staticmethod
    def _format_json_content(content: Union[str, Dict, List]) -> str:
        """Format content as JSON."""
        if isinstance(content, str):
            # Validate it's valid JSON
            json.loads(content)
            return content
        else:
            return json.dumps(content, indent=2)
    
    @staticmethod
    def _format_markdown_content(content: Union[str, Dict, List]) -> str:
        """Format content as markdown."""
        if isinstance(content, str):
            return content
        else:
            # Convert non-string to JSON first, then wrap in markdown
            json_str = json.dumps(content, indent=2)
            return f"```json\n{json_str}\n```"
    
    @staticmethod
    def _format_text_content(content: Union[str, Dict, List]) -> str:
        """Format content as plain text."""
        if isinstance(content, str):
            return content
        else:
            return json.dumps(content, indent=2)


class ToolResponseFactory:
    """Factory for creating tool-specific responses."""
    
    # Registry of tool names to response classes
    TOOL_RESPONSE_MAPPING = {
        "query_knowledge_base": QueryResponse,
        "manage_collections": CollectionResponse,
        "ingest_documents": IngestResponse,
        "manage_projects": ProjectResponse,
        "augment_knowledge": AugmentResponse
    }
    
    @classmethod
    def create_response(
        cls,
        tool_name: str,
        request_id: str,
        data: Dict[str, Any]
    ) -> Union[QueryResponse, CollectionResponse, IngestResponse, ProjectResponse, AugmentResponse]:
        """
        Create appropriate tool response based on tool name.
        
        Args:
            tool_name: Name of the tool
            request_id: Request ID
            data: Tool response data
            
        Returns:
            Appropriate tool response object
        """
        if tool_name not in cls.TOOL_RESPONSE_MAPPING:
            raise ResponseFormattingError(f"Unknown tool name: {tool_name}")
        
        response_class = cls.TOOL_RESPONSE_MAPPING[tool_name]
        
        try:
            if tool_name == "query_knowledge_base":
                return cls._create_query_response(request_id, data)
            elif tool_name == "manage_collections":
                return cls._create_collection_response(request_id, data)
            elif tool_name == "ingest_documents":
                return cls._create_ingest_response(request_id, data)
            elif tool_name == "manage_projects":
                return cls._create_project_response(request_id, data)
            elif tool_name == "augment_knowledge":
                return cls._create_augment_response(request_id, data)
        except Exception as e:
            logger.error(f"Error creating {tool_name} response: {e}")
            raise ResponseFormattingError(f"Failed to create {tool_name} response: {e}")
    
    @staticmethod
    def _create_query_response(request_id: str, data: Dict[str, Any]) -> QueryResponse:
        """Create QueryResponse instance."""
        return QueryResponse(
            id=request_id,
            status=data.get("status", ""),
            results=data.get("results", []),
            query_refinement=data.get("query_refinement", {})
        )
    
    @staticmethod
    def _create_collection_response(request_id: str, data: Dict[str, Any]) -> CollectionResponse:
        """Create CollectionResponse instance."""
        return CollectionResponse(
            id=request_id,
            status=data.get("status", ""),
            action=data.get("action", ""),
            collections=data.get("collections", []),
            collection_info=data.get("collection_info")
        )
    
    @staticmethod
    def _create_ingest_response(request_id: str, data: Dict[str, Any]) -> IngestResponse:
        """Create IngestResponse instance."""
        return IngestResponse(
            id=request_id,
            status=data.get("status", ""),
            processed_files=data.get("processed_files", []),
            total_chunks=data.get("total_chunks", 0),
            collection=data.get("collection", ""),
            processing_time=data.get("processing_time", 0.0)
        )
    
    @staticmethod
    def _create_project_response(request_id: str, data: Dict[str, Any]) -> ProjectResponse:
        """Create ProjectResponse instance."""
        return ProjectResponse(
            id=request_id,
            status=data.get("status", ""),
            action=data.get("action", ""),
            projects=data.get("projects", []),
            project_info=data.get("project_info")
        )
    
    @staticmethod
    def _create_augment_response(request_id: str, data: Dict[str, Any]) -> AugmentResponse:
        """Create AugmentResponse instance."""
        return AugmentResponse(
            id=request_id,
            status=data.get("status", ""),
            content_added=data.get("content_added", False),
            collection=data.get("collection", ""),
            source=data.get("source", "")
        )


class ResponseFormatter:
    """
    Formats responses according to MCP protocol and JSON-RPC 2.0 standards.
    
    Provides methods for formatting success responses, error responses,
    and tool-specific responses with proper content type handling.
    """
    
    def __init__(self):
        """Initialize the response formatter."""
        self.content_formatter = ContentFormatter()
        self.tool_factory = ToolResponseFactory()
        logger.debug("ResponseFormatter initialized")
    
    def format_success_response(
        self, 
        request_id: str, 
        content: Union[str, Dict, List],
        content_type: str = "text"
    ) -> MCPSuccessResponse:
        """
        Format a success response according to MCP protocol.
        
        Args:
            request_id: The request ID to include in response
            content: The content to include in the response
            content_type: Type of content (text, markdown, json)
            
        Returns:
            MCPSuccessResponse object
        """
        try:
            # Format content based on type
            formatted_content = self.content_formatter.format_content(content, content_type)
            
            # Create MCP-compliant result structure
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": formatted_content
                    }
                ]
            }
            
            return MCPSuccessResponse(
                id=request_id,
                result=result
            )
            
        except Exception as e:
            logger.error(f"Error formatting success response: {e}")
            raise ResponseFormattingError(f"Failed to format success response: {e}")
    
    def format_error_response(
        self,
        request_id: str,
        error_code: int,
        error_message: str,
        error_data: Optional[Dict[str, Any]] = None
    ) -> MCPErrorResponse:
        """
        Format an error response according to JSON-RPC 2.0.
        
        Args:
            request_id: The request ID to include in response
            error_code: JSON-RPC error code
            error_message: Human-readable error message
            error_data: Additional error data
            
        Returns:
            MCPErrorResponse object
        """
        try:
            error = {
                "code": error_code,
                "message": error_message
            }
            
            if error_data:
                error["data"] = error_data
            
            return MCPErrorResponse(
                id=request_id,
                error=error
            )
            
        except Exception as e:
            logger.error(f"Error formatting error response: {e}")
            # Return a basic error response even if formatting fails
            return MCPErrorResponse(
                id=request_id,
                error={
                    "code": -32000,
                    "message": "Internal formatting error"
                }
            )
    
    def format_tool_response(
        self,
        tool_name: str,
        request_id: str,
        data: Dict[str, Any]
    ) -> Union[QueryResponse, CollectionResponse, IngestResponse, ProjectResponse, AugmentResponse]:
        """
        Format a tool-specific response using the factory pattern.
        
        Args:
            tool_name: Name of the tool
            request_id: Request ID
            data: Tool response data
            
        Returns:
            Appropriate tool response object
        """
        return self.tool_factory.create_response(tool_name, request_id, data)
    
    def format_progress_response(
        self,
        request_id: str,
        progress_data: Dict[str, Any]
    ) -> ProgressResponse:
        """
        Format a progress update response.
        
        Args:
            request_id: Request ID
            progress_data: Progress information
            
        Returns:
            ProgressResponse object
        """
        try:
            return ProgressResponse(
                id=request_id,
                operation=progress_data.get("operation", ""),
                progress=progress_data.get("progress", 0.0),
                current_file=progress_data.get("current_file", ""),
                files_processed=progress_data.get("files_processed", 0),
                total_files=progress_data.get("total_files", 0),
                estimated_time_remaining=progress_data.get("estimated_time_remaining", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error formatting progress response: {e}")
            raise ResponseFormattingError(f"Failed to format progress response: {e}")
    
    def format_status_response(
        self,
        request_id: str,
        status_data: Dict[str, Any]
    ) -> StatusResponse:
        """
        Format a status update response.
        
        Args:
            request_id: Request ID
            status_data: Status information
            
        Returns:
            StatusResponse object
        """
        try:
            return StatusResponse(
                id=request_id,
                operation=status_data.get("operation", ""),
                status=status_data.get("status", ""),
                stage=status_data.get("stage", ""),
                message=status_data.get("message", "")
            )
            
        except Exception as e:
            logger.error(f"Error formatting status response: {e}")
            raise ResponseFormattingError(f"Failed to format status response: {e}")
    
    # Legacy method for backward compatibility
    def _format_content(self, content: Union[str, Dict, List], content_type: str) -> str:
        """
        Legacy method for content formatting - delegates to ContentFormatter.
        
        Args:
            content: Content to format
            content_type: Type of content
            
        Returns:
            Formatted content as string
        """
        return self.content_formatter.format_content(content, content_type) 