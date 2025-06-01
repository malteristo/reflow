"""
Protocol module for Research Agent MCP Server.

Contains message handling, error handling, and response formatting
components for MCP protocol compliance.
"""

from .message_handler import MessageHandler
from .error_handler import ErrorHandler, MCPErrorCode
from .response_formatter import (
    ResponseFormatter,
    ResponseFormattingError,
    MCPResponse,
    MCPSuccessResponse,
    MCPErrorResponse,
    QueryResponse,
    CollectionResponse,
    IngestResponse,
    ProjectResponse,
    AugmentResponse,
    ProgressResponse,
    StatusResponse,
    ContentFormatter,
    ToolResponseFactory
)

__all__ = [
    "MessageHandler",
    "ErrorHandler", 
    "MCPErrorCode",
    "ResponseFormatter",
    "ResponseFormattingError",
    "MCPResponse",
    "MCPSuccessResponse",
    "MCPErrorResponse",
    "QueryResponse",
    "CollectionResponse",
    "IngestResponse",
    "ProjectResponse",
    "AugmentResponse",
    "ProgressResponse",
    "StatusResponse",
    "ContentFormatter",
    "ToolResponseFactory"
] 