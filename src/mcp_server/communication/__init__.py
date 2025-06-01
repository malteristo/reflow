"""
Communication module for Research Agent MCP Server.

Provides STDIO communication handlers and message processing
for the MCP protocol implementation.
"""

from .stdio_handler import StdioHandler
from .message_processor import MessageProcessor, ParsedRequest

__all__ = [
    "StdioHandler",
    "MessageProcessor", 
    "ParsedRequest"
] 