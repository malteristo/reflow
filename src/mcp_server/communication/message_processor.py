"""
Message Processor for Research Agent MCP Server.

Handles JSON-RPC 2.0 message parsing, validation, and formatting
according to the MCP protocol specification.

Implements subtask 15.2: STDIO Communication Layer.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedRequest:
    """Represents a parsed JSON-RPC request."""
    jsonrpc: str
    method: str
    params: Dict[str, Any]
    id: Any


class MessageProcessor:
    """
    Processes JSON-RPC 2.0 messages for MCP protocol.
    
    Handles parsing, validation, and formatting of requests and responses
    according to JSON-RPC 2.0 specification and MCP protocol requirements.
    """
    
    def __init__(self):
        """Initialize the message processor."""
        logger.debug("MessageProcessor initialized")
    
    def parse_request(self, message: Dict[str, Any]) -> ParsedRequest:
        """
        Parse and validate a JSON-RPC request message.
        
        Args:
            message: Raw message dictionary
            
        Returns:
            ParsedRequest object with validated fields
            
        Raises:
            ValueError: If message is invalid JSON-RPC 2.0
        """
        # Validate JSON-RPC version
        if message.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC version, must be '2.0'")
        
        # Validate required fields
        if "method" not in message:
            raise ValueError("Missing required field 'method'")
        
        if "id" not in message:
            raise ValueError("Missing required field 'id'")
        
        # Extract params (optional)
        params = message.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Field 'params' must be an object")
        
        return ParsedRequest(
            jsonrpc=message["jsonrpc"],
            method=message["method"],
            params=params,
            id=message["id"]
        )
    
    async def parse_request_from_string(self, message_str: str) -> ParsedRequest:
        """
        Asynchronously parse and validate a JSON-RPC request message from string.
        
        Args:
            message_str: Raw JSON message string
            
        Returns:
            ParsedRequest object with validated fields
            
        Raises:
            ValueError: If message is invalid JSON-RPC 2.0
        """
        try:
            message = json.loads(message_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")
        
        return self.parse_request(message)
    
    async def validate_mcp_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate that a response follows MCP protocol requirements.
        
        Args:
            response: Response message to validate
            
        Returns:
            True if valid MCP response
        """
        # Check JSON-RPC 2.0 compliance
        if response.get("jsonrpc") != "2.0":
            return False
        
        if "id" not in response:
            return False
        
        # Must have either result or error
        has_result = "result" in response
        has_error = "error" in response
        
        if not (has_result or has_error) or (has_result and has_error):
            return False
        
        # If has result, check MCP content format
        if has_result:
            result = response["result"]
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            return False
                        if "type" not in item or "text" not in item:
                            return False
                        if item["type"] not in ["text", "markdown", "json"]:
                            return False
        
        return True
    
    def format_response(self, result: Any, request_id: Any) -> Dict[str, Any]:
        """
        Format a successful JSON-RPC response.
        
        Args:
            result: Result data to include in response
            request_id: ID from the original request
            
        Returns:
            Formatted JSON-RPC response
        """
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
    
    def format_error_response(
        self, 
        error_code: int, 
        error_message: str, 
        error_data: Optional[Dict[str, Any]] = None,
        request_id: Any = None
    ) -> Dict[str, Any]:
        """
        Format a JSON-RPC error response.
        
        Args:
            error_code: JSON-RPC error code
            error_message: Human-readable error message
            error_data: Optional additional error data
            request_id: ID from the original request
            
        Returns:
            Formatted JSON-RPC error response
        """
        error_obj = {
            "code": error_code,
            "message": error_message
        }
        
        if error_data is not None:
            error_obj["data"] = error_data
        
        return {
            "jsonrpc": "2.0",
            "error": error_obj,
            "id": request_id
        }
    
    async def process_message_async(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously process a message.
        
        This is a placeholder for actual message processing logic.
        In the full implementation, this would route to appropriate handlers.
        
        Args:
            message: Raw message to process
            
        Returns:
            Response message
            
        Raises:
            NotImplementedError: This is a placeholder implementation
        """
        try:
            parsed = self.parse_request(message)
            logger.debug(f"Processing async message: {parsed.method}")
            
            # Placeholder - in real implementation, this would route to tool handlers
            if parsed.method == "ping":
                return self.format_response({"message": "pong"}, parsed.id)
            else:
                # For now, return not implemented error
                return self.format_error_response(
                    -32601,  # Method not found
                    f"Method '{parsed.method}' not implemented",
                    {"method": parsed.method},
                    parsed.id
                )
                
        except ValueError as e:
            logger.error(f"Invalid request: {e}")
            return self.format_error_response(
                -32600,  # Invalid Request
                str(e),
                None,
                message.get("id")
            )
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self.format_error_response(
                -32603,  # Internal error
                "Internal error",
                {"details": str(e)},
                message.get("id")
            ) 