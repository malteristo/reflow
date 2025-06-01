"""
STDIO Communication Handler for Research Agent MCP Server.

Handles standard input/output communication for the MCP protocol,
providing synchronous and asynchronous message handling capabilities.

Implements subtask 15.2: STDIO Communication Layer.
"""

import asyncio
import json
import sys
import uuid
import logging
from typing import Dict, Any, Optional, AsyncIterator, TextIO
from io import TextIOWrapper, StringIO

from .message_processor import MessageProcessor


logger = logging.getLogger(__name__)


class StdioHandler:
    """
    Handles STDIO communication for MCP protocol.
    
    Provides both synchronous and asynchronous interfaces for reading
    from stdin and writing to stdout/stderr, with proper message framing
    and session management.
    """
    
    def __init__(
        self, 
        timeout: float = 30.0,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
        error_stream: Optional[TextIO] = None
    ):
        """
        Initialize STDIO handler.
        
        Args:
            timeout: Default timeout for operations in seconds
            input_stream: Custom input stream (defaults to sys.stdin)
            output_stream: Custom output stream (defaults to sys.stdout)
            error_stream: Custom error stream (defaults to sys.stderr)
        """
        self.stdin = input_stream or sys.stdin
        self.stdout = output_stream or sys.stdout
        self.stderr = error_stream or sys.stderr
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())
        self.is_active = True
        self.message_processor = MessageProcessor()
        
        logger.info(f"StdioHandler initialized with session_id: {self.session_id}")
    
    async def process_message(self, raw_message: str) -> Dict[str, Any]:
        """
        Process a raw message string and return a response.
        
        Args:
            raw_message: Raw JSON message string
            
        Returns:
            Response message dictionary
        """
        try:
            # Parse the message
            message = json.loads(raw_message)
            
            # Use message processor to parse and validate
            parsed = self.message_processor.parse_request(message)
            
            # For integration testing, return a simple response
            if parsed.method == "tools/list":
                return self.message_processor.format_response(
                    {"tools": ["ping", "server_info"]},
                    parsed.id
                )
            elif parsed.method == "ping":
                return self.message_processor.format_response(
                    {"message": "pong"},
                    parsed.id
                )
            else:
                return self.message_processor.format_error_response(
                    -32601,  # Method not found
                    f"Method '{parsed.method}' not found",
                    None,
                    parsed.id
                )
                
        except json.JSONDecodeError as e:
            return self.message_processor.format_error_response(
                -32700,  # Parse error
                f"Invalid JSON: {str(e)}",
                None,
                None
            )
        except ValueError as e:
            return self.message_processor.format_error_response(
                -32600,  # Invalid Request
                str(e),
                None,
                None
            )
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self.message_processor.format_error_response(
                -32603,  # Internal error
                "Internal error",
                {"details": str(e)},
                None
            )
    
    def read_message(self) -> Optional[Dict[str, Any]]:
        """
        Read a single message from stdin.
        
        Returns:
            Parsed JSON message or None if EOF/error
        """
        try:
            line = self.stdin.readline()
            if not line:  # EOF
                return None
            
            line = line.strip()
            if not line:  # Empty line
                return None
            
            message = json.loads(line)
            logger.debug(f"Read message: {message}")
            return message
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading message: {e}")
            return None
    
    def write_message(self, message: Dict[str, Any]) -> None:
        """
        Write a message to stdout.
        
        Args:
            message: JSON-serializable message to write
        """
        try:
            framed_message = self.frame_message(message)
            self.stdout.write(framed_message)
            self.stdout.flush()
            logger.debug(f"Wrote message: {message}")
            
        except Exception as e:
            logger.error(f"Error writing message: {e}")
            raise
    
    def write_error(self, error_message: str) -> None:
        """
        Write an error message to stderr.
        
        Args:
            error_message: Error message to write
        """
        try:
            self.stderr.write(f"ERROR: {error_message}\n")
            self.stderr.flush()
            logger.error(f"Wrote error: {error_message}")
            
        except Exception as e:
            logger.error(f"Error writing error message: {e}")
    
    def frame_message(self, message: Dict[str, Any]) -> str:
        """
        Frame a message for STDIO transport.
        
        Args:
            message: Message to frame
            
        Returns:
            Framed message string with proper termination
        """
        json_str = json.dumps(message, separators=(',', ':'))
        return f"{json_str}\n"
    
    async def read_line_async(self) -> Optional[str]:
        """
        Asynchronously read a line from stdin.
        
        Returns:
            Line from stdin or None if EOF
        """
        try:
            # Use asyncio to read from stdin without blocking
            loop = asyncio.get_event_loop()
            line = await loop.run_in_executor(None, self.stdin.readline)
            return line if line else None
            
        except Exception as e:
            logger.error(f"Error reading line async: {e}")
            return None
    
    async def read_messages_async(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Asynchronously read messages from stdin.
        
        Yields:
            Parsed JSON messages
        """
        while self.is_active:
            try:
                line = await self.read_line_async()
                if not line:  # EOF
                    break
                
                line = line.strip()
                if not line:  # Empty line
                    continue
                
                message = json.loads(line)
                logger.debug(f"Read async message: {message}")
                yield message
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in async read: {e}")
                continue
            except Exception as e:
                logger.error(f"Error in async message reading: {e}")
                break
    
    async def write_message_async(self, message: Dict[str, Any]) -> None:
        """
        Asynchronously write a message to stdout.
        
        Args:
            message: JSON-serializable message to write
        """
        try:
            framed_message = self.frame_message(message)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_to_stdout, framed_message)
            logger.debug(f"Wrote async message: {message}")
            
        except Exception as e:
            logger.error(f"Error writing async message: {e}")
            raise
    
    def _write_to_stdout(self, message: str) -> None:
        """
        Helper method to write to stdout (for executor).
        
        Args:
            message: Message string to write
        """
        self.stdout.write(message)
        self.stdout.flush()
    
    def shutdown(self) -> None:
        """
        Shutdown the STDIO handler.
        """
        self.is_active = False
        logger.info(f"StdioHandler session {self.session_id} shutdown")
    
    def cleanup(self) -> None:
        """
        Clean up the STDIO handler session.
        """
        self.is_active = False
        logger.info(f"StdioHandler session {self.session_id} cleaned up") 