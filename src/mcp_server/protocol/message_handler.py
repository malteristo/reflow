"""
Message Handler for Research Agent MCP Server.

Provides parameter validation, sanitization, and security checks
for MCP protocol messages.

Implements subtask 15.2: STDIO Communication Layer.
"""

import re
import logging
from typing import Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class MessageHandler:
    """
    Handles message validation and sanitization for MCP protocol.
    
    Provides security checks, parameter validation, and input sanitization
    to prevent common security vulnerabilities.
    """
    
    def __init__(self):
        """Initialize the message handler."""
        logger.debug("MessageHandler initialized")
    
    def sanitize_path(self, path: str) -> str:
        """
        Sanitize a file path to prevent path traversal attacks.
        
        Args:
            path: Raw file path
            
        Returns:
            Sanitized path
        """
        if not isinstance(path, str):
            raise ValueError("Path must be a string")
        
        # Remove path traversal patterns
        sanitized = path.replace("../", "").replace("..\\", "")
        
        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")
        
        # Normalize the path
        try:
            normalized = str(Path(sanitized).resolve())
            # Ensure the path doesn't start with traversal patterns
            if normalized.startswith("../") or normalized.startswith("..\\"):
                raise ValueError("Path traversal detected")
            return normalized
        except Exception:
            # If path normalization fails, return a safe default
            return sanitized.lstrip("./\\")
    
    def sanitize_query(self, query: str) -> str:
        """
        Sanitize a query string to prevent command injection.
        
        Args:
            query: Raw query string
            
        Returns:
            Sanitized query
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        # Remove potentially dangerous characters
        dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "<", ">"]
        sanitized = query
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        
        # Limit length to prevent DoS
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def validate_string_param(self, value: Any) -> bool:
        """
        Validate that a parameter is a string.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid string, False otherwise
        """
        return isinstance(value, str)
    
    def validate_number_param(self, value: Any) -> bool:
        """
        Validate that a parameter is a number.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid number, False otherwise
        """
        return isinstance(value, (int, float))
    
    def validate_boolean_param(self, value: Any) -> bool:
        """
        Validate that a parameter is a boolean.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid boolean, False otherwise
        """
        return isinstance(value, bool)
    
    def validate_dict_param(self, value: Any) -> bool:
        """
        Validate that a parameter is a dictionary.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid dictionary, False otherwise
        """
        return isinstance(value, dict)
    
    def validate_list_param(self, value: Any) -> bool:
        """
        Validate that a parameter is a list.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid list, False otherwise
        """
        return isinstance(value, list)
    
    def validate_email(self, email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email string to validate
            
        Returns:
            True if valid email format, False otherwise
        """
        if not isinstance(email, str):
            return False
        
        # Simple email regex pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_collection_name(self, name: str) -> bool:
        """
        Validate collection name format.
        
        Args:
            name: Collection name to validate
            
        Returns:
            True if valid collection name, False otherwise
        """
        if not isinstance(name, str):
            return False
        
        # Collection names should be alphanumeric with underscores/hyphens
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, name)) and len(name) <= 100
    
    def sanitize_collection_name(self, name: str) -> str:
        """
        Sanitize a collection name.
        
        Args:
            name: Raw collection name
            
        Returns:
            Sanitized collection name
        """
        if not isinstance(name, str):
            raise ValueError("Collection name must be a string")
        
        # Remove invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', name)
        
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        # Ensure it's not empty
        if not sanitized:
            raise ValueError("Collection name cannot be empty after sanitization")
        
        return sanitized
    
    def validate_tool_parameters(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate parameters for a specific tool.
        
        Args:
            tool_name: Name of the tool
            params: Parameters to validate
            
        Returns:
            Dictionary of validation errors (empty if all valid)
        """
        errors = {}
        
        if tool_name == "query_knowledge_base":
            if "query" not in params:
                errors["query"] = "Required parameter 'query' is missing"
            elif not self.validate_string_param(params["query"]):
                errors["query"] = "Parameter 'query' must be a string"
            
            if "collections" in params and params["collections"] is not None:
                if not self.validate_string_param(params["collections"]):
                    errors["collections"] = "Parameter 'collections' must be a string"
            
            if "top_k" in params:
                if not self.validate_number_param(params["top_k"]):
                    errors["top_k"] = "Parameter 'top_k' must be a number"
                elif params["top_k"] <= 0 or params["top_k"] > 100:
                    errors["top_k"] = "Parameter 'top_k' must be between 1 and 100"
        
        elif tool_name == "manage_collections":
            if "action" not in params:
                errors["action"] = "Required parameter 'action' is missing"
            elif params["action"] not in ["create", "list", "delete", "info"]:
                errors["action"] = "Parameter 'action' must be one of: create, list, delete, info"
            
            if params.get("action") in ["create", "delete", "info"]:
                if "collection_name" not in params:
                    errors["collection_name"] = "Parameter 'collection_name' is required for this action"
                elif not self.validate_collection_name(params["collection_name"]):
                    errors["collection_name"] = "Invalid collection name format"
        
        elif tool_name == "ingest_documents":
            if "path" not in params:
                errors["path"] = "Required parameter 'path' is missing"
            elif not self.validate_string_param(params["path"]):
                errors["path"] = "Parameter 'path' must be a string"
            
            if "collection" not in params:
                errors["collection"] = "Required parameter 'collection' is missing"
            elif not self.validate_collection_name(params["collection"]):
                errors["collection"] = "Invalid collection name format"
        
        return errors 