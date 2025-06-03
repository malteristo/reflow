"""
Dedicated Collection Management Tools for MCP Server.

Replaces action-based manage_collections tool with dedicated tools that map
directly to CLI commands for better parameter alignment and natural usage.

Addresses Task 32 Critical Issue #3: Action-Based vs Dedicated Tools.
"""

import json
import logging
import subprocess
import asyncio
from typing import Dict, Any, List, Optional

from .base_tool import BaseMCPTool, ToolValidationError

logger = logging.getLogger(__name__)


class CreateCollectionTool(BaseMCPTool):
    """
    MCP tool for creating new collections.
    
    Maps directly to CLI command:
    research-agent collections create <name> --description="..." --type="..."
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the create collection tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "create_collection"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Create a new collection in the knowledge base with optional description and type. "
            "Collections organize documents by topic, project, or purpose."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["name"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the collection to create",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "minLength": 1,
                    "maxLength": 50
                },
                "description": {
                    "type": ["string", "null"],
                    "description": "Optional description of the collection's purpose"
                },
                "collection_type": {
                    "type": "string",
                    "enum": ["general", "project-specific", "fundamental", "reference", "temporary"],
                    "default": "general",
                    "description": "Type of collection to create"
                }
            },
            "required": ["name"]
        }
    
    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """Validate tool-specific parameters."""
        errors = []
        
        name = parameters.get("name")
        if name:
            # Validate collection name format
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                errors.append(ToolValidationError(
                    parameter="name",
                    message="Collection name can only contain letters, numbers, hyphens, and underscores",
                    value=name
                ))
            
            if len(name) > 50:
                errors.append(ToolValidationError(
                    parameter="name",
                    message="Collection name must be 50 characters or less",
                    value=name
                ))
        
        collection_type = parameters.get("collection_type")
        if collection_type:
            valid_types = ["general", "project-specific", "fundamental", "reference", "temporary"]
            if collection_type not in valid_types:
                errors.append(ToolValidationError(
                    parameter="collection_type",
                    message=f"Collection type must be one of: {', '.join(valid_types)}",
                    value=collection_type
                ))
        
        return errors
    
    async def invoke_cli_async(self, args: List[str]) -> Dict[str, Any]:
        """Asynchronously invoke the CLI command."""
        try:
            process = await asyncio.create_subprocess_exec(
                self.cli_path, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "CLI command failed"
                raise subprocess.CalledProcessError(
                    process.returncode, 
                    [self.cli_path] + args, 
                    error_msg
                )
            
            # Parse JSON output if available
            output = stdout.decode().strip()
            if not output:
                return {"status": "success"}
            
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                # CLI might not return JSON, create success response
                return {"status": "success", "message": output}
            
        except Exception as e:
            logger.error(f"CLI invocation failed: {e}")
            raise
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the create collection tool."""
        try:
            # Build CLI command
            args = ["collections", "create", parameters["name"]]
            
            if parameters.get("description"):
                args.extend(["--description", parameters["description"]])
            
            if parameters.get("collection_type"):
                args.extend(["--type", parameters["collection_type"]])
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Format standardized response
            return self.format_success_response(
                data={
                    "collection_name": parameters["name"],
                    "collection_type": parameters.get("collection_type", "general"),
                    "description": parameters.get("description"),
                    "created": True
                },
                message=f"Successfully created collection '{parameters['name']}'",
                operation="create_collection"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to create collection: {e}",
                operation="create_collection"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error creating collection: {e}",
                operation="create_collection"
            )


class ListCollectionsTool(BaseMCPTool):
    """
    MCP tool for listing all collections.
    
    Maps directly to CLI command:
    research-agent collections list --type="..." --stats
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the list collections tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "list_collections"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "List all collections in the knowledge base with optional filtering by type "
            "and statistics display."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return []  # No required parameters
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "collection_type": {
                    "type": ["string", "null"],
                    "enum": [None, "general", "project-specific", "fundamental", "reference", "temporary"],
                    "description": "Filter by collection type"
                },
                "show_stats": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include collection statistics (document count, size, etc.)"
                }
            }
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the list collections tool."""
        try:
            # Build CLI command
            args = ["collections", "list"]
            
            if parameters.get("collection_type"):
                args.extend(["--type", parameters["collection_type"]])
            
            if parameters.get("show_stats", False):
                args.append("--stats")
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Format standardized response
            collections_data = cli_result.get("collections", [])
            
            return self.format_success_response(
                data={
                    "collections": collections_data,
                    "total_count": len(collections_data),
                    "filtered_by_type": parameters.get("collection_type"),
                    "includes_stats": parameters.get("show_stats", False)
                },
                message=f"Found {len(collections_data)} collections",
                operation="list_collections"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to list collections: {e}",
                operation="list_collections"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error listing collections: {e}",
                operation="list_collections"
            )


class DeleteCollectionTool(BaseMCPTool):
    """
    MCP tool for deleting collections.
    
    Maps directly to CLI command:
    research-agent collections delete <name> --confirm
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the delete collection tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "delete_collection"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Delete a collection from the knowledge base. This is a destructive operation "
            "that removes the collection and optionally its documents."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["name"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the collection to delete",
                    "minLength": 1
                },
                "confirm": {
                    "type": "boolean",
                    "default": True,
                    "description": "Skip confirmation prompt (auto-confirm deletion)"
                },
                "keep_documents": {
                    "type": "boolean",
                    "default": False,
                    "description": "Keep documents but remove from collection"
                }
            },
            "required": ["name"]
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the delete collection tool."""
        try:
            # Build CLI command
            args = ["collections", "delete", parameters["name"]]
            
            if parameters.get("confirm", True):
                args.append("--confirm")
            
            if parameters.get("keep_documents", False):
                args.append("--keep-documents")
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Format standardized response
            return self.format_success_response(
                data={
                    "collection_name": parameters["name"],
                    "deleted": True,
                    "documents_kept": parameters.get("keep_documents", False)
                },
                message=f"Successfully deleted collection '{parameters['name']}'",
                operation="delete_collection"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to delete collection: {e}",
                operation="delete_collection"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error deleting collection: {e}",
                operation="delete_collection"
            )


class GetCollectionInfoTool(BaseMCPTool):
    """
    MCP tool for getting detailed information about a collection.
    
    Maps directly to CLI command:
    research-agent collections info <name>
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the collection info tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "get_collection_info"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Get detailed information about a specific collection including "
            "document count, metadata, and configuration."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["name"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the collection to get information about",
                    "minLength": 1
                }
            },
            "required": ["name"]
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the collection info tool."""
        try:
            # Build CLI command
            args = ["collections", "info", parameters["name"]]
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Extract collection info from CLI result
            collection_info = cli_result.get("collection_info", {})
            
            # Format standardized response
            return self.format_success_response(
                data=collection_info,
                message=f"Retrieved information for collection '{parameters['name']}'",
                operation="get_collection_info"
            )
            
        except subprocess.CalledProcessError as e:
            if "not found" in str(e).lower():
                return self.format_not_found("collection", parameters["name"])
            return self.format_error(
                message=f"Failed to get collection info: {e}",
                operation="get_collection_info"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error getting collection info: {e}",
                operation="get_collection_info"
            )


# Additional dedicated tools for other collection operations

class RenameCollectionTool(BaseMCPTool):
    """
    MCP tool for renaming collections.
    
    Maps directly to CLI command:
    research-agent collections rename <old_name> <new_name>
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the rename collection tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "rename_collection"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return "Rename an existing collection while preserving all documents and metadata."
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["old_name", "new_name"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "old_name": {
                    "type": "string",
                    "description": "Current name of the collection",
                    "minLength": 1
                },
                "new_name": {
                    "type": "string",
                    "description": "New name for the collection",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "minLength": 1,
                    "maxLength": 50
                }
            },
            "required": ["old_name", "new_name"]
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the rename collection tool."""
        try:
            # Build CLI command
            args = ["collections", "rename", parameters["old_name"], parameters["new_name"]]
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Format standardized response
            return self.format_success_response(
                data={
                    "old_name": parameters["old_name"],
                    "new_name": parameters["new_name"],
                    "renamed": True
                },
                message=f"Successfully renamed collection from '{parameters['old_name']}' to '{parameters['new_name']}'",
                operation="rename_collection"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to rename collection: {e}",
                operation="rename_collection"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error renaming collection: {e}",
                operation="rename_collection"
            ) 