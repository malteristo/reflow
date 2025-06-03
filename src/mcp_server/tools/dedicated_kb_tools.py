"""
Dedicated Knowledge Base Tools for MCP Server.

Implements missing high-priority knowledge base tools that were identified
in the CLI-MCP integration audit.

Addresses Task 32 - High Priority Missing Tools:
- list_documents
- document_status  
- rebuild_index
"""

import json
import logging
import subprocess
import asyncio
from typing import Dict, Any, List, Optional

from .base_tool import BaseMCPTool, ToolValidationError

logger = logging.getLogger(__name__)


class ListDocumentsTool(BaseMCPTool):
    """
    MCP tool for listing documents in the knowledge base.
    
    Maps directly to CLI command:
    research-agent kb list-documents --collection="..." --limit=50
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the list documents tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "list_documents"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "List documents in the knowledge base with optional filtering by collection "
            "and limit on number of results."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return []  # No required parameters
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": ["string", "null"],
                    "description": "Filter by collection name (optional)"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "default": 50,
                    "description": "Maximum number of documents to show"
                },
                "include_metadata": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include document metadata in results"
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["created", "modified", "title", "size"],
                    "default": "created",
                    "description": "Sort documents by specified field"
                }
            }
        }
    
    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """Validate tool-specific parameters."""
        errors = []
        
        limit = parameters.get("limit")
        if limit is not None:
            if not isinstance(limit, int) or limit < 1 or limit > 500:
                errors.append(ToolValidationError(
                    parameter="limit",
                    message="Limit must be an integer between 1 and 500",
                    value=limit
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
            
            # Parse JSON output
            output = stdout.decode().strip()
            if not output:
                return {"documents": [], "total_count": 0}
            
            try:
                return json.loads(output)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse CLI JSON output: {e}")
                raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Invalid JSON output: {e}")
            
        except Exception as e:
            logger.error(f"CLI invocation failed: {e}")
            raise
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the list documents tool."""
        try:
            # Build CLI command
            args = ["kb", "list-documents"]
            
            if parameters.get("collection"):
                args.extend(["--collection", parameters["collection"]])
            
            limit = parameters.get("limit", 50)
            args.extend(["--limit", str(limit)])
            
            # Add JSON output flag if available
            args.append("--json")
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Extract documents data
            documents = cli_result.get("documents", [])
            total_count = cli_result.get("total_count", len(documents))
            
            # Format standardized response
            return self.format_success_response(
                data={
                    "documents": documents,
                    "total_count": total_count,
                    "filtered_by_collection": parameters.get("collection"),
                    "limit_applied": limit,
                    "returned_count": len(documents)
                },
                message=f"Found {len(documents)} documents" + (f" in collection '{parameters.get('collection')}'" if parameters.get("collection") else ""),
                operation="list_documents"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to list documents: {e}",
                operation="list_documents"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error listing documents: {e}",
                operation="list_documents"
            )


class GetKnowledgeBaseStatusTool(BaseMCPTool):
    """
    MCP tool for getting knowledge base status information.
    
    Maps directly to CLI command:
    research-agent kb status
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the KB status tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "get_knowledge_base_status"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Get comprehensive status information about the knowledge base including "
            "document counts, collection statistics, and system health metrics."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return []  # No required parameters
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "include_detailed_stats": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include detailed per-collection statistics"
                },
                "check_index_health": {
                    "type": "boolean",
                    "default": True,
                    "description": "Perform index health checks"
                }
            }
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the knowledge base status tool."""
        try:
            # Build CLI command
            args = ["kb", "status"]
            
            if parameters.get("include_detailed_stats", False):
                args.append("--detailed")
            
            # Add JSON output flag
            args.append("--json")
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Extract status information
            status_data = cli_result.get("status", {})
            
            # Format standardized response
            return self.format_success_response(
                data=status_data,
                message="Knowledge base status retrieved successfully",
                operation="get_knowledge_base_status"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to get knowledge base status: {e}",
                operation="get_knowledge_base_status"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error getting knowledge base status: {e}",
                operation="get_knowledge_base_status"
            )


class RebuildIndexTool(BaseMCPTool):
    """
    MCP tool for rebuilding the knowledge base index.
    
    Maps directly to CLI command:
    research-agent kb rebuild-index --collection="..." --confirm
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the rebuild index tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "rebuild_index"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Rebuild the vector index for the knowledge base or specific collection. "
            "This may take time for large collections but improves search performance."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return []  # No required parameters
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "collection": {
                    "type": ["string", "null"],
                    "description": "Rebuild index for specific collection only (optional)"
                },
                "confirm": {
                    "type": "boolean",
                    "default": True,
                    "description": "Skip confirmation prompt"
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force rebuild even if index appears healthy"
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100,
                    "description": "Batch size for processing documents"
                }
            }
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the rebuild index tool."""
        try:
            # Build CLI command
            args = ["kb", "rebuild-index"]
            
            if parameters.get("collection"):
                args.extend(["--collection", parameters["collection"]])
            
            if parameters.get("confirm", True):
                args.append("--confirm")
            
            if parameters.get("force", False):
                args.append("--force")
            
            batch_size = parameters.get("batch_size", 100)
            args.extend(["--batch-size", str(batch_size)])
            
            # Add JSON output flag
            args.append("--json")
            
            # Execute CLI command (this may take a while)
            cli_result = await self.invoke_cli_async(args)
            
            # Extract rebuild information
            rebuild_data = cli_result.get("rebuild_result", {})
            
            # Format standardized response
            return self.format_success_response(
                data=rebuild_data,
                message="Index rebuild completed successfully",
                operation="rebuild_index"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to rebuild index: {e}",
                operation="rebuild_index"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error rebuilding index: {e}",
                operation="rebuild_index"
            )


class AddDocumentTool(BaseMCPTool):
    """
    MCP tool for adding a single document to the knowledge base.
    
    Maps directly to CLI command:
    research-agent kb add-document <path> --collection="..." --force
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the add document tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "add_document"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Add a single document to the knowledge base with automatic text extraction "
            "and embedding generation."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["file_path"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file to add",
                    "minLength": 1
                },
                "collection": {
                    "type": "string",
                    "default": "default",
                    "description": "Collection to add the document to"
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Overwrite existing document if it exists"
                },
                "title": {
                    "type": ["string", "null"],
                    "description": "Optional custom title for the document"
                },
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Optional tags to associate with the document"
                }
            },
            "required": ["file_path"]
        }
    
    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """Validate tool-specific parameters."""
        errors = []
        
        file_path = parameters.get("file_path")
        if file_path:
            # Basic path validation
            if not isinstance(file_path, str) or not file_path.strip():
                errors.append(ToolValidationError(
                    parameter="file_path",
                    message="File path must be a non-empty string",
                    value=file_path
                ))
        
        return errors
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the add document tool."""
        try:
            # Build CLI command
            args = ["kb", "add-document", parameters["file_path"]]
            
            collection = parameters.get("collection", "default")
            args.extend(["--collection", collection])
            
            if parameters.get("force", False):
                args.append("--force")
            
            if parameters.get("title"):
                args.extend(["--title", parameters["title"]])
            
            if parameters.get("tags"):
                tags_str = ",".join(parameters["tags"])
                args.extend(["--tags", tags_str])
            
            # Add JSON output flag
            args.append("--json")
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Extract document information
            document_data = cli_result.get("document", {})
            
            # Format standardized response
            return self.format_success_response(
                data=document_data,
                message=f"Successfully added document to collection '{collection}'",
                operation="add_document"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to add document: {e}",
                operation="add_document"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error adding document: {e}",
                operation="add_document"
            )


class RemoveDocumentTool(BaseMCPTool):
    """
    MCP tool for removing a document from the knowledge base.
    
    Maps directly to CLI command:
    research-agent kb remove-document <document_id> --confirm
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the remove document tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "remove_document"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Remove a document from the knowledge base permanently. "
            "This action cannot be undone."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["document_id"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "ID of the document to remove",
                    "minLength": 1
                },
                "confirm": {
                    "type": "boolean",
                    "default": True,
                    "description": "Skip confirmation prompt"
                }
            },
            "required": ["document_id"]
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the remove document tool."""
        try:
            # Build CLI command
            args = ["kb", "remove-document", parameters["document_id"]]
            
            if parameters.get("confirm", True):
                args.append("--confirm")
            
            # Add JSON output flag
            args.append("--json")
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Format standardized response
            return self.format_success_response(
                data={
                    "document_id": parameters["document_id"],
                    "removed": True
                },
                message=f"Successfully removed document '{parameters['document_id']}'",
                operation="remove_document"
            )
            
        except subprocess.CalledProcessError as e:
            if "not found" in str(e).lower():
                return self.format_not_found("document", parameters["document_id"])
            return self.format_error(
                message=f"Failed to remove document: {e}",
                operation="remove_document"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error removing document: {e}",
                operation="remove_document"
            ) 