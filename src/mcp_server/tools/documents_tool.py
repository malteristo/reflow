"""
Ingest Documents MCP Tool for Research Agent.

Maps CLI knowledge base document commands to MCP protocol operations,
providing document ingestion and management functionality.

Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
"""

import json
import logging
import subprocess
import asyncio
from typing import Dict, Any, List

from .base_tool import BaseMCPTool, ToolValidationError

logger = logging.getLogger(__name__)


class IngestDocumentsTool(BaseMCPTool):
    """
    MCP tool for ingesting documents into the knowledge base.
    
    Maps to CLI commands:
    - research-agent kb add-document <path> [options]
    - research-agent kb ingest-folder <path> [options]
    - research-agent kb remove-document <id>
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the ingest documents tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "ingest_documents"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Add documents or folders to the knowledge base. "
            "Supports single document ingestion, folder batch processing, and document removal."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["action"]
    
    def get_supported_actions(self) -> List[str]:
        """Get supported document actions."""
        return ["add_document", "ingest_folder", "remove_document"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": self.get_supported_actions(),
                    "description": "Action to perform with documents"
                },
                "path": {
                    "type": "string",
                    "description": "File or folder path (required for add_document/ingest_folder)"
                },
                "document_id": {
                    "type": "string",
                    "description": "Document ID (required for remove_document)"
                },
                "collection": {
                    "type": "string",
                    "description": "Target collection name"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Process folders recursively (for ingest_folder)",
                    "default": False
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to include (e.g., ['.md', '.txt'])"
                }
            },
            "required": ["action"]
        }
    
    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """Validate tool-specific parameters."""
        errors = []
        
        action = parameters.get("action")
        if not action:
            errors.append(ToolValidationError(
                parameter="action",
                message="Action is required",
                value=action
            ))
            return errors
        
        # Validate action-specific parameters
        if action in ["add_document", "ingest_folder"]:
            if not parameters.get("path"):
                errors.append(ToolValidationError(
                    parameter="path",
                    message=f"path is required for action '{action}'",
                    value=parameters.get("path")
                ))
        
        if action == "remove_document":
            if not parameters.get("document_id"):
                errors.append(ToolValidationError(
                    parameter="document_id",
                    message="document_id is required for remove_document action",
                    value=parameters.get("document_id")
                ))
        
        return errors
    
    async def invoke_cli_async(self, args: List[str]) -> Dict[str, Any]:
        """
        Asynchronously invoke the CLI command.
        
        Args:
            args: Command line arguments
            
        Returns:
            CLI output as dictionary
            
        Raises:
            subprocess.CalledProcessError: If CLI command fails
        """
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
                return {"status": "success"}
            
            return json.loads(output)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CLI JSON output: {e}")
            raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Invalid JSON output: {e}")
        except Exception as e:
            logger.error(f"CLI invocation failed: {e}")
            raise
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the ingest documents tool."""
        try:
            action = parameters["action"]
            
            # Build CLI command based on action
            if action == "add_document":
                args = ["kb", "add-document", parameters["path"]]
                if parameters.get("collection"):
                    args.extend(["--collection", parameters["collection"]])
                args.append("--json")
                
            elif action == "ingest_folder":
                args = ["kb", "ingest-folder", parameters["path"]]
                if parameters.get("collection"):
                    args.extend(["--collection", parameters["collection"]])
                if parameters.get("recursive", False):
                    args.append("--recursive")
                if parameters.get("file_types"):
                    args.extend(["--file-types", ",".join(parameters["file_types"])])
                args.append("--json")
                
            elif action == "remove_document":
                args = ["kb", "remove-document", parameters["document_id"], "--json"]
                
            else:
                return self.format_error(f"Unsupported action: {action}")
            
            # Execute CLI command asynchronously
            try:
                cli_data = await self.invoke_cli_async(args)
                
                # Format response
                formatted_response = self._format_documents_response(action, cli_data)
                return self.format_success_response(formatted_response)
                
            except subprocess.CalledProcessError as e:
                error_msg = str(e)
                return self.format_error(f"Document operation failed: {error_msg}")
            except subprocess.TimeoutExpired:
                return self.format_error("Document operation timeout - processing took longer than 2 minutes")
            except json.JSONDecodeError as e:
                return self.format_error(f"Invalid CLI response format: {e}")
                
        except Exception as e:
            logger.error(f"Documents tool execution failed: {e}", exc_info=True)
            return self.format_error(f"Documents tool execution failed: {str(e)}")
    
    def _format_documents_response(self, action: str, cli_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format CLI response for document operations."""
        if action == "add_document":
            return {
                "action": action,
                "document_id": cli_data.get("document_id", ""),
                "path": cli_data.get("path", ""),
                "collection": cli_data.get("collection", ""),
                "chunks_created": cli_data.get("chunks_created", 0),
                "status": cli_data.get("status", "success"),
                "message": cli_data.get("message", "Document added successfully")
            }
        elif action == "ingest_folder":
            return {
                "action": action,
                "folder_path": cli_data.get("folder_path", ""),
                "collection": cli_data.get("collection", ""),
                "documents_processed": cli_data.get("documents_processed", 0),
                "total_chunks": cli_data.get("total_chunks", 0),
                "status": cli_data.get("status", "success"),
                "processing_summary": cli_data.get("processing_summary", [])
            }
        elif action == "remove_document":
            return {
                "action": action,
                "document_id": cli_data.get("document_id", ""),
                "status": cli_data.get("status", "success"),
                "message": cli_data.get("message", "Document removed successfully")
            }
        else:
            return cli_data 