"""
Manage Collections MCP Tool for Research Agent.

Maps CLI collections commands to MCP protocol operations, providing
collection management functionality.

Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
"""

import json
import logging
import subprocess
from typing import Dict, Any, List

from .base_tool import BaseMCPTool, ToolValidationError

logger = logging.getLogger(__name__)


class ManageCollectionsTool(BaseMCPTool):
    """
    MCP tool for managing collections.
    
    Maps to CLI commands:
    - research-agent collections create <name> [options]
    - research-agent collections list
    - research-agent collections delete <name>
    - research-agent collections info <name>
    - research-agent collections rename <old> <new>
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the manage collections tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "manage_collections"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Create, list, delete, and manage knowledge collections. "
            "Supports collection information retrieval and renaming operations."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["action"]
    
    def get_supported_actions(self) -> List[str]:
        """Get supported collection actions."""
        return ["create", "list", "delete", "info", "rename"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": self.get_supported_actions(),
                    "description": "Action to perform on collections"
                },
                "collection_name": {
                    "type": "string",
                    "description": "Name of the collection (required for create/delete/info)"
                },
                "new_name": {
                    "type": "string",
                    "description": "New name for collection (required for rename action)"
                },
                "description": {
                    "type": "string",
                    "description": "Description for the collection (optional for create)"
                },
                "collection_type": {
                    "type": "string",
                    "enum": ["general", "project", "research"],
                    "description": "Type of collection (optional for create)"
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
        if action in ["create", "delete", "info"]:
            if not parameters.get("collection_name"):
                errors.append(ToolValidationError(
                    parameter="collection_name",
                    message=f"collection_name is required for action '{action}'",
                    value=parameters.get("collection_name")
                ))
        
        if action == "rename":
            if not parameters.get("collection_name"):
                errors.append(ToolValidationError(
                    parameter="collection_name",
                    message="collection_name (old name) is required for rename action",
                    value=parameters.get("collection_name")
                ))
            if not parameters.get("new_name"):
                errors.append(ToolValidationError(
                    parameter="new_name",
                    message="new_name is required for rename action",
                    value=parameters.get("new_name")
                ))
        
        return errors
    
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the manage collections tool."""
        try:
            action = parameters["action"]
            
            # Build CLI command based on action
            if action == "list":
                args = ["collections", "list", "--json"]
            elif action == "create":
                args = ["collections", "create", parameters["collection_name"]]
                if parameters.get("description"):
                    args.extend(["--description", parameters["description"]])
                if parameters.get("collection_type"):
                    args.extend(["--type", parameters["collection_type"]])
                args.append("--json")
            elif action == "delete":
                args = ["collections", "delete", parameters["collection_name"], "--json"]
            elif action == "info":
                args = ["collections", "info", parameters["collection_name"], "--json"]
            elif action == "rename":
                args = ["collections", "rename", parameters["collection_name"], parameters["new_name"], "--json"]
            else:
                return self.format_error(f"Unsupported action: {action}")
            
            # Execute CLI command
            try:
                result = subprocess.run(
                    [self.cli_path] + args,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
                
                # Parse CLI output
                if result.stdout.strip():
                    cli_data = json.loads(result.stdout.strip())
                else:
                    cli_data = {"status": "success"}
                
                # Format response
                formatted_response = self._format_collections_response(action, cli_data)
                return self.format_success_response(formatted_response)
                
            except subprocess.TimeoutExpired:
                return self.format_error("Collections operation timeout")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else "Collections operation failed"
                return self.format_error(f"Collections operation failed: {error_msg}")
            except json.JSONDecodeError as e:
                return self.format_error(f"Invalid CLI response format: {e}")
                
        except Exception as e:
            logger.error(f"Collections tool execution failed: {e}", exc_info=True)
            return self.format_error(f"Collections tool execution failed: {str(e)}")
    
    def _format_collections_response(self, action: str, cli_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format CLI response for collections operations."""
        if action == "list":
            return {
                "action": action,
                "collections": cli_data.get("collections", []),
                "total_collections": len(cli_data.get("collections", []))
            }
        elif action == "create":
            return {
                "action": action,
                "collection_name": cli_data.get("collection_name", ""),
                "status": cli_data.get("status", "success"),
                "message": cli_data.get("message", "Collection created successfully")
            }
        elif action == "delete":
            return {
                "action": action,
                "collection_name": cli_data.get("collection_name", ""),
                "status": cli_data.get("status", "success"),
                "message": cli_data.get("message", "Collection deleted successfully")
            }
        elif action == "info":
            return {
                "action": action,
                "collection_info": cli_data.get("collection_info", {}),
                "documents_count": cli_data.get("documents_count", 0),
                "chunks_count": cli_data.get("chunks_count", 0)
            }
        elif action == "rename":
            return {
                "action": action,
                "old_name": cli_data.get("old_name", ""),
                "new_name": cli_data.get("new_name", ""),
                "status": cli_data.get("status", "success"),
                "message": cli_data.get("message", "Collection renamed successfully")
            }
        else:
            return cli_data 