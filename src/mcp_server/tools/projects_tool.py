"""
Manage Projects MCP Tool for Research Agent.

Maps CLI projects commands to MCP protocol operations, providing
project management functionality.

Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
"""

import json
import logging
import subprocess
from typing import Dict, Any, List

from .base_tool import BaseMCPTool, ToolValidationError

logger = logging.getLogger(__name__)


class ManageProjectsTool(BaseMCPTool):
    """
    MCP tool for managing projects.
    
    Maps to CLI commands:
    - research-agent projects create <name> [options]
    - research-agent projects list
    - research-agent projects info <name>
    - research-agent projects activate <name>
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the manage projects tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "manage_projects"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Create, list, and manage research projects. "
            "Supports project activation and information retrieval."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["action"]
    
    def get_supported_actions(self) -> List[str]:
        """Get supported project actions."""
        return ["create", "list", "info", "activate"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": self.get_supported_actions(),
                    "description": "Action to perform on projects"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name of the project (required for create/info/activate)"
                },
                "description": {
                    "type": "string",
                    "description": "Description for the project (optional for create)"
                },
                "collections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Collections to link to the project (optional for create)"
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
        if action in ["create", "info", "activate"]:
            if not parameters.get("project_name"):
                errors.append(ToolValidationError(
                    parameter="project_name",
                    message=f"project_name is required for action '{action}'",
                    value=parameters.get("project_name")
                ))
        
        return errors
    
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the manage projects tool."""
        try:
            action = parameters["action"]
            
            # Build CLI command based on action
            if action == "list":
                args = ["projects", "list", "--json"]
            elif action == "create":
                args = ["projects", "create", parameters["project_name"]]
                if parameters.get("description"):
                    args.extend(["--description", parameters["description"]])
                if parameters.get("collections"):
                    args.extend(["--collections", ",".join(parameters["collections"])])
                args.append("--json")
            elif action == "info":
                args = ["projects", "info", parameters["project_name"], "--json"]
            elif action == "activate":
                args = ["projects", "activate", parameters["project_name"], "--json"]
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
                formatted_response = self._format_projects_response(action, cli_data)
                return self.format_success_response(formatted_response)
                
            except subprocess.TimeoutExpired:
                return self.format_error("Projects operation timeout")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else "Projects operation failed"
                return self.format_error(f"Projects operation failed: {error_msg}")
            except json.JSONDecodeError as e:
                return self.format_error(f"Invalid CLI response format: {e}")
                
        except Exception as e:
            logger.error(f"Projects tool execution failed: {e}", exc_info=True)
            return self.format_error(f"Projects tool execution failed: {str(e)}")
    
    def _format_projects_response(self, action: str, cli_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format CLI response for project operations."""
        if action == "list":
            return {
                "action": action,
                "projects": cli_data.get("projects", []),
                "active_project": cli_data.get("active_project"),
                "total_projects": len(cli_data.get("projects", []))
            }
        elif action == "create":
            return {
                "action": action,
                "project_name": cli_data.get("project_name", ""),
                "status": cli_data.get("status", "success"),
                "message": cli_data.get("message", "Project created successfully")
            }
        elif action == "info":
            return {
                "action": action,
                "project_info": cli_data.get("project_info", {}),
                "linked_collections": cli_data.get("linked_collections", []),
                "documents_count": cli_data.get("documents_count", 0)
            }
        elif action == "activate":
            return {
                "action": action,
                "project_name": cli_data.get("project_name", ""),
                "status": cli_data.get("status", "success"),
                "message": cli_data.get("message", "Project activated successfully")
            }
        else:
            return cli_data 