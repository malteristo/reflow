"""
Augment Knowledge MCP Tool for Research Agent.

Maps CLI augmentation commands to MCP protocol operations, providing
knowledge base augmentation functionality.

Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
"""

import json
import logging
import subprocess
from typing import Dict, Any, List
import asyncio

from .base_tool import BaseMCPTool, ToolValidationError

logger = logging.getLogger(__name__)


class AugmentKnowledgeTool(BaseMCPTool):
    """
    MCP tool for augmenting knowledge base.
    
    Maps to CLI commands:
    - research-agent augment <text> [options]
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the augment knowledge tool."""
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "augment_knowledge"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Augment the knowledge base with enhanced content from external sources. "
            "Provides research-backed knowledge expansion capabilities."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["text"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text or topic to augment with additional knowledge"
                },
                "collection": {
                    "type": "string",
                    "description": "Target collection for augmented content"
                },
                "depth": {
                    "type": "string",
                    "enum": ["shallow", "medium", "deep"],
                    "description": "Depth of augmentation research",
                    "default": "medium"
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred knowledge sources for augmentation"
                }
            },
            "required": ["text"]
        }
    
    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """Validate tool-specific parameters."""
        errors = []
        
        # Validate text
        text = parameters.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            errors.append(ToolValidationError(
                parameter="text",
                message="text must be a non-empty string",
                value=text
            ))
        
        return errors
    
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the augment knowledge tool."""
        try:
            # Build CLI command
            args = ["augment", parameters["text"]]
            
            if parameters.get("collection"):
                args.extend(["--collection", parameters["collection"]])
            
            if parameters.get("depth"):
                args.extend(["--depth", parameters["depth"]])
            
            if parameters.get("sources"):
                args.extend(["--sources", ",".join(parameters["sources"])])
            
            # Add JSON output format
            args.append("--json")
            
            # Execute CLI command
            try:
                result = subprocess.run(
                    [self.cli_path] + args,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=180  # 3 minutes for augmentation
                )
                
                # Parse CLI output
                if result.stdout.strip():
                    cli_data = json.loads(result.stdout.strip())
                else:
                    cli_data = {"status": "success"}
                
                # Format response
                formatted_response = self._format_augment_response(cli_data)
                return self.format_success_response(formatted_response)
                
            except subprocess.TimeoutExpired:
                return self.format_error("Augmentation timeout - operation took longer than 3 minutes")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else "Augmentation failed"
                return self.format_error(f"Augmentation failed: {error_msg}")
            except json.JSONDecodeError as e:
                return self.format_error(f"Invalid CLI response format: {e}")
                
        except Exception as e:
            logger.error(f"Augment tool execution failed: {e}", exc_info=True)
            return self.format_error(f"Augment tool execution failed: {str(e)}")
    
    def _format_augment_response(self, cli_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format CLI response for augmentation operations."""
        return {
            "status": cli_data.get("status", "success"),
            "original_text": cli_data.get("original_text", ""),
            "augmented_content": cli_data.get("augmented_content", []),
            "collection": cli_data.get("collection", ""),
            "chunks_added": cli_data.get("chunks_added", 0),
            "sources_used": cli_data.get("sources_used", []),
            "processing_time": cli_data.get("processing_time", 0),
            "message": cli_data.get("message", "Knowledge augmentation completed successfully")
        }


class SubmitFeedbackTool(BaseMCPTool):
    """Tool for submitting user feedback on knowledge base chunks."""
    
    @property
    def name(self) -> str:
        return "submit_feedback"
    
    @property
    def description(self) -> str:
        return "Submit user feedback (thumbs up/down) for a knowledge base chunk to improve content quality"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "ID of the chunk to provide feedback on"
                },
                "rating": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "Rating for the chunk: positive (thumbs up), negative (thumbs down), or neutral"
                },
                "reason": {
                    "type": "string", 
                    "description": "Reason for the rating (e.g., 'very-relevant', 'incorrect-information', 'outdated-info')"
                },
                "comment": {
                    "type": "string",
                    "description": "Optional detailed comment about the chunk quality",
                    "default": None
                },
                "user_id": {
                    "type": "string",
                    "description": "Optional user ID for non-anonymous feedback",
                    "default": None
                }
            },
            "required": ["chunk_id", "rating", "reason"]
        }
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feedback submission."""
        try:
            chunk_id = params["chunk_id"]
            rating = params["rating"]
            reason = params["reason"]
            comment = params.get("comment")
            user_id = params.get("user_id")
            
            # Build CLI command
            cmd = [
                "python", "-m", "research_agent_backend.cli.cli",
                "kb", "feedback",
                "--chunk-id", chunk_id,
                "--rating", rating,
                "--reason", reason
            ]
            
            if comment:
                cmd.extend(["--comment", comment])
            if user_id:
                cmd.extend(["--user-id", user_id])
            
            # Execute CLI command
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Feedback recorded for chunk {chunk_id}",
                    "rating": rating,
                    "reason": reason,
                    "chunk_id": chunk_id,
                    "impact": "Quality score updated based on your feedback"
                }
            else:
                error_msg = stderr.decode() if stderr else "Unknown error occurred"
                return {
                    "status": "error",
                    "message": f"Failed to submit feedback: {error_msg}",
                    "chunk_id": chunk_id
                }
                
        except Exception as e:
            logger.error(f"Error in submit_feedback: {e}")
            return {
                "status": "error", 
                "message": f"Failed to submit feedback: {str(e)}"
            } 