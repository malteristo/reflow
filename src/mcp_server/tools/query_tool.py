"""
Query Knowledge Base MCP Tool for Research Agent.

Maps CLI query commands to MCP protocol operations, providing search
functionality with re-ranking and query refinement.

Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, Any, List, Optional

from .base_tool import BaseMCPTool, ToolValidationError

logger = logging.getLogger(__name__)


class QueryKnowledgeBaseTool(BaseMCPTool):
    """
    MCP tool for querying the knowledge base.
    
    Maps to CLI commands:
    - research-agent query <query> [options]
    - research-agent ask <query> [options]
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """
        Initialize the query knowledge base tool.
        
        Args:
            cli_path: Path to the research agent CLI executable
        """
        super().__init__()
        self.cli_path = cli_path
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "query_knowledge_base"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Search the knowledge base with semantic similarity and re-ranking. "
            "Supports collection filtering, document context, and query refinement."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return ["query"]
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User's search query"
                },
                "collections": {
                    "type": ["string", "null"],
                    "description": "Comma-separated collection names to search"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                },
                "document_context": {
                    "type": ["string", "null"],
                    "description": "Current document context for enhanced search"
                }
            },
            "required": ["query"]
        }
    
    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """Validate tool-specific parameters."""
        errors = []
        
        # Validate query
        if "query" in parameters:
            query = parameters["query"]
            if not isinstance(query, str) or not query.strip():
                errors.append(ToolValidationError(
                    parameter="query",
                    message="Query must be a non-empty string",
                    value=query
                ))
        
        # Validate top_k
        if "top_k" in parameters:
            top_k = parameters["top_k"]
            if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
                errors.append(ToolValidationError(
                    parameter="top_k",
                    message="top_k must be an integer between 1 and 100",
                    value=top_k
                ))
        
        # Validate collections format
        if "collections" in parameters and parameters["collections"] is not None:
            collections = parameters["collections"]
            if not isinstance(collections, str):
                errors.append(ToolValidationError(
                    parameter="collections",
                    message="collections must be a string or null",
                    value=collections
                ))
        
        return errors
    
    def get_cli_command_pattern(self) -> str:
        """Get the CLI command pattern for this tool."""
        return 'research-agent query "{query}" --collections="{collections}" --top-k={top_k}'
    
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
                return {"results": [], "total_results": 0}
            
            return json.loads(output)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CLI JSON output: {e}")
            raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Invalid JSON output: {e}")
        except Exception as e:
            logger.error(f"CLI invocation failed: {e}")
            raise
    
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the query knowledge base tool.
        
        Args:
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            # Extract parameters
            query = parameters["query"]
            collections = parameters.get("collections")
            top_k = parameters.get("top_k", 10)
            document_context = parameters.get("document_context")
            
            # Build CLI command
            args = ["query", query]
            
            if collections:
                args.extend(["--collections", collections])
            
            args.extend(["--top-k", str(top_k)])
            
            if document_context:
                args.extend(["--context", document_context])
            
            # Add JSON output format
            args.append("--json")
            
            # Execute CLI command synchronously for now
            # TODO: Use async version when MCP supports it
            try:
                result = subprocess.run(
                    [self.cli_path] + args,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30  # 30 second timeout
                )
                
                # Parse CLI output
                if result.stdout.strip():
                    cli_data = json.loads(result.stdout.strip())
                else:
                    cli_data = {"results": [], "total_results": 0}
                
                # Format response according to MCP protocol specification
                formatted_response = self._format_query_response(cli_data)
                
                return self.format_success_response(formatted_response)
                
            except subprocess.TimeoutExpired:
                return self.format_error("Query timeout - operation took longer than 30 seconds")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else "CLI query failed"
                return self.format_error(f"Query failed: {error_msg}")
            except json.JSONDecodeError as e:
                return self.format_error(f"Invalid CLI response format: {e}")
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            return self.format_error(f"Query execution failed: {str(e)}")
    
    def _format_query_response(self, cli_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format CLI response according to MCP protocol specification.
        
        Args:
            cli_data: Raw CLI response data
            
        Returns:
            Formatted response according to protocol spec
        """
        # Extract results
        results = cli_data.get("results", [])
        formatted_results = []
        
        for result in results:
            formatted_result = {
                "content": result.get("content", ""),
                "relevance_score": result.get("score", 0.0),
                "relevance_label": self._get_relevance_label(result.get("score", 0.0)),
                "source_document": result.get("document_id", ""),
                "header_path": result.get("header_path", ""),
                "metadata": {
                    "document_title": result.get("document_title", ""),
                    "content_type": result.get("content_type", "prose"),
                    "chunk_sequence_id": result.get("chunk_id", 0),
                    "collection": result.get("collection", "")
                }
            }
            formatted_results.append(formatted_result)
        
        # Query refinement feedback
        query_refinement = {
            "status": cli_data.get("query_status", "optimal"),
            "suggestions": cli_data.get("suggestions", []),
            "message": cli_data.get("refinement_message", "")
        }
        
        return {
            "status": "success",
            "results": formatted_results,
            "query_refinement": query_refinement,
            "total_results": len(formatted_results)
        }
    
    def _get_relevance_label(self, score: float) -> str:
        """
        Convert relevance score to human-readable label.
        
        Args:
            score: Relevance score (0.0 to 1.0)
            
        Returns:
            Human-readable relevance label
        """
        if score >= 0.9:
            return "Highly Relevant"
        elif score >= 0.7:
            return "Very Relevant"
        elif score >= 0.5:
            return "Relevant"
        elif score >= 0.3:
            return "Somewhat Relevant"
        else:
            return "Low Relevance" 