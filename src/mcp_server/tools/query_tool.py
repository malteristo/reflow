"""
Query Knowledge Base Tool for MCP Server.

This tool provides enhanced query capabilities with rich result formatting,
including keyword highlighting, relevance indicators, and metadata display.
"""

import json
import logging
import subprocess
import asyncio
from typing import Dict, List, Any, Optional

from mcp_server.tools.base_tool import BaseMCPTool, ToolValidationError

# Import the new result formatter
from research_agent_backend.services.result_formatter import (
    format_results_for_cursor,
    create_result_markdown,
    ResultFormatter,
    FormattingOptions,
    DisplayFormat
)

logger = logging.getLogger(__name__)


class QueryKnowledgeBaseTool(BaseMCPTool):
    """
    Tool for querying the Research Agent knowledge base with enhanced formatting.
    
    Provides semantic search capabilities with rich result presentation including:
    - Keyword highlighting in content
    - Relevance score indicators  
    - Rich metadata display
    - Source document information
    - User feedback UI elements
    """

    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the query tool with CLI path."""
        super().__init__()
        self.cli_path = cli_path

    def get_tool_name(self) -> str:
        """Get the name of this tool."""
        return "query_knowledge_base"

    def get_tool_description(self) -> str:
        """Get the description of this tool."""
        return (
            "Search the Research Agent knowledge base for relevant information with "
            "rich formatting including keyword highlighting, relevance indicators, "
            "and metadata display. Supports filtering by collections and controlling "
            "result count."
        )

    def get_required_parameters(self) -> List[str]:
        """Get the list of required parameters."""
        return ["query"]

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute against the knowledge base"
                },
                "collections": {
                    "type": ["string", "null"],
                    "description": "Comma-separated list of collection names to search in (optional)"
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Maximum number of results to return"
                },
                "document_context": {
                    "type": ["string", "null"],
                    "description": "Additional context about the type of documents to search for"
                },
                "format_options": {
                    "type": "object",
                    "description": "Formatting options for result display",
                    "properties": {
                        "compact": {
                            "type": "boolean",
                            "default": False,
                            "description": "Use compact display mode"
                        },
                        "highlight_keywords": {
                            "type": "boolean", 
                            "default": True,
                            "description": "Enable keyword highlighting in results"
                        },
                        "show_metadata": {
                            "type": "boolean",
                            "default": True,
                            "description": "Show document metadata in results"
                        },
                        "show_relevance_scores": {
                            "type": "boolean",
                            "default": True,
                            "description": "Display relevance scores and indicators"
                        }
                    }
                }
            },
            "required": ["query"]
        }

    def validate_tool_parameters(self, parameters: Dict[str, Any]) -> List[ToolValidationError]:
        """Validate the tool parameters."""
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

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the query knowledge base tool with enhanced formatting.
        
        Args:
            parameters: Tool parameters
            
        Returns:
            Tool execution result with formatted results
        """
        try:
            # Extract parameters
            query = parameters["query"]
            collections = parameters.get("collections")
            top_k = parameters.get("top_k", 10)
            document_context = parameters.get("document_context")
            format_options = parameters.get("format_options", {})
            
            # Build CLI command
            args = ["query", query]
            
            if collections:
                if isinstance(collections, list):
                    args.extend(["--collections", ",".join(collections)])
                else:
                    args.extend(["--collections", str(collections)])
            
            args.extend(["--top-k", str(top_k)])
            
            if document_context:
                args.extend(["--context", document_context])
            
            # Add JSON output format
            args.append("--json")
            
            # Execute CLI command asynchronously
            try:
                cli_data = await self.invoke_cli_async(args)
                
                # Format response with enhanced formatting
                formatted_response = self._format_query_response_enhanced(
                    cli_data, query, format_options
                )
                
                return self.format_success_response(formatted_response)
                
            except subprocess.CalledProcessError as e:
                error_msg = str(e)
                return self.format_error(f"Query failed: {error_msg}")
            except subprocess.TimeoutExpired:
                return self.format_error("Query timeout - operation took longer than 30 seconds")
            except json.JSONDecodeError as e:
                return self.format_error(f"Invalid CLI response format: {e}")
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            return self.format_error(f"Query execution failed: {str(e)}")

    def _format_query_response_enhanced(
        self, 
        cli_data: Dict[str, Any], 
        query: str,
        format_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format CLI response with enhanced presentation using ResultFormatter.
        
        Args:
            cli_data: Raw CLI response data
            query: Original search query
            format_options: User-specified formatting options
            
        Returns:
            Enhanced formatted response
        """
        # Extract results
        raw_results = cli_data.get("results", [])
        
        if not raw_results:
            return {
                "status": "success",
                "summary": f"## No Results Found\n\nNo documents matched your query: **{query}**",
                "results": [],
                "query_refinement": {
                    "status": "no_results",
                    "suggestions": ["Try broader search terms", "Check spelling", "Use synonyms"],
                    "message": "No matching documents found"
                },
                "total_results": 0
            }
        
        # Create formatting options from user preferences
        compact_mode = format_options.get("compact", False)
        
        # Use the new result formatter for enhanced presentation
        try:
            formatted_results = format_results_for_cursor(
                raw_results, query, compact=compact_mode
            )
            
            # Create markdown representation for each result
            markdown_results = []
            structured_results = []
            
            for i, formatted_result in enumerate(formatted_results):
                # Create complete markdown for display
                result_markdown = create_result_markdown(formatted_result, i + 1)
                markdown_results.append(result_markdown)
                
                # Create structured data for programmatic access
                structured_result = {
                    "content": formatted_result.content,
                    "relevance_score": formatted_result.relevance_info.get("score", 0.0),
                    "relevance_label": formatted_result.relevance_info.get("label", "Unknown"),
                    "relevance_icon": formatted_result.relevance_info.get("icon", "📄"),
                    "source_document": formatted_result.raw_result.get("document_id", ""),
                    "header_path": formatted_result.raw_result.get("header_path", ""),
                    "metadata": {
                        "document_title": formatted_result.raw_result.get("metadata", {}).get("document_title", ""),
                        "content_type": formatted_result.raw_result.get("metadata", {}).get("content_type", "prose"),
                        "chunk_sequence_id": formatted_result.raw_result.get("metadata", {}).get("chunk_sequence_id", 0),
                        "collection": formatted_result.raw_result.get("collection", "")
                    },
                    "content_truncated": formatted_result.content_truncated,
                    "highlights_count": formatted_result.highlights_count
                }
                structured_results.append(structured_result)
            
            # Create query summary using the formatter
            formatter = ResultFormatter()
            summary = formatter.format_query_summary(
                formatted_results, query, len(raw_results)
            )
            
            # Query refinement feedback
            query_refinement = {
                "status": cli_data.get("query_status", "optimal"),
                "suggestions": cli_data.get("suggestions", []),
                "message": cli_data.get("refinement_message", "")
            }
            
            # Combine everything into enhanced response
            return {
                "status": "success",
                "summary": summary,
                "results_markdown": "\n".join(markdown_results),
                "results": structured_results,
                "query_refinement": query_refinement,
                "total_results": len(formatted_results),
                "formatting_info": {
                    "format_type": "enhanced_markdown",
                    "compact_mode": compact_mode,
                    "keyword_highlighting": True,
                    "relevance_indicators": True,
                    "metadata_display": True
                }
            }
            
        except Exception as e:
            logger.warning(f"Enhanced formatting failed, falling back to basic: {e}")
            # Fallback to basic formatting if enhanced formatting fails
            return self._format_query_response_basic(cli_data, query)

    def _format_query_response_basic(self, cli_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Fallback basic formatting method (maintains original functionality).
        
        Args:
            cli_data: Raw CLI response data
            query: Original search query
            
        Returns:
            Basic formatted response
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
        Convert relevance score to human-readable label (fallback method).
        
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