"""
Model Management Tools for MCP Server.

Implements model change detection and reindexing management tools that integrate
with the existing model change detection system and enhanced collection metadata.

Addresses Task 35 - Model Change Detection Integration:
- detect_model_changes
- reindex_collection  
- get_reindex_status
"""

import json
import logging
import subprocess
import asyncio
from typing import Dict, Any, List, Optional

from .base_tool import BaseMCPTool, ToolValidationError

logger = logging.getLogger(__name__)


class DetectModelChangesTool(BaseMCPTool):
    """
    MCP tool for detecting model changes and identifying collections requiring re-indexing.
    
    Leverages the existing ModelChangeDetector singleton and enhanced CollectionMetadata
    to provide comprehensive change detection and impact assessment.
    
    Maps to CLI command:
    research-agent model check-changes --show-collections --auto-register
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the detect model changes tool."""
        super().__init__()
        self.cli_path = cli_path
        # Ensure legacy response format for compatibility with tests
        self.standardized_responses = False
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "detect_model_changes"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Detect embedding model changes and identify collections that require re-indexing. "
            "Uses the model change detection system to compare current model fingerprints "
            "with stored collection metadata and provides impact assessment."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return []  # No required parameters
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "auto_register": {
                    "type": "boolean",
                    "default": True,
                    "description": "Automatically register detected model changes"
                },
                "show_collections": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include affected collections in the response"
                },
                "check_compatibility": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check model compatibility with existing collections"
                }
            }
        }
    
    async def invoke_cli_async(self, args: List[str], timeout: int = 300) -> Dict[str, Any]:
        """
        Asynchronously invoke the CLI command.
        
        Args:
            args: Command line arguments
            timeout: Command timeout in seconds
            
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
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
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
            
        except asyncio.TimeoutError:
            logger.error(f"CLI command timed out after {timeout} seconds")
            raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Command timed out after {timeout}s")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CLI JSON output: {e}")
            raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Invalid JSON output: {e}")
        except Exception as e:
            logger.error(f"CLI invocation failed: {e}")
            raise

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the detect model changes tool."""
        try:
            # Build CLI command
            args = ["model", "check-changes"]
            
            if parameters.get("auto_register", True):
                args.append("--auto-register")
            
            if parameters.get("show_collections", True):
                args.append("--show-collections")
            
            # Add JSON output flag
            args.append("--json")
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Extract model change information
            change_data = cli_result.get("model_change", {})
            
            # Format standardized response
            response_data = {
                "change_detected": change_data.get("change_detected", False),
                "model_info": change_data.get("current_model", {}),
                "change_type": change_data.get("change_type"),
                "requires_reindexing": change_data.get("requires_reindexing", False),
                "affected_collections": change_data.get("affected_collections", []),
                "total_collections_affected": len(change_data.get("affected_collections", [])),
                "impact_level": change_data.get("impact_level", "low"),
                "auto_registered": parameters.get("auto_register", True) and change_data.get("change_detected", False)
            }
            
            # Add compatibility information if requested
            if parameters.get("check_compatibility", True):
                response_data["compatibility_check"] = change_data.get("compatibility_info", {})
            
            message = "No model changes detected"
            if change_data.get("change_detected", False):
                affected_count = len(change_data.get("affected_collections", []))
                message = f"Model change detected affecting {affected_count} collection(s)"
            
            return self.format_success_response(
                data=response_data,
                message=message,
                operation="detect_model_changes"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to detect model changes: {e}",
                operation="detect_model_changes"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error detecting model changes: {e}",
                operation="detect_model_changes"
            )


class ReindexCollectionTool(BaseMCPTool):
    """
    MCP tool for triggering collection re-indexing with the current model.
    
    Integrates with existing reindexing infrastructure and enhanced CollectionMetadata
    to provide progress tracking and status management.
    
    Maps to CLI command:
    research-agent model reindex --collections="..." --parallel --force
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the reindex collection tool."""
        super().__init__()
        self.cli_path = cli_path
        # Ensure legacy response format for compatibility with tests
        self.standardized_responses = False
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "reindex_collection"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Re-index one or more collections with the current embedding model. "
            "Supports parallel processing, progress tracking, and status management "
            "using enhanced collection metadata."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters for this tool."""
        return []  # No required parameters - defaults to all collections
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "collections": {
                    "type": ["string", "null"],
                    "description": "Comma-separated list of collections to re-index (default: all collections requiring re-indexing)"
                },
                "parallel": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use parallel processing for improved performance"
                },
                "workers": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                    "description": "Number of worker threads for parallel processing (default: auto)"
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 50,
                    "description": "Batch size for processing documents"
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip confirmation prompts and force re-indexing"
                },
                "track_progress": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable progress tracking using enhanced metadata"
                }
            }
        }

    async def invoke_cli_async(self, args: List[str], timeout: int = 3600) -> Dict[str, Any]:
        """
        Asynchronously invoke the CLI command with extended timeout for reindexing.
        
        Args:
            args: Command line arguments
            timeout: Command timeout in seconds (default: 1 hour)
            
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
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
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
            
        except asyncio.TimeoutError:
            logger.error(f"CLI command timed out after {timeout} seconds")
            raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Command timed out after {timeout}s")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CLI JSON output: {e}")
            raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Invalid JSON output: {e}")
        except Exception as e:
            logger.error(f"CLI invocation failed: {e}")
            raise
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the reindex collection tool."""
        try:
            # Build CLI command
            args = ["model", "reindex"]
            
            if parameters.get("collections"):
                args.extend(["--collections", parameters["collections"]])
            
            if parameters.get("parallel", True):
                args.append("--parallel")
            else:
                args.append("--sequential")
            
            if parameters.get("workers"):
                args.extend(["--workers", str(parameters["workers"])])
            
            batch_size = parameters.get("batch_size", 50)
            args.extend(["--batch-size", str(batch_size)])
            
            if parameters.get("force", False):
                args.append("--force")
            
            # Add JSON output flag
            args.append("--json")
            
            # Execute CLI command (this may take a while for large collections)
            cli_result = await self.invoke_cli_async(args, timeout=3600)  # 1 hour timeout
            
            # Extract reindexing results
            reindex_data = cli_result.get("reindex_result", {})
            
            # Format standardized response
            response_data = {
                "success": reindex_data.get("success", False),
                "collections_processed": reindex_data.get("successful_collections", 0),
                "collections_failed": reindex_data.get("failed_collections", 0),
                "total_documents_processed": reindex_data.get("total_documents", 0),
                "processing_time_seconds": reindex_data.get("elapsed_time_seconds", 0),
                "performance_metrics": reindex_data.get("performance", {}),
                "collection_results": reindex_data.get("collection_results", {}),
                "errors": reindex_data.get("errors", []),
                "parallel_processing": parameters.get("parallel", True),
                "workers_used": reindex_data.get("workers_used", 1),
                "batch_size": batch_size
            }
            
            # Calculate performance metrics
            if response_data["total_documents_processed"] > 0 and response_data["processing_time_seconds"] > 0:
                response_data["documents_per_second"] = (
                    response_data["total_documents_processed"] / response_data["processing_time_seconds"]
                )
            
            if reindex_data.get("success", False):
                message = f"Successfully re-indexed {response_data['collections_processed']} collection(s)"
                if response_data["collections_failed"] > 0:
                    message += f" with {response_data['collections_failed']} failure(s)"
            else:
                message = f"Re-indexing failed for {response_data['collections_failed']} collection(s)"
            
            return self.format_success_response(
                data=response_data,
                message=message,
                operation="reindex_collection"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to re-index collections: {e}",
                operation="reindex_collection"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error during re-indexing: {e}",
                operation="reindex_collection"
            )


class GetReindexStatusTool(BaseMCPTool):
    """
    MCP tool for monitoring re-indexing progress and status.
    
    Leverages enhanced CollectionMetadata to provide real-time status updates,
    progress tracking, and completion estimates.
    
    Maps to CLI functionality but provides structured status information.
    """
    
    def __init__(self, cli_path: str = "research-agent-cli"):
        """Initialize the get reindex status tool."""
        super().__init__()
        self.cli_path = cli_path
        # Ensure legacy response format for compatibility with tests
        self.standardized_responses = False
    
    def get_tool_name(self) -> str:
        """Get the tool name for MCP registration."""
        return "get_reindex_status"
    
    def get_tool_description(self) -> str:
        """Get the tool description."""
        return (
            "Get the current re-indexing status for collections using enhanced metadata. "
            "Provides progress tracking, completion estimates, and detailed status information "
            "for ongoing or completed re-indexing operations."
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
                    "description": "Specific collection name to check status for (default: all collections)"
                },
                "show_completed": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include completed re-indexing operations in the response"
                },
                "show_progress_details": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include detailed progress information and metrics"
                },
                "include_history": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include historical re-indexing information"
                }
            }
        }
    
    async def invoke_cli_async(self, args: List[str], timeout: int = 60) -> Dict[str, Any]:
        """
        Asynchronously invoke the CLI command.
        
        Args:
            args: Command line arguments
            timeout: Command timeout in seconds
            
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
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
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
            
        except asyncio.TimeoutError:
            logger.error(f"CLI command timed out after {timeout} seconds")
            raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Command timed out after {timeout}s")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CLI JSON output: {e}")
            raise subprocess.CalledProcessError(1, [self.cli_path] + args, f"Invalid JSON output: {e}")
        except Exception as e:
            logger.error(f"CLI invocation failed: {e}")
            raise

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the get reindex status tool."""
        try:
            # Build CLI command to get collection status
            args = ["collections", "list", "--stats"]
            
            if parameters.get("collection"):
                # Get specific collection info
                args = ["collections", "info", parameters["collection"]]
            
            # Add JSON output flag
            args.append("--json")
            
            # Execute CLI command
            cli_result = await self.invoke_cli_async(args)
            
            # Process collection status information
            if parameters.get("collection"):
                # Single collection status
                collection_data = cli_result.get("collection", {})
                collections_status = [collection_data] if collection_data else []
            else:
                # All collections status
                collections_status = cli_result.get("collections", [])
            
            # Extract reindex status information from enhanced metadata
            status_summary = {
                "total_collections": len(collections_status),
                "pending": 0,
                "in_progress": 0,
                "completed": 0,
                "failed": 0,
                "not_required": 0,
                "collections": []
            }
            
            for collection in collections_status:
                # Extract enhanced metadata fields
                reindex_status = collection.get("reindex_status", "not_required")
                collection_status = {
                    "collection_name": collection.get("collection_name", ""),
                    "reindex_status": reindex_status,
                    "model_fingerprint": collection.get("embedding_model_fingerprint"),
                    "model_name": collection.get("model_name"),
                    "model_version": collection.get("model_version"),
                    "last_reindex": collection.get("last_reindex_timestamp"),
                    "document_count": collection.get("document_count", 0),
                    "original_document_count": collection.get("original_document_count", 0),
                    "requires_reindexing": collection.get("requires_reindexing", False)
                }
                
                # Add progress details if requested
                if parameters.get("show_progress_details", True):
                    collection_status["progress_info"] = {
                        "completion_percentage": self._calculate_progress(collection),
                        "documents_processed": collection.get("document_count", 0),
                        "estimated_total": collection.get("original_document_count", 0),
                        "processing_time": self._calculate_processing_time(collection),
                        "last_updated": collection.get("updated_at")
                    }
                
                # Count status types
                status_summary[reindex_status] = status_summary.get(reindex_status, 0) + 1
                
                # Include in response based on filters
                if (parameters.get("show_completed", True) or reindex_status != "completed"):
                    status_summary["collections"].append(collection_status)
            
            # Add overall summary
            response_data = {
                "status_summary": status_summary,
                "has_active_operations": status_summary["in_progress"] > 0,
                "requires_attention": status_summary["pending"] + status_summary["failed"] > 0,
                "timestamp": cli_result.get("timestamp"),
                "model_info": cli_result.get("current_model", {})
            }
            
            # Create descriptive message
            active_count = status_summary["in_progress"]
            pending_count = status_summary["pending"] 
            failed_count = status_summary["failed"]
            
            if active_count > 0:
                message = f"{active_count} collection(s) currently being re-indexed"
            elif pending_count > 0:
                message = f"{pending_count} collection(s) pending re-indexing"
            elif failed_count > 0:
                message = f"{failed_count} collection(s) failed re-indexing"
            else:
                message = "All collections are up to date"
            
            return self.format_success_response(
                data=response_data,
                message=message,
                operation="get_reindex_status"
            )
            
        except subprocess.CalledProcessError as e:
            return self.format_error(
                message=f"Failed to get reindex status: {e}",
                operation="get_reindex_status"
            )
        except Exception as e:
            return self.format_error(
                message=f"Unexpected error getting reindex status: {e}",
                operation="get_reindex_status"
            )
    
    def _calculate_progress(self, collection: Dict[str, Any]) -> float:
        """Calculate reindexing progress percentage."""
        original_count = collection.get("original_document_count", 0)
        current_count = collection.get("document_count", 0)
        
        if original_count == 0:
            return 0.0
        
        return min(100.0, (current_count / original_count) * 100.0)
    
    def _calculate_processing_time(self, collection: Dict[str, Any]) -> Optional[str]:
        """Calculate processing time if available."""
        last_reindex = collection.get("last_reindex_timestamp")
        if not last_reindex:
            return None
        
        # This would need additional timestamp tracking for accurate calculation
        # For now, return the timestamp as-is
        return last_reindex 