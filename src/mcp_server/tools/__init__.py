"""
MCP Tools module for Research Agent.

Provides all MCP tools that map CLI commands to MCP protocol operations.
Implements subtask 15.3: Map CLI Tools to MCP Resources and Actions.
Enhanced with model change detection tools for Task 35.
"""

from .base_tool import BaseMCPTool, ToolValidationError
from .query_tool import QueryKnowledgeBaseTool
from .collections_tool import ManageCollectionsTool
from .documents_tool import IngestDocumentsTool
from .projects_tool import ManageProjectsTool
from .augment_tool import AugmentKnowledgeTool
from .model_management_tools import (
    DetectModelChangesTool,
    ReindexCollectionTool,
    GetReindexStatusTool
)

__all__ = [
    "BaseMCPTool",
    "ToolValidationError",
    "QueryKnowledgeBaseTool",
    "ManageCollectionsTool", 
    "IngestDocumentsTool",
    "ManageProjectsTool",
    "AugmentKnowledgeTool",
    # Model change detection tools
    "DetectModelChangesTool",
    "ReindexCollectionTool", 
    "GetReindexStatusTool"
]
