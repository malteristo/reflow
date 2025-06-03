"""
Tests for Model Management MCP Tools.

Tests the model change detection tools that integrate with the existing
model change detection system and enhanced collection metadata.

Implements testing for Task 35 - Model Change Detection Integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
import subprocess

from src.mcp_server.tools.model_management_tools import (
    DetectModelChangesTool,
    ReindexCollectionTool,
    GetReindexStatusTool
)
from src.mcp_server.tools.base_tool import ToolValidationError


class TestDetectModelChangesTool:
    """Test the DetectModelChangesTool MCP tool."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.tool = DetectModelChangesTool()
    
    def test_tool_metadata(self):
        """Test tool metadata is correct."""
        assert self.tool.get_tool_name() == "detect_model_changes"
        
        description = self.tool.get_tool_description()
        assert "detect" in description.lower()
        assert "model changes" in description.lower()
        assert "collection" in description.lower()
        
        required_params = self.tool.get_required_parameters()
        assert required_params == []  # No required parameters
    
    def test_parameter_schema(self):
        """Test parameter schema is well-formed."""
        schema = self.tool.get_parameter_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        
        # Check expected properties
        properties = schema["properties"]
        assert "auto_register" in properties
        assert "show_collections" in properties
        assert "check_compatibility" in properties
        
        # Check property types
        assert properties["auto_register"]["type"] == "boolean"
        assert properties["show_collections"]["type"] == "boolean"
        assert properties["check_compatibility"]["type"] == "boolean"
        
        # Check defaults
        assert properties["auto_register"]["default"] is True
        assert properties["show_collections"]["default"] is True
        assert properties["check_compatibility"]["default"] is True
    
    @pytest.mark.asyncio
    async def test_execute_no_changes_detected(self):
        """Test execute when no model changes are detected."""
        # Mock CLI result for no changes
        mock_cli_result = {
            "model_change": {
                "change_detected": False,
                "current_model": {"name": "test-model", "version": "1.0"},
                "requires_reindexing": False,
                "affected_collections": [],
                "impact_level": "low"
            }
        }
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result):
            result = await self.tool.execute({"auto_register": True})
        
        assert result["status"] == "success"
        assert result["data"]["change_detected"] is False
        assert result["data"]["requires_reindexing"] is False
        assert result["data"]["affected_collections"] == []
        assert result["data"]["total_collections_affected"] == 0
        assert result["message"] == "No model changes detected"
    
    @pytest.mark.asyncio
    async def test_execute_changes_detected(self):
        """Test execute when model changes are detected."""
        # Mock CLI result for detected changes
        mock_cli_result = {
            "model_change": {
                "change_detected": True,
                "current_model": {"name": "new-model", "version": "2.0"},
                "change_type": "model_update",
                "requires_reindexing": True,
                "affected_collections": ["collection1", "collection2"],
                "impact_level": "medium",
                "compatibility_info": {"status": "requires_reindex"}
            }
        }
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result):
            result = await self.tool.execute({
                "auto_register": True,
                "show_collections": True,
                "check_compatibility": True
            })
        
        assert result["status"] == "success"
        assert result["data"]["change_detected"] is True
        assert result["data"]["requires_reindexing"] is True
        assert result["data"]["affected_collections"] == ["collection1", "collection2"]
        assert result["data"]["total_collections_affected"] == 2
        assert result["data"]["impact_level"] == "medium"
        assert result["data"]["auto_registered"] is True
        assert result["data"]["compatibility_check"]["status"] == "requires_reindex"
        assert "affecting 2 collection(s)" in result["message"]
    
    @pytest.mark.asyncio
    async def test_execute_cli_error(self):
        """Test handling of CLI errors."""
        tool = DetectModelChangesTool()
        
        # Mock CLI call to raise an error
        with patch.object(tool, 'invoke_cli_async') as mock_cli:
            mock_cli.side_effect = subprocess.CalledProcessError(1, ["research-agent-cli"], "CLI failed")
            
            result = await tool.execute({})
            
            assert result["status"] == "error"
            assert "Failed to detect model changes" in result["message"]
    
    @pytest.mark.asyncio
    async def test_cli_command_building(self):
        """Test that CLI commands are built correctly."""
        mock_cli_result = {"model_change": {"change_detected": False}}
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result) as mock_cli:
            await self.tool.execute({
                "auto_register": True,
                "show_collections": True,
                "check_compatibility": False
            })
        
        # Verify CLI was called with correct arguments
        mock_cli.assert_called_once()
        args = mock_cli.call_args[0][0]
        
        assert "model" in args
        assert "check-changes" in args
        assert "--auto-register" in args
        assert "--show-collections" in args
        assert "--json" in args


class TestReindexCollectionTool:
    """Test the ReindexCollectionTool MCP tool."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.tool = ReindexCollectionTool()
    
    def test_tool_metadata(self):
        """Test tool metadata is correct."""
        assert self.tool.get_tool_name() == "reindex_collection"
        
        description = self.tool.get_tool_description()
        assert "re-index" in description.lower()
        assert "collection" in description.lower()
        assert "parallel" in description.lower()
        
        required_params = self.tool.get_required_parameters()
        assert required_params == []  # No required parameters
    
    def test_parameter_schema(self):
        """Test parameter schema is well-formed."""
        schema = self.tool.get_parameter_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        
        # Check expected properties
        properties = schema["properties"]
        assert "collections" in properties
        assert "parallel" in properties
        assert "workers" in properties
        assert "batch_size" in properties
        assert "force" in properties
        assert "track_progress" in properties
        
        # Check property constraints
        assert properties["workers"]["minimum"] == 1
        assert properties["workers"]["maximum"] == 16
        assert properties["batch_size"]["minimum"] == 1
        assert properties["batch_size"]["maximum"] == 1000
        assert properties["batch_size"]["default"] == 50
    
    @pytest.mark.asyncio
    async def test_execute_successful_reindex(self):
        """Test execute with successful reindexing."""
        # Mock CLI result for successful reindex
        mock_cli_result = {
            "reindex_result": {
                "success": True,
                "successful_collections": 2,
                "failed_collections": 0,
                "total_documents": 150,
                "elapsed_time_seconds": 30.5,
                "performance": {"throughput": "4.9 docs/sec"},
                "collection_results": {"collection1": "success", "collection2": "success"},
                "errors": [],
                "workers_used": 4
            }
        }
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result):
            result = await self.tool.execute({
                "collections": "collection1,collection2",
                "parallel": True,
                "workers": 4,
                "batch_size": 25,
                "force": True
            })
        
        assert result["status"] == "success"
        assert result["data"]["success"] is True
        assert result["data"]["collections_processed"] == 2
        assert result["data"]["collections_failed"] == 0
        assert result["data"]["total_documents_processed"] == 150
        assert result["data"]["processing_time_seconds"] == 30.5
        assert result["data"]["parallel_processing"] is True
        assert result["data"]["workers_used"] == 4
        assert result["data"]["batch_size"] == 25
        assert "documents_per_second" in result["data"]
        assert result["message"] == "Successfully re-indexed 2 collection(s)"
    
    @pytest.mark.asyncio
    async def test_execute_partial_failure(self):
        """Test execute with partial failures."""
        # Mock CLI result for partial failure
        mock_cli_result = {
            "reindex_result": {
                "success": True,
                "successful_collections": 1,
                "failed_collections": 1,
                "total_documents": 75,
                "elapsed_time_seconds": 20.0,
                "performance": {},
                "collection_results": {"collection1": "success", "collection2": "failed"},
                "errors": ["Collection2 failed: timeout"],
                "workers_used": 2
            }
        }
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result):
            result = await self.tool.execute({"collections": "collection1,collection2"})
        
        assert result["status"] == "success"
        assert result["data"]["collections_processed"] == 1
        assert result["data"]["collections_failed"] == 1
        assert result["data"]["errors"] == ["Collection2 failed: timeout"]
        assert "with 1 failure(s)" in result["message"]
    
    @pytest.mark.asyncio
    async def test_execute_complete_failure(self):
        """Test execute with complete failure."""
        # Mock CLI result for complete failure
        mock_cli_result = {
            "reindex_result": {
                "success": False,
                "successful_collections": 0,
                "failed_collections": 2,
                "total_documents": 0,
                "elapsed_time_seconds": 5.0,
                "errors": ["Database connection failed", "Vector store unavailable"]
            }
        }
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result):
            result = await self.tool.execute({})
        
        assert result["status"] == "success"  # Tool succeeded, but reindexing failed
        assert result["data"]["success"] is False
        assert result["data"]["collections_failed"] == 2
        assert len(result["data"]["errors"]) == 2
        assert "failed for 2 collection(s)" in result["message"]
    
    @pytest.mark.asyncio
    async def test_cli_command_building_detailed(self):
        """Test that CLI commands are built with all parameters."""
        mock_cli_result = {"reindex_result": {"success": True}}
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result) as mock_cli:
            await self.tool.execute({
                "collections": "test-collection",
                "parallel": False,
                "workers": 8,
                "batch_size": 100,
                "force": True
            })
        
        # Verify CLI was called with correct arguments
        args = mock_cli.call_args[0][0]
        
        assert "model" in args
        assert "reindex" in args
        assert "--collections" in args
        assert "test-collection" in args
        assert "--sequential" in args  # parallel=False
        assert "--workers" in args
        assert "8" in args
        assert "--batch-size" in args
        assert "100" in args
        assert "--force" in args
        assert "--json" in args


class TestGetReindexStatusTool:
    """Test the GetReindexStatusTool MCP tool."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.tool = GetReindexStatusTool()
    
    def test_tool_metadata(self):
        """Test tool metadata is correct."""
        assert self.tool.get_tool_name() == "get_reindex_status"
        
        description = self.tool.get_tool_description()
        assert "status" in description.lower()
        assert "re-indexing" in description.lower()
        assert "progress" in description.lower()
        
        required_params = self.tool.get_required_parameters()
        assert required_params == []  # No required parameters
    
    def test_parameter_schema(self):
        """Test parameter schema is well-formed."""
        schema = self.tool.get_parameter_schema()
        
        assert schema["type"] == "object"
        assert "properties" in schema
        
        # Check expected properties
        properties = schema["properties"]
        assert "collection" in properties
        assert "show_completed" in properties
        assert "show_progress_details" in properties
        assert "include_history" in properties
        
        # Check defaults
        assert properties["show_completed"]["default"] is True
        assert properties["show_progress_details"]["default"] is True
        assert properties["include_history"]["default"] is False
    
    @pytest.mark.asyncio
    async def test_execute_all_collections_status(self):
        """Test getting status for all collections."""
        # Mock CLI result with collection status
        mock_cli_result = {
            "collections": [
                {
                    "collection_name": "collection1",
                    "reindex_status": "completed",
                    "embedding_model_fingerprint": "fingerprint1",
                    "model_name": "model1",
                    "model_version": "1.0",
                    "last_reindex_timestamp": "2024-01-01T10:00:00Z",
                    "document_count": 25,
                    "original_document_count": 50,
                    "requires_reindexing": False,
                    "updated_at": "2024-01-01T10:00:00Z"
                },
                {
                    "collection_name": "collection2", 
                    "reindex_status": "in_progress",
                    "document_count": 50,
                    "original_document_count": 75,
                    "requires_reindexing": True
                },
                {
                    "collection_name": "collection3",
                    "reindex_status": "pending",
                    "document_count": 40,
                    "original_document_count": 50,
                    "requires_reindexing": True
                }
            ],
            "timestamp": "2024-01-01T12:00:00Z",
            "current_model": {"name": "current-model", "version": "2.0"}
        }
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result):
            result = await self.tool.execute({"show_progress_details": True})
        
        assert result["status"] == "success"
        
        # Check that all status types are counted correctly
        status_summary = result["data"]["status_summary"]
        assert status_summary["total_collections"] == 3
        assert status_summary["in_progress"] == 1
        assert status_summary["completed"] == 1
        assert status_summary["pending"] == 1
        
        # Check that individual collection details are correct
        collections = status_summary["collections"]
        assert len(collections) == 3
        
        # Collection 1: completed (25/50 documents) - 50%
        collection1 = collections[0]
        assert collection1["reindex_status"] == "completed"
        assert collection1["progress_info"]["completion_percentage"] == 50.0  # 25/50 * 100
        
        # Collection 2: in_progress (50/75 documents) - 66.67%
        collection2 = collections[1]
        assert collection2["reindex_status"] == "in_progress"
        assert collection2["progress_info"]["completion_percentage"] == pytest.approx(66.67, rel=1e-2)  # 50/75 * 100
        
        # Collection 3: pending (40/50 documents) - 80%
        collection3 = collections[2]
        assert collection3["reindex_status"] == "pending"
        assert collection3["progress_info"]["completion_percentage"] == 80.0  # 40/50 * 100
        
        assert "1 collection(s) currently being re-indexed" in result["message"]
    
    @pytest.mark.asyncio
    async def test_execute_single_collection_status(self):
        """Test getting status for a specific collection."""
        # Mock CLI result for single collection
        mock_cli_result = {
            "collection": {
                "collection_name": "test-collection",
                "reindex_status": "failed",
                "last_reindex_timestamp": "2024-01-01T10:00:00Z",
                "document_count": 0,
                "original_document_count": 50,
                "requires_reindexing": True
            }
        }
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result):
            result = await self.tool.execute({"collection": "test-collection"})
        
        assert result["status"] == "success"
        
        status_summary = result["data"]["status_summary"]
        assert status_summary["total_collections"] == 1
        assert status_summary["failed"] == 1
        assert result["data"]["requires_attention"] is True
        
        assert "1 collection(s) failed re-indexing" in result["message"]
    
    @pytest.mark.asyncio
    async def test_execute_all_up_to_date(self):
        """Test status when all collections are up to date."""
        # Mock CLI result with all collections completed
        mock_cli_result = {
            "collections": [
                {
                    "collection_name": "collection1",
                    "reindex_status": "completed",
                    "requires_reindexing": False
                },
                {
                    "collection_name": "collection2",
                    "reindex_status": "not_required",
                    "requires_reindexing": False
                }
            ]
        }
        
        with patch.object(self.tool, 'invoke_cli_async', return_value=mock_cli_result):
            result = await self.tool.execute({})
        
        assert result["status"] == "success"
        
        status_summary = result["data"]["status_summary"]
        assert status_summary["in_progress"] == 0
        assert status_summary["pending"] == 0
        assert status_summary["failed"] == 0
        assert result["data"]["has_active_operations"] is False
        assert result["data"]["requires_attention"] is False
        
        assert "All collections are up to date" in result["message"]
    
    def test_calculate_progress(self):
        """Test progress calculation helper method."""
        # Test normal progress
        collection = {"original_document_count": 100, "document_count": 75}
        progress = self.tool._calculate_progress(collection)
        assert progress == 75.0
        
        # Test zero original count
        collection = {"original_document_count": 0, "document_count": 50}
        progress = self.tool._calculate_progress(collection)
        assert progress == 0.0
        
        # Test over 100% (should cap at 100)
        collection = {"original_document_count": 50, "document_count": 75}
        progress = self.tool._calculate_progress(collection)
        assert progress == 100.0
    
    def test_calculate_processing_time(self):
        """Test processing time calculation helper method."""
        # Test with timestamp
        collection = {"last_reindex_timestamp": "2024-01-01T10:00:00Z"}
        processing_time = self.tool._calculate_processing_time(collection)
        assert processing_time == "2024-01-01T10:00:00Z"
        
        # Test without timestamp
        collection = {}
        processing_time = self.tool._calculate_processing_time(collection)
        assert processing_time is None


class TestModelManagementToolsIntegration:
    """Test integration between model management tools."""
    
    @pytest.mark.asyncio
    async def test_detect_and_reindex_workflow(self):
        """Test the complete workflow from detection to reindexing."""
        detect_tool = DetectModelChangesTool()
        reindex_tool = ReindexCollectionTool()
        status_tool = GetReindexStatusTool()
        
        # Step 1: Detect changes
        detect_result = {
            "model_change": {
                "change_detected": True,
                "affected_collections": ["collection1", "collection2"],
                "requires_reindexing": True
            }
        }
        
        with patch.object(detect_tool, 'invoke_cli_async', return_value=detect_result):
            detection = await detect_tool.execute({})
        
        assert detection["data"]["change_detected"] is True
        affected_collections = detection["data"]["affected_collections"]
        
        # Step 2: Reindex affected collections
        reindex_result = {
            "reindex_result": {
                "success": True,
                "successful_collections": len(affected_collections),
                "failed_collections": 0
            }
        }
        
        collections_param = ",".join(affected_collections)
        with patch.object(reindex_tool, 'invoke_cli_async', return_value=reindex_result):
            reindexing = await reindex_tool.execute({"collections": collections_param})
        
        assert reindexing["data"]["success"] is True
        assert reindexing["data"]["collections_processed"] == 2
        
        # Step 3: Check final status
        status_result = {
            "collections": [
                {"collection_name": "collection1", "reindex_status": "completed"},
                {"collection_name": "collection2", "reindex_status": "completed"}
            ]
        }
        
        with patch.object(status_tool, 'invoke_cli_async', return_value=status_result):
            status = await status_tool.execute({})
        
        assert status["data"]["status_summary"]["completed"] == 2
        assert status["data"]["has_active_operations"] is False 