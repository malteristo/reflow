"""
Test suite for Structured Feedback and Progress Reporting.

Tests the implementation of subtask 15.7: Implement Structured Feedback 
and Progress Reporting for the FastMCP server.

Follows TDD methodology - GREEN PHASE.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import asdict
import tempfile

from src.mcp_server.protocol.response_formatter import (
    ResponseFormatter, ProgressResponse, StatusResponse
)

# Import the implemented components
from src.mcp_server.feedback.progress_event_system import (
    ProgressEventSystem,
    ProgressEvent,
    ProgressTracker,
    OperationType,
    ProgressPhase
)

# Import the implemented status update protocol components
from src.mcp_server.feedback.status_update_protocol import (
    StatusUpdateProtocol,
    StatusMessage,
    OperationTracker,
    OperationStatus
)

# Import the implemented contextual feedback system components
from src.mcp_server.feedback.contextual_feedback_system import (
    ContextualFeedbackSystem,
    FeedbackAnalyzer,
    QueryRefinementSuggestion,
    ContextualFeedback,
    FeedbackType,
    SuggestionConfidence
)


class TestProgressEventSystem:
    """Test progress event generation and transmission during long-running operations."""
    
    def test_progress_event_system_initialization(self):
        """Test that progress event system initializes correctly with configuration."""
        # Test default initialization
        system = ProgressEventSystem()
        assert system.update_interval == 1.0
        assert system.batch_size == 10
        assert system.max_events_in_memory == 1000
        assert len(system._trackers) == 0
        assert len(system._event_queue) == 0
        assert len(system._event_callbacks) == 0
        
        # Test custom initialization
        system_custom = ProgressEventSystem(
            update_interval=0.5,
            batch_size=5,
            max_events_in_memory=500
        )
        assert system_custom.update_interval == 0.5
        assert system_custom.batch_size == 5
        assert system_custom.max_events_in_memory == 500
    
    def test_progress_event_generation(self):
        """Test progress event generation with proper data structure."""
        system = ProgressEventSystem(update_interval=0.1)
        
        # Create a tracker
        tracker = system.create_tracker(
            operation_type=OperationType.DOCUMENT_INGESTION,
            total_items=10
        )
        
        # Generate progress event
        event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=3,
            current_item="test_file.md",
            phase=ProgressPhase.PROCESSING,
            message="Processing document",
            metadata={"file_size": 1024}
        )
        
        # Verify event structure
        assert event is not None
        assert event.operation_id == tracker.operation_id
        assert event.operation_type == OperationType.DOCUMENT_INGESTION
        assert event.progress_percentage == 30.0  # 3/10 * 100
        assert event.current_phase == ProgressPhase.PROCESSING
        assert event.current_item == "test_file.md"
        assert event.items_processed == 3
        assert event.total_items == 10
        assert event.message == "Processing document"
        assert event.metadata["file_size"] == 1024
        assert event.elapsed_time >= 0
        
        # Test event to_dict conversion
        event_dict = event.to_dict()
        assert isinstance(event_dict, dict)
        assert event_dict["operation_id"] == tracker.operation_id
        assert event_dict["operation_type"] == "document_ingestion"
        assert event_dict["progress_percentage"] == 30.0
    
    def test_progress_event_transmission(self):
        """Test that progress events are transmitted correctly via callbacks."""
        system = ProgressEventSystem(batch_size=2, update_interval=0.1)
        
        # Mock callback to capture transmitted events
        transmitted_events = []
        def mock_callback(events: List[ProgressEvent]):
            transmitted_events.extend(events)
        
        system.add_event_callback(mock_callback)
        
        # Create tracker and generate events
        tracker = system.create_tracker(
            operation_type=OperationType.QUERY_PROCESSING,
            total_items=5
        )
        
        # Generate first event
        event1 = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=1,
            current_item="step1"
        )
        
        # Should not trigger callback yet (batch_size=2)
        assert len(transmitted_events) == 0
        
        # Generate second event
        time.sleep(0.11)  # Ensure enough time passes for update
        event2 = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=2,
            current_item="step2"
        )
        
        # Should trigger callback now (batch_size=2 reached)
        assert len(transmitted_events) == 2
        assert transmitted_events[0].items_processed == 1
        assert transmitted_events[1].items_processed == 2
    
    def test_progress_batching_mechanism(self):
        """Test that progress events are batched appropriately to avoid flooding."""
        system = ProgressEventSystem(batch_size=3, update_interval=0.05)
        
        callback_invocations = []
        def mock_callback(events: List[ProgressEvent]):
            callback_invocations.append(len(events))
        
        system.add_event_callback(mock_callback)
        
        tracker = system.create_tracker(
            operation_type=OperationType.EMBEDDING_GENERATION,
            total_items=10
        )
        
        # Generate 5 events with sufficient time gaps
        for i in range(5):
            time.sleep(0.06)  # Ensure update_interval is exceeded
            system.update_progress(
                operation_id=tracker.operation_id,
                items_processed=i + 1,
                current_item=f"item_{i+1}"
            )
        
        # Should have one callback with 3 events, and 2 events still in queue
        assert len(callback_invocations) == 1
        assert callback_invocations[0] == 3
        assert len(system._event_queue) == 2
        
        # Flush remaining events
        system.flush_events()
        assert len(callback_invocations) == 2
        assert callback_invocations[1] == 2
        assert len(system._event_queue) == 0
    
    def test_progress_estimation_accuracy(self):
        """Test that progress estimation calculates accurate ETAs and percentages."""
        system = ProgressEventSystem(update_interval=0.01)
        
        tracker = system.create_tracker(
            operation_type=OperationType.VECTOR_SEARCH,
            total_items=100
        )
        
        # Simulate progress over time
        start_time = time.time()
        
        # First update - 20% complete
        time.sleep(0.02)
        event1 = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=20
        )
        
        assert event1.progress_percentage == 20.0
        assert event1.estimated_time_remaining >= 0
        
        # Second update - 50% complete
        time.sleep(0.02)
        event2 = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=50
        )
        
        assert event2.progress_percentage == 50.0
        assert event2.elapsed_time > event1.elapsed_time
        
        # Third update - 80% complete
        time.sleep(0.02)
        event3 = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=80
        )
        
        assert event3.progress_percentage == 80.0
        # ETA should be reasonable for remaining 20%
        assert event3.estimated_time_remaining >= 0
        assert event3.estimated_time_remaining < 10.0  # Should be reasonable
        
        # Complete operation
        final_event = system.complete_operation(
            operation_id=tracker.operation_id,
            message="Search completed"
        )
        
        assert final_event.current_phase == ProgressPhase.COMPLETED
        assert final_event.message == "Search completed"


class TestDocumentIngestionProgressTracking:
    """Test progress tracking during document ingestion operations."""
    
    def test_document_ingestion_progress_start(self):
        """Test progress tracking initialization for document ingestion."""
        # Create progress system and tracker for document ingestion
        system = ProgressEventSystem(update_interval=0.1)
        
        # Test starting document ingestion operation
        tracker = system.create_tracker(
            operation_type=OperationType.DOCUMENT_INGESTION,
            total_items=5  # 5 files to process
        )
        
        # Verify tracker creation
        assert tracker.operation_type == OperationType.DOCUMENT_INGESTION
        assert tracker.total_items == 5
        assert tracker.items_processed == 0
        assert tracker.current_phase == ProgressPhase.INITIALIZING
        
        # Test initial progress event
        initial_event = system.update_progress(
            operation_id=tracker.operation_id,
            phase=ProgressPhase.PROCESSING,
            message="Starting document ingestion",
            metadata={"folder_path": "/test/docs", "file_count": 5},
            force_update=True
        )
        
        assert initial_event is not None
        assert initial_event.operation_type == OperationType.DOCUMENT_INGESTION
        assert initial_event.current_phase == ProgressPhase.PROCESSING
        assert initial_event.message == "Starting document ingestion"
        assert initial_event.metadata["folder_path"] == "/test/docs"
        assert initial_event.metadata["file_count"] == 5
    
    def test_document_ingestion_file_progress(self):
        """Test progress updates during individual file processing."""
        system = ProgressEventSystem(update_interval=0.1)
        tracker = system.create_tracker(
            operation_type=OperationType.DOCUMENT_INGESTION,
            total_items=3
        )
        
        # Test processing first file
        time.sleep(0.11)  # Ensure update interval passes
        file1_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=1,
            current_item="document1.md",
            phase=ProgressPhase.PROCESSING,
            message="Processing document1.md",
            metadata={"file_size": 2048, "file_path": "/docs/document1.md"}
        )
        
        assert file1_event.items_processed == 1
        assert file1_event.current_item == "document1.md"
        assert file1_event.progress_percentage == 33.33333333333333  # 1/3 * 100
        assert file1_event.metadata["file_size"] == 2048
        
        # Test processing second file
        time.sleep(0.11)
        file2_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=2,
            current_item="document2.md",
            message="Processing document2.md",
            metadata={"file_size": 1024, "file_path": "/docs/document2.md"}
        )
        
        assert file2_event.items_processed == 2
        assert file2_event.current_item == "document2.md"
        assert file2_event.progress_percentage == 66.66666666666666  # 2/3 * 100
        assert file2_event.metadata["file_size"] == 1024
        
        # Test completing final file
        time.sleep(0.11)
        file3_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=3,
            current_item="document3.md",
            phase=ProgressPhase.FINALIZING,
            message="Finalizing document3.md",
            metadata={"file_size": 4096, "file_path": "/docs/document3.md"}
        )
        
        assert file3_event.items_processed == 3
        assert file3_event.progress_percentage == 100.0
        assert file3_event.current_phase == ProgressPhase.FINALIZING
    
    def test_document_ingestion_chunk_progress(self):
        """Test progress reporting for chunking operations within files."""
        system = ProgressEventSystem(update_interval=0.05)
        
        # Track chunking process for a large document
        tracker = system.create_tracker(
            operation_type=OperationType.DOCUMENT_INGESTION,
            total_items=1  # Single large document
        )
        
        # Test chunking progress within the document
        time.sleep(0.06)
        chunk_start_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=0,
            current_item="large_document.md",
            phase=ProgressPhase.PROCESSING,
            message="Chunking document: analyzing structure",
            metadata={
                "chunking_phase": "structure_analysis",
                "document_size": 50000,
                "expected_chunks": 25
            }
        )
        
        assert chunk_start_event.metadata["chunking_phase"] == "structure_analysis"
        assert chunk_start_event.metadata["expected_chunks"] == 25
        
        # Test chunk generation progress
        time.sleep(0.06)
        chunk_progress_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=0,  # Still processing the same document
            current_item="large_document.md",
            message="Chunking document: generating chunks",
            metadata={
                "chunking_phase": "chunk_generation",
                "chunks_created": 15,
                "expected_chunks": 25,
                "current_section": "Implementation Details"
            }
        )
        
        assert chunk_progress_event.metadata["chunks_created"] == 15
        assert chunk_progress_event.metadata["current_section"] == "Implementation Details"
        
        # Test chunk completion
        time.sleep(0.06)
        chunk_complete_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=1,  # Document processing complete
            current_item="large_document.md",
            message="Chunking completed: 25 chunks created",
            metadata={
                "chunking_phase": "completed",
                "chunks_created": 25,
                "total_chunk_size": 52000
            }
        )
        
        assert chunk_complete_event.items_processed == 1
        assert chunk_complete_event.progress_percentage == 100.0
        assert chunk_complete_event.metadata["chunks_created"] == 25
    
    def test_document_ingestion_embedding_progress(self):
        """Test progress tracking during embedding generation."""
        system = ProgressEventSystem(update_interval=0.05)
        
        # Track embedding generation for multiple chunks
        tracker = system.create_tracker(
            operation_type=OperationType.EMBEDDING_GENERATION,
            total_items=20  # 20 chunks to embed
        )
        
        # Test batch embedding progress
        time.sleep(0.06)
        embed_start_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=0,
            phase=ProgressPhase.PROCESSING,
            message="Starting embedding generation",
            metadata={
                "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
                "batch_size": 5,
                "total_chunks": 20
            }
        )
        
        assert embed_start_event.metadata["embedding_model"] == "multi-qa-MiniLM-L6-cos-v1"
        assert embed_start_event.metadata["batch_size"] == 5
        
        # Test batch processing progress
        time.sleep(0.06)
        batch1_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=5,
            current_item="batch_1",
            message="Processed embedding batch 1/4",
            metadata={
                "batch_number": 1,
                "batch_size": 5,
                "embedding_time": 2.5
            }
        )
        
        assert batch1_event.items_processed == 5
        assert batch1_event.progress_percentage == 25.0  # 5/20 * 100
        assert batch1_event.metadata["batch_number"] == 1
        
        # Test final batch completion
        time.sleep(0.06)
        final_batch_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=20,
            current_item="batch_4",
            phase=ProgressPhase.FINALIZING,
            message="Completed all embedding batches",
            metadata={
                "batch_number": 4,
                "total_embedding_time": 9.2,
                "embeddings_created": 20
            }
        )
        
        assert final_batch_event.items_processed == 20
        assert final_batch_event.progress_percentage == 100.0
        assert final_batch_event.metadata["embeddings_created"] == 20
    
    def test_document_ingestion_storage_progress(self):
        """Test progress tracking during vector storage operations."""
        system = ProgressEventSystem(update_interval=0.05)
        
        # Track vector storage operation
        tracker = system.create_tracker(
            operation_type=OperationType.VECTOR_SEARCH,  # Using existing enum value
            total_items=15  # 15 vectors to store
        )
        
        # Test storage initialization
        time.sleep(0.06)
        storage_start_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=0,
            phase=ProgressPhase.PROCESSING,
            message="Initializing vector storage",
            metadata={
                "collection_name": "research_docs",
                "vector_dimension": 384,
                "storage_backend": "chromadb"
            }
        )
        
        assert storage_start_event.metadata["collection_name"] == "research_docs"
        assert storage_start_event.metadata["vector_dimension"] == 384
        
        # Test batch storage progress
        time.sleep(0.06)
        storage_progress_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=10,
            current_item="vector_batch_2",
            message="Storing vector batch 2/3",
            metadata={
                "vectors_stored": 10,
                "batch_size": 5,
                "storage_time": 1.2
            }
        )
        
        assert storage_progress_event.items_processed == 10
        assert storage_progress_event.progress_percentage == 66.66666666666666  # 10/15 * 100
        assert storage_progress_event.metadata["vectors_stored"] == 10
        
        # Test storage completion
        completion_event = system.complete_operation(
            operation_id=tracker.operation_id,
            message="Vector storage completed successfully"
        )
        
        assert completion_event.current_phase == ProgressPhase.COMPLETED
        assert completion_event.message == "Vector storage completed successfully"
    
    def test_document_ingestion_error_progress_handling(self):
        """Test progress reporting when errors occur during ingestion."""
        system = ProgressEventSystem(update_interval=0.05)
        
        # Mock callback to capture events
        captured_events = []
        def capture_events(events):
            captured_events.extend(events)
        
        system.add_event_callback(capture_events)
        
        # Track ingestion with some failures
        tracker = system.create_tracker(
            operation_type=OperationType.DOCUMENT_INGESTION,
            total_items=4
        )
        
        # Test successful file processing
        time.sleep(0.06)
        success_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=1,
            current_item="good_file.md",
            message="Successfully processed good_file.md",
            metadata={"status": "success", "chunks_created": 5}
        )
        
        assert success_event.metadata["status"] == "success"
        
        # Test file processing with error
        time.sleep(0.06)
        error_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=1,  # Still count as processed (failed)
            current_item="bad_file.md",
            message="Failed to process bad_file.md: Invalid format",
            metadata={
                "status": "error",
                "error_type": "invalid_format",
                "error_message": "Unsupported file format"
            }
        )
        
        assert error_event.metadata["status"] == "error"
        assert error_event.metadata["error_type"] == "invalid_format"
        
        # Test recovery and continuation
        time.sleep(0.06)
        recovery_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=2,
            current_item="recovery_file.md",
            message="Continuing with next file after error",
            metadata={"status": "success", "chunks_created": 3}
        )
        
        assert recovery_event.items_processed == 2
        assert recovery_event.metadata["status"] == "success"
        
        # Test final operation with partial success
        final_event = system.complete_operation(
            operation_id=tracker.operation_id,
            message="Ingestion completed with 1 error"
        )
        
        assert final_event.current_phase == ProgressPhase.COMPLETED
        assert "error" in final_event.message


class TestQueryOperationProgressTracking:
    """Test progress tracking during knowledge base query operations."""
    
    def test_query_operation_progress_start(self):
        """Test progress tracking initialization for query operations."""
        system = ProgressEventSystem(update_interval=0.1)
        
        # Test starting query operation
        tracker = system.create_tracker(
            operation_type=OperationType.QUERY_PROCESSING,
            total_items=4  # 4 major phases: embedding, search, rerank, format
        )
        
        # Verify tracker creation
        assert tracker.operation_type == OperationType.QUERY_PROCESSING
        assert tracker.total_items == 4
        assert tracker.items_processed == 0
        assert tracker.current_phase == ProgressPhase.INITIALIZING
        
        # Test initial query processing event
        initial_event = system.update_progress(
            operation_id=tracker.operation_id,
            phase=ProgressPhase.PROCESSING,
            message="Starting query processing",
            metadata={
                "query": "research methodology for AI",
                "collections": ["research_docs", "fundamentals"],
                "query_length": 26
            },
            force_update=True
        )
        
        assert initial_event is not None
        assert initial_event.operation_type == OperationType.QUERY_PROCESSING
        assert initial_event.current_phase == ProgressPhase.PROCESSING
        assert initial_event.message == "Starting query processing"
        assert initial_event.metadata["query"] == "research methodology for AI"
        assert initial_event.metadata["collections"] == ["research_docs", "fundamentals"]
        assert initial_event.metadata["query_length"] == 26
    
    def test_query_embedding_progress(self):
        """Test progress reporting during query embedding generation."""
        system = ProgressEventSystem(update_interval=0.05)
        tracker = system.create_tracker(
            operation_type=OperationType.EMBEDDING_GENERATION,
            total_items=1  # Single query to embed
        )
        
        # Test query embedding start
        time.sleep(0.06)
        embed_start_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=0,
            current_item="query_embedding",
            phase=ProgressPhase.PROCESSING,
            message="Generating query embedding",
            metadata={
                "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
                "query": "machine learning algorithms",
                "query_tokens": 3,
                "embedding_dimension": 384
            }
        )
        
        assert embed_start_event.current_item == "query_embedding"
        assert embed_start_event.metadata["embedding_model"] == "multi-qa-MiniLM-L6-cos-v1"
        assert embed_start_event.metadata["query_tokens"] == 3
        assert embed_start_event.metadata["embedding_dimension"] == 384
        
        # Test query embedding completion
        time.sleep(0.06)
        embed_complete_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=1,
            current_item="query_embedding",
            phase=ProgressPhase.FINALIZING,
            message="Query embedding generated successfully",
            metadata={
                "embedding_time": 0.45,
                "embedding_vector_norm": 0.95
            }
        )
        
        assert embed_complete_event.items_processed == 1
        assert embed_complete_event.progress_percentage == 100.0
        assert embed_complete_event.current_phase == ProgressPhase.FINALIZING
        assert embed_complete_event.metadata["embedding_time"] == 0.45
        assert embed_complete_event.metadata["embedding_vector_norm"] == 0.95
    
    def test_vector_search_progress(self):
        """Test progress tracking during vector similarity search."""
        system = ProgressEventSystem(update_interval=0.05)
        tracker = system.create_tracker(
            operation_type=OperationType.VECTOR_SEARCH,
            total_items=3  # 3 collections to search
        )
        
        # Test search start
        time.sleep(0.06)
        search_start_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=0,
            phase=ProgressPhase.PROCESSING,
            message="Starting vector similarity search",
            metadata={
                "collections": ["research_docs", "fundamentals", "advanced"],
                "search_top_k": 50,
                "search_metric": "cosine"
            }
        )
        
        assert search_start_event.metadata["collections"] == ["research_docs", "fundamentals", "advanced"]
        assert search_start_event.metadata["search_top_k"] == 50
        assert search_start_event.metadata["search_metric"] == "cosine"
        
        # Test search progress for first collection
        time.sleep(0.06)
        collection1_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=1,
            current_item="research_docs",
            message="Searching collection: research_docs",
            metadata={
                "current_collection": "research_docs",
                "collection_size": 1500,
                "candidates_found": 42,
                "search_time": 0.25
            }
        )
        
        assert collection1_event.items_processed == 1
        assert collection1_event.current_item == "research_docs"
        assert collection1_event.progress_percentage == 33.33333333333333  # 1/3 * 100
        assert collection1_event.metadata["candidates_found"] == 42
        
        # Test search completion for all collections
        time.sleep(0.06)
        search_complete_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=3,
            current_item="advanced",
            phase=ProgressPhase.FINALIZING,
            message="Vector search completed across all collections",
            metadata={
                "total_candidates": 125,
                "total_search_time": 0.78,
                "collections_searched": 3
            }
        )
        
        assert search_complete_event.items_processed == 3
        assert search_complete_event.progress_percentage == 100.0
        assert search_complete_event.metadata["total_candidates"] == 125
        assert search_complete_event.metadata["collections_searched"] == 3
    
    def test_reranking_progress(self):
        """Test progress reporting during result re-ranking operations."""
        system = ProgressEventSystem(update_interval=0.05)
        tracker = system.create_tracker(
            operation_type=OperationType.RERANKING,
            total_items=50  # 50 candidates to rerank
        )
        
        # Test reranking start
        time.sleep(0.06)
        rerank_start_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=0,
            phase=ProgressPhase.PROCESSING,
            message="Starting result re-ranking",
            metadata={
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                "candidates_to_rerank": 50,
                "target_top_k": 10,
                "batch_size": 10
            }
        )
        
        assert rerank_start_event.metadata["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert rerank_start_event.metadata["candidates_to_rerank"] == 50
        assert rerank_start_event.metadata["target_top_k"] == 10
        
        # Test reranking batch progress
        time.sleep(0.06)
        batch_progress_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=30,
            current_item="rerank_batch_3",
            message="Re-ranking batch 3/5",
            metadata={
                "current_batch": 3,
                "batch_size": 10,
                "processed_candidates": 30,
                "batch_rerank_time": 1.2
            }
        )
        
        assert batch_progress_event.items_processed == 30
        assert batch_progress_event.progress_percentage == 60.0  # 30/50 * 100
        assert batch_progress_event.metadata["current_batch"] == 3
        assert batch_progress_event.metadata["processed_candidates"] == 30
        
        # Test reranking completion
        final_event = system.complete_operation(
            operation_id=tracker.operation_id,
            message="Re-ranking completed, top 10 results selected"
        )
        
        assert final_event.current_phase == ProgressPhase.COMPLETED
        assert final_event.message == "Re-ranking completed, top 10 results selected"
    
    def test_result_formatting_progress(self):
        """Test progress tracking during result formatting and response preparation."""
        system = ProgressEventSystem(update_interval=0.05)
        tracker = system.create_tracker(
            operation_type=OperationType.RESULT_FORMATTING,
            total_items=10  # 10 top results to format
        )
        
        # Test formatting start
        time.sleep(0.06)
        format_start_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=0,
            phase=ProgressPhase.PROCESSING,
            message="Starting result formatting",
            metadata={
                "results_to_format": 10,
                "include_metadata": True,
                "include_snippets": True,
                "response_format": "structured"
            }
        )
        
        assert format_start_event.metadata["results_to_format"] == 10
        assert format_start_event.metadata["include_metadata"] is True
        assert format_start_event.metadata["response_format"] == "structured"
        
        # Test metadata enrichment progress
        time.sleep(0.06)
        metadata_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=5,
            current_item="metadata_enrichment",
            message="Enriching result metadata",
            metadata={
                "enrichment_phase": "metadata_extraction",
                "results_enriched": 5,
                "avg_enrichment_time": 0.12
            }
        )
        
        assert metadata_event.items_processed == 5
        assert metadata_event.progress_percentage == 50.0
        assert metadata_event.metadata["enrichment_phase"] == "metadata_extraction"
        assert metadata_event.metadata["results_enriched"] == 5
        
        # Test snippet generation progress
        time.sleep(0.06)
        snippet_event = system.update_progress(
            operation_id=tracker.operation_id,
            items_processed=8,
            current_item="snippet_generation",
            message="Generating result snippets",
            metadata={
                "enrichment_phase": "snippet_generation",
                "snippets_generated": 8,
                "avg_snippet_length": 200
            }
        )
        
        assert snippet_event.items_processed == 8
        assert snippet_event.progress_percentage == 80.0
        assert snippet_event.metadata["snippets_generated"] == 8
        
        # Test formatting completion
        final_event = system.complete_operation(
            operation_id=tracker.operation_id,
            message="Result formatting completed successfully"
        )
        
        assert final_event.current_phase == ProgressPhase.COMPLETED
        assert final_event.message == "Result formatting completed successfully"


class TestStatusUpdateProtocol:
    """Test structured status update protocol implementation."""
    
    def test_status_update_initialization(self):
        """Test status update system initialization with protocol compliance."""
        # Test default initialization
        protocol = StatusUpdateProtocol()
        assert protocol.enable_persistence is True
        assert protocol.max_history_per_operation == 100
        assert protocol.cleanup_completed_after_seconds == 3600
        assert len(protocol._operations) == 0
        assert len(protocol._status_callbacks) == 0
        
        # Test custom initialization
        protocol_custom = StatusUpdateProtocol(
            enable_persistence=False,
            max_history_per_operation=50,
            cleanup_completed_after_seconds=1800
        )
        assert protocol_custom.enable_persistence is False
        assert protocol_custom.max_history_per_operation == 50
        assert protocol_custom.cleanup_completed_after_seconds == 1800
    
    def test_operation_status_lifecycle(self):
        """Test complete status lifecycle: started, processing, completed."""
        protocol = StatusUpdateProtocol()
        
        # Start operation
        operation_id = protocol.start_operation(
            operation_type="test_operation",
            initial_message="Starting test",
            context={"test": True},
            metadata={"version": "1.0"}
        )
        
        assert operation_id is not None
        assert len(protocol._operations) == 1
        
        # Check initial status
        status = protocol.get_operation_status(operation_id)
        assert status is not None
        assert status.status == OperationStatus.STARTED
        assert status.message == "Starting test"
        assert status.operation_id == operation_id
        assert status.context["test"] is True
        assert status.metadata["version"] == "1.0"
        
        # Update to processing
        processing_status = protocol.update_operation_status(
            operation_id=operation_id,
            status=OperationStatus.PROCESSING,
            message="Processing data",
            progress_percentage=50.0,
            metadata={"processed_items": 5}
        )
        
        assert processing_status is not None
        assert processing_status.status == OperationStatus.PROCESSING
        assert processing_status.progress_percentage == 50.0
        assert processing_status.metadata["processed_items"] == 5
        
        # Complete operation
        completed_status = protocol.complete_operation(
            operation_id=operation_id,
            message="Operation completed successfully",
            metadata={"final_count": 10}
        )
        
        assert completed_status is not None
        assert completed_status.status == OperationStatus.COMPLETED
        assert completed_status.progress_percentage == 100.0
        assert completed_status.message == "Operation completed successfully"
        assert completed_status.metadata["final_count"] == 10
        
        # Check history
        history = protocol.get_operation_history(operation_id)
        assert len(history) == 3  # started, processing, completed
        assert history[0].status == OperationStatus.STARTED
        assert history[1].status == OperationStatus.PROCESSING
        assert history[2].status == OperationStatus.COMPLETED
    
    def test_status_message_formatting(self):
        """Test that status messages are properly formatted per MCP protocol."""
        protocol = StatusUpdateProtocol()
        operation_id = protocol.start_operation(operation_type="format_test")
        
        status = protocol.get_operation_status(operation_id)
        
        # Test dictionary format
        status_dict = status.to_dict()
        assert isinstance(status_dict, dict)
        assert "message_id" in status_dict
        assert "operation_id" in status_dict
        assert "timestamp" in status_dict
        assert "status" in status_dict
        assert "message" in status_dict
        assert "progress_percentage" in status_dict
        assert "metadata" in status_dict
        assert "context" in status_dict
        
        # Test MCP format
        mcp_format = status.to_mcp_format()
        assert isinstance(mcp_format, dict)
        assert mcp_format["jsonrpc"] == "2.0"
        assert mcp_format["method"] == "notifications/status"
        assert "params" in mcp_format
        
        params = mcp_format["params"]
        assert params["operation_id"] == operation_id
        assert params["status"] == "started"
        assert "timestamp" in params
        assert "progress_percentage" in params
        
        # Test error message formatting
        error_status = protocol.error_operation(
            operation_id=operation_id,
            error_message="Test error occurred",
            error_code="TEST_ERROR",
            error_details="Detailed error information"
        )
        
        error_mcp = error_status.to_mcp_format()
        assert "error" in error_mcp["params"]
        assert error_mcp["params"]["error"]["code"] == "TEST_ERROR"
        assert error_mcp["params"]["error"]["message"] == "Detailed error information"
    
    def test_concurrent_operation_status_tracking(self):
        """Test status tracking for multiple concurrent operations."""
        protocol = StatusUpdateProtocol()
        
        # Start multiple operations
        op1_id = protocol.start_operation(operation_type="operation_1", initial_message="Op 1 started")
        op2_id = protocol.start_operation(operation_type="operation_2", initial_message="Op 2 started")
        op3_id = protocol.start_operation(operation_type="operation_3", initial_message="Op 3 started")
        
        assert len(protocol._operations) == 3
        assert op1_id != op2_id != op3_id
        
        # Update different operations with different statuses
        protocol.update_operation_status(op1_id, OperationStatus.PROCESSING, "Op 1 processing")
        protocol.update_operation_status(op2_id, OperationStatus.PROCESSING, "Op 2 processing")
        protocol.error_operation(op3_id, "Op 3 failed", "OP3_ERROR")
        
        # Check individual statuses
        op1_status = protocol.get_operation_status(op1_id)
        op2_status = protocol.get_operation_status(op2_id)
        op3_status = protocol.get_operation_status(op3_id)
        
        assert op1_status.status == OperationStatus.PROCESSING
        assert op2_status.status == OperationStatus.PROCESSING
        assert op3_status.status == OperationStatus.ERROR
        assert op3_status.error_code == "OP3_ERROR"
        
        # Check active operations
        active_ops = protocol.get_active_operations()
        assert len(active_ops) == 2  # op1 and op2 are still active
        
        # Complete one operation
        protocol.complete_operation(op1_id, "Op 1 completed")
        
        # Check active operations again
        active_ops = protocol.get_active_operations()
        assert len(active_ops) == 1  # only op2 is still active
        
        # Check all operations
        all_ops = protocol.get_all_operations()
        assert len(all_ops) == 3  # all operations are tracked
    
    def test_status_error_reporting(self):
        """Test status updates when operations encounter errors."""
        protocol = StatusUpdateProtocol()
        operation_id = protocol.start_operation(operation_type="error_test")
        
        # Report error with full details
        error_status = protocol.error_operation(
            operation_id=operation_id,
            error_message="Critical error occurred",
            error_code="CRITICAL_ERROR",
            error_details="Detailed error information for debugging",
            metadata={"error_type": "critical", "retry_count": 3}
        )
        
        assert error_status is not None
        assert error_status.status == OperationStatus.ERROR
        assert error_status.message == "Critical error occurred"
        assert error_status.error_code == "CRITICAL_ERROR"
        assert error_status.error_details == "Detailed error information for debugging"
        assert error_status.metadata["error_type"] == "critical"
        assert error_status.metadata["retry_count"] == 3
        
        # Check MCP format includes error information
        mcp_format = error_status.to_mcp_format()
        assert "error" in mcp_format["params"]
        assert mcp_format["params"]["error"]["code"] == "CRITICAL_ERROR"
        assert mcp_format["params"]["error"]["message"] == "Detailed error information for debugging"
        
        # Check operation is no longer active
        active_ops = protocol.get_active_operations()
        assert len(active_ops) == 0
        
        # Check statistics reflect the error
        stats = protocol.get_operation_statistics()
        assert stats["failed_operations"] == 1
        assert stats["active_operations"] == 0
    
    def test_status_persistence_across_requests(self):
        """Test that status information persists correctly across request boundaries."""
        # Create protocol with persistence enabled
        protocol = StatusUpdateProtocol(enable_persistence=True)
        
        # Start operation and update status
        operation_id = protocol.start_operation(
            operation_type="persistent_test",
            initial_message="Persistent operation started",
            metadata={"session_id": "abc123"}
        )
        
        protocol.update_operation_status(
            operation_id=operation_id,
            status=OperationStatus.PROCESSING,
            message="Processing data",
            progress_percentage=25.0,
            metadata={"processed_files": 5}
        )
        
        # Simulate request boundary - status should persist
        current_status = protocol.get_operation_status(operation_id)
        assert current_status is not None
        assert current_status.status == OperationStatus.PROCESSING
        assert current_status.progress_percentage == 25.0
        assert current_status.metadata["session_id"] == "abc123"
        assert current_status.metadata["processed_files"] == 5
        
        # Check full history is maintained
        history = protocol.get_operation_history(operation_id)
        assert len(history) == 2  # started + processing
        
        # Test with persistence disabled
        protocol_no_persist = StatusUpdateProtocol(enable_persistence=False)
        
        # Operations should still work but cleanup behavior might differ
        temp_op_id = protocol_no_persist.start_operation(operation_type="temp_test")
        temp_status = protocol_no_persist.get_operation_status(temp_op_id)
        assert temp_status is not None
        assert temp_status.status == OperationStatus.STARTED


class TestContextualFeedbackSystem:
    """Test contextual feedback and query analysis system."""
    
    def test_contextual_feedback_initialization(self):
        """Test contextual feedback system initialization and configuration."""
        # Test default initialization
        system = ContextualFeedbackSystem()
        assert system.enable_analysis is True
        assert system.feedback_threshold == 3
        assert system.analyzer is not None
        assert isinstance(system._feedback_history, list)
        assert len(system._feedback_history) == 0
        
        # Test custom initialization
        custom_system = ContextualFeedbackSystem(
            enable_analysis=False, 
            feedback_threshold=5
        )
        assert custom_system.enable_analysis is False
        assert custom_system.feedback_threshold == 5
        
        # Test analyzer initialization
        analyzer = FeedbackAnalyzer()
        assert analyzer._common_terms is not None
        assert analyzer._collection_keywords is not None
        assert "machine learning" in analyzer._common_terms
        assert "research_papers" in analyzer._collection_keywords
    
    def test_query_refinement_suggestions(self):
        """Test generation of query refinement suggestions."""
        system = ContextualFeedbackSystem()
        
        # Test short query analysis
        short_query = "AI"
        suggestions = system.get_query_refinement_suggestions(short_query)
        assert len(suggestions) >= 1
        
        # Find the query length suggestion
        length_suggestion = next(
            (s for s in suggestions if "more specific terms" in s.suggestion_text), 
            None
        )
        assert length_suggestion is not None
        assert length_suggestion.confidence == SuggestionConfidence.HIGH
        assert "AI methodology implementation" in length_suggestion.example_query
        
        # Test ambiguous query analysis
        ambiguous_query = "How does it work?"
        ambiguous_suggestions = system.get_query_refinement_suggestions(ambiguous_query)
        assert len(ambiguous_suggestions) >= 1
        
        # Find the ambiguity suggestion
        ambiguity_suggestion = next(
            (s for s in ambiguous_suggestions if s.suggestion_type == FeedbackType.AMBIGUOUS_QUERY),
            None
        )
        assert ambiguity_suggestion is not None
        assert ambiguity_suggestion.confidence == SuggestionConfidence.MEDIUM
        assert "[specific technology]" in ambiguity_suggestion.example_query
        
        # Test non-technical query analysis  
        non_technical_query = "How to solve problems"
        tech_suggestions = system.get_query_refinement_suggestions(non_technical_query)
        
        # Find the technical depth suggestion
        tech_suggestion = next(
            (s for s in tech_suggestions if s.suggestion_type == FeedbackType.SEARCH_STRATEGY),
            None
        )
        assert tech_suggestion is not None
        assert "domain-specific terminology" in tech_suggestion.suggestion_text
        assert "machine learning algorithm" in tech_suggestion.example_query
    
    def test_insufficient_results_feedback(self):
        """Test feedback generation for insufficient search results."""
        system = ContextualFeedbackSystem(feedback_threshold=5)
        
        # Test zero results scenario
        zero_results_feedback = system.generate_feedback(
            query="very specific obscure term",
            result_count=0,
            collections_searched=["research_papers"]
        )
        
        assert len(zero_results_feedback) >= 1
        insufficient_feedback = next(
            (f for f in zero_results_feedback if f.feedback_type == FeedbackType.INSUFFICIENT_RESULTS),
            None
        )
        assert insufficient_feedback is not None
        assert "Found only 0 results" in insufficient_feedback.message
        assert len(insufficient_feedback.suggestions) >= 1
        
        # Check broadening suggestion
        broadening_suggestion = next(
            (s for s in insufficient_feedback.suggestions 
             if s.suggestion_type == FeedbackType.INSUFFICIENT_RESULTS),
            None
        )
        assert broadening_suggestion is not None
        assert "broader terms" in broadening_suggestion.suggestion_text
        assert broadening_suggestion.confidence == SuggestionConfidence.HIGH
        
        # Test limited collections scenario
        limited_collections_feedback = system.generate_feedback(
            query="machine learning",
            result_count=2,
            collections_searched=["fundamentals"]
        )
        
        # Should have collection recommendation
        collection_feedback = next(
            (f for f in limited_collections_feedback 
             if f.feedback_type == FeedbackType.COLLECTION_RECOMMENDATION),
            None
        )
        assert collection_feedback is not None
        
        # Check collection expansion suggestion
        collection_suggestion = next(
            (s for s in collection_feedback.suggestions
             if "additional collections" in s.suggestion_text),
            None
        )
        if collection_suggestion:  # May be in insufficient results feedback instead
            assert collection_suggestion.confidence == SuggestionConfidence.MEDIUM
    
    def test_ambiguous_query_feedback(self):
        """Test detection and feedback for ambiguous queries."""
        system = ContextualFeedbackSystem()
        analyzer = system.analyzer
        
        # Test query with pronouns
        ambiguous_query = "How does this algorithm work with that dataset?"
        feedback = analyzer.analyze_ambiguous_query(ambiguous_query)
        
        assert feedback.feedback_type == FeedbackType.AMBIGUOUS_QUERY
        assert "ambiguous terms" in feedback.message
        assert len(feedback.suggestions) >= 1
        
        # Check for 'this' replacement suggestion
        this_suggestion = next(
            (s for s in feedback.suggestions if "'this'" in s.suggestion_text),
            None
        )
        assert this_suggestion is not None
        assert this_suggestion.confidence == SuggestionConfidence.HIGH
        assert "particular method" in this_suggestion.example_query
        
        # Check for 'that' replacement suggestion  
        that_suggestion = next(
            (s for s in feedback.suggestions if "'that'" in s.suggestion_text),
            None
        )
        assert that_suggestion is not None
        assert "specific technique" in that_suggestion.example_query
        
        # Test query without ambiguous terms
        clear_query = "How does the ResNet architecture work with the ImageNet dataset?"
        clear_feedback = analyzer.analyze_ambiguous_query(clear_query)
        assert len(clear_feedback.suggestions) == 0  # No ambiguous terms to fix
    
    def test_collection_recommendation_feedback(self):
        """Test collection recommendation based on query content."""
        system = ContextualFeedbackSystem()
        analyzer = system.analyzer
        
        # Test research paper query
        research_query = "latest research on neural networks"
        research_feedback = analyzer.recommend_collections(research_query)
        
        assert research_feedback.feedback_type == FeedbackType.COLLECTION_RECOMMENDATION
        assert "collections may contain relevant content" in research_feedback.message
        assert "research_papers" in research_feedback.metadata["recommended_collections"]
        
        # Check research papers suggestion
        research_suggestion = next(
            (s for s in research_feedback.suggestions 
             if "research_papers" in s.suggestion_text),
            None
        )
        assert research_suggestion is not None
        assert research_suggestion.confidence == SuggestionConfidence.MEDIUM
        assert "research_papers" in research_suggestion.metadata["recommended_collection"]
        
        # Test documentation query
        docs_query = "API documentation for TensorFlow"
        docs_feedback = analyzer.recommend_collections(docs_query)
        assert "documentation" in docs_feedback.metadata["recommended_collections"]
        
        # Test code query  
        code_query = "implementation example for gradient descent"
        code_feedback = analyzer.recommend_collections(code_query)
        assert "code_examples" in code_feedback.metadata["recommended_collections"]
        
        # Test general query (should get defaults)
        general_query = "information about stuff"
        general_feedback = analyzer.recommend_collections(general_query)
        assert "research_papers" in general_feedback.metadata["recommended_collections"]
        assert "fundamentals" in general_feedback.metadata["recommended_collections"]
    
    def test_search_strategy_feedback(self):
        """Test search strategy optimization suggestions."""
        system = ContextualFeedbackSystem()
        analyzer = system.analyzer
        
        # Test long query
        long_query = "how to implement advanced deep learning neural network architectures for computer vision tasks"
        long_feedback = analyzer.suggest_search_strategy(long_query, previous_results=10)
        
        assert long_feedback.feedback_type == FeedbackType.SEARCH_STRATEGY
        assert len(long_feedback.suggestions) >= 1
        
        # Check for query shortening suggestion
        shorten_suggestion = next(
            (s for s in long_feedback.suggestions 
             if "breaking long queries" in s.suggestion_text),
            None
        )
        assert shorten_suggestion is not None
        assert shorten_suggestion.confidence == SuggestionConfidence.MEDIUM
        assert len(shorten_suggestion.example_query.split()) <= 5
        
        # Test phrase query (should suggest quotes)
        phrase_query = "machine learning algorithm"
        phrase_feedback = analyzer.suggest_search_strategy(phrase_query, previous_results=10)
        
        quote_suggestion = next(
            (s for s in phrase_feedback.suggestions 
             if "quotes around key phrases" in s.suggestion_text),
            None
        )
        assert quote_suggestion is not None
        assert quote_suggestion.confidence == SuggestionConfidence.HIGH
        assert '"' in quote_suggestion.example_query
        
        # Test low results scenario
        low_results_query = "specialized topic"
        low_results_feedback = analyzer.suggest_search_strategy(low_results_query, previous_results=2)
        
        synonym_suggestion = next(
            (s for s in low_results_feedback.suggestions 
             if "alternative terminology" in s.suggestion_text),
            None
        )
        assert synonym_suggestion is not None
        assert "OR" in synonym_suggestion.example_query
    
    def test_knowledge_gap_feedback(self):
        """Test knowledge gap detection and learning guidance."""
        system = ContextualFeedbackSystem()
        analyzer = system.analyzer
        
        # Test advanced query with basic collections
        advanced_query = "advanced optimization techniques for deep neural networks"
        basic_collections = ["fundamentals", "introduction"]
        gap_feedback = analyzer.detect_knowledge_gap(advanced_query, basic_collections)
        
        assert gap_feedback.feedback_type == FeedbackType.KNOWLEDGE_GAP
        assert "Knowledge building suggestions" in gap_feedback.message
        assert gap_feedback.metadata["has_advanced_terms"] is True
        
        # Check for prerequisite suggestion
        prereq_suggestion = next(
            (s for s in gap_feedback.suggestions 
             if "prerequisite topics" in s.suggestion_text),
            None
        )
        if prereq_suggestion:  # May suggest prerequisites for advanced topics
            assert prereq_suggestion.confidence == SuggestionConfidence.MEDIUM
            assert "introduction to" in prereq_suggestion.example_query
        
        # Test regular query (should get general learning suggestion)
        regular_query = "machine learning basics"
        regular_collections = ["research_papers", "documentation"]
        regular_feedback = analyzer.detect_knowledge_gap(regular_query, regular_collections)
        
        assert len(regular_feedback.suggestions) >= 1
        foundation_suggestion = next(
            (s for s in regular_feedback.suggestions 
             if "foundational concepts" in s.suggestion_text),
            None
        )
        assert foundation_suggestion is not None
        assert foundation_suggestion.confidence == SuggestionConfidence.LOW
        assert "fundamentals basics" in foundation_suggestion.example_query


class TestInteractiveFeedbackFlows:
    """Test interactive feedback flows for multi-step operations."""
    
    def test_interactive_collection_management_flow(self):
        """Test interactive feedback during collection management operations."""
        from src.mcp_server.feedback.interactive_feedback_flows import (
            InteractiveFeedbackFlows, InteractionType, InteractionState
        )
        
        flows = InteractiveFeedbackFlows()
        
        # Start collection management flow
        context = {
            "available_collections": ["documents", "research", "projects"],
            "operation": "select_collection"
        }
        session = flows.start_collection_management_flow(context)
        
        assert session.interaction_type == InteractionType.COLLECTION_MANAGEMENT
        assert session.state == InteractionState.IN_PROGRESS
        assert len(session.interactions) == 1
        
        # Verify initial interaction
        initial_interaction = session.interactions[0]
        assert initial_interaction.state == InteractionState.WAITING_USER_INPUT
        assert "Select collection management action" in initial_interaction.prompt
        assert "Create new collection" in initial_interaction.options
        
        # Process user response - create new collection
        response = flows.process_user_response(session.session_id, "Create new collection")
        assert response is not None
        assert response.user_response == "Create new collection"
        
        # Verify follow-up interaction for naming
        assert len(session.interactions) == 2
        naming_interaction = session.interactions[1]
        assert "Enter name for new collection" in naming_interaction.prompt
        
        # Provide collection name
        flows.process_user_response(session.session_id, "test_collection")
        
        # Verify type selection interaction
        assert len(session.interactions) == 3
        type_interaction = session.interactions[2]
        assert "Select collection type" in type_interaction.prompt
        assert "general" in type_interaction.options
        
        # Complete workflow
        flows.process_user_response(session.session_id, "general")
        
        # Verify workflow completion
        completed_session = flows.get_session(session.session_id)
        assert completed_session.state == InteractionState.COMPLETED
        assert completed_session.context["new_collection_name"] == "test_collection"
        assert completed_session.context["collection_type"] == "general"
        assert completed_session.context["collection_created"] is True
    
    def test_interactive_document_conflict_resolution(self):
        """Test interactive feedback for resolving document conflicts."""
        from src.mcp_server.feedback.interactive_feedback_flows import (
            InteractiveFeedbackFlows, InteractionType, InteractionState
        )
        
        flows = InteractiveFeedbackFlows()
        
        # Start document conflict resolution
        context = {
            "conflict_type": "duplicate",
            "existing_document": "doc1.md",
            "new_document": "doc1_new.md"
        }
        session = flows.start_document_conflict_resolution(context)
        
        assert session.interaction_type == InteractionType.DOCUMENT_CONFLICT_RESOLUTION
        assert session.state == InteractionState.IN_PROGRESS
        
        # Verify conflict resolution options
        conflict_interaction = session.interactions[0]
        assert "Document conflict detected" in conflict_interaction.prompt
        assert "Keep existing document" in conflict_interaction.options
        assert "Merge documents" in conflict_interaction.options
        
        # Choose merge option
        flows.process_user_response(session.session_id, "Merge documents")
        
        # Verify merge strategy interaction
        assert len(session.interactions) == 2
        merge_interaction = session.interactions[1]
        assert "Select merge strategy" in merge_interaction.prompt
        assert "Combine content sections" in merge_interaction.options
        
        # Complete merge workflow
        flows.process_user_response(session.session_id, "Combine content sections")
        
        # Verify resolution
        completed_session = flows.get_session(session.session_id)
        assert completed_session.state == InteractionState.COMPLETED
        assert completed_session.context["resolution"] == "merge"
        assert completed_session.context["merge_strategy"] == "Combine content sections"
    
    def test_interactive_query_refinement_flow(self):
        """Test interactive query refinement with user collaboration."""
        from src.mcp_server.feedback.interactive_feedback_flows import (
            InteractiveFeedbackFlows, InteractionType, InteractionState
        )
        
        flows = InteractiveFeedbackFlows()
        
        # Start query refinement flow
        context = {
            "query": "machine learning",
            "result_count": 5,
            "available_collections": ["ai_research", "tutorials", "papers"]
        }
        session = flows.start_query_refinement_flow(context)
        
        assert session.interaction_type == InteractionType.QUERY_REFINEMENT
        assert session.state == InteractionState.IN_PROGRESS
        
        # Verify refinement options
        refinement_interaction = session.interactions[0]
        assert "Query results may be improved" in refinement_interaction.prompt
        assert "Make query more specific" in refinement_interaction.options
        assert "Add related terms" in refinement_interaction.options
        
        # Choose to add related terms
        flows.process_user_response(session.session_id, "Add related terms")
        
        # Verify term input interaction
        assert len(session.interactions) == 2
        terms_interaction = session.interactions[1]
        assert "Enter additional search terms" in terms_interaction.prompt
        
        # Provide additional terms
        flows.process_user_response(session.session_id, "neural networks, deep learning")
        
        # Verify completion
        completed_session = flows.get_session(session.session_id)
        assert completed_session.state == InteractionState.COMPLETED
        assert completed_session.context["refinement"] == "add_terms"
        assert "neural networks" in completed_session.context["additional_terms"]
        assert "deep learning" in completed_session.context["additional_terms"]
    
    def test_interactive_error_recovery_flow(self):
        """Test interactive feedback for error recovery procedures."""
        from src.mcp_server.feedback.interactive_feedback_flows import (
            InteractiveFeedbackFlows, InteractionType, InteractionState
        )
        
        flows = InteractiveFeedbackFlows()
        
        # Start error recovery flow
        context = {
            "error_type": "timeout",
            "error_message": "Connection timeout during document processing",
            "recoverable": True,
            "error_details": "Network connection lost after 30 seconds"
        }
        session = flows.start_error_recovery_flow(context)
        
        assert session.interaction_type == InteractionType.ERROR_RECOVERY
        assert session.state == InteractionState.IN_PROGRESS
        
        # Verify error recovery options
        recovery_interaction = session.interactions[0]
        assert "Error occurred" in recovery_interaction.prompt
        assert "Retry operation" in recovery_interaction.options
        assert "Modify parameters and retry" in recovery_interaction.options
        
        # Choose to modify parameters
        flows.process_user_response(session.session_id, "Modify parameters and retry")
        
        # Verify parameter modification interaction
        assert len(session.interactions) == 2
        param_interaction = session.interactions[1]
        assert "Which parameters would you like to modify" in param_interaction.prompt
        assert "Timeout settings" in param_interaction.options
        
        # Select timeout settings
        flows.process_user_response(session.session_id, "Timeout settings")
        
        # Verify value input interaction
        assert len(session.interactions) == 3
        value_interaction = session.interactions[2]
        assert "Enter new parameter value" in value_interaction.prompt
        
        # Provide new timeout value
        flows.process_user_response(session.session_id, "60")
        
        # Verify completion
        completed_session = flows.get_session(session.session_id)
        assert completed_session.state == InteractionState.COMPLETED
        assert completed_session.context["recovery_action"] == "modify_and_retry"
        assert completed_session.context["modified_parameters"]["Timeout settings"] == "60"


class TestFeedbackProtocolCompliance:
    """Test protocol compliance for all feedback and progress reporting."""
    
    def test_feedback_json_rpc_compliance(self):
        """Test that all feedback follows JSON-RPC 2.0 protocol."""
        from src.mcp_server.feedback.protocol_compliance import (
            FeedbackProtocolManager, JsonRpcMessage
        )
        
        manager = FeedbackProtocolManager()
        
        # Test progress message JSON-RPC compliance
        progress_msg = manager.create_compliant_progress_message(
            operation_id="test_op_001",
            progress=50.0,
            message="Processing documents",
            request_id="req_123"
        )
        
        # Validate JSON-RPC structure
        assert progress_msg.jsonrpc == "2.0"
        assert progress_msg.id == "req_123"
        assert progress_msg.method == "feedback/message"
        assert progress_msg.params is not None
        
        # Validate JSON serialization
        json_str = progress_msg.to_json()
        assert '"jsonrpc": "2.0"' in json_str
        assert '"method": "feedback/message"' in json_str
        
        # Test status message JSON-RPC compliance
        status_msg = manager.create_compliant_status_message(
            operation_id="test_op_001",
            status="completed",
            request_id="req_124"
        )
        
        assert status_msg.jsonrpc == "2.0"
        assert status_msg.id == "req_124"
        assert status_msg.method == "feedback/message"
        
        # Test error message JSON-RPC compliance
        error_msg = manager.create_compliant_error_message(
            error_code=-32601,
            error_message="Method not found",
            request_id="req_125"
        )
        
        assert error_msg.jsonrpc == "2.0"
        assert error_msg.id == "req_125"
        assert error_msg.error is not None
        assert error_msg.error["code"] == -32601
        assert error_msg.error["message"] == "Method not found"
    
    def test_feedback_mcp_protocol_compliance(self):
        """Test that feedback follows MCP protocol specifications."""
        from src.mcp_server.feedback.protocol_compliance import (
            FeedbackProtocolManager, FeedbackMessageType, ContentType
        )
        
        manager = FeedbackProtocolManager()
        
        # Test MCP feedback message structure
        progress_msg = manager.create_compliant_progress_message(
            operation_id="mcp_test_001",
            progress=75.0,
            message="MCP protocol test",
            metadata={"test": "value"}
        )
        
        # Validate MCP-specific parameters
        params = progress_msg.params
        assert params["type"] == FeedbackMessageType.PROGRESS.value
        assert params["content"] == "MCP protocol test"
        assert params["contentType"] == ContentType.TEXT.value
        assert params["operationId"] == "mcp_test_001"
        assert params["progress"] == 75.0
        assert params["metadata"]["test"] == "value"
        
        # Test validation
        is_valid, errors = manager.validator.validate_feedback_message(progress_msg)
        assert is_valid, f"MCP validation failed: {errors}"
        
        # Test different content types
        status_msg = manager.create_compliant_status_message(
            operation_id="mcp_test_002",
            status="in_progress",
            details={"step": "processing", "estimated_time": 30}
        )
        
        status_params = status_msg.params
        assert status_params["type"] == FeedbackMessageType.STATUS.value
        assert status_params["contentType"] == ContentType.JSON.value
        assert isinstance(status_params["content"], dict)
        
        # Validate MCP compliance
        is_valid, errors = manager.validator.validate_feedback_message(status_msg)
        assert is_valid, f"MCP status validation failed: {errors}"
    
    def test_feedback_content_type_handling(self):
        """Test proper content type handling in feedback messages."""
        from src.mcp_server.feedback.protocol_compliance import (
            ProtocolCompliantFeedbackFormatter, ContentType, FeedbackMessageType, McpFeedbackMessage
        )
        
        formatter = ProtocolCompliantFeedbackFormatter()
        
        # Test TEXT content type
        text_feedback = McpFeedbackMessage(
            message_type=FeedbackMessageType.PROGRESS,
            content="Simple text message",
            content_type=ContentType.TEXT
        )
        
        assert formatter.validate_content_type(text_feedback.content, ContentType.TEXT)
        
        # Test JSON content type
        json_feedback = McpFeedbackMessage(
            message_type=FeedbackMessageType.STATUS,
            content={"status": "running", "details": {"progress": 45}},
            content_type=ContentType.JSON
        )
        
        assert formatter.validate_content_type(json_feedback.content, ContentType.JSON)
        
        # Test MARKDOWN content type
        markdown_feedback = McpFeedbackMessage(
            message_type=FeedbackMessageType.NOTIFICATION,
            content="# Processing Complete\n\n**Results**: 100 documents processed",
            content_type=ContentType.MARKDOWN
        )
        
        assert formatter.validate_content_type(markdown_feedback.content, ContentType.MARKDOWN)
        
        # Test invalid content type mismatches
        assert not formatter.validate_content_type({"dict": "content"}, ContentType.TEXT)
        assert not formatter.validate_content_type("text content", ContentType.JSON)
        
        # Test validation through full message pipeline
        jsonrpc_msg = text_feedback.to_jsonrpc("test_req")
        is_valid, errors = formatter.validate_mcp_compliance(jsonrpc_msg)
        assert is_valid, f"Content type validation failed: {errors}"
    
    def test_feedback_error_handling_compliance(self):
        """Test that feedback error handling follows protocol standards."""
        from src.mcp_server.feedback.protocol_compliance import (
            FeedbackProtocolManager, JsonRpcMessage
        )
        
        manager = FeedbackProtocolManager()
        
        # Test standard JSON-RPC error codes
        standard_errors = [
            (-32700, "Parse error"),
            (-32600, "Invalid Request"),
            (-32601, "Method not found"),
            (-32602, "Invalid params"),
            (-32603, "Internal error")
        ]
        
        for error_code, error_message in standard_errors:
            error_msg = manager.create_compliant_error_message(
                error_code=error_code,
                error_message=error_message,
                request_id="error_test"
            )
            
            # Validate error structure
            assert error_msg.error is not None
            assert error_msg.error["code"] == error_code
            assert error_msg.error["message"] == error_message
            
            # Validate JSON-RPC compliance
            is_valid, errors = manager.validator.validate_feedback_message(error_msg)
            assert is_valid, f"Error message validation failed: {errors}"
        
        # Test custom error with additional data
        custom_error = manager.create_compliant_error_message(
            error_code=-50000,
            error_message="Custom operation failed",
            error_data={
                "operation_id": "custom_op_001",
                "failure_reason": "insufficient_resources",
                "retry_after": 300
            },
            request_id="custom_error_test"
        )
        
        assert custom_error.error["data"]["operation_id"] == "custom_op_001"
        assert custom_error.error["data"]["retry_after"] == 300
        
        # Validate custom error compliance
        is_valid, errors = manager.validator.validate_feedback_message(custom_error)
        assert is_valid, f"Custom error validation failed: {errors}"
    
    def test_feedback_response_timing_compliance(self):
        """Test that feedback responses are delivered with minimal latency."""
        import time
        from src.mcp_server.feedback.protocol_compliance import (
            FeedbackProtocolManager, ProtocolCompliantFeedbackFormatter
        )
        
        manager = FeedbackProtocolManager()
        formatter = ProtocolCompliantFeedbackFormatter()
        
        # Test message within timing constraints
        current_msg = manager.create_compliant_progress_message(
            operation_id="timing_test_001",
            progress=25.0,
            message="Current message"
        )
        
        # Should pass timing validation immediately
        is_valid, errors = formatter.validate_timing_compliance(current_msg)
        assert is_valid, f"Current message timing failed: {errors}"
        
        # Test message with custom timestamp (simulating age)
        old_timestamp = time.time() - 10.0  # 10 seconds ago
        old_msg = manager.create_compliant_progress_message(
            operation_id="timing_test_002",
            progress=50.0,
            message="Old message"
        )
        old_msg.timestamp = old_timestamp
        
        # Should fail timing validation (exceeds max response time of 5 seconds)
        is_valid, errors = formatter.validate_timing_compliance(old_msg)
        assert not is_valid
        assert any("exceeds max response time" in error for error in errors)
        
        # Test message batching for timing compliance
        messages = []
        for i in range(150):  # Exceeds max batch size of 100
            msg = manager.create_compliant_progress_message(
                operation_id=f"batch_test_{i:03d}",
                progress=float(i),
                message=f"Batch message {i}"
            )
            messages.append(msg)
        
        # Should create multiple batches
        batches = formatter.batch_messages(messages)
        assert len(batches) == 2  # 100 + 50
        assert len(batches[0]) == 100
        assert len(batches[1]) == 50
        
        # Each batch should be within constraints
        for batch in batches:
            assert len(batch) <= formatter.timing_constraints["max_batch_size"]


class TestAsynchronousProgressReporting:
    """Test asynchronous progress reporting capabilities."""
    
    @pytest.mark.asyncio
    async def test_async_progress_reporting_initialization(self):
        """Test initialization of asynchronous progress reporting system."""
        from src.mcp_server.feedback.async_progress_reporter import (
            AsyncProgressReporter, ProgressEventType
        )
        
        # Test default initialization
        reporter = AsyncProgressReporter()
        assert reporter.max_concurrent_operations == 100
        assert len(reporter.active_operations) == 0
        assert len(reporter.event_consumers) == 0
        assert len(reporter.global_callbacks) == 0
        assert not reporter.is_shutdown
        
        # Test custom initialization
        custom_reporter = AsyncProgressReporter(max_concurrent_operations=50)
        assert custom_reporter.max_concurrent_operations == 50
        
        # Test operation initialization
        operation_id = "test_async_op_001"
        tracker = await reporter.initialize_operation(
            operation_id,
            metadata={"test": True, "async": True}
        )
        
        assert tracker.operation_id == operation_id
        assert tracker.current_progress == 0.0
        assert tracker.status == "running"
        assert not tracker.is_cancelled
        assert tracker.metadata["test"] is True
        assert tracker.metadata["async"] is True
        
        # Verify operation is tracked
        assert operation_id in reporter.active_operations
        assert operation_id in reporter.event_consumers
        
        # Clean up
        await reporter.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_progress_tracking(self):
        """Test progress tracking for multiple concurrent operations."""
        from src.mcp_server.feedback.async_progress_reporter import (
            AsyncProgressReporter, ProgressEventType
        )
        
        reporter = AsyncProgressReporter(max_concurrent_operations=5)
        
        # Initialize multiple operations
        operation_ids = [f"concurrent_op_{i:03d}" for i in range(3)]
        trackers = {}
        
        for op_id in operation_ids:
            tracker = await reporter.initialize_operation(
                op_id,
                metadata={"operation_index": operation_ids.index(op_id)}
            )
            trackers[op_id] = tracker
        
        # Verify all operations are tracked
        assert len(reporter.active_operations) == 3
        
        # Report progress for different operations concurrently
        progress_tasks = []
        for i, op_id in enumerate(operation_ids):
            progress = (i + 1) * 25.0  # 25%, 50%, 75%
            task = asyncio.create_task(
                reporter.report_progress(
                    op_id,
                    progress,
                    f"Progress update {i+1}",
                    {"step": i+1}
                )
            )
            progress_tasks.append(task)
        
        # Wait for all progress updates
        await asyncio.gather(*progress_tasks)
        
        # Verify individual progress
        for i, op_id in enumerate(operation_ids):
            expected_progress = (i + 1) * 25.0
            status = await reporter.get_operation_status(op_id)
            assert status["progress"] == expected_progress
            assert status["status"] == "running"
        
        # Complete operations concurrently
        completion_tasks = []
        for op_id in operation_ids:
            task = asyncio.create_task(
                reporter.complete_operation(
                    op_id,
                    f"Completed {op_id}",
                    {"completed": True}
                )
            )
            completion_tasks.append(task)
        
        await asyncio.gather(*completion_tasks)
        
        # Verify all operations are completed and cleaned up
        assert len(reporter.active_operations) == 0
        
        # Clean up
        await reporter.shutdown()
    
    @pytest.mark.asyncio
    async def test_async_progress_event_generation(self):
        """Test asynchronous generation of progress events."""
        from src.mcp_server.feedback.async_progress_reporter import (
            AsyncProgressReporter, ProgressEventType, AsyncProgressEvent
        )
        
        reporter = AsyncProgressReporter()
        operation_id = "async_event_test"
        
        # Capture emitted events
        emitted_events = []
        
        async def event_callback(event: AsyncProgressEvent):
            emitted_events.append(event)
        
        reporter.add_global_callback(event_callback)
        
        # Initialize operation (should emit STARTED event)
        await reporter.initialize_operation(operation_id, {"test": "async_events"})
        
        # Wait a bit for async processing
        await asyncio.sleep(0.01)
        
        # Report progress (should emit PROGRESS event)
        await reporter.report_progress(
            operation_id,
            30.0,
            "Async progress update",
            {"async_test": True}
        )
        
        await asyncio.sleep(0.01)
        
        # Report more progress
        await reporter.report_progress(
            operation_id,
            75.0,
            "Another async update",
            {"step": 2}
        )
        
        await asyncio.sleep(0.01)
        
        # Complete operation (should emit COMPLETED event)
        await reporter.complete_operation(
            operation_id,
            "Async operation completed",
            {"final": True}
        )
        
        await asyncio.sleep(0.01)
        
        # Verify events were generated
        assert len(emitted_events) >= 4  # STARTED + 2 PROGRESS + COMPLETED
        
        # Verify event types and sequence
        event_types = [event.event_type for event in emitted_events]
        assert ProgressEventType.STARTED in event_types
        assert ProgressEventType.PROGRESS in event_types
        assert ProgressEventType.COMPLETED in event_types
        
        # Verify event content
        started_event = next(e for e in emitted_events if e.event_type == ProgressEventType.STARTED)
        assert started_event.operation_id == operation_id
        assert started_event.progress_percentage == 0.0
        assert started_event.metadata["test"] == "async_events"
        
        progress_events = [e for e in emitted_events if e.event_type == ProgressEventType.PROGRESS]
        assert len(progress_events) >= 2
        assert progress_events[0].progress_percentage == 30.0
        assert progress_events[1].progress_percentage == 75.0
        
        completed_event = next(e for e in emitted_events if e.event_type == ProgressEventType.COMPLETED)
        assert completed_event.operation_id == operation_id
        assert completed_event.progress_percentage == 100.0
        assert completed_event.metadata["final"] is True
        
        # Clean up
        await reporter.shutdown()
    
    @pytest.mark.asyncio
    async def test_async_progress_event_consumption(self):
        """Test asynchronous consumption of progress events by clients."""
        from src.mcp_server.feedback.async_progress_reporter import (
            AsyncProgressReporter, ProgressEventType, AsyncProgressEvent
        )
        
        reporter = AsyncProgressReporter()
        operation_id = "consumer_test"
        
        # Initialize operation
        await reporter.initialize_operation(operation_id, {"consumer_test": True})
        
        # Create consumer
        consumer = await reporter.create_consumer(operation_id)
        
        # Start consuming events
        consumed_events = []
        
        async def consume_callback(event: AsyncProgressEvent):
            consumed_events.append(event)
        
        await consumer.start_consuming(consume_callback)
        
        # Give consumer time to start
        await asyncio.sleep(0.01)
        
        # Report progress (consumer should receive events)
        await reporter.report_progress(
            operation_id,
            40.0,
            "Consumer test progress",
            {"consumed": True}
        )
        
        await asyncio.sleep(0.01)
        
        await reporter.report_progress(
            operation_id,
            80.0,
            "More consumer progress",
            {"step": 2}
        )
        
        await asyncio.sleep(0.01)
        
        # Complete operation
        await reporter.complete_operation(
            operation_id,
            "Consumer test completed"
        )
        
        # Give time for final event consumption
        await asyncio.sleep(0.01)
        
        # Verify consumer received events
        assert len(consumed_events) >= 2  # Should have progress events
        
        # Verify consumed event content
        progress_events = [e for e in consumed_events if e.event_type == ProgressEventType.PROGRESS]
        assert len(progress_events) >= 2
        assert progress_events[0].progress_percentage == 40.0
        assert progress_events[0].metadata["consumed"] is True
        assert progress_events[1].progress_percentage == 80.0
        assert progress_events[1].metadata["step"] == 2
        
        # Verify consumer internal state
        consumer_events = consumer.get_consumed_events()
        assert len(consumer_events) >= 2
        
        # Stop consumer
        await consumer.stop_consuming()
        
        # Clean up
        await reporter.shutdown()
    
    @pytest.mark.asyncio
    async def test_async_error_handling_in_progress_reporting(self):
        """Test error handling in asynchronous progress reporting."""
        from src.mcp_server.feedback.async_progress_reporter import (
            AsyncProgressReporter, ProgressEventType, AsyncProgressEvent
        )
        
        reporter = AsyncProgressReporter()
        operation_id = "error_test_async"
        
        # Capture error events
        error_events = []
        
        async def error_callback(event: AsyncProgressEvent):
            if event.event_type == ProgressEventType.ERROR:
                error_events.append(event)
        
        reporter.add_global_callback(error_callback)
        
        # Initialize operation
        await reporter.initialize_operation(operation_id, {"error_test": True})
        
        # Report some progress
        await reporter.report_progress(
            operation_id,
            25.0,
            "Progress before error"
        )
        
        # Report error
        test_error = ValueError("Async test error")
        await reporter.report_error(
            operation_id,
            test_error,
            "Test error occurred",
            {"error_context": "async_test"}
        )
        
        await asyncio.sleep(0.01)
        
        # Verify error event was generated
        assert len(error_events) == 1
        error_event = error_events[0]
        
        assert error_event.operation_id == operation_id
        assert error_event.event_type == ProgressEventType.ERROR
        assert error_event.message == "Test error occurred"
        assert error_event.metadata["error_context"] == "async_test"
        assert error_event.error_details is not None
        assert error_event.error_details["error_type"] == "ValueError"
        assert error_event.error_details["error_message"] == "Async test error"
        
        # Verify operation was cleaned up
        status = await reporter.get_operation_status(operation_id)
        assert status is None  # Should be cleaned up after error
        
        # Test cancellation error handling
        cancel_operation_id = "cancel_test_async"
        await reporter.initialize_operation(cancel_operation_id)
        
        # Report progress
        await reporter.report_progress(cancel_operation_id, 50.0, "Before cancellation")
        
        # Cancel operation
        await reporter.cancel_operation(cancel_operation_id, "Async cancellation test")
        
        # Verify operation was cancelled and cleaned up
        cancel_status = await reporter.get_operation_status(cancel_operation_id)
        assert cancel_status is None  # Should be cleaned up after cancellation
        
        # Test error in callback (should not crash the system)
        def bad_callback(event):
            raise RuntimeError("Callback error")
        
        reporter.add_global_callback(bad_callback)
        
        error_test_id = "callback_error_test"
        await reporter.initialize_operation(error_test_id)
        await reporter.report_progress(error_test_id, 30.0, "Testing callback error")
        
        # Should not crash, operation should continue
        await reporter.complete_operation(error_test_id, "Completed despite callback error")
        
        # Clean up
        await reporter.shutdown()


class TestProgressReportingIntegration:
    """Test integration of progress reporting with existing MCP components."""
    
    def test_integration_with_stdio_communication(self):
        """Test integration with STDIO communication layer."""
        from src.mcp_server.feedback.progress_integration import (
            StdioProgressCommunicator, StdioMessage
        )
        
        communicator = StdioProgressCommunicator({
            "enable_batching": True,
            "max_queue_size": 100
        })
        
        # Test progress message creation
        progress_msg = communicator.send_progress_message(
            "stdio_test_001",
            45.0,
            "Processing documents via STDIO",
            {"batch": 1, "total_batches": 4}
        )
        
        assert isinstance(progress_msg, StdioMessage)
        assert progress_msg.jsonrpc == "2.0"
        assert progress_msg.method == "progress/update"
        assert progress_msg.params["operation_id"] == "stdio_test_001"
        assert progress_msg.params["progress"] == 45.0
        assert progress_msg.params["message"] == "Processing documents via STDIO"
        assert progress_msg.params["metadata"]["batch"] == 1
        
        # Test status message creation
        status_msg = communicator.send_status_message(
            "stdio_test_001",
            "in_progress",
            {"phase": "document_processing", "estimated_completion": "2m"}
        )
        
        assert isinstance(status_msg, StdioMessage)
        assert status_msg.method == "status/update"
        assert status_msg.params["operation_id"] == "stdio_test_001"
        assert status_msg.params["status"] == "in_progress"
        assert status_msg.params["details"]["phase"] == "document_processing"
        
        # Test message queue functionality
        pending_messages = communicator.get_pending_messages()
        assert len(pending_messages) == 2  # progress + status
        
        # Verify JSON serialization
        progress_json = progress_msg.to_json()
        assert '"jsonrpc":"2.0"' in progress_json
        assert '"method":"progress/update"' in progress_json
        
        # Queue should be cleared after getting messages
        pending_again = communicator.get_pending_messages()
        assert len(pending_again) == 0
    
    def test_integration_with_response_formatter(self):
        """Test integration with response formatting system."""
        from src.mcp_server.feedback.progress_integration import (
            ResponseFormatterIntegration, ResponseFormat, ErrorContext
        )
        
        formatter = ResponseFormatterIntegration({
            "include_timestamps": True,
            "format_version": "1.0"
        })
        
        # Test response formatting with progress
        operation_data = {"result": "document_processed", "id": "doc_123"}
        progress_info = {"progress": 80.0, "current_step": "embedding_generation"}
        feedback_info = {"suggestions": ["refine_query"], "confidence": "high"}
        
        response = formatter.format_response_with_progress(
            operation_data,
            progress_info,
            feedback_info
        )
        
        assert isinstance(response, ResponseFormat)
        assert response.status == "success"
        assert response.data == operation_data
        assert response.progress == progress_info
        assert response.feedback == feedback_info
        assert "formatted_at" in response.metadata
        assert response.metadata["formatter"] == "ResponseFormatterIntegration"
        
        # Test error response formatting
        error_context = ErrorContext(
            operation_id="error_test_001",
            error_type="ValidationError",
            error_message="Invalid parameter value",
            user_message="Please check your input parameters",
            recovery_suggestions=["Verify parameter format", "Use valid range"],
            metadata={"parameter": "chunk_size", "provided_value": -1}
        )
        
        error_response = formatter.format_error_response_with_context(error_context)
        
        assert error_response.status == "error"
        assert error_response.data["error_type"] == "ValidationError"
        assert error_response.data["user_message"] == "Please check your input parameters"
        assert len(error_response.data["recovery_suggestions"]) == 2
        assert error_response.metadata["operation_id"] == "error_test_001"
        assert error_response.metadata["error_context"]["parameter"] == "chunk_size"
        
        # Test formatter cache
        cached_formats = formatter.get_cached_formats()
        assert isinstance(cached_formats, dict)
    
    def test_integration_with_error_handler(self):
        """Test integration with error handling system."""
        from src.mcp_server.feedback.progress_integration import (
            ErrorHandlerIntegration, ErrorContext
        )
        
        error_handler = ErrorHandlerIntegration({
            "max_error_history": 50,
            "enable_recovery_strategies": True
        })
        
        # Test error handling
        test_error = ValueError("Test validation error")
        context = {"operation": "document_ingestion", "batch_id": "batch_001"}
        
        error_context = error_handler.handle_progress_error(
            "error_integration_test",
            test_error,
            context
        )
        
        assert isinstance(error_context, ErrorContext)
        assert error_context.operation_id == "error_integration_test"
        assert error_context.error_type == "ValueError"
        assert error_context.error_message == "Test validation error"
        assert error_context.user_message == "An error occurred in operation error_integration_test"
        assert len(error_context.recovery_suggestions) > 0
        assert error_context.metadata == context
        
        # Test recovery strategy registration and execution
        def recovery_strategy(error_ctx: ErrorContext):
            return f"Recovered from {error_ctx.error_type} in {error_ctx.operation_id}"
        
        error_handler.register_recovery_strategy("ValueError", recovery_strategy)
        
        recovery_result = error_handler.execute_recovery(error_context)
        assert recovery_result == "Recovered from ValueError in error_integration_test"
        
        # Test error statistics
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 1
        assert "ValueError" in stats["error_types"]
        assert stats["error_types"]["ValueError"] == 1
        assert len(stats["recent_errors"]) == 1
    
    def test_integration_with_mcp_tools(self):
        """Test integration with existing MCP tools."""
        from src.mcp_server.feedback.progress_integration import McpToolsIntegration
        
        tools_integration = McpToolsIntegration()
        
        # Mock MCP tool function
        def mock_search_tool(query: str, collection: str = "default"):
            """Mock search tool for testing."""
            import time
            time.sleep(0.01)  # Simulate processing time
            return {"results": [f"result_for_{query}"], "collection": collection}
        
        # Register tool for tracking
        tracked_search = tools_integration.register_tool_for_tracking(
            "search_knowledge_base",
            mock_search_tool
        )
        
        # Execute tracked tool
        result = tracked_search("test query", collection="documents")
        
        # Verify tool result
        assert result["results"] == ["result_for_test query"]
        assert result["collection"] == "documents"
        
        # Verify tracking
        search_progress = tools_integration.get_tool_progress("search_knowledge_base")
        assert len(search_progress) == 1
        
        operation = search_progress[0]
        assert operation["tool_name"] == "search_knowledge_base"
        assert operation["status"] == "completed"
        assert operation["result"] == result
        assert "start_time" in operation
        assert "end_time" in operation
        
        # Test all tool operations
        all_operations = tools_integration.get_all_tool_operations()
        assert len(all_operations) == 1
        
        # Test error handling in tool tracking
        def failing_tool():
            raise RuntimeError("Tool execution failed")
        
        tracked_failing = tools_integration.register_tool_for_tracking(
            "failing_tool",
            failing_tool
        )
        
        try:
            tracked_failing()
            assert False, "Should have raised exception"
        except RuntimeError:
            pass  # Expected
        
        # Verify error was tracked
        failing_progress = tools_integration.get_tool_progress("failing_tool")
        assert len(failing_progress) == 1
        assert failing_progress[0]["status"] == "error"
        assert "Tool execution failed" in failing_progress[0]["error"]
        
        # Test cleanup
        tools_integration.cleanup_completed_operations(max_age_seconds=0)
        remaining_operations = tools_integration.get_all_tool_operations()
        assert len(remaining_operations) == 0  # All should be cleaned up
    
    def test_integration_with_validation_system(self):
        """Test integration with parameter validation system."""
        from src.mcp_server.feedback.progress_integration import ValidationSystemIntegration
        
        validation_integration = ValidationSystemIntegration({
            "strict_validation": True,
            "log_validation_results": True
        })
        
        # Register validation rules
        validation_integration.register_validation_rule(
            "progress",
            lambda x: isinstance(x, (int, float)) and 0 <= x <= 100,
            "Progress must be a number between 0 and 100"
        )
        
        validation_integration.register_validation_rule(
            "operation_id",
            lambda x: isinstance(x, str) and len(x) > 0,
            "Operation ID must be a non-empty string"
        )
        
        # Test valid progress parameters
        valid_params = {
            "progress": 75.5,
            "operation_id": "valid_operation_123",
            "message": "Processing documents"
        }
        
        validation_result = validation_integration.validate_progress_parameters(valid_params)
        
        assert validation_result["valid"] is True
        assert len(validation_result["errors"]) == 0
        assert validation_result["validated_parameters"] == valid_params
        
        # Test invalid progress parameters
        invalid_params = {
            "progress": 150,  # Invalid: > 100
            "operation_id": "",  # Invalid: empty string
            "message": "Invalid progress"
        }
        
        invalid_result = validation_integration.validate_progress_parameters(invalid_params)
        
        assert invalid_result["valid"] is False
        assert len(invalid_result["errors"]) == 2
        
        # Check specific errors
        progress_error = next(e for e in invalid_result["errors"] if e["parameter"] == "progress")
        assert "between 0 and 100" in progress_error["message"]
        
        operation_id_error = next(e for e in invalid_result["errors"] if e["parameter"] == "operation_id")
        assert "non-empty string" in operation_id_error["message"]
        
        # Test feedback validation
        valid_feedback = {
            "type": "suggestion",
            "message": "Consider refining your query",
            "confidence": "medium"
        }
        
        feedback_result = validation_integration.validate_feedback_parameters(valid_feedback)
        assert feedback_result["valid"] is True
        
        # Test invalid feedback (missing required fields)
        invalid_feedback = {
            "confidence": "high"
            # Missing 'type' and 'message'
        }
        
        invalid_feedback_result = validation_integration.validate_feedback_parameters(invalid_feedback)
        assert invalid_feedback_result["valid"] is False
        assert len(invalid_feedback_result["errors"]) == 2
        
        # Test validation statistics
        stats = validation_integration.get_validation_statistics()
        assert stats["total_validations"] == 4  # 2 progress + 2 feedback
        assert stats["successful_validations"] == 2  # Only the valid ones
        assert stats["success_rate"] == 0.5  # 2/4
        assert len(stats["recent_validations"]) <= 10


class TestFeedbackConfiguration:
    """Test configuration and customization of feedback and progress reporting."""
    
    def test_feedback_configuration_loading(self):
        """Test loading of feedback configuration from config files."""
        import tempfile
        import json
        from src.mcp_server.feedback.feedback_configuration import (
            FeedbackConfiguration, VerbosityLevel, FeedbackType
        )
        
        # Create temporary config file
        config_data = {
            "frequency": {
                "update_interval_seconds": 2.0,
                "batch_size": 20,
                "max_frequency_hz": 5.0,
                "enable_adaptive_frequency": False
            },
            "verbosity": {
                "global_verbosity": "detailed",
                "per_type_verbosity": {
                    "progress": "verbose",
                    "error": "debug"
                },
                "include_timestamps": False,
                "include_debug_info": True
            },
            "system": {
                "enabled": False,
                "max_concurrent_operations": 50,
                "enable_caching": False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test loading configuration
            config = FeedbackConfiguration(config_path)
            
            # Verify frequency config
            freq_config = config.get_frequency_config()
            assert freq_config.update_interval_seconds == 2.0
            assert freq_config.batch_size == 20
            assert freq_config.max_frequency_hz == 5.0
            assert freq_config.enable_adaptive_frequency is False
            
            # Verify verbosity config
            verb_config = config.get_verbosity_config()
            assert verb_config.global_verbosity == VerbosityLevel.DETAILED
            assert verb_config.get_verbosity_for_type(FeedbackType.PROGRESS) == VerbosityLevel.VERBOSE
            assert verb_config.get_verbosity_for_type(FeedbackType.ERROR) == VerbosityLevel.DEBUG
            assert verb_config.include_timestamps is False
            assert verb_config.include_debug_info is True
            
            # Verify system config
            sys_config = config.get_system_config()
            assert sys_config.enabled is False
            assert sys_config.max_concurrent_operations == 50
            assert sys_config.enable_caching is False
            
        finally:
            import os
            os.unlink(config_path)
    
    def test_progress_reporting_frequency_configuration(self):
        """Test configuration of progress reporting frequency."""
        from src.mcp_server.feedback.feedback_configuration import (
            FeedbackConfiguration, ProgressReportingFrequencyConfig
        )
        
        config = FeedbackConfiguration()
        
        # Test default frequency config
        freq_config = config.get_frequency_config()
        assert freq_config.update_interval_seconds == 1.0
        assert freq_config.batch_size == 10
        assert freq_config.max_frequency_hz == 10.0
        assert freq_config.enable_adaptive_frequency is True
        
        # Test setting new frequency config
        new_freq_config = ProgressReportingFrequencyConfig(
            update_interval_seconds=0.5,
            batch_size=5,
            max_frequency_hz=20.0,
            enable_adaptive_frequency=False
        )
        
        config.set_frequency_config(new_freq_config)
        updated_config = config.get_frequency_config()
        
        assert updated_config.update_interval_seconds == 0.5
        assert updated_config.batch_size == 5
        assert updated_config.max_frequency_hz == 20.0
        assert updated_config.enable_adaptive_frequency is False
        
        # Test adaptive frequency calculation
        adaptive_config = ProgressReportingFrequencyConfig(enable_adaptive_frequency=True)
        
        # Test short operation (no change)
        short_interval = adaptive_config.get_effective_update_interval(30.0)
        assert short_interval == 1.0
        
        # Test medium operation (increase interval)
        medium_interval = adaptive_config.get_effective_update_interval(120.0)
        assert medium_interval == 2.0
        
        # Test long operation (further increase)
        long_interval = adaptive_config.get_effective_update_interval(600.0)
        assert long_interval == 5.0
    
    def test_feedback_verbosity_configuration(self):
        """Test configuration of feedback verbosity levels."""
        from src.mcp_server.feedback.feedback_configuration import (
            FeedbackConfiguration, VerbosityLevel, FeedbackType
        )
        
        config = FeedbackConfiguration()
        
        # Test default verbosity config
        verb_config = config.get_verbosity_config()
        assert verb_config.global_verbosity == VerbosityLevel.STANDARD
        assert verb_config.include_timestamps is True
        assert verb_config.include_operation_context is True
        assert verb_config.include_performance_metrics is False
        assert verb_config.include_debug_info is False
        
        # Test setting global verbosity level
        config.set_verbosity_level(VerbosityLevel.VERBOSE)
        updated_config = config.get_verbosity_config()
        assert updated_config.global_verbosity == VerbosityLevel.VERBOSE
        
        # Test setting type-specific verbosity
        config.set_verbosity_level(VerbosityLevel.DEBUG, FeedbackType.ERROR)
        config.set_verbosity_level(VerbosityLevel.MINIMAL, FeedbackType.PROGRESS)
        
        verb_config = config.get_verbosity_config()
        assert verb_config.get_verbosity_for_type(FeedbackType.ERROR) == VerbosityLevel.DEBUG
        assert verb_config.get_verbosity_for_type(FeedbackType.PROGRESS) == VerbosityLevel.MINIMAL
        assert verb_config.get_verbosity_for_type(FeedbackType.STATUS) == VerbosityLevel.VERBOSE  # Global default
        
        # Test detail inclusion based on verbosity
        assert verb_config.should_include_detail(FeedbackType.ERROR, "debug_info") is True  # DEBUG level
        assert verb_config.should_include_detail(FeedbackType.PROGRESS, "timestamps") is True  # MINIMAL includes timestamps
        assert verb_config.should_include_detail(FeedbackType.PROGRESS, "performance_metrics") is False  # MINIMAL doesn't include metrics
    
    def test_custom_feedback_template_configuration(self):
        """Test configuration of custom feedback templates."""
        from src.mcp_server.feedback.feedback_configuration import (
            FeedbackConfiguration, FeedbackTemplate
        )
        
        config = FeedbackConfiguration()
        template_config = config.get_template_config()
        
        # Test default templates are loaded
        assert len(template_config.templates) == 4  # Default templates
        assert template_config.get_template("progress_update") is not None
        assert template_config.get_template("error_message") is not None
        
        # Test adding custom template
        custom_template = FeedbackTemplate(
            name="custom_progress",
            pattern="Custom: {status} - {details} ({timestamp})",
            required_variables=["status", "details"],
            optional_variables=["timestamp"],
            description="Custom progress template"
        )
        
        config.add_custom_template(custom_template)
        
        # Verify custom template was added
        retrieved_template = template_config.get_template("custom_progress")
        assert retrieved_template is not None
        assert retrieved_template.name == "custom_progress"
        assert retrieved_template.pattern == "Custom: {status} - {details} ({timestamp})"
        assert "status" in retrieved_template.required_variables
        assert "timestamp" in retrieved_template.optional_variables
        
        # Test template formatting
        variables = {
            "status": "In Progress",
            "details": "Processing documents",
            "timestamp": "2024-01-01 12:00:00"
        }
        
        formatted = retrieved_template.format_message(variables)
        expected = "Custom: In Progress - Processing documents (2024-01-01 12:00:00)"
        assert formatted == expected
        
        # Test template validation (missing required variable)
        try:
            retrieved_template.format_message({"details": "test"})  # Missing 'status'
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Missing required variables" in str(e)
        
        # Test disabling custom templates
        template_config.enable_custom_templates = False
        try:
            config.add_custom_template(FeedbackTemplate(name="test", pattern="test"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Custom templates are disabled" in str(e)


class TestFeedbackPerformanceOptimization:
    """Test performance optimization for feedback and progress reporting."""
    
    def test_feedback_performance_under_load(self):
        """Test feedback system performance under high load."""
        # Should maintain good performance with many concurrent operations
        assert False, "Feedback performance under load not implemented"
    
    def test_progress_reporting_memory_efficiency(self):
        """Test memory efficiency of progress reporting system."""
        # Should use memory efficiently even with many tracked operations
        assert False, "Progress reporting memory efficiency not implemented"
    
    def test_feedback_response_time_optimization(self):
        """Test that feedback responses are delivered with minimal latency."""
        # Should deliver feedback with low latency even under load
        assert False, "Feedback response time optimization not implemented"
    
    def test_progress_data_cleanup(self):
        """Test cleanup of progress data for completed operations."""
        # Should clean up progress tracking data for completed operations
        assert False, "Progress data cleanup not implemented" 