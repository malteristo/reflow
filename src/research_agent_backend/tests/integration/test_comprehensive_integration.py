"""
Comprehensive integration tests for end-to-end workflows and performance validation.

This module implements the comprehensive integration test suite identified in the RED PHASE,
covering performance benchmarks, load testing, error handling, and cross-component validation.

Following TDD methodology: RED → GREEN → REFACTOR
REFACTOR PHASE: Optimized test infrastructure with performance improvements and enhanced maintainability.
"""

import pytest
import asyncio
import time
import threading
import tempfile
import shutil
import psutil
import gc
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Research Agent imports
from src.research_agent_backend.core.integration_pipeline import (
    DocumentProcessingPipeline, 
    IntegratedSearchEngine,
    DataPreparationManager,
    CollectionTypeManager,
    apply_integration_patches,
    remove_integration_patches
)
from src.research_agent_backend.core.vector_store import ChromaDBManager
from src.research_agent_backend.core.query_manager import QueryManager
from src.research_agent_backend.core.document_insertion import DocumentInsertionManager
from src.research_agent_backend.core.data_preparation import DataPreparationManager as CoreDataPreparationManager
from src.research_agent_backend.core.collection_type_manager import CollectionTypeManager as CoreCollectionTypeManager
from src.research_agent_backend.utils.config import ConfigManager
from src.research_agent_backend.exceptions.vector_store_exceptions import VectorStoreError


# REFACTOR PHASE: Enhanced test utilities and performance monitoring
@dataclass
class IntegrationTestMetrics:
    """Enhanced test metrics collection for performance analysis."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    operations_per_second: float = 0.0
    success_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "execution_time": round(self.execution_time, 3),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "cpu_usage_percent": round(self.cpu_usage_percent, 2),
            "operations_per_second": round(self.operations_per_second, 2),
            "success_rate": round(self.success_rate, 4)
        }


class PerformanceMonitor:
    """REFACTOR PHASE: Centralized performance monitoring utility."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0.0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def get_metrics(self, operation_count: int = 1) -> IntegrationTestMetrics:
        """Get comprehensive performance metrics."""
        if self.start_time is None:
            return IntegrationTestMetrics()
            
        execution_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_usage = current_memory - self.start_memory
        cpu_usage = self.process.cpu_percent()
        ops_per_second = operation_count / execution_time if execution_time > 0 else 0
        
        return IntegrationTestMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=self.peak_memory - self.start_memory,
            cpu_usage_percent=cpu_usage,
            operations_per_second=ops_per_second,
            success_rate=1.0  # Default success rate
        )


class IntegrationTestUtilities:
    """REFACTOR PHASE: Centralized test utilities for better maintainability."""
    
    @staticmethod
    def create_test_documents(count: int, content_template: str = "Test document {i} content") -> List[Dict[str, Any]]:
        """Create test documents efficiently."""
        return [
            {
                "content": content_template.format(i=i),
                "metadata": {
                    "source": f"docs/test_{i}.md",
                    "document_type": "test",
                    "index": i
                }
            }
            for i in range(count)
        ]
    
    @staticmethod
    def validate_processing_results(results: List[Any], expected_count: int, expected_status: str = "success") -> bool:
        """Validate processing results efficiently."""
        if len(results) != expected_count:
            return False
        
        return all(
            hasattr(result, 'status') and result.status == expected_status
            for result in results
        )
    
    @staticmethod
    def cleanup_test_environment():
        """Clean up test environment resources."""
        # Force garbage collection
        gc.collect()
        
        # Remove integration patches
        try:
            remove_integration_patches()
        except Exception:
            pass  # Patches may not be applied


# REFACTOR PHASE: Enhanced fixtures with better resource management
@pytest.fixture
def integration_config():
    """Enhanced integration configuration."""
    return {
        "vector_store": {
            "provider": "chromadb",
            "in_memory": True,
            "collection_name": "integration_test_collection",
            "distance_metric": "cosine"
        },
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 256,
            "chunk_overlap": 32
        },
        "search": {
            "default_top_k": 10,
            "min_relevance": 0.1
        },
        "processing": {
            "batch_size": 50,
            "max_concurrent": 4
        },
        "performance": {
            "enable_monitoring": True,
            "metrics_collection": True
        }
    }


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    monitor = PerformanceMonitor()
    yield monitor
    # Cleanup happens automatically


@pytest.fixture(scope="function")
def test_utilities():
    """Test utilities fixture with cleanup."""
    utilities = IntegrationTestUtilities()
    yield utilities
    utilities.cleanup_test_environment()


@pytest.fixture
def sample_documents():
    """REFACTOR PHASE: More efficient sample document generation."""
    base_documents = [
        {"content": "# Configuration Guide\n\nThis guide covers system configuration and setup procedures for optimal performance.", 
         "metadata": {"source": "docs/guide.md", "type": "documentation"}},
        {"content": "# API Reference\n\nComplete API documentation with examples and best practices for developers.", 
         "metadata": {"source": "docs/api.md", "type": "reference"}},
        {"content": "# Configuration Manual\n\nDetailed configuration options and troubleshooting guide for administrators.", 
         "metadata": {"source": "docs/config.md", "type": "manual"}}
    ]
    return base_documents


# REFACTOR PHASE: Enhanced test classes with improved organization
class TestEndToEndWorkflows:
    """REFACTOR PHASE: Enhanced end-to-end workflow testing with performance optimization."""

    def test_complete_document_ingestion_pipeline(self, integration_config, sample_documents, performance_monitor, test_utilities):
        """
        REFACTOR PHASE: Optimized complete document ingestion pipeline test.
        Tests the full document processing workflow with performance monitoring.
        """
        performance_monitor.start_monitoring()
        
        # Apply integration patches for testing
        apply_integration_patches()
        
        try:
            # Initialize pipeline components
            pipeline = DocumentProcessingPipeline(integration_config)
            
            # Process documents through complete pipeline
            results = asyncio.run(pipeline.process_documents(sample_documents))
            
            # Validate results efficiently
            assert test_utilities.validate_processing_results(results, len(sample_documents))
            
            # Verify processing metrics
            for result in results:
                assert result.chunks_created > 0
                assert result.embeddings_generated > 0
                assert result.processing_time >= 0
                
            # Performance validation
            metrics = performance_monitor.get_metrics(len(sample_documents))
            assert metrics.execution_time < 5.0, f"Pipeline took too long: {metrics.execution_time}s"
            assert metrics.operations_per_second > 0.5, f"Low throughput: {metrics.operations_per_second} ops/s"
            
        finally:
            remove_integration_patches()

    def test_collection_lifecycle_workflow(self, integration_config, test_utilities):
        """
        REFACTOR PHASE: Enhanced collection lifecycle workflow test.
        """
        apply_integration_patches()
        
        try:
            # Initialize components
            config_manager = ConfigManager()
            vector_store = ChromaDBManager(config_manager, in_memory=True)
            collection_manager = CollectionTypeManager(integration_config)
            
            # Test collection creation
            collection_config = collection_manager.get_collection_config("FUNDAMENTAL")
            assert "max_documents" in collection_config
            assert collection_config["max_documents"] == 5000
            
            # Test document insertion workflow
            test_docs = test_utilities.create_test_documents(5)
            
            # Process and validate
            data_manager = DataPreparationManager(integration_config)
            prepared_data = data_manager.prepare_for_storage(test_docs)
            
            assert len(prepared_data) == len(test_docs)
            
        finally:
            remove_integration_patches()

    def test_cross_component_data_flow(self, integration_config, sample_documents, test_utilities):
        """
        REFACTOR PHASE: Enhanced cross-component data flow validation.
        """
        apply_integration_patches()
        
        try:
            # Initialize all components
            pipeline = DocumentProcessingPipeline(integration_config)
            search_engine = IntegratedSearchEngine(integration_config)
            collection_manager = CollectionTypeManager(integration_config)
            
            # Test data flow: Processing → Storage → Retrieval
            processing_results = asyncio.run(pipeline.process_documents(sample_documents))
            assert test_utilities.validate_processing_results(processing_results, len(sample_documents))
            
            # Test search functionality
            search_results = asyncio.run(search_engine.search("configuration guide", top_k=5))
            assert len(search_results) > 0
            assert all(result.relevance_score >= 0.1 for result in search_results)
            
            # Test collection configuration consistency
            collection_config = collection_manager.get_collection_config("INTEGRATION_TEST")
            assert "max_documents" in collection_config
            
        finally:
            remove_integration_patches()


class TestPerformanceAndLoadTesting:
    """REFACTOR PHASE: Enhanced performance and load testing with detailed metrics."""

    def test_bulk_insertion_performance_benchmarks(self, integration_config, performance_monitor, test_utilities):
        """
        REFACTOR PHASE: Optimized bulk insertion performance benchmarks.
        """
        performance_monitor.start_monitoring()
        apply_integration_patches()
        
        try:
            # Generate larger document set for bulk testing
            bulk_documents = test_utilities.create_test_documents(
                100, 
                "Bulk test document {i} with substantial content for realistic performance testing. " * 5
            )
            
            # Initialize pipeline
            pipeline = DocumentProcessingPipeline(integration_config)
            
            # Perform bulk processing
            results = asyncio.run(pipeline.process_documents(bulk_documents))
            
            # Validate results
            assert test_utilities.validate_processing_results(results, len(bulk_documents))
            
            # Performance validation
            metrics = performance_monitor.get_metrics(len(bulk_documents))
            
            # Enhanced performance assertions
            assert metrics.execution_time < 10.0, f"Bulk processing too slow: {metrics.execution_time}s"
            assert metrics.operations_per_second > 5.0, f"Low bulk throughput: {metrics.operations_per_second} ops/s"
            assert metrics.memory_usage_mb < 100.0, f"Excessive memory usage: {metrics.memory_usage_mb}MB"
            
        finally:
            remove_integration_patches()

    def test_concurrent_query_load_handling(self, integration_config, performance_monitor):
        """
        REFACTOR PHASE: Enhanced concurrent query load handling test.
        """
        performance_monitor.start_monitoring()
        apply_integration_patches()
        
        try:
            search_engine = IntegratedSearchEngine(integration_config)
            
            # Concurrent query execution
            queries = [
                "configuration setup",
                "api documentation", 
                "troubleshooting guide",
                "performance optimization",
                "security settings"
            ]
            
            async def execute_concurrent_queries():
                tasks = []
                for query in queries:
                    for _ in range(3):  # 3 iterations per query
                        task = search_engine.search(query, top_k=5)
                        tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                return results
            
            # Execute concurrent queries
            all_results = asyncio.run(execute_concurrent_queries())
            
            # Validate concurrent execution
            assert len(all_results) == len(queries) * 3
            successful_queries = sum(1 for result in all_results if len(result) > 0)
            success_rate = successful_queries / len(all_results)
            
            assert success_rate >= 0.8, f"Low concurrent query success rate: {success_rate}"
            
            # Performance validation
            metrics = performance_monitor.get_metrics(len(all_results))
            assert metrics.execution_time < 15.0, f"Concurrent queries too slow: {metrics.execution_time}s"
            
        finally:
            remove_integration_patches()

    def test_memory_resource_management(self, integration_config, performance_monitor, test_utilities):
        """
        REFACTOR PHASE: Enhanced memory resource management test with monitoring.
        """
        performance_monitor.start_monitoring()
        apply_integration_patches()
        
        try:
            # Baseline memory measurement
            baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Process documents in batches to test memory management
            batch_size = 25
            total_docs = 100
            pipeline = DocumentProcessingPipeline(integration_config)
            
            all_results = []
            for batch_start in range(0, total_docs, batch_size):
                batch_docs = test_utilities.create_test_documents(
                    batch_size, 
                    f"Memory test batch document {{i}} iteration {batch_start // batch_size}. " * 10
                )
                
                batch_results = asyncio.run(pipeline.process_documents(batch_docs))
                all_results.extend(batch_results)
                
                # Update peak memory tracking
                performance_monitor.update_peak_memory()
                
                # Force garbage collection between batches
                gc.collect()
            
            # Final validation
            assert test_utilities.validate_processing_results(all_results, total_docs)
            
            # Memory management validation
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = final_memory - baseline_memory
            
            # Enhanced memory assertions
            assert memory_growth < 50.0, f"Excessive memory growth: {memory_growth}MB"
            
            metrics = performance_monitor.get_metrics(total_docs)
            assert metrics.peak_memory_mb < 75.0, f"Peak memory too high: {metrics.peak_memory_mb}MB"
            
        finally:
            remove_integration_patches()


class TestErrorHandlingAndEdgeCases:
    """REFACTOR PHASE: Enhanced error handling and edge case testing."""

    def test_database_connection_failure_recovery(self, integration_config, sample_documents, test_utilities):
        """
        REFACTOR PHASE: Enhanced database connection failure recovery test.
        """
        apply_integration_patches()
        
        try:
            # Mock unhealthy vector store  
            with patch('src.research_agent_backend.core.vector_store.ChromaDBManager.health_check') as mock_health:
                mock_health.return_value = {"status": "unhealthy", "error": "Connection failed"}
                
                pipeline = DocumentProcessingPipeline(integration_config)
                results = asyncio.run(pipeline.process_documents(sample_documents))
                
                # Verify error handling
                assert len(results) == len(sample_documents)
                assert all(result.status == "error" for result in results)
                # Check for either "Connection failure" or "connection" in error messages
                assert all(
                    ("Connection failure" in result.error_message or "connection" in result.error_message.lower())
                    for result in results
                )
                
        finally:
            remove_integration_patches()

    def test_invalid_document_cross_component_handling(self, integration_config, test_utilities):
        """
        REFACTOR PHASE: Enhanced invalid document handling across components.
        """
        apply_integration_patches()
        
        try:
            # Create invalid documents
            invalid_docs = [
                {"content": "", "metadata": {"source": "empty.md"}},  # Empty content
                {"content": "x" * 60000, "metadata": {"source": "huge.md"}},  # Too large
                {"metadata": {"source": "no_content.md"}},  # Missing content
                {"content": "valid content", "metadata": {}},  # Missing metadata
            ]
            
            pipeline = DocumentProcessingPipeline(integration_config)
            results = asyncio.run(pipeline.process_documents(invalid_docs))
            
            # Validate error handling for each case
            assert len(results) == len(invalid_docs)
            
            # Check specific error patterns
            error_results = [r for r in results if r.status == "error"]
            assert len(error_results) >= 2, "Should have at least 2 validation errors"
            
        finally:
            remove_integration_patches()

    def test_resource_exhaustion_handling(self, integration_config, performance_monitor, test_utilities):
        """
        REFACTOR PHASE: Enhanced resource exhaustion handling test.
        """
        performance_monitor.start_monitoring()
        apply_integration_patches()
        
        try:
            # Simulate resource exhaustion with very large document set
            large_doc_set = test_utilities.create_test_documents(
                200,
                "Large content document {i} with extensive text content. " * 50
            )
            
            pipeline = DocumentProcessingPipeline(integration_config)
            
            # Process with resource monitoring
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            results = asyncio.run(pipeline.process_documents(large_doc_set))
            
            # Validate that system handled large load
            assert len(results) == len(large_doc_set)
            successful_results = sum(1 for r in results if r.status == "success")
            success_rate = successful_results / len(large_doc_set)
            
            # Should handle at least 80% successfully
            assert success_rate >= 0.8, f"Low success rate under load: {success_rate}"
            
            # Memory should not grow excessively
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = final_memory - start_memory
            assert memory_growth < 200.0, f"Excessive memory growth under load: {memory_growth}MB"
            
        finally:
            remove_integration_patches()


class TestCrossComponentIntegration:
    """REFACTOR PHASE: Enhanced cross-component integration testing."""

    def test_chromadb_query_manager_integration(self, integration_config, test_utilities):
        """
        REFACTOR PHASE: Enhanced ChromaDB and QueryManager integration test.
        """
        apply_integration_patches()
        
        try:
            # Initialize core components
            config_manager = ConfigManager()
            vector_store = ChromaDBManager(config_manager, in_memory=True)
            query_manager = QueryManager(vector_store, config_manager)
            
            # Test basic integration
            health_status = vector_store.health_check()
            assert health_status["status"] == "healthy"
            
            # Test query manager integration - check for actual methods
            test_query = "integration test query"
            assert hasattr(query_manager, 'similarity_search')
            assert hasattr(query_manager, 'get_available_collections')
            assert hasattr(query_manager, 'search_with_text_query')
            
        finally:
            remove_integration_patches()

    def test_data_preparation_insertion_workflow(self, integration_config, sample_documents, test_utilities):
        """
        REFACTOR PHASE: Enhanced data preparation and insertion workflow test.
        """
        apply_integration_patches()
        
        try:
            # Initialize components
            data_prep = DataPreparationManager(integration_config)
            
            # Test data preparation
            prepared_data = data_prep.prepare_for_storage(sample_documents)
            assert len(prepared_data) == len(sample_documents)
            
            # Validate prepared data structure
            for item in prepared_data:
                assert isinstance(item, dict)
                assert "normalized_content" in item or "content" in item
                
        finally:
            remove_integration_patches()

    def test_collection_type_manager_system_integration(self, integration_config, test_utilities):
        """
        REFACTOR PHASE: Enhanced collection type manager system integration test.
        """
        apply_integration_patches()
        
        try:
            collection_manager = CollectionTypeManager(integration_config)
            
            # Test various collection types
            test_types = ["FUNDAMENTAL", "PROJECT_SPECIFIC", "EXPERIMENTAL", "documentation", "code"]
            
            for collection_type in test_types:
                config = collection_manager.get_collection_config(collection_type)
                
                # Validate required fields
                assert "max_documents" in config, f"Missing max_documents for {collection_type}"
                assert "type" in config
                assert "embedding_function" in config
                assert "distance_metric" in config
                assert config["max_documents"] > 0
                
        finally:
            remove_integration_patches()

    def test_configuration_system_cross_component(self, integration_config, test_utilities):
        """
        REFACTOR PHASE: Enhanced configuration system cross-component test.
        """
        apply_integration_patches()
        
        try:
            # Initialize components with shared configuration
            config_manager = ConfigManager()
            vector_store = ChromaDBManager(config_manager, in_memory=True)
            pipeline = DocumentProcessingPipeline(integration_config)
            search_engine = IntegratedSearchEngine(integration_config)
            
            # Test configuration consistency across components
            assert vector_store.config_manager is not None
            assert pipeline.chunking_strategy == integration_config["chunking"]["strategy"]
            assert search_engine.default_top_k == integration_config["search"]["default_top_k"]
            
            # Test configuration propagation
            health_status = vector_store.health_check()
            assert health_status["status"] == "healthy"
            
        finally:
            remove_integration_patches() 