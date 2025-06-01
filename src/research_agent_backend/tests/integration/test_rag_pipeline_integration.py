"""
Integration tests for RAG pipeline with complete caching and optimization layer.

This module contains comprehensive integration tests that validate the end-to-end
RAG pipeline with all caching and optimization components working together:
- Embedding cache integration with LocalEmbeddingService
- Query cache integration with RAG query pipeline  
- Batch processing optimization integration
- Lazy loading integration across components
- Cache invalidation integration with document updates
- Performance validation under realistic workloads

Following TDD RED PHASE: These tests should initially FAIL until complete integration is implemented.
"""

import pytest
import asyncio
import time
import threading
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.research_agent_backend.core.local_embedding_service import LocalEmbeddingService
from src.research_agent_backend.core.query_manager.manager import QueryManager
from src.research_agent_backend.core.vector_store import ChromaDBManager
from src.research_agent_backend.core.enhanced_caching import (
    MultiLevelCacheManager, 
    ModelAwareCacheManager,
    IntelligentCacheWarmer
)
from src.research_agent_backend.core.query_manager.cache import QueryCache
from src.research_agent_backend.core.rag_cache_integration import RAGCacheIntegration
from src.research_agent_backend.core.enhanced_integration import OptimizedPipelineCoordinator
from src.research_agent_backend.core.performance_benchmark import EmbeddingCacheBenchmark
from src.research_agent_backend.core.comprehensive_benchmark import RAGPipelineBenchmark
from src.research_agent_backend.core.integrated_rag_pipeline import IntegratedRAGPipeline


@pytest.fixture
def temp_db_path():
    """Create temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def integrated_rag_pipeline(temp_db_path):
    """Create fully integrated RAG pipeline with all optimizations."""
    # This fixture should FAIL initially - missing complete integration
    pipeline = IntegratedRAGPipeline(
        db_path=temp_db_path,
        enable_embedding_cache=True,
        enable_query_cache=True,
        enable_batch_optimization=True,
        enable_lazy_loading=True,
        enable_cache_invalidation=True
    )
    yield pipeline
    pipeline.cleanup()


class TestEndToEndRAGPipelineIntegration:
    """Test complete RAG pipeline with all optimizations integrated."""
    
    def test_complete_query_flow_with_caching(self, integrated_rag_pipeline):
        """Test complete query flow from input to result with all caches active."""
        # This test should FAIL initially - missing integrated pipeline
        pipeline = integrated_rag_pipeline
        
        # Ingest test documents
        test_documents = [
            {"content": "Machine learning is a subset of artificial intelligence.", "metadata": {"source": "ml_doc.md"}},
            {"content": "Deep learning uses neural networks with multiple layers.", "metadata": {"source": "dl_doc.md"}},
            {"content": "Natural language processing deals with text understanding.", "metadata": {"source": "nlp_doc.md"}}
        ]
        
        # Test document ingestion with cache population
        ingestion_result = pipeline.ingest_documents(test_documents, "test_collection")
        assert ingestion_result.success == True, "Document ingestion should succeed"
        assert ingestion_result.cache_populated == True, "Caches should be populated during ingestion"
        assert ingestion_result.documents_processed == 3, "Should process all documents"
        
        # Test first query (cache miss expected)
        query1_result = pipeline.query("What is machine learning?", ["test_collection"])
        assert query1_result.success == True, "Query should succeed"
        assert len(query1_result.results) > 0, "Should return relevant results"
        assert query1_result.from_embedding_cache == False, "First query should miss embedding cache"
        assert query1_result.from_query_cache == False, "First query should miss query cache"
        
        # Test identical query (cache hit expected)
        query2_result = pipeline.query("What is machine learning?", ["test_collection"])
        assert query2_result.success == True, "Cached query should succeed"
        assert query2_result.from_embedding_cache == True, "Should hit embedding cache"
        assert query2_result.from_query_cache == True, "Should hit query cache"
        assert query2_result.response_time < query1_result.response_time, "Cached query should be faster"
        
        # Test cache statistics
        cache_stats = pipeline.get_cache_statistics()
        assert cache_stats['embedding_cache']['hit_rate'] > 0, "Should have embedding cache hits"
        assert cache_stats['query_cache']['hit_rate'] > 0, "Should have query cache hits"
        assert cache_stats['overall_performance']['speedup_factor'] > 1.5, "Should show performance improvement"
    
    def test_multi_collection_query_with_cache_coordination(self, integrated_rag_pipeline):
        """Test queries across multiple collections with proper cache coordination."""
        # This test should FAIL initially - missing multi-collection cache coordination
        pipeline = integrated_rag_pipeline
        
        # Create multiple collections
        collection1_docs = [
            {"content": "Python is a programming language.", "metadata": {"source": "python.md"}},
            {"content": "JavaScript is used for web development.", "metadata": {"source": "js.md"}}
        ]
        collection2_docs = [
            {"content": "Machine learning algorithms learn from data.", "metadata": {"source": "ml.md"}},
            {"content": "Data science involves statistical analysis.", "metadata": {"source": "ds.md"}}
        ]
        
        # Ingest into separate collections
        pipeline.ingest_documents(collection1_docs, "programming")
        pipeline.ingest_documents(collection2_docs, "datascience")
        
        # Query single collection
        single_result = pipeline.query("Python programming", ["programming"])
        assert single_result.success == True, "Single collection query should succeed"
        assert len(single_result.collections_searched) == 1, "Should search one collection"
        
        # Query multiple collections
        multi_result = pipeline.query("Python data analysis", ["programming", "datascience"])
        assert multi_result.success == True, "Multi-collection query should succeed"
        assert len(multi_result.collections_searched) == 2, "Should search both collections"
        assert multi_result.cache_coordination_used == True, "Should coordinate caches across collections"
        
        # Verify cache isolation
        cache_stats = pipeline.get_cache_statistics()
        assert "programming" in cache_stats['collection_caches'], "Should have programming collection cache"
        assert "datascience" in cache_stats['collection_caches'], "Should have datascience collection cache"
        assert cache_stats['cache_isolation_maintained'] == True, "Should maintain cache isolation"
    
    def test_document_update_with_cache_invalidation(self, integrated_rag_pipeline):
        """Test document updates trigger proper cache invalidation."""
        # This test should FAIL initially - missing document update integration
        pipeline = integrated_rag_pipeline
        
        # Initial document ingestion
        initial_docs = [
            {"content": "Machine learning is AI subset.", "metadata": {"source": "ml.md", "version": 1}}
        ]
        pipeline.ingest_documents(initial_docs, "test_collection")
        
        # Query to populate caches
        initial_query = pipeline.query("machine learning", ["test_collection"])
        assert initial_query.from_embedding_cache == False, "First query should miss cache"
        
        # Repeat query to hit cache
        cached_query = pipeline.query("machine learning", ["test_collection"])
        assert cached_query.from_query_cache == True, "Should hit query cache"
        
        # Update document (should trigger cache invalidation)
        updated_docs = [
            {"content": "Machine learning is artificial intelligence subset with advanced algorithms.", "metadata": {"source": "ml.md", "version": 2}}
        ]
        update_result = pipeline.update_documents(updated_docs, "test_collection")
        assert update_result.success == True, "Document update should succeed"
        assert update_result.caches_invalidated == True, "Should invalidate relevant caches"
        assert "test_collection" in update_result.invalidated_collections, "Should invalidate collection cache"
        
        # Query after update (should miss cache due to invalidation)
        post_update_query = pipeline.query("machine learning", ["test_collection"])
        assert post_update_query.from_query_cache == False, "Should miss cache after invalidation"
        assert post_update_query.content_updated == True, "Should reflect updated content"
        
        # Verify invalidation metrics
        invalidation_stats = pipeline.get_invalidation_statistics()
        assert invalidation_stats['total_invalidations'] > 0, "Should track invalidations"
        assert invalidation_stats['document_update_invalidations'] > 0, "Should track document update invalidations"
        assert invalidation_stats['average_invalidation_latency'] < 0.1, "Invalidation should be fast"
    
    def test_error_handling_with_cache_fallbacks(self, integrated_rag_pipeline):
        """Test error handling and fallback mechanisms with caching."""
        # This test should FAIL initially - missing error handling integration
        pipeline = integrated_rag_pipeline
        
        # Simulate cache failure scenarios
        with patch.object(pipeline._embedding_cache, 'get', side_effect=Exception("Cache failure")):
            # Query should still work with cache failure
            result = pipeline.query("test query", ["test_collection"])
            assert result.success == True, "Query should succeed despite cache failure"
            assert result.cache_fallback_used == True, "Should use cache fallback"
            assert result.embedding_generated_fresh == True, "Should generate fresh embeddings"
        
        # Simulate vector store failure with cache backup
        with patch.object(pipeline._vector_store, 'query', side_effect=Exception("Vector store failure")):
            # Should try cache-based fallback
            result = pipeline.query("cached query", ["test_collection"])
            # May succeed or fail depending on cache availability
            if result.success:
                assert result.cache_only_result == True, "Should use cache-only result"
            else:
                assert "vector store" in result.error_message.lower(), "Should indicate vector store error"
        
        # Test recovery after errors
        recovery_result = pipeline.query("recovery test", ["test_collection"])
        assert recovery_result.success == True, "Should recover after errors"
        assert recovery_result.error_recovery_used == False, "Should not need error recovery for normal operation"


class TestCacheIntegrationValidation:
    """Test integration of all caching components."""
    
    def test_embedding_cache_integration(self, integrated_rag_pipeline):
        """Test embedding cache properly integrated with LocalEmbeddingService."""
        # This test should FAIL initially - incomplete embedding cache integration
        pipeline = integrated_rag_pipeline
        embedding_service = pipeline.get_embedding_service()
        
        # Test cache integration
        assert hasattr(embedding_service, '_cache_manager'), "Should have cache manager integrated"
        assert embedding_service._cache_manager is not None, "Cache manager should be initialized"
        
        # Test embedding caching behavior
        text = "test embedding integration"
        embedding1 = embedding_service.embed_text(text)
        embedding2 = embedding_service.embed_text(text)
        
        # Should be identical due to caching
        assert embedding1 == embedding2, "Cached embeddings should be identical"
        
        # Test cache statistics
        cache_stats = embedding_service.get_cache_stats()
        assert cache_stats['hit_rate'] > 0, "Should have cache hits"
        assert cache_stats['total_requests'] >= 2, "Should track requests"
        
        # Test model fingerprint integration
        model_fingerprint = embedding_service.get_model_fingerprint()
        assert model_fingerprint is not None, "Should generate model fingerprint"
        assert len(model_fingerprint) > 0, "Model fingerprint should not be empty"
    
    def test_query_cache_integration_with_rag_pipeline(self, integrated_rag_pipeline):
        """Test query cache integration with complete RAG pipeline."""
        # This test should FAIL initially - incomplete query cache integration
        pipeline = integrated_rag_pipeline
        
        # Ingest test documents
        test_docs = [
            {"content": "Integration testing validates system components.", "metadata": {"source": "testing.md"}}
        ]
        pipeline.ingest_documents(test_docs, "integration_test")
        
        # First query (should populate query cache)
        query1 = pipeline.query("integration testing", ["integration_test"])
        assert query1.success == True, "Query should succeed"
        assert query1.from_query_cache == False, "First query should not hit cache"
        
        # Second identical query (should hit cache)
        query2 = pipeline.query("integration testing", ["integration_test"])
        assert query2.success == True, "Cached query should succeed"
        assert query2.from_query_cache == True, "Should hit query cache"
        assert query2.response_time < query1.response_time, "Cached query should be faster"
        
        # Test query cache with different parameters
        query3 = pipeline.query("integration testing", ["integration_test"], top_k=5)
        assert query3.from_query_cache == False, "Different parameters should miss cache"
        
        # Test query cache statistics
        query_cache_stats = pipeline.get_query_cache_statistics()
        assert query_cache_stats['hit_rate'] > 0, "Should have query cache hits"
        assert query_cache_stats['unique_queries'] >= 2, "Should track unique queries"
        assert query_cache_stats['total_queries'] >= 3, "Should track total queries"
    
    def test_batch_processing_integration(self, integrated_rag_pipeline):
        """Test batch processing optimization integration."""
        # This test should FAIL initially - incomplete batch processing integration
        pipeline = integrated_rag_pipeline
        
        # Large batch of documents for batch processing
        batch_docs = [
            {"content": f"Document {i} contains information about topic {i}.", "metadata": {"source": f"doc{i}.md"}}
            for i in range(50)
        ]
        
        # Test batch ingestion with optimization
        batch_result = pipeline.ingest_documents_batch(batch_docs, "batch_test")
        assert batch_result.success == True, "Batch ingestion should succeed"
        assert batch_result.batch_optimization_used == True, "Should use batch optimization"
        assert batch_result.parallel_processing_used == True, "Should use parallel processing"
        assert batch_result.documents_per_second > 5, "Should achieve good throughput"
        
        # Test batch query processing
        queries = [f"topic {i}" for i in range(10)]
        batch_query_result = pipeline.query_batch(queries, ["batch_test"])
        assert batch_query_result.success == True, "Batch queries should succeed"
        assert batch_query_result.batch_optimization_used == True, "Should optimize batch queries"
        assert len(batch_query_result.results) == 10, "Should return all query results"
        
        # Test batch processing statistics
        batch_stats = pipeline.get_batch_processing_statistics()
        assert batch_stats['parallel_efficiency'] > 0.7, "Should show good parallel efficiency"
        assert batch_stats['memory_efficiency'] > 0.8, "Should be memory efficient"
        assert batch_stats['batch_speedup'] > 1.5, "Should show batch processing speedup"
    
    def test_lazy_loading_integration(self, integrated_rag_pipeline):
        """Test lazy loading integration across all components."""
        # This test should FAIL initially - incomplete lazy loading integration
        pipeline = integrated_rag_pipeline
        
        # Test lazy loading initialization
        initialization_stats = pipeline.get_initialization_statistics()
        assert initialization_stats['lazy_loading_enabled'] == True, "Should use lazy loading"
        assert initialization_stats['components_loaded_on_demand'] > 0, "Should load components on demand"
        
        # Test component loading times
        embedding_load_time = pipeline.measure_component_load_time('embedding_service')
        vector_store_load_time = pipeline.measure_component_load_time('vector_store')
        query_manager_load_time = pipeline.measure_component_load_time('query_manager')
        
        assert embedding_load_time > 0, "Should measure embedding service load time"
        assert vector_store_load_time > 0, "Should measure vector store load time"
        assert query_manager_load_time > 0, "Should measure query manager load time"
        
        # Test lazy loading performance benefits
        lazy_stats = pipeline.get_lazy_loading_statistics()
        assert lazy_stats['initialization_time_saved'] > 0, "Should save initialization time"
        assert lazy_stats['memory_usage_reduced'] > 0, "Should reduce memory usage"
        assert lazy_stats['components_loaded_lazily'] > 0, "Should have lazy-loaded components"


class TestPerformanceIntegrationTesting:
    """Test performance with all optimizations integrated."""
    
    def test_end_to_end_performance_optimization(self, integrated_rag_pipeline):
        """Test complete pipeline performance with all optimizations."""
        # This test should FAIL initially - missing performance integration
        pipeline = integrated_rag_pipeline
        
        # Baseline measurement (first run, all caches cold)
        baseline_docs = [
            {"content": f"Performance document {i} contains benchmark data.", "metadata": {"source": f"perf{i}.md"}}
            for i in range(20)
        ]
        pipeline.ingest_documents(baseline_docs, "performance_test")
        
        # Measure baseline performance
        baseline_start = time.time()
        baseline_results = []
        for i in range(10):
            result = pipeline.query(f"benchmark data {i}", ["performance_test"])
            baseline_results.append(result)
        baseline_time = time.time() - baseline_start
        
        # Warm up caches
        for i in range(5):
            pipeline.query(f"benchmark data {i}", ["performance_test"])
        
        # Measure optimized performance (with warm caches)
        optimized_start = time.time()
        optimized_results = []
        for i in range(10):
            result = pipeline.query(f"benchmark data {i}", ["performance_test"])
            optimized_results.append(result)
        optimized_time = time.time() - optimized_start
        
        # Performance validation
        performance_improvement = baseline_time / optimized_time
        assert performance_improvement > 1.5, f"Should show >50% performance improvement: {performance_improvement}"
        
        # Cache hit rate validation
        cache_hit_rate = sum(1 for r in optimized_results if r.from_query_cache) / len(optimized_results)
        assert cache_hit_rate > 0.5, f"Should achieve >50% cache hit rate: {cache_hit_rate}"
        
        # Memory usage validation
        memory_stats = pipeline.get_memory_usage_statistics()
        assert memory_stats['peak_memory_mb'] < 500, "Should use reasonable memory"
        assert memory_stats['memory_leaks_detected'] == False, "Should not have memory leaks"
    
    def test_concurrent_user_performance_integration(self, integrated_rag_pipeline):
        """Test performance under concurrent user load."""
        # This test should FAIL initially - missing concurrent performance handling
        pipeline = integrated_rag_pipeline
        
        # Setup test data
        concurrent_docs = [
            {"content": f"Concurrent document {i} for load testing.", "metadata": {"source": f"concurrent{i}.md"}}
            for i in range(30)
        ]
        pipeline.ingest_documents(concurrent_docs, "concurrent_test")
        
        def simulate_user_queries(user_id: int, query_count: int):
            """Simulate queries from one user."""
            user_results = []
            for i in range(query_count):
                query = f"user {user_id} query {i}"
                result = pipeline.query(query, ["concurrent_test"])
                user_results.append(result)
            return user_results
        
        # Test concurrent users
        user_count = 5
        queries_per_user = 10
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=user_count) as executor:
            futures = [
                executor.submit(simulate_user_queries, user_id, queries_per_user)
                for user_id in range(user_count)
            ]
            all_results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        total_queries = user_count * queries_per_user
        
        # Performance validation
        average_query_time = total_time / total_queries
        assert average_query_time < 0.5, f"Average query time should be <0.5s: {average_query_time}"
        
        # Success rate validation
        success_count = sum(len([r for r in user_results if r.success]) for user_results in all_results)
        success_rate = success_count / total_queries
        assert success_rate > 0.95, f"Success rate should be >95%: {success_rate}"
        
        # Concurrent cache coordination validation
        concurrent_stats = pipeline.get_concurrent_performance_statistics()
        assert concurrent_stats['cache_contention_detected'] == False, "Should not have cache contention"
        assert concurrent_stats['thread_safety_maintained'] == True, "Should maintain thread safety"
        assert concurrent_stats['concurrent_cache_hit_rate'] > 0.3, "Should achieve reasonable cache hits under load"
    
    def test_memory_usage_stability_integration(self, integrated_rag_pipeline):
        """Test memory usage stability during extended operations."""
        # This test should FAIL initially - missing memory stability integration
        pipeline = integrated_rag_pipeline
        
        # Setup for extended operation test
        extended_docs = [
            {"content": f"Extended operation document {i}.", "metadata": {"source": f"extended{i}.md"}}
            for i in range(100)
        ]
        pipeline.ingest_documents(extended_docs, "extended_test")
        
        # Track memory usage over extended operations
        memory_samples = []
        operation_count = 200
        
        for i in range(operation_count):
            # Perform various operations
            if i % 4 == 0:
                # Query operation
                result = pipeline.query(f"extended document {i % 100}", ["extended_test"])
                assert result.success == True, f"Query {i} should succeed"
            elif i % 4 == 1:
                # Batch query operation
                queries = [f"document {j}" for j in range(i % 10 + 1)]
                batch_result = pipeline.query_batch(queries, ["extended_test"])
                assert batch_result.success == True, f"Batch query {i} should succeed"
            elif i % 4 == 2:
                # Cache statistics operation
                stats = pipeline.get_cache_statistics()
                assert 'embedding_cache' in stats, "Should provide cache statistics"
            else:
                # Memory cleanup operation
                pipeline.cleanup_temporary_resources()
            
            # Sample memory usage
            if i % 20 == 0:
                memory_usage = pipeline.get_current_memory_usage()
                memory_samples.append(memory_usage)
        
        # Memory stability validation
        max_memory = max(memory_samples)
        min_memory = min(memory_samples)
        memory_variance = max_memory - min_memory
        
        assert max_memory < 1000, f"Peak memory should be reasonable: {max_memory}MB"
        assert memory_variance < 200, f"Memory variance should be low: {memory_variance}MB"
        
        # Memory leak detection
        final_memory = pipeline.get_current_memory_usage()
        initial_memory = memory_samples[0] if memory_samples else final_memory
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 50, f"Memory growth should be minimal: {memory_growth}MB"
        
        # Cache size stability
        cache_stats = pipeline.get_cache_statistics()
        assert cache_stats['cache_size_stable'] == True, "Cache sizes should be stable"
        assert cache_stats['cache_eviction_working'] == True, "Cache eviction should be working"


class TestSystemStabilityAndCorrectness:
    """Test system stability and correctness with all components integrated."""
    
    def test_cache_consistency_validation(self, integrated_rag_pipeline):
        """Test cache consistency across all layers."""
        # This test should FAIL initially - missing cache consistency validation
        pipeline = integrated_rag_pipeline
        
        # Setup test documents
        consistency_docs = [
            {"content": "Cache consistency document.", "metadata": {"source": "consistency.md", "id": "doc1"}}
        ]
        pipeline.ingest_documents(consistency_docs, "consistency_test")
        
        # Query to populate all caches
        initial_query = pipeline.query("cache consistency", ["consistency_test"])
        assert initial_query.success == True, "Initial query should succeed"
        
        # Verify all cache layers are populated
        cache_consistency = pipeline.validate_cache_consistency()
        assert cache_consistency['embedding_cache_consistent'] == True, "Embedding cache should be consistent"
        assert cache_consistency['query_cache_consistent'] == True, "Query cache should be consistent"
        assert cache_consistency['vector_store_cache_consistent'] == True, "Vector store cache should be consistent"
        
        # Update document (should invalidate consistently across all caches)
        updated_docs = [
            {"content": "Updated cache consistency document.", "metadata": {"source": "consistency.md", "id": "doc1"}}
        ]
        update_result = pipeline.update_documents(updated_docs, "consistency_test")
        assert update_result.success == True, "Document update should succeed"
        
        # Verify consistent invalidation
        post_update_consistency = pipeline.validate_cache_consistency()
        assert post_update_consistency['all_caches_invalidated_consistently'] == True, "All caches should be invalidated consistently"
        assert post_update_consistency['no_stale_data_detected'] == True, "No stale data should be detected"
        
        # Query after update should reflect changes in all caches
        post_update_query = pipeline.query("cache consistency", ["consistency_test"])
        assert post_update_query.success == True, "Post-update query should succeed"
        assert post_update_query.content_reflects_update == True, "Should reflect document update"
    
    def test_resource_cleanup_and_leak_prevention(self, integrated_rag_pipeline):
        """Test proper resource cleanup and memory leak prevention."""
        # This test should FAIL initially - missing resource management
        pipeline = integrated_rag_pipeline
        
        # Test resource tracking
        initial_resources = pipeline.get_resource_usage()
        assert 'file_handles' in initial_resources, "Should track file handles"
        assert 'thread_count' in initial_resources, "Should track thread count"
        assert 'memory_usage' in initial_resources, "Should track memory usage"
        
        # Perform operations that create resources
        for i in range(20):
            # Create temporary collections
            temp_docs = [{"content": f"Temporary document {i}.", "metadata": {"source": f"temp{i}.md"}}]
            pipeline.ingest_documents(temp_docs, f"temp_collection_{i}")
            
            # Query temporary collections
            result = pipeline.query(f"temporary {i}", [f"temp_collection_{i}"])
            assert result.success == True, f"Temporary query {i} should succeed"
        
        # Check resource usage during operations
        peak_resources = pipeline.get_resource_usage()
        assert peak_resources['file_handles'] < 100, "Should limit file handle usage"
        assert peak_resources['thread_count'] < 20, "Should limit thread count"
        
        # Cleanup resources
        cleanup_result = pipeline.cleanup_all_resources()
        assert cleanup_result['success'] == True, "Resource cleanup should succeed"
        assert cleanup_result['resources_freed'] > 0, "Should free resources"
        
        # Verify resource cleanup
        final_resources = pipeline.get_resource_usage()
        assert final_resources['file_handles'] <= initial_resources['file_handles'] + 5, "Should cleanup file handles"
        assert final_resources['thread_count'] <= initial_resources['thread_count'] + 2, "Should cleanup threads"
        
        # Memory leak detection
        memory_growth = final_resources['memory_usage'] - initial_resources['memory_usage']
        assert memory_growth < 100, f"Should not have significant memory growth: {memory_growth}MB"
    
    def test_thread_safety_validation(self, integrated_rag_pipeline):
        """Test thread safety for concurrent operations."""
        # This test should FAIL initially - missing thread safety validation
        pipeline = integrated_rag_pipeline
        
        # Setup test data
        thread_safety_docs = [
            {"content": f"Thread safety document {i}.", "metadata": {"source": f"thread{i}.md"}}
            for i in range(10)
        ]
        pipeline.ingest_documents(thread_safety_docs, "thread_safety_test")
        
        # Concurrent operations
        results = []
        errors = []
        
        def concurrent_operation(operation_id: int):
            """Perform concurrent operations."""
            try:
                for i in range(10):
                    # Mix different operations
                    if i % 3 == 0:
                        # Query operation
                        result = pipeline.query(f"thread safety {operation_id}", ["thread_safety_test"])
                        results.append(result)
                    elif i % 3 == 1:
                        # Cache statistics operation
                        stats = pipeline.get_cache_statistics()
                        assert 'embedding_cache' in stats, "Should provide stats thread-safely"
                    else:
                        # Resource usage operation
                        resources = pipeline.get_resource_usage()
                        assert 'memory_usage' in resources, "Should provide resource info thread-safely"
                
            except Exception as e:
                errors.append(f"Operation {operation_id}: {str(e)}")
        
        # Run concurrent operations
        thread_count = 8
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(concurrent_operation, i)
                for i in range(thread_count)
            ]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # Validate thread safety
        assert len(errors) == 0, f"Should not have thread safety errors: {errors}"
        assert len(results) > 0, "Should have successful concurrent results"
        
        # Validate cache integrity after concurrent access
        cache_integrity = pipeline.validate_cache_integrity()
        assert cache_integrity['cache_corruption_detected'] == False, "Should not have cache corruption"
        assert cache_integrity['concurrent_access_successful'] == True, "Should handle concurrent access"
        assert cache_integrity['thread_safety_maintained'] == True, "Should maintain thread safety"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 