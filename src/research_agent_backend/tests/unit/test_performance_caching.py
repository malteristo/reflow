"""
Test suite for performance optimization and caching layer.

This module contains comprehensive tests for:
- Embedding cache with LRU eviction and TTL
- Query result cache with content-based invalidation  
- Batch processing optimizations
- Cache integration with embedding services
- Performance benchmarking functionality

Following TDD RED PHASE: These tests should initially FAIL until implementation is complete.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.research_agent_backend.core.enhanced_caching import (
    MultiLevelCacheManager,
    ModelAwareCacheManager,
    IntelligentCacheWarmer,
    CachePerformanceOptimizer
)
from src.research_agent_backend.core.query_manager.cache import QueryCache
from src.research_agent_backend.core.local_embedding_service import LocalEmbeddingService


class TestEmbeddingCacheIntegration:
    """Test embedding cache integration with actual embedding services."""
    
    def test_embedding_service_with_cache_integration(self):
        """Test that embedding service properly integrates with caching layer."""
        # This test should FAIL initially - no cache integration in embedding service
        service = LocalEmbeddingService()
        
        # Check if service has cache manager integrated
        assert hasattr(service, '_cache_manager'), "Embedding service should have cache manager"
        assert service._cache_manager is not None, "Cache manager should be initialized"
        
        # Test cache usage in embedding generation
        text = "test embedding text"
        embedding1 = service.embed_text(text)
        embedding2 = service.embed_text(text)  # Should hit cache
        
        # Should be same reference if cached properly
        assert embedding1 == embedding2, "Cached embeddings should be identical"
        
        # Verify cache statistics
        cache_stats = service.get_cache_stats()
        assert cache_stats['hit_rate'] > 0, "Should have cache hits for repeated text"
        assert cache_stats['total_requests'] >= 2, "Should track request count"
    
    def test_embedding_cache_memory_efficiency(self):
        """Test that embedding cache doesn't consume excessive memory."""
        # This test should FAIL initially - no memory management in cache
        service = LocalEmbeddingService()
        
        # Generate many embeddings to test memory limits
        large_texts = [f"test text {i}" for i in range(1000)]
        embeddings = []
        
        for text in large_texts:
            embeddings.append(service.embed_text(text))
        
        # Check cache size doesn't exceed limits
        cache_stats = service.get_cache_stats()
        assert cache_stats['memory_usage_mb'] < 100, "Cache should limit memory usage"
        assert cache_stats['entries_count'] <= service._cache_manager.max_size
    
    def test_embedding_cache_ttl_expiration(self):
        """Test that cached embeddings expire after TTL."""
        # This test should FAIL initially - no TTL implementation
        service = LocalEmbeddingService(cache_ttl=1)  # 1 second TTL
        
        text = "test expiring embedding"
        embedding1 = service.embed_text(text)
        
        # Wait for TTL expiration
        time.sleep(1.1)
        
        embedding2 = service.embed_text(text)
        cache_stats = service.get_cache_stats()
        
        # Should have cache miss due to expiration
        assert cache_stats['miss_count'] > 0, "Should have cache miss after TTL expiration"
    
    def test_batch_embedding_cache_optimization(self):
        """Test that batch processing leverages caching efficiently."""
        # This test should FAIL initially - no batch cache optimization
        service = LocalEmbeddingService()
        
        texts = ["text 1", "text 2", "text 1", "text 3", "text 2"]  # Duplicates
        
        # Process batch
        batch_embeddings = service.embed_batch(texts)
        
        # Verify cache optimization stats
        cache_stats = service.get_cache_stats()
        assert cache_stats['batch_cache_hits'] >= 2, "Should cache duplicate texts in batch"
        assert cache_stats['batch_efficiency'] > 0.5, "Should improve batch efficiency"


class TestQueryResultCache:
    """Test query result caching with TTL and invalidation."""
    
    def test_query_cache_stores_complete_results(self):
        """Test that query cache stores complete query results with metadata."""
        # This test should FAIL initially - incomplete result caching
        cache = QueryCache(max_size=100, ttl_seconds=300)
        
        # Mock complete query result
        query_result = Mock()
        query_result.results = [{"content": "test doc 1"}]
        query_result.total_results = 1
        query_result.similarity_scores = [0.95]
        query_result.metadata = [{"source": "test.md"}]
        query_result.performance_metrics = Mock()
        query_result.performance_metrics.total_execution_time = 0.5
        
        key = "test_query_key"
        cache.put(key, query_result)
        
        # Retrieve and verify completeness
        cached_result = cache.get(key)
        assert cached_result is not None, "Should retrieve cached result"
        assert cached_result.results == query_result.results
        assert cached_result.similarity_scores == query_result.similarity_scores
        assert cached_result.metadata == query_result.metadata
        assert hasattr(cached_result, 'from_cache'), "Should mark result as from cache"
        assert cached_result.from_cache == True, "Should indicate cache hit"
    
    def test_query_cache_content_based_invalidation(self):
        """Test content-based cache invalidation when documents change."""
        # This test should FAIL initially - no content invalidation
        cache = QueryCache()
        
        # Cache query result
        key = cache._generate_key([0.1, 0.2], ["collection1"], {"filter": "value"})
        mock_result = Mock()
        cache.put(key, mock_result, ["collection1"])
        
        # Simulate document update in collection
        cache.invalidate_by_content_change("collection1", ["doc1", "doc2"])
        
        # Should invalidate related cache entries
        cached_result = cache.get(key)
        assert cached_result is None, "Should invalidate cache on content change"
    
    def test_query_cache_collection_filtering(self):
        """Test cache invalidation by collection."""
        # This test should FAIL initially - collection-based invalidation not working
        cache = QueryCache()
        
        # Cache multiple queries for different collections
        key1 = cache._generate_key([0.1, 0.2], ["collection1"])
        key2 = cache._generate_key([0.3, 0.4], ["collection2"])
        key3 = cache._generate_key([0.5, 0.6], ["collection1", "collection2"])
        
        cache.put(key1, Mock(), ["collection1"])
        cache.put(key2, Mock(), ["collection2"]) 
        cache.put(key3, Mock(), ["collection1", "collection2"])
        
        # Invalidate by collection
        cache.invalidate_by_collections(["collection1"])
        
        # Verify selective invalidation
        assert cache.get(key1) is None, "Should invalidate collection1 queries"
        assert cache.get(key2) is not None, "Should keep collection2 queries"
        assert cache.get(key3) is None, "Should invalidate multi-collection queries"
    
    def test_query_cache_performance_tracking(self):
        """Test query cache performance metrics tracking."""
        # This test should FAIL initially - no performance tracking
        cache = QueryCache()
        
        # Perform cache operations
        key = "test_key"
        mock_result = Mock()
        
        cache.put(key, mock_result)
        cache.get(key)  # Hit
        cache.get("nonexistent")  # Miss
        
        # Check performance metrics
        metrics = cache.get_performance_metrics()
        assert metrics['total_requests'] == 2, "Should track total requests"
        assert metrics['hit_rate'] == 0.5, "Should calculate hit rate correctly"
        assert metrics['miss_rate'] == 0.5, "Should calculate miss rate correctly"
        assert metrics['average_response_time'] > 0, "Should track response time"


class TestBatchProcessingOptimization:
    """Test batch processing optimizations."""
    
    def test_parallel_batch_processing(self):
        """Test parallel processing improves batch performance."""
        # This test should FAIL initially - no parallel processing
        from src.research_agent_backend.core.enhanced_integration import OptimizedPipelineCoordinator
        
        coordinator = OptimizedPipelineCoordinator({
            "parallel_processing": True,
            "batch_optimization": True,
            "max_workers": 4
        })
        
        # Large batch for parallel processing
        documents = [{"content": f"Document {i}", "metadata": {}} for i in range(100)]
        
        # Process with parallel optimization
        result = coordinator.process_documents_optimized(documents, "test_collection")
        
        assert result.success == True, "Should successfully process batch"
        assert result.optimization_metrics['parallel_processing_used'] == True
        assert result.optimization_metrics['documents_per_second'] > 10, "Should achieve good throughput"
        assert result.optimization_metrics['parallel_efficiency'] > 0.7, "Should show parallel benefit"
    
    def test_adaptive_batch_sizing(self):
        """Test adaptive batch size optimization."""
        # This test should FAIL initially - no adaptive batching
        from src.research_agent_backend.core.enhanced_integration import OptimizedPipelineCoordinator
        
        coordinator = OptimizedPipelineCoordinator({
            "adaptive_batching": True,
            "auto_tune_batch_size": True
        })
        
        # Test different document counts
        small_batch = [{"content": f"Doc {i}"} for i in range(10)]
        large_batch = [{"content": f"Doc {i}"} for i in range(1000)]
        
        small_result = coordinator.process_documents_optimized(small_batch, "test")
        large_result = coordinator.process_documents_optimized(large_batch, "test")
        
        # Should use different optimal batch sizes
        small_batch_size = small_result.optimization_metrics['optimal_batch_size']
        large_batch_size = large_result.optimization_metrics['optimal_batch_size']
        
        assert small_batch_size < large_batch_size, "Should use larger batches for more documents"
        assert small_batch_size <= 20, "Should use small batches for few documents"
        assert large_batch_size >= 50, "Should use larger batches for many documents"
    
    def test_memory_efficient_batch_processing(self):
        """Test memory-efficient processing of large batches."""
        # This test should FAIL initially - no memory management
        from src.research_agent_backend.core.enhanced_integration import OptimizedPipelineCoordinator
        
        coordinator = OptimizedPipelineCoordinator({
            "memory_efficient": True,
            "max_memory_mb": 256
        })
        
        # Very large batch that could exceed memory
        large_batch = [{"content": "Large document content " * 1000} for i in range(500)]
        
        result = coordinator.process_documents_optimized(large_batch, "test")
        
        assert result.success == True, "Should handle large batch without memory issues"
        assert result.optimization_metrics['memory_usage_mb'] < 256, "Should respect memory limits"
        assert result.optimization_metrics['memory_efficient_processing'] == True


class TestCachePerformanceBenchmarking:
    """Test performance benchmarking functionality."""
    
    def test_embedding_cache_performance_benchmark(self):
        """Test embedding cache performance benchmarking."""
        # This test should FAIL initially - no benchmarking system
        from src.research_agent_backend.core.performance_benchmark import EmbeddingCacheBenchmark
        
        benchmark = EmbeddingCacheBenchmark()
        
        # Run benchmark with different cache configurations
        configs = [
            {"cache_size": 100, "ttl": 300},
            {"cache_size": 1000, "ttl": 300},
            {"cache_size": 100, "ttl": 60}
        ]
        
        results = benchmark.run_benchmark(configs, test_duration_seconds=10)
        
        assert len(results) == 3, "Should benchmark all configurations"
        
        for result in results:
            assert 'cache_hit_rate' in result, "Should measure hit rate"
            assert 'average_response_time' in result, "Should measure response time"
            assert 'memory_usage' in result, "Should measure memory usage"
            assert 'throughput_ops_per_second' in result, "Should measure throughput"
    
    def test_query_cache_performance_comparison(self):
        """Test query cache performance comparison."""
        # This test should FAIL initially - no performance comparison
        from src.research_agent_backend.core.performance_benchmark import QueryCacheBenchmark
        
        benchmark = QueryCacheBenchmark()
        
        # Compare cached vs non-cached performance
        comparison = benchmark.compare_cached_vs_uncached(
            query_count=100,
            collection_size=1000
        )
        
        assert comparison['cached_performance']['average_response_time'] < \
               comparison['uncached_performance']['average_response_time'], \
               "Cached queries should be faster"
        
        assert comparison['cached_performance']['throughput'] > \
               comparison['uncached_performance']['throughput'], \
               "Cached queries should have higher throughput"
        
        assert comparison['performance_improvement'] > 2.0, \
               "Should show significant performance improvement"
    
    def test_cache_memory_usage_benchmark(self):
        """Test cache memory usage benchmarking."""
        # This test should FAIL initially - no memory benchmarking
        from src.research_agent_backend.core.performance_benchmark import CacheMemoryBenchmark
        
        benchmark = CacheMemoryBenchmark()
        
        # Test memory usage under load
        memory_profile = benchmark.profile_memory_usage(
            cache_operations=10000,
            embedding_size=384,
            result_size_kb=10
        )
        
        assert 'peak_memory_mb' in memory_profile, "Should track peak memory"
        assert 'average_memory_mb' in memory_profile, "Should track average memory"
        assert 'memory_efficiency' in memory_profile, "Should calculate efficiency"
        assert memory_profile['memory_leaks_detected'] == False, "Should detect memory leaks"


class TestCacheIntegrationWithRAGPipeline:
    """Test cache integration with complete RAG pipeline."""
    
    def test_end_to_end_cached_query_pipeline(self):
        """Test complete query pipeline with caching at all levels."""
        # This test should FAIL initially - incomplete pipeline integration
        from src.research_agent_backend.core.rag_cache_integration import RAGCacheIntegration
        
        # Initialize RAG cache integration with performance monitoring
        rag_integration = RAGCacheIntegration({
            "invalidation_policy": "content_based",
            "enable_performance_monitoring": True,
            "batch_invalidation_size": 100
        })
        
        # Register cache components for end-to-end testing
        from src.research_agent_backend.core.enhanced_caching import ModelAwareCacheManager
        from src.research_agent_backend.core.query_manager.cache import QueryCache
        
        embedding_cache = ModelAwareCacheManager()
        query_cache = QueryCache(max_size=1000, ttl_seconds=300)
        
        rag_integration.register_embedding_cache(embedding_cache)
        rag_integration.register_query_cache(query_cache)
        
        # Test end-to-end performance metrics
        performance_metrics = rag_integration.get_end_to_end_performance()
        
        assert 'cache_hit_rate' in performance_metrics, "Should measure overall cache hit rate"
        assert 'query_response_time' in performance_metrics, "Should measure query response time"
        assert 'memory_usage_mb' in performance_metrics, "Should measure memory usage"
        assert 'throughput_qps' in performance_metrics, "Should measure throughput"
        assert 'invalidation_latency' in performance_metrics, "Should measure invalidation performance"
        
        # Verify cache layer integration
        assert 'cache_layers' in performance_metrics, "Should report on cache layers"
        cache_layers = performance_metrics['cache_layers']
        assert 'embedding_cache' in cache_layers, "Should include embedding cache metrics"
        assert 'query_cache' in cache_layers, "Should include query cache metrics"
        
        # Test cache performance expectations
        assert performance_metrics['cache_hit_rate'] >= 0.0, "Should have valid hit rate"
        assert performance_metrics['query_response_time'] > 0, "Should have measurable response time"
        assert performance_metrics['memory_usage_mb'] > 0, "Should have measurable memory usage"

    def test_cache_invalidation_on_document_update(self):
        """Test cache invalidation when documents are updated."""
        # This test should FAIL initially - no document update integration
        from src.research_agent_backend.core.rag_cache_integration import RAGCacheIntegration
        from src.research_agent_backend.core.vector_store import ChromaDBManager
        
        # Initialize RAG cache integration
        rag_integration = RAGCacheIntegration({
            "invalidation_policy": "content_based",
            "enable_performance_monitoring": True
        })
        
        # Initialize vector store manager
        vector_store = ChromaDBManager(in_memory=True)
        
        # Register caches with integration (mock caches for testing)
        from src.research_agent_backend.core.enhanced_caching import ModelAwareCacheManager
        from src.research_agent_backend.core.query_manager.cache import QueryCache
        
        embedding_cache = ModelAwareCacheManager()
        query_cache = QueryCache(max_size=100, ttl_seconds=300)
        
        rag_integration.register_embedding_cache(embedding_cache)
        rag_integration.register_query_cache(query_cache)
        rag_integration.register_vector_store_cache(vector_store)
        
        # Test cache invalidation on document update
        invalidation_result = rag_integration.invalidate_on_document_update(
            collection_names=["test_collection"],
            document_ids=["doc1", "doc2"],
            content_hash="hash123"
        )
        
        assert invalidation_result['success'] == True, "Should successfully invalidate caches"
        assert "test_collection" in invalidation_result['collections_invalidated'], "Should invalidate specified collection"
        assert invalidation_result['invalidation_latency'] > 0, "Should track invalidation timing"
        assert 'content_based' in invalidation_result['policy'], "Should use correct invalidation policy"
        
        # Verify cache invalidation was coordinated across cache layers
        assert 'cache_results' in invalidation_result, "Should have cache invalidation results"


class TestComprehensivePerformanceBenchmarking:
    """Test comprehensive performance benchmarking across all optimization components."""
    
    def test_baseline_vs_enhanced_pipeline_comparison(self):
        """Test performance comparison between baseline and enhanced RAG pipeline."""
        # This test should FAIL initially - no comprehensive benchmarking system
        from src.research_agent_backend.core.comprehensive_benchmark import RAGPipelineBenchmark
        
        benchmark = RAGPipelineBenchmark()
        
        # Run baseline benchmark (no optimizations)
        baseline_results = benchmark.run_baseline_benchmark(
            query_count=100,
            document_count=1000,
            test_duration_seconds=30
        )
        
        # Run enhanced benchmark (all optimizations enabled)
        enhanced_results = benchmark.run_enhanced_benchmark(
            query_count=100,
            document_count=1000,
            test_duration_seconds=30,
            enable_embedding_cache=True,
            enable_query_cache=True,
            enable_batch_optimization=True,
            enable_lazy_loading=True,
            enable_cache_invalidation=True
        )
        
        # Verify performance improvements
        assert enhanced_results['average_response_time'] < baseline_results['average_response_time'], \
               "Enhanced pipeline should be faster"
        assert enhanced_results['throughput_qps'] > baseline_results['throughput_qps'], \
               "Enhanced pipeline should have higher throughput"
        assert enhanced_results['memory_efficiency'] > baseline_results['memory_efficiency'], \
               "Enhanced pipeline should be more memory efficient"
        
        # Verify specific optimization benefits
        performance_improvement = benchmark.calculate_performance_improvement(baseline_results, enhanced_results)
        assert performance_improvement['response_time_improvement'] > 1.5, \
               "Should show at least 50% response time improvement"
        assert performance_improvement['throughput_improvement'] > 2.0, \
               "Should show at least 100% throughput improvement"
        assert performance_improvement['cache_hit_rate'] > 0.7, \
               "Should achieve high cache hit rate"
    
    def test_component_level_performance_metrics(self):
        """Test individual component performance metrics."""
        # This test should FAIL initially - no component-level benchmarking
        from src.research_agent_backend.core.comprehensive_benchmark import ComponentBenchmark
        
        component_benchmark = ComponentBenchmark()
        
        # Benchmark embedding cache performance
        embedding_metrics = component_benchmark.benchmark_embedding_cache(
            operation_count=1000,
            cache_size=500,
            hit_rate_target=0.8
        )
        
        assert embedding_metrics['cache_hit_rate'] >= 0.7, "Should achieve high cache hit rate"
        assert embedding_metrics['cache_response_time'] < embedding_metrics['no_cache_response_time'], \
               "Cached embeddings should be faster"
        assert embedding_metrics['memory_usage_mb'] < 200, "Should use reasonable memory"
        
        # Benchmark query cache performance  
        query_metrics = component_benchmark.benchmark_query_cache(
            query_count=500,
            unique_queries=100,
            ttl_seconds=300
        )
        
        assert query_metrics['cache_hit_rate'] >= 0.6, "Should achieve good query cache hit rate"
        assert query_metrics['invalidation_accuracy'] >= 0.95, "Should maintain cache consistency"
        assert query_metrics['collection_invalidation_latency'] < 0.1, "Should invalidate quickly"
        
        # Benchmark batch processing optimization
        batch_metrics = component_benchmark.benchmark_batch_processing(
            batch_sizes=[10, 50, 100, 200],
            parallel_workers=[1, 2, 4, 8]
        )
        
        assert batch_metrics['optimal_batch_size'] > 0, "Should identify optimal batch size"
        assert batch_metrics['parallel_efficiency'] > 0.7, "Should show parallel processing benefit"
        assert batch_metrics['memory_efficient_processing'] == True, "Should respect memory limits"
    
    def test_end_to_end_pipeline_scalability(self):
        """Test end-to-end pipeline scalability under different workloads."""
        # This test should FAIL initially - no scalability benchmarking
        from src.research_agent_backend.core.comprehensive_benchmark import ScalabilityBenchmark
        
        scalability_benchmark = ScalabilityBenchmark()
        
        # Test scalability with increasing data sizes
        scalability_results = scalability_benchmark.test_data_size_scalability(
            data_sizes=[100, 500, 1000, 2000],
            queries_per_size=50
        )
        
        # Verify performance scales reasonably
        for i in range(len(scalability_results) - 1):
            current = scalability_results[i]
            next_size = scalability_results[i + 1]
            
            # Response time should scale sub-linearly
            response_time_ratio = next_size['response_time'] / current['response_time']
            data_size_ratio = next_size['data_size'] / current['data_size']
            assert response_time_ratio < data_size_ratio, \
                   f"Response time should scale better than data size: {response_time_ratio} vs {data_size_ratio}"
        
        # Test concurrent user scalability
        concurrent_results = scalability_benchmark.test_concurrent_users(
            user_counts=[1, 5, 10, 20],
            queries_per_user=20,
            test_duration_seconds=30
        )
        
        # Verify system handles concurrent load
        for result in concurrent_results:
            assert result['error_rate'] < 0.05, f"Error rate should be low: {result['error_rate']}"
            assert result['cache_hit_rate'] > 0.5, f"Cache should be effective under load: {result['cache_hit_rate']}"
            assert result['average_response_time'] < 2.0, f"Response time should be reasonable: {result['average_response_time']}"
    
    def test_realistic_workload_simulation(self):
        """Test performance under realistic workload scenarios."""
        # This test should FAIL initially - no realistic workload simulation
        from src.research_agent_backend.core.comprehensive_benchmark import WorkloadSimulator
        
        workload_simulator = WorkloadSimulator()
        
        # Simulate knowledge base updates with cache invalidation
        update_scenario = workload_simulator.simulate_document_update_workload(
            initial_documents=1000,
            update_frequency_seconds=10,
            query_rate_qps=5,
            test_duration_seconds=60
        )
        
        assert update_scenario['cache_invalidation_events'] > 0, "Should trigger cache invalidations"
        assert update_scenario['cache_consistency_maintained'] == True, "Should maintain cache consistency"
        assert update_scenario['query_success_rate'] > 0.95, "Should maintain high query success rate"
        assert update_scenario['average_invalidation_latency'] < 0.1, "Should invalidate caches quickly"
        
        # Simulate mixed query patterns (popular vs rare queries)
        query_pattern_scenario = workload_simulator.simulate_query_patterns(
            popular_queries=20,    # 20% of queries are popular (repeated)
            rare_queries=80,       # 80% of queries are unique
            total_queries=500,
            popularity_distribution="zipf"  # Realistic distribution
        )
        
        assert query_pattern_scenario['popular_query_hit_rate'] > 0.8, \
               "Popular queries should have high cache hit rate"
        assert query_pattern_scenario['overall_cache_efficiency'] > 0.4, \
               "Overall cache should be reasonably effective"
        assert query_pattern_scenario['cache_size_efficiency'] > 0.6, \
               "Cache should use space efficiently"
        
        # Simulate cache warmup vs cold cache scenarios
        warmup_comparison = workload_simulator.compare_warmup_scenarios(
            query_count=100,
            warmup_queries=50
        )
        
        assert warmup_comparison['warm_cache_performance']['response_time'] < \
               warmup_comparison['cold_cache_performance']['response_time'], \
               "Warm cache should be faster than cold cache"
        assert warmup_comparison['cache_warmup_benefit'] > 1.5, \
               "Cache warmup should provide significant benefit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 