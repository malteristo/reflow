"""
Test suite for QueryManager - Query Optimization and Implementation.

This module contains comprehensive tests for the QueryManager class following
strict TDD methodology (RED-GREEN-REFACTOR). All tests are designed to fail
initially and drive the implementation of the QueryManager functionality.

Tests cover:
- Advanced vector similarity search with configurable parameters
- Enhanced metadata filtering with complex boolean logic
- Query execution optimization and performance monitoring
- Pagination and result limiting for large result sets
- Query performance benchmarking and caching
- Integration with existing manager classes
- Comprehensive error handling and graceful degradation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime, timedelta

# Import test infrastructure
from .conftest import (
    sample_embeddings, sample_metadata, temp_directory,
    in_memory_chroma_manager, collection_type_manager,
    data_preparation_manager, config_manager
)

# Import modules we'll be testing (these will fail initially)
from ..core.query_manager import (
    QueryManager,
    QueryConfig,
    QueryResult,
    PaginationConfig,
    FilterConfig,
    PerformanceMetrics,
    QueryOptimizer,
    QueryCache,
    BatchQueryProcessor
)
from ..core.vector_store import ChromaDBManager
from ..core.collection_type_manager import CollectionTypeManager
from ..core.data_preparation import DataPreparationManager
from ..utils.config import ConfigManager
from ..exceptions.query_exceptions import (
    QueryManagerError,
    QueryOptimizationError,
    PaginationError,
    FilterValidationError,
    PerformanceError,
    CacheError
)


class TestQueryManagerInitialization:
    """Test QueryManager initialization and configuration."""

    def test_query_manager_basic_initialization(self, in_memory_chroma_manager, config_manager):
        """Test basic QueryManager initialization with required dependencies."""
        # This will fail initially - QueryManager class doesn't exist
        query_manager = QueryManager(
            chroma_manager=in_memory_chroma_manager,
            config_manager=config_manager
        )
        
        assert query_manager.chroma_manager is in_memory_chroma_manager
        assert query_manager.config_manager is config_manager
        assert query_manager.is_initialized is True
        assert query_manager.query_cache is not None
        assert query_manager.performance_monitor is not None

    def test_query_manager_with_all_dependencies(
        self, 
        in_memory_chroma_manager, 
        config_manager,
        collection_type_manager,
        data_preparation_manager
    ):
        """Test QueryManager initialization with all optional dependencies."""
        query_manager = QueryManager(
            chroma_manager=in_memory_chroma_manager,
            config_manager=config_manager,
            collection_type_manager=collection_type_manager,
            data_preparation_manager=data_preparation_manager
        )
        
        assert query_manager.collection_type_manager is collection_type_manager
        assert query_manager.data_preparation_manager is data_preparation_manager
        assert query_manager.query_optimizer is not None
        assert query_manager.batch_processor is not None

    def test_query_manager_configuration_validation(self, config_manager):
        """Test QueryManager validates configuration during initialization."""
        # Test with invalid configuration
        with patch.object(config_manager, 'get') as mock_get:
            mock_get.return_value = {"invalid_config": True}
            
            with pytest.raises(QueryManagerError, match="Invalid query configuration"):
                QueryManager(
                    chroma_manager=Mock(),
                    config_manager=config_manager
                )

    def test_query_manager_default_configuration(self, in_memory_chroma_manager):
        """Test QueryManager creates default configuration when none provided."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        assert query_manager.config_manager is not None
        assert query_manager.default_query_config is not None
        assert query_manager.default_query_config.max_results == 100
        assert query_manager.default_query_config.similarity_threshold == 0.0
        assert query_manager.default_query_config.enable_caching is True


class TestAdvancedVectorSimilaritySearch:
    """Test advanced vector similarity search capabilities."""

    def test_configurable_similarity_search(self, in_memory_chroma_manager, sample_embeddings):
        """Test vector similarity search with configurable parameters."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        query_config = QueryConfig(
            similarity_threshold=0.8,
            max_results=20,
            embedding_model="custom-model",
            search_strategy="optimized"
        )
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            config=query_config
        )
        
        assert isinstance(result, QueryResult)
        assert result.total_results <= 20
        assert all(score >= 0.8 for score in result.similarity_scores)
        assert result.embedding_model == "custom-model"
        assert result.search_strategy == "optimized"

    def test_multi_collection_search(self, in_memory_chroma_manager, sample_embeddings):
        """Test vector similarity search across multiple collections."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["collection1", "collection2", "collection3"],
            merge_strategy="score_weighted",
            collection_weights={"collection1": 1.0, "collection2": 0.8, "collection3": 0.6}
        )
        
        assert isinstance(result, QueryResult)
        assert len(result.collection_results) == 3
        assert result.merge_strategy == "score_weighted"
        assert "collection1" in result.collection_results
        assert "collection2" in result.collection_results
        assert "collection3" in result.collection_results

    def test_embedding_model_selection(self, in_memory_chroma_manager, sample_embeddings):
        """Test dynamic embedding model selection for queries."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Test with different embedding models
        models_to_test = ["sentence-transformers", "openai-ada", "custom-model"]
        
        for model in models_to_test:
            query_config = QueryConfig(embedding_model=model)
            result = query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                config=query_config
            )
            
            assert result.embedding_model == model
            assert result.embedding_dimension > 0

    def test_vector_space_optimization(self, in_memory_chroma_manager, sample_embeddings):
        """Test vector space optimization for improved search performance."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        query_config = QueryConfig(
            enable_vector_optimization=True,
            optimization_strategy="dimension_reduction",
            target_dimensions=256
        )
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            config=query_config
        )
        
        assert result.optimization_applied is True
        assert result.optimized_dimensions == 256
        assert result.performance_metrics.optimization_time > 0


class TestEnhancedMetadataFiltering:
    """Test enhanced metadata filtering capabilities."""

    def test_complex_boolean_logic_filtering(self, in_memory_chroma_manager, sample_embeddings):
        """Test metadata filtering with complex boolean logic (AND/OR/NOT)."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        complex_filter = FilterConfig(
            logic_operator="AND",
            filters=[
                {"field": "document_type", "operator": "eq", "value": "markdown"},
                {
                    "logic_operator": "OR",
                    "filters": [
                        {"field": "priority", "operator": "gte", "value": 5},
                        {"field": "team_id", "operator": "in", "value": ["team1", "team2"]}
                    ]
                },
                {
                    "logic_operator": "NOT",
                    "filters": [
                        {"field": "status", "operator": "eq", "value": "archived"}
                    ]
                }
            ]
        )
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            metadata_filter=complex_filter
        )
        
        assert isinstance(result, QueryResult)
        assert result.filter_applied is True
        assert result.filtered_count < result.total_available

    def test_range_queries(self, in_memory_chroma_manager, sample_embeddings):
        """Test range-based metadata queries."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        range_filter = FilterConfig(
            filters=[
                {"field": "created_at", "operator": "between", "value": ["2024-01-01", "2024-12-31"]},
                {"field": "chunk_sequence_id", "operator": "gte", "value": 1},
                {"field": "chunk_sequence_id", "operator": "lte", "value": 100},
                {"field": "content_length", "operator": "gt", "value": 50}
            ]
        )
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            metadata_filter=range_filter
        )
        
        assert result.filter_applied is True
        assert all("created_at" in meta for meta in result.metadata)

    def test_field_validation(self, in_memory_chroma_manager, sample_embeddings):
        """Test metadata field validation during filtering."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Test with invalid field
        invalid_filter = FilterConfig(
            filters=[{"field": "nonexistent_field", "operator": "eq", "value": "test"}]
        )
        
        with pytest.raises(FilterValidationError, match="Field 'nonexistent_field' not found"):
            query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                metadata_filter=invalid_filter
            )

    def test_nested_metadata_filtering(self, in_memory_chroma_manager, sample_embeddings):
        """Test filtering on nested metadata structures."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        nested_filter = FilterConfig(
            filters=[
                {"field": "header_hierarchy.level", "operator": "eq", "value": 2},
                {"field": "header_hierarchy.title", "operator": "contains", "value": "Implementation"},
                {"field": "metadata.tags", "operator": "contains_any", "value": ["python", "ml"]}
            ]
        )
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            metadata_filter=nested_filter
        )
        
        assert result.filter_applied is True
        assert result.nested_field_count > 0


class TestQueryExecutionOptimization:
    """Test query execution optimization and performance monitoring."""

    def test_performance_monitoring(self, in_memory_chroma_manager, sample_embeddings):
        """Test query performance monitoring and metrics collection."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        start_time = time.time()
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            enable_performance_monitoring=True
        )
        end_time = time.time()
        
        assert isinstance(result.performance_metrics, PerformanceMetrics)
        assert result.performance_metrics.total_execution_time > 0
        assert result.performance_metrics.total_execution_time <= (end_time - start_time)
        assert result.performance_metrics.vector_search_time > 0
        assert result.performance_metrics.filter_time >= 0
        assert result.performance_metrics.result_formatting_time > 0

    def test_execution_strategy_selection(self, in_memory_chroma_manager, sample_embeddings):
        """Test automatic execution strategy selection based on query characteristics."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Test with small result set - should use "precise" strategy
        small_query_config = QueryConfig(max_results=10)
        result_small = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            config=small_query_config
        )
        
        assert result_small.execution_strategy == "precise"
        
        # Test with large result set - should use "fast" strategy  
        large_query_config = QueryConfig(max_results=1000)
        result_large = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            config=large_query_config
        )
        
        assert result_large.execution_strategy == "fast"

    def test_query_plan_analysis(self, in_memory_chroma_manager, sample_embeddings):
        """Test query plan analysis and optimization recommendations."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        result = query_manager.analyze_query_plan(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            metadata_filter=FilterConfig(filters=[{"field": "document_type", "operator": "eq", "value": "markdown"}])
        )
        
        assert isinstance(result, dict)
        assert "estimated_cost" in result
        assert "optimization_recommendations" in result
        assert "execution_plan" in result
        assert len(result["optimization_recommendations"]) >= 0

    def test_optimization_recommendation_engine(self, in_memory_chroma_manager, sample_embeddings):
        """Test query optimization recommendation engine."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Execute a potentially inefficient query
        inefficient_config = QueryConfig(
            max_results=10000,
            similarity_threshold=0.0,  # Very low threshold
            enable_vector_optimization=False
        )
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            config=inefficient_config,
            analyze_performance=True
        )
        
        assert len(result.optimization_recommendations) > 0
        assert any("similarity_threshold" in rec for rec in result.optimization_recommendations)
        assert any("vector_optimization" in rec for rec in result.optimization_recommendations)


class TestPaginationSystem:
    """Test pagination system for large result sets."""

    def test_offset_limit_pagination(self, in_memory_chroma_manager, sample_embeddings):
        """Test offset/limit based pagination."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        pagination_config = PaginationConfig(
            type="offset_limit",
            page_size=10,
            current_page=1
        )
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            pagination=pagination_config
        )
        
        assert isinstance(result, QueryResult)
        assert len(result.results) <= 10
        assert result.pagination_info.current_page == 1
        assert result.pagination_info.page_size == 10
        assert result.pagination_info.total_pages >= 1
        assert result.pagination_info.has_next_page is not None

    def test_cursor_based_pagination(self, in_memory_chroma_manager, sample_embeddings):
        """Test cursor-based pagination for large datasets."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # First page
        pagination_config = PaginationConfig(
            type="cursor",
            page_size=10,
            cursor=None
        )
        
        result_page1 = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            pagination=pagination_config
        )
        
        assert result_page1.pagination_info.next_cursor is not None
        
        # Second page using cursor from first page
        pagination_config.cursor = result_page1.pagination_info.next_cursor
        result_page2 = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            pagination=pagination_config
        )
        
        assert len(result_page2.results) <= 10
        assert result_page2.pagination_info.previous_cursor is not None

    def test_large_result_set_handling(self, in_memory_chroma_manager, sample_embeddings):
        """Test pagination with very large result sets."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Simulate large result set
        pagination_config = PaginationConfig(
            type="offset_limit",
            page_size=50,
            current_page=100  # Deep pagination
        )
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            pagination=pagination_config
        )
        
        assert isinstance(result, QueryResult)
        assert result.pagination_info.performance_warning is not None
        assert "deep_pagination" in result.pagination_info.performance_warning

    def test_pagination_state_management(self, in_memory_chroma_manager, sample_embeddings):
        """Test pagination state management and consistency."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        session_id = "test_session_123"
        pagination_config = PaginationConfig(
            type="stateful",
            page_size=10,
            session_id=session_id
        )
        
        # Multiple pages in same session
        results = []
        for page in range(3):
            pagination_config.current_page = page + 1
            result = query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                pagination=pagination_config
            )
            results.append(result)
        
        # Verify no duplicate results across pages
        all_ids = []
        for result in results:
            all_ids.extend(result.document_ids)
        
        assert len(all_ids) == len(set(all_ids))  # No duplicates


class TestResultLimitingFeatures:
    """Test result limiting and ranking features."""

    def test_configurable_result_limits(self, in_memory_chroma_manager, sample_embeddings):
        """Test configurable result limits with various strategies."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        limit_configs = [
            {"max_results": 5, "strategy": "top_k"},
            {"max_results": 10, "strategy": "threshold_based", "threshold": 0.8},
            {"max_results": 20, "strategy": "diversified", "diversity_factor": 0.5}
        ]
        
        for config in limit_configs:
            result = query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                result_limiting=config
            )
            
            assert len(result.results) <= config["max_results"]
            assert result.limiting_strategy == config["strategy"]

    def test_ranking_strategies(self, in_memory_chroma_manager, sample_embeddings):
        """Test different ranking strategies for result ordering."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        ranking_strategies = ["similarity_score", "relevance_score", "hybrid", "custom"]
        
        for strategy in ranking_strategies:
            result = query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                ranking_strategy=strategy
            )
            
            assert result.ranking_strategy == strategy
            assert len(result.similarity_scores) == len(result.results)
            # Verify results are properly sorted
            if len(result.similarity_scores) > 1:
                assert all(result.similarity_scores[i] >= result.similarity_scores[i+1] 
                          for i in range(len(result.similarity_scores)-1))

    def test_performance_aware_limiting(self, in_memory_chroma_manager, sample_embeddings):
        """Test performance-aware result limiting."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Test with performance constraints
        performance_config = {
            "max_execution_time": 1.0,  # 1 second
            "max_memory_usage": 100,    # 100 MB
            "adaptive_limiting": True
        }
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            performance_constraints=performance_config
        )
        
        assert result.performance_metrics.total_execution_time <= 1.0
        assert result.adaptive_limiting_applied is True
        assert result.final_result_count <= result.requested_result_count

    def test_dynamic_limit_adjustment(self, in_memory_chroma_manager, sample_embeddings):
        """Test dynamic result limit adjustment based on query performance."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Start with high limit
        initial_config = QueryConfig(max_results=1000)
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            config=initial_config,
            enable_dynamic_limiting=True
        )
        
        if result.performance_metrics.total_execution_time > 0.5:  # If query was slow
            assert result.suggested_limit < initial_config.max_results
            assert "performance_optimization" in result.limit_adjustment_reason


class TestQueryPerformanceBenchmarking:
    """Test query performance benchmarking and monitoring."""

    def test_execution_time_tracking(self, in_memory_chroma_manager, sample_embeddings):
        """Test detailed execution time tracking for query components."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            enable_detailed_timing=True
        )
        
        metrics = result.performance_metrics
        assert metrics.vector_search_time > 0
        assert metrics.filter_time >= 0
        assert metrics.result_formatting_time > 0
        assert metrics.total_execution_time > 0
        assert (metrics.vector_search_time + metrics.filter_time + 
                metrics.result_formatting_time) <= metrics.total_execution_time

    def test_optimization_recommendations(self, in_memory_chroma_manager, sample_embeddings):
        """Test performance optimization recommendations."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Run multiple queries to build performance history
        for _ in range(5):
            query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"]
            )
        
        recommendations = query_manager.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 0
        if recommendations:
            assert all("recommendation" in rec for rec in recommendations)
            assert all("confidence" in rec for rec in recommendations)
            assert all("impact" in rec for rec in recommendations)

    def test_performance_regression_detection(self, in_memory_chroma_manager, sample_embeddings):
        """Test detection of performance regressions over time."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Establish baseline performance
        baseline_results = []
        for _ in range(10):
            result = query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"]
            )
            baseline_results.append(result.performance_metrics.total_execution_time)
        
        # Simulate performance regression (mock slower execution)
        with patch.object(query_manager, '_execute_query') as mock_execute:
            mock_execute.return_value = Mock()
            mock_execute.return_value.performance_metrics.total_execution_time = max(baseline_results) * 2
            
            regression_analysis = query_manager.detect_performance_regression()
            
            assert regression_analysis["regression_detected"] is True
            assert regression_analysis["severity"] in ["minor", "major", "critical"]
            assert regression_analysis["baseline_avg"] > 0

    def test_benchmark_reporting(self, in_memory_chroma_manager, sample_embeddings):
        """Test comprehensive benchmark reporting."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Run benchmark suite
        benchmark_config = {
            "test_types": ["latency", "throughput", "memory"],
            "iterations": 10,
            "collection_sizes": [100, 1000, 10000]
        }
        
        benchmark_report = query_manager.run_benchmark_suite(benchmark_config)
        
        assert isinstance(benchmark_report, dict)
        assert "latency_results" in benchmark_report
        assert "throughput_results" in benchmark_report
        assert "memory_usage" in benchmark_report
        assert "recommendations" in benchmark_report


class TestBatchQueryProcessing:
    """Test batch query processing capabilities."""

    def test_batch_query_execution(self, in_memory_chroma_manager, sample_embeddings):
        """Test batch processing of multiple queries."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Prepare batch of queries
        batch_queries = [
            {
                "query_embedding": sample_embeddings[i % len(sample_embeddings)],
                "collections": ["test_collection"],
                "query_id": f"query_{i}"
            }
            for i in range(10)
        ]
        
        batch_result = query_manager.process_batch_queries(batch_queries)
        
        assert isinstance(batch_result, dict)
        assert len(batch_result["results"]) == 10
        assert batch_result["total_processed"] == 10
        assert batch_result["failed_count"] == 0
        assert batch_result["batch_execution_time"] > 0

    def test_parallel_query_processing(self, in_memory_chroma_manager, sample_embeddings):
        """Test parallel processing of batch queries."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        batch_queries = [
            {
                "query_embedding": sample_embeddings[0],
                "collections": ["test_collection"],
                "query_id": f"parallel_query_{i}"
            }
            for i in range(20)
        ]
        
        # Process with parallelization
        parallel_config = {
            "max_workers": 4,
            "enable_parallel": True,
            "chunk_size": 5
        }
        
        batch_result = query_manager.process_batch_queries(
            batch_queries, 
            parallel_config=parallel_config
        )
        
        assert batch_result["parallel_execution"] is True
        assert batch_result["workers_used"] == 4
        assert batch_result["chunks_processed"] == 4

    def test_batch_optimization_strategies(self, in_memory_chroma_manager, sample_embeddings):
        """Test optimization strategies for batch query processing."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Test different batch strategies
        strategies = ["sequential", "parallel", "adaptive", "optimized"]
        
        for strategy in strategies:
            batch_queries = [
                {
                    "query_embedding": sample_embeddings[0],
                    "collections": ["test_collection"],
                    "query_id": f"{strategy}_query_{i}"
                }
                for i in range(5)
            ]
            
            batch_result = query_manager.process_batch_queries(
                batch_queries,
                strategy=strategy
            )
            
            assert batch_result["strategy_used"] == strategy
            assert batch_result["total_processed"] == 5

    def test_batch_result_aggregation(self, in_memory_chroma_manager, sample_embeddings):
        """Test aggregation of batch query results."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        batch_queries = [
            {
                "query_embedding": sample_embeddings[0],
                "collections": ["test_collection"],
                "query_id": f"agg_query_{i}",
                "aggregation_group": "group_1" if i < 5 else "group_2"
            }
            for i in range(10)
        ]
        
        batch_result = query_manager.process_batch_queries(
            batch_queries,
            enable_aggregation=True,
            aggregation_strategy="by_group"
        )
        
        assert "aggregated_results" in batch_result
        assert "group_1" in batch_result["aggregated_results"]
        assert "group_2" in batch_result["aggregated_results"]
        assert len(batch_result["aggregated_results"]["group_1"]) == 5
        assert len(batch_result["aggregated_results"]["group_2"]) == 5


class TestQueryCachingSystem:
    """Test query result caching capabilities."""

    def test_query_result_caching(self, in_memory_chroma_manager, sample_embeddings):
        """Test basic query result caching functionality."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # First query - should hit database
        result1 = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            enable_caching=True
        )
        
        assert result1.from_cache is False
        assert result1.cache_key is not None
        
        # Second identical query - should hit cache
        result2 = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            enable_caching=True
        )
        
        assert result2.from_cache is True
        assert result2.cache_key == result1.cache_key
        assert result2.cache_hit_time < result1.performance_metrics.total_execution_time

    def test_cache_invalidation(self, in_memory_chroma_manager, sample_embeddings):
        """Test cache invalidation strategies."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Cache a query result
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            enable_caching=True
        )
        
        cache_key = result.cache_key
        
        # Test time-based invalidation
        query_manager.invalidate_cache(strategy="time_based", max_age_seconds=1)
        time.sleep(1.1)
        
        assert not query_manager.query_cache.is_valid(cache_key)
        
        # Test manual invalidation
        query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"],
            enable_caching=True
        )
        
        query_manager.invalidate_cache(strategy="manual", cache_keys=[cache_key])
        assert not query_manager.query_cache.is_valid(cache_key)

    def test_cache_performance_optimization(self, in_memory_chroma_manager, sample_embeddings):
        """Test cache performance optimization features."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        cache_config = {
            "max_cache_size": 100,
            "eviction_strategy": "LRU",
            "compression_enabled": True,
            "memory_threshold": 80  # Percent
        }
        
        query_manager.configure_cache(cache_config)
        
        # Fill cache with queries
        for i in range(150):  # Exceed max_cache_size
            query_manager.similarity_search(
                query_embedding=sample_embeddings[i % len(sample_embeddings)],
                collections=["test_collection"],
                enable_caching=True,
                cache_key_suffix=f"_{i}"
            )
        
        cache_stats = query_manager.get_cache_statistics()
        
        assert cache_stats["current_size"] <= 100
        assert cache_stats["evictions"] > 0
        assert cache_stats["hit_rate"] >= 0
        assert cache_stats["compression_ratio"] > 0

    def test_cache_memory_management(self, in_memory_chroma_manager, sample_embeddings):
        """Test cache memory management and cleanup."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Monitor memory usage during caching
        initial_memory = query_manager.get_memory_usage()
        
        # Cache many large results
        for i in range(50):
            query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                config=QueryConfig(max_results=1000),  # Large results
                enable_caching=True,
                cache_key_suffix=f"_large_{i}"
            )
        
        peak_memory = query_manager.get_memory_usage()
        
        # Trigger cache cleanup
        query_manager.cleanup_cache(aggressive=True)
        
        final_memory = query_manager.get_memory_usage()
        
        assert peak_memory > initial_memory
        assert final_memory < peak_memory
        assert query_manager.query_cache.size == 0


class TestIntegrationWithExistingManagers:
    """Test integration with existing manager classes."""

    def test_chroma_manager_integration(
        self, 
        in_memory_chroma_manager, 
        collection_type_manager,
        sample_embeddings
    ):
        """Test seamless integration with ChromaDBManager."""
        query_manager = QueryManager(
            chroma_manager=in_memory_chroma_manager,
            collection_type_manager=collection_type_manager
        )
        
        # Test that QueryManager can leverage ChromaDBManager's collection operations
        collections = query_manager.get_available_collections()
        
        assert isinstance(collections, list)
        assert query_manager.chroma_manager is in_memory_chroma_manager
        
        # Test query delegation
        result = query_manager.similarity_search(
            query_embedding=sample_embeddings[0],
            collections=["test_collection"]
        )
        
        assert isinstance(result, QueryResult)

    def test_collection_type_manager_integration(
        self,
        in_memory_chroma_manager,
        collection_type_manager,
        sample_embeddings
    ):
        """Test integration with CollectionTypeManager."""
        query_manager = QueryManager(
            chroma_manager=in_memory_chroma_manager,
            collection_type_manager=collection_type_manager
        )
        
        # Test collection type-aware querying
        result = query_manager.similarity_search_by_type(
            query_embedding=sample_embeddings[0],
            collection_type="FUNDAMENTAL",
            project_name="test_project"
        )
        
        assert isinstance(result, QueryResult)
        assert result.collection_type == "FUNDAMENTAL"
        assert "test_project" in result.collections_searched

    def test_data_preparation_manager_integration(
        self,
        in_memory_chroma_manager,
        data_preparation_manager,
        sample_embeddings
    ):
        """Test integration with DataPreparationManager."""
        query_manager = QueryManager(
            chroma_manager=in_memory_chroma_manager,
            data_preparation_manager=data_preparation_manager
        )
        
        # Test query preprocessing
        raw_query_text = "What is machine learning?"
        
        result = query_manager.search_with_text_query(
            query_text=raw_query_text,
            collections=["test_collection"],
            preprocess_query=True
        )
        
        assert isinstance(result, QueryResult)
        assert result.preprocessed_query_text != raw_query_text
        assert result.preprocessing_applied is True

    def test_cross_manager_coordination(
        self,
        in_memory_chroma_manager,
        collection_type_manager,
        data_preparation_manager,
        sample_embeddings
    ):
        """Test coordination between multiple managers."""
        query_manager = QueryManager(
            chroma_manager=in_memory_chroma_manager,
            collection_type_manager=collection_type_manager,
            data_preparation_manager=data_preparation_manager
        )
        
        # Test complex query that requires all managers
        result = query_manager.comprehensive_search(
            query_text="Python machine learning tutorial",
            collection_types=["FUNDAMENTAL", "PROJECT_SPECIFIC"],
            preprocess_query=True,
            auto_select_collections=True
        )
        
        assert isinstance(result, QueryResult)
        assert result.managers_involved == ["ChromaDBManager", "CollectionTypeManager", "DataPreparationManager"]
        assert len(result.collections_searched) > 0


class TestComprehensiveErrorHandling:
    """Test comprehensive error handling and graceful degradation."""

    def test_query_timeout_handling(self, in_memory_chroma_manager, sample_embeddings):
        """Test handling of query timeouts."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        query_config = QueryConfig(timeout_seconds=0.001)  # Very short timeout
        
        with pytest.raises(QueryManagerError, match="Query timeout"):
            query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                config=query_config
            )

    def test_database_connection_failures(self, sample_embeddings):
        """Test handling of database connection failures."""
        # Create QueryManager with disconnected ChromaDBManager
        failed_chroma_manager = Mock()
        failed_chroma_manager.client.side_effect = ConnectionError("Database connection failed")
        
        query_manager = QueryManager(chroma_manager=failed_chroma_manager)
        
        with pytest.raises(QueryManagerError, match="Database connection failed"):
            query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"]
            )

    def test_invalid_embedding_dimensions(self, in_memory_chroma_manager):
        """Test handling of invalid embedding dimensions."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        invalid_embedding = [0.1, 0.2]  # Wrong dimension
        
        with pytest.raises(QueryManagerError, match="Invalid embedding dimension"):
            query_manager.similarity_search(
                query_embedding=invalid_embedding,
                collections=["test_collection"]
            )

    def test_graceful_degradation(self, in_memory_chroma_manager, sample_embeddings):
        """Test graceful degradation when some features fail."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        # Mock partial failures
        with patch.object(query_manager, 'query_cache') as mock_cache:
            mock_cache.get.side_effect = CacheError("Cache unavailable")
            
            # Should still work without cache
            result = query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                enable_caching=True,
                enable_graceful_degradation=True
            )
            
            assert isinstance(result, QueryResult)
            assert result.from_cache is False
            assert "cache_unavailable" in result.warnings

    def test_error_recovery_mechanisms(self, in_memory_chroma_manager, sample_embeddings):
        """Test error recovery and retry mechanisms."""
        query_manager = QueryManager(chroma_manager=in_memory_chroma_manager)
        
        retry_config = {
            "max_retries": 3,
            "retry_delay": 0.1,
            "exponential_backoff": True
        }
        
        # Mock intermittent failures
        call_count = 0
        def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise QueryError("Temporary failure")
            return Mock()  # Success on third try
        
        with patch.object(in_memory_chroma_manager, 'query_collection', side_effect=mock_query):
            result = query_manager.similarity_search(
                query_embedding=sample_embeddings[0],
                collections=["test_collection"],
                retry_config=retry_config
            )
            
            assert call_count == 3
            assert result is not None


# Test fixtures and utilities
@pytest.fixture
def query_manager(in_memory_chroma_manager, config_manager):
    """Create a QueryManager instance for testing."""
    return QueryManager(
        chroma_manager=in_memory_chroma_manager,
        config_manager=config_manager
    )


@pytest.fixture 
def sample_query_configs():
    """Sample query configurations for testing."""
    return [
        QueryConfig(max_results=10, similarity_threshold=0.8),
        QueryConfig(max_results=50, similarity_threshold=0.5, enable_caching=True),
        QueryConfig(max_results=100, enable_vector_optimization=True)
    ]


@pytest.fixture
def sample_filter_configs():
    """Sample filter configurations for testing."""
    return [
        FilterConfig(filters=[{"field": "document_type", "operator": "eq", "value": "markdown"}]),
        FilterConfig(
            logic_operator="AND",
            filters=[
                {"field": "priority", "operator": "gte", "value": 5},
                {"field": "team_id", "operator": "in", "value": ["team1", "team2"]}
            ]
        )
    ]


@pytest.fixture
def sample_pagination_configs():
    """Sample pagination configurations for testing."""
    return [
        PaginationConfig(type="offset_limit", page_size=10, current_page=1),
        PaginationConfig(type="cursor", page_size=20, cursor=None),
        PaginationConfig(type="stateful", page_size=15, session_id="test_session")
    ] 