"""
Comprehensive unit tests for QueryManager module.

Tests query optimization, caching, batch processing, performance monitoring,
and all advanced search capabilities of the Research Agent query system.

Following TDD methodology - comprehensive coverage for all components.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from research_agent_backend.core.query_manager import (
    QueryManager,
    QueryConfig,
    FilterConfig,
    PaginationConfig,
    PerformanceMetrics,
    PaginationInfo,
    QueryResult,
    QueryOptimizer,
    QueryCache,
    BatchQueryProcessor,
    QueryManagerError,
    QueryOptimizationError,
    PaginationError,
    FilterValidationError,
    PerformanceError,
    CacheError,
)
from research_agent_backend.core.vector_store import ChromaDBManager
from research_agent_backend.core.collection_type_manager import CollectionTypeManager
from research_agent_backend.core.data_preparation import DataPreparationManager
from research_agent_backend.utils.config import ConfigManager


class TestQueryConfig:
    """Test QueryConfig dataclass."""
    
    def test_query_config_creation_default(self):
        """Test creating QueryConfig with default values."""
        config = QueryConfig()
        assert config.max_results == 100
        assert config.similarity_threshold == 0.0
        assert config.embedding_model == "default"
        assert config.search_strategy == "precise"
        assert config.enable_caching == True
        assert config.enable_vector_optimization == False
        assert config.optimization_strategy == "none"
        assert config.target_dimensions is None
        assert config.timeout_seconds == 30.0
    
    def test_query_config_creation_custom(self):
        """Test creating QueryConfig with custom values."""
        config = QueryConfig(
            max_results=50,
            similarity_threshold=0.7,
            embedding_model="advanced",
            search_strategy="fast",
            enable_caching=False,
            enable_vector_optimization=True,
            optimization_strategy="pca",
            target_dimensions=256,
            timeout_seconds=15.0
        )
        assert config.max_results == 50
        assert config.similarity_threshold == 0.7
        assert config.embedding_model == "advanced"
        assert config.search_strategy == "fast"
        assert config.enable_caching == False
        assert config.enable_vector_optimization == True
        assert config.optimization_strategy == "pca"
        assert config.target_dimensions == 256
        assert config.timeout_seconds == 15.0


class TestFilterConfig:
    """Test FilterConfig dataclass."""
    
    def test_filter_config_creation_default(self):
        """Test creating FilterConfig with default values."""
        config = FilterConfig()
        assert config.logic_operator == "AND"
        assert config.filters == []
    
    def test_filter_config_creation_custom(self):
        """Test creating FilterConfig with custom filters."""
        filters = [
            {"field": "source", "operator": "equals", "value": "doc1.md"},
            {"field": "section", "operator": "contains", "value": "introduction"}
        ]
        config = FilterConfig(logic_operator="OR", filters=filters)
        assert config.logic_operator == "OR"
        assert len(config.filters) == 2
        assert config.filters[0]["field"] == "source"
        assert config.filters[1]["field"] == "section"


class TestPaginationConfig:
    """Test PaginationConfig dataclass."""
    
    def test_pagination_config_creation_default(self):
        """Test creating PaginationConfig with default values."""
        config = PaginationConfig()
        assert config.type == "offset_limit"
        assert config.page_size == 10
        assert config.current_page == 1
        assert config.cursor is None
        assert config.session_id is None
    
    def test_pagination_config_creation_custom(self):
        """Test creating PaginationConfig with custom values."""
        config = PaginationConfig(
            type="cursor",
            page_size=25,
            current_page=3,
            cursor="abc123",
            session_id="session_456"
        )
        assert config.type == "cursor"
        assert config.page_size == 25
        assert config.current_page == 3
        assert config.cursor == "abc123"
        assert config.session_id == "session_456"


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation_default(self):
        """Test creating PerformanceMetrics with default values."""
        metrics = PerformanceMetrics()
        assert metrics.total_execution_time == 0.0
        assert metrics.vector_search_time == 0.0
        assert metrics.filter_time == 0.0
        assert metrics.result_formatting_time == 0.0
        assert metrics.optimization_time == 0.0
    
    def test_performance_metrics_creation_custom(self):
        """Test creating PerformanceMetrics with custom values."""
        metrics = PerformanceMetrics(
            total_execution_time=2.5,
            vector_search_time=1.2,
            filter_time=0.8,
            result_formatting_time=0.3,
            optimization_time=0.2
        )
        assert metrics.total_execution_time == 2.5
        assert metrics.vector_search_time == 1.2
        assert metrics.filter_time == 0.8
        assert metrics.result_formatting_time == 0.3
        assert metrics.optimization_time == 0.2


class TestPaginationInfo:
    """Test PaginationInfo dataclass."""
    
    def test_pagination_info_creation_default(self):
        """Test creating PaginationInfo with default values."""
        info = PaginationInfo()
        assert info.type == "none"
        assert info.current_page is None
        assert info.page_size is None
        assert info.total_pages is None
        assert info.has_next_page is None
        assert info.has_previous_page is None
        assert info.next_cursor is None
        assert info.previous_cursor is None
        assert info.total_results is None
        assert info.performance_warning is None
        assert info.session_id is None


class TestQueryResult:
    """Test QueryResult dataclass."""
    
    def test_query_result_creation_default(self):
        """Test creating QueryResult with default values."""
        result = QueryResult()
        assert result.results == []
        assert result.total_results == 0
        assert result.similarity_scores == []
        assert result.metadata == []
        assert result.document_ids == []
        assert isinstance(result.performance_metrics, PerformanceMetrics)
        assert result.embedding_model == "default"
        assert result.search_strategy == "precise"
        assert result.filter_applied == False
        assert result.from_cache == False
        assert result.warnings == []
    
    def test_query_result_creation_with_data(self):
        """Test creating QueryResult with actual data."""
        results_data = [
            {"content": "Test document 1", "score": 0.95},
            {"content": "Test document 2", "score": 0.89}
        ]
        similarity_scores = [0.95, 0.89]
        metadata = [{"source": "doc1.md"}, {"source": "doc2.md"}]
        document_ids = ["doc1_chunk1", "doc2_chunk1"]
        
        result = QueryResult(
            results=results_data,
            total_results=2,
            similarity_scores=similarity_scores,
            metadata=metadata,
            document_ids=document_ids,
            embedding_model="advanced",
            search_strategy="fast",
            filter_applied=True,
            from_cache=True
        )
        
        assert len(result.results) == 2
        assert result.total_results == 2
        assert result.similarity_scores == [0.95, 0.89]
        assert len(result.metadata) == 2
        assert len(result.document_ids) == 2
        assert result.embedding_model == "advanced"
        assert result.search_strategy == "fast"
        assert result.filter_applied == True
        assert result.from_cache == True


class TestQueryOptimizer:
    """Test QueryOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = QueryOptimizer()
    
    def test_optimizer_initialization(self):
        """Test QueryOptimizer initialization."""
        assert self.optimizer is not None
        assert self.optimizer.performance_history == []
        assert isinstance(self.optimizer.optimization_strategies, dict)
        assert "precise" in self.optimizer.optimization_strategies
        assert "fast" in self.optimizer.optimization_strategies
        assert "balanced" in self.optimizer.optimization_strategies
    
    def test_select_strategy_precise(self):
        """Test strategy selection for small result sets."""
        strategy = self.optimizer.select_strategy(max_results=25)
        assert strategy == "precise"
        
        strategy = self.optimizer.select_strategy(max_results=50)
        assert strategy == "precise"
    
    def test_select_strategy_fast(self):
        """Test strategy selection for large result sets."""
        strategy = self.optimizer.select_strategy(max_results=500)
        assert strategy == "fast"
        
        strategy = self.optimizer.select_strategy(max_results=1000)
        assert strategy == "fast"
    
    def test_select_strategy_balanced(self):
        """Test strategy selection for medium result sets."""
        strategy = self.optimizer.select_strategy(max_results=100)
        assert strategy == "balanced"
        
        strategy = self.optimizer.select_strategy(max_results=300)
        assert strategy == "balanced"
    
    def test_analyze_performance_slow_execution(self):
        """Test performance analysis for slow queries."""
        metrics = PerformanceMetrics(total_execution_time=6.0)
        config = QueryConfig(max_results=1000)
        
        recommendations = self.optimizer.analyze_performance(metrics, config)
        assert len(recommendations) > 0
        assert any("Consider reducing max_results for better performance" in rec for rec in recommendations)
    
    def test_analyze_performance_low_threshold(self):
        """Test performance analysis for low similarity threshold."""
        metrics = PerformanceMetrics(total_execution_time=1.0)
        config = QueryConfig(similarity_threshold=0.3)
        
        recommendations = self.optimizer.analyze_performance(metrics, config)
        assert any("similarity_threshold" in rec for rec in recommendations)
    
    def test_analyze_performance_no_optimization(self):
        """Test performance analysis when optimization is disabled."""
        metrics = PerformanceMetrics(total_execution_time=1.0)
        config = QueryConfig(enable_vector_optimization=False)
        
        recommendations = self.optimizer.analyze_performance(metrics, config)
        assert any("vector_optimization" in rec for rec in recommendations)
    
    def test_analyze_performance_optimal_config(self):
        """Test performance analysis with optimal configuration."""
        metrics = PerformanceMetrics(total_execution_time=1.0)
        config = QueryConfig(
            similarity_threshold=0.8,
            enable_vector_optimization=True,
            max_results=50
        )
        
        recommendations = self.optimizer.analyze_performance(metrics, config)
        # Should have no recommendations for optimal config
        assert len(recommendations) == 0


class TestQueryCache:
    """Test QueryCache class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = QueryCache(max_size=5, ttl_seconds=3600)
    
    def test_cache_initialization(self):
        """Test QueryCache initialization."""
        assert self.cache.max_size == 5
        assert self.cache.ttl_seconds == 3600
        assert self.cache.cache == {}
        assert self.cache.access_times == {}
    
    def test_generate_key(self):
        """Test cache key generation."""
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["collection1", "collection2"]
        filters = {"source": "doc1.md"}
        
        key = self.cache._generate_key(query_embedding, collections, filters)
        assert isinstance(key, str)
        assert len(key) > 0
        
        # Same inputs should generate same key
        key2 = self.cache._generate_key(query_embedding, collections, filters)
        assert key == key2
        
        # Different inputs should generate different keys
        key3 = self.cache._generate_key([0.1, 0.2, 0.4], collections, filters)
        assert key != key3
    
    def test_cache_put_and_get(self):
        """Test putting and getting items from cache."""
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["collection1"]
        key = self.cache._generate_key(query_embedding, collections)
        
        # Create test result
        result = QueryResult(
            results=[{"content": "test"}],
            total_results=1,
            from_cache=False
        )
        
        # Put in cache
        self.cache.put(key, result)
        assert self.cache.size == 1
        
        # Get from cache
        cached_result = self.cache.get(key)
        assert cached_result is not None
        assert cached_result.from_cache == True
        assert cached_result.cache_key == key
        assert len(cached_result.results) == 1
    
    def test_cache_miss(self):
        """Test cache miss scenario."""
        nonexistent_key = "nonexistent_key"
        result = self.cache.get(nonexistent_key)
        assert result is None
    
    def test_cache_size_limit(self):
        """Test cache size limiting and LRU eviction."""
        # Fill cache to max size
        for i in range(6):  # One more than max_size
            key = f"key_{i}"
            result = QueryResult(results=[{"id": i}])
            self.cache.put(key, result)
        
        # Cache should not exceed max size
        assert self.cache.size <= 5
        
        # First item should have been evicted (LRU)
        first_result = self.cache.get("key_0")
        assert first_result is None
        
        # Last item should still be there
        last_result = self.cache.get("key_5")
        assert last_result is not None
    
    def test_cache_invalidation_all(self):
        """Test invalidating all cache entries."""
        # Add some items
        for i in range(3):
            key = f"key_{i}"
            result = QueryResult(results=[{"id": i}])
            self.cache.put(key, result)
        
        assert self.cache.size == 3
        
        # Invalidate all
        self.cache.invalidate(strategy="all")
        assert self.cache.size == 0
    
    def test_cache_invalidation_by_collection(self):
        """Test invalidating cache entries by collection."""
        # Add items with different collections
        collections1 = ["collection1"]
        collections2 = ["collection2"]
        
        key1 = self.cache._generate_key([0.1, 0.2], collections1)
        key2 = self.cache._generate_key([0.1, 0.3], collections2)
        
        result1 = QueryResult(collections_searched=collections1)
        result2 = QueryResult(collections_searched=collections2)
        
        self.cache.put(key1, result1)
        self.cache.put(key2, result2)
        
        assert self.cache.size == 2
        
        # Invalidate by collection
        self.cache.invalidate(strategy="collection", collection="collection1")
        
        # Only collection2 result should remain
        assert self.cache.get(key1) is None
        assert self.cache.get(key2) is not None
    
    @patch('time.time')
    def test_cache_ttl_expiration(self, mock_time):
        """Test cache TTL expiration."""
        # Set initial time
        mock_time.return_value = 1000
        
        key = "test_key"
        result = QueryResult()
        self.cache.put(key, result)
        
        # Item should be valid
        assert self.cache.is_valid(key) == True
        
        # Move time forward beyond TTL
        mock_time.return_value = 1000 + 3601  # Beyond 3600 seconds TTL
        
        # Item should be expired
        assert self.cache.is_valid(key) == False
        
        # Getting expired item should return None
        assert self.cache.get(key) is None


class TestBatchQueryProcessor:
    """Test BatchQueryProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchQueryProcessor(max_workers=2)
        self.mock_query_manager = Mock()
    
    def test_processor_initialization(self):
        """Test BatchQueryProcessor initialization."""
        assert self.processor.max_workers == 2
        assert self.processor.executor is not None
    
    def test_process_batch_sequential(self):
        """Test sequential batch processing."""
        queries = [
            {"query_embedding": [0.1, 0.2], "collections": ["col1"]},
            {"query_embedding": [0.2, 0.3], "collections": ["col2"]}
        ]
        
        # Mock query manager responses
        mock_result1 = QueryResult(total_results=5)
        mock_result2 = QueryResult(total_results=3)
        self.mock_query_manager.similarity_search.side_effect = [mock_result1, mock_result2]
        
        result = self.processor.process_batch(
            queries, 
            self.mock_query_manager,
            strategy="sequential"
        )
        
        assert result["execution_strategy"] == "sequential"
        assert result["total_queries"] == 2
        assert len(result["results"]) == 2
        assert result["total_results"] == 8  # 5 + 3
        assert result["successful_queries"] == 2
        assert result["failed_queries"] == 0
    
    def test_process_batch_parallel(self):
        """Test parallel batch processing."""
        queries = [
            {"query_embedding": [0.1, 0.2], "collections": ["col1"]},
            {"query_embedding": [0.2, 0.3], "collections": ["col2"]}
        ]
        
        # Mock query manager responses
        mock_result1 = QueryResult(total_results=5)
        mock_result2 = QueryResult(total_results=3)
        self.mock_query_manager.similarity_search.side_effect = [mock_result1, mock_result2]
        
        result = self.processor.process_batch(
            queries,
            self.mock_query_manager,
            strategy="parallel",
            parallel_config={"max_workers": 2}
        )
        
        assert result["execution_strategy"] == "parallel"
        assert result["total_queries"] == 2
        assert len(result["results"]) == 2
        assert result["successful_queries"] == 2
        assert result["failed_queries"] == 0
    
    def test_process_batch_with_failures(self):
        """Test batch processing with some query failures."""
        queries = [
            {"query_embedding": [0.1, 0.2], "collections": ["col1"]},
            {"query_embedding": [0.2, 0.3], "collections": ["col2"]}
        ]
        
        # First succeeds, second fails
        mock_result = QueryResult(total_results=5)
        self.mock_query_manager.similarity_search.side_effect = [
            mock_result,
            Exception("Query failed")
        ]
        
        result = self.processor.process_batch(
            queries,
            self.mock_query_manager,
            strategy="sequential"
        )
        
        assert result["successful_queries"] == 1
        assert result["failed_queries"] == 1
        assert len(result["errors"]) == 1
        assert "Query failed" in str(result["errors"][0])
    
    def test_process_batch_empty_queries(self):
        """Test batch processing with empty query list."""
        result = self.processor.process_batch(
            [],
            self.mock_query_manager,
            strategy="sequential"
        )
        
        assert result["total_queries"] == 0
        assert result["successful_queries"] == 0
        assert result["failed_queries"] == 0
        assert len(result["results"]) == 0


class TestQueryManager:
    """Test QueryManager class - main functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_chroma_manager = Mock(spec=ChromaDBManager)
        self.mock_config_manager = Mock(spec=ConfigManager)
        self.mock_collection_type_manager = Mock(spec=CollectionTypeManager)
        self.mock_data_preparation_manager = Mock(spec=DataPreparationManager)
        
        # Initialize query manager with mocks
        self.query_manager = QueryManager(
            chroma_manager=self.mock_chroma_manager,
            config_manager=self.mock_config_manager,
            collection_type_manager=self.mock_collection_type_manager,
            data_preparation_manager=self.mock_data_preparation_manager
        )
    
    def test_query_manager_initialization(self):
        """Test QueryManager initialization."""
        assert self.query_manager.chroma_manager == self.mock_chroma_manager
        assert self.query_manager.config_manager == self.mock_config_manager
        assert self.query_manager.collection_type_manager == self.mock_collection_type_manager
        assert self.query_manager.data_preparation_manager == self.mock_data_preparation_manager
        assert isinstance(self.query_manager.cache, QueryCache)
        assert isinstance(self.query_manager.optimizer, QueryOptimizer)
        assert isinstance(self.query_manager.batch_processor, BatchQueryProcessor)
    
    def test_query_manager_initialization_minimal(self):
        """Test QueryManager initialization with minimal dependencies."""
        manager = QueryManager(chroma_manager=self.mock_chroma_manager)
        assert manager.chroma_manager == self.mock_chroma_manager
        assert manager.config_manager is None
        assert manager.collection_type_manager is None
        assert manager.data_preparation_manager is None
    
    @patch('research_agent_backend.core.query_manager.time.time')
    def test_similarity_search_basic(self, mock_time):
        """Test basic similarity search functionality."""
        # Setup mock time
        mock_time.return_value = 1000
        
        # Setup mock responses
        mock_results = [
            {"content": "Test content 1", "score": 0.95},
            {"content": "Test content 2", "score": 0.89}
        ]
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Test content 1", "Test content 2"]],
            "metadatas": [[{"source": "doc1.md"}, {"source": "doc2.md"}]],
            "distances": [[0.05, 0.11]],
            "ids": [["id1", "id2"]]
        }
        self.mock_chroma_manager.get_collection.return_value = mock_collection
        
        # Execute query
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["test_collection"]
        
        result = self.query_manager.similarity_search(
            query_embedding=query_embedding,
            collections=collections
        )
        
        # Verify results
        assert isinstance(result, QueryResult)
        assert len(result.results) == 2
        assert result.total_results == 2
        assert len(result.similarity_scores) == 2
        assert result.similarity_scores[0] == 0.95  # 1 - 0.05
        assert result.similarity_scores[1] == 0.89  # 1 - 0.11
        assert result.embedding_model == "default"
        assert result.search_strategy == "precise"
        assert result.collections_searched == collections
    
    def test_similarity_search_with_config(self):
        """Test similarity search with custom configuration."""
        # Setup mock collection
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Test content"]],
            "metadatas": [[{"source": "doc1.md"}]],
            "distances": [[0.05]],
            "ids": [["id1"]]
        }
        self.mock_chroma_manager.get_collection.return_value = mock_collection
        
        # Create custom config
        config = QueryConfig(
            max_results=50,
            similarity_threshold=0.8,
            embedding_model="advanced",
            search_strategy="fast"
        )
        
        result = self.query_manager.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            collections=["test_collection"],
            config=config
        )
        
        assert result.embedding_model == "advanced"
        assert result.execution_strategy == "fast"  # Should be selected by optimizer
    
    def test_similarity_search_with_filters(self):
        """Test similarity search with metadata filters."""
        # Setup mock collection
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Filtered content"]],
            "metadatas": [[{"source": "doc1.md", "section": "intro"}]],
            "distances": [[0.05]],
            "ids": [["id1"]]
        }
        self.mock_chroma_manager.get_collection.return_value = mock_collection
        
        # Create filter config
        filter_config = FilterConfig(
            logic_operator="AND",
            filters=[
                {"field": "source", "operator": "equals", "value": "doc1.md"},
                {"field": "section", "operator": "equals", "value": "intro"}
            ]
        )
        
        result = self.query_manager.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            collections=["test_collection"],
            metadata_filter=filter_config
        )
        
        assert result.filter_applied == True
        assert len(result.results) == 1
        # Verify that chroma query was called with filters
        self.mock_chroma_manager.get_collection.assert_called_once()
        mock_collection.query.assert_called_once()
    
    def test_similarity_search_with_pagination(self):
        """Test similarity search with pagination."""
        # Setup mock collection with multiple results
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Content 1", "Content 2", "Content 3", "Content 4", "Content 5"]],
            "metadatas": [[{"id": i} for i in range(5)]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "ids": [[f"id{i}" for i in range(5)]]
        }
        self.mock_chroma_manager.get_collection.return_value = mock_collection
        
        # Create pagination config
        pagination_config = PaginationConfig(
            type="offset_limit",
            page_size=2,
            current_page=2
        )
        
        result = self.query_manager.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            collections=["test_collection"],
            pagination=pagination_config
        )
        
        assert result.pagination_info is not None
        assert result.pagination_info.type == "offset_limit"
        assert result.pagination_info.current_page == 2
        assert result.pagination_info.page_size == 2
        assert result.pagination_info.has_next_page == True
        assert result.pagination_info.has_previous_page == True
    
    def test_similarity_search_caching(self):
        """Test similarity search caching functionality."""
        # Setup mock collection
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Cached content"]],
            "metadatas": [[{"source": "doc1.md"}]],
            "distances": [[0.05]],
            "ids": [["id1"]]
        }
        self.mock_chroma_manager.get_collection.return_value = mock_collection
        
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["test_collection"]
        
        # First call - should hit the database
        result1 = self.query_manager.similarity_search(
            query_embedding=query_embedding,
            collections=collections,
            enable_caching=True
        )
        assert result1.from_cache == False
        
        # Second call - should hit the cache
        result2 = self.query_manager.similarity_search(
            query_embedding=query_embedding,
            collections=collections,
            enable_caching=True
        )
        assert result2.from_cache == True
        assert result2.cache_key is not None
        
        # Database should only have been called once
        assert self.mock_chroma_manager.get_collection.call_count == 1
    
    def test_similarity_search_performance_monitoring(self):
        """Test similarity search with performance monitoring."""
        # Setup mock collection
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Performance test content"]],
            "metadatas": [[{"source": "doc1.md"}]],
            "distances": [[0.05]],
            "ids": [["id1"]]
        }
        self.mock_chroma_manager.get_collection.return_value = mock_collection
        
        result = self.query_manager.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            collections=["test_collection"],
            enable_performance_monitoring=True,
            analyze_performance=True
        )
        
        assert isinstance(result.performance_metrics, PerformanceMetrics)
        assert result.performance_metrics.total_execution_time > 0
        assert isinstance(result.optimization_recommendations, list)
    
    def test_similarity_search_error_handling(self):
        """Test similarity search error handling."""
        # Setup mock to raise exception
        self.mock_chroma_manager.get_collection.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            self.query_manager.similarity_search(
                query_embedding=[0.1, 0.2, 0.3],
                collections=["test_collection"]
            )
    
    def test_similarity_search_graceful_degradation(self):
        """Test similarity search with graceful degradation."""
        # Setup mock to fail on first collection, succeed on second
        mock_collection1 = Mock()
        mock_collection1.query.side_effect = Exception("Collection 1 error")
        
        mock_collection2 = Mock()
        mock_collection2.query.return_value = {
            "documents": [["Fallback content"]],
            "metadatas": [[{"source": "doc2.md"}]],
            "distances": [[0.05]],
            "ids": [["id1"]]
        }
        
        self.mock_chroma_manager.get_collection.side_effect = [mock_collection1, mock_collection2]
        
        result = self.query_manager.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            collections=["failing_collection", "working_collection"],
            enable_graceful_degradation=True
        )
        
        # Should succeed with partial results
        assert len(result.results) == 1
        assert len(result.warnings) > 0
        assert "failing_collection" in result.warnings[0]
    
    def test_get_available_collections(self):
        """Test getting available collections."""
        # Setup mock response
        mock_collections = [
            {"name": "collection1", "metadata": {"type": "fundamental"}},
            {"name": "collection2", "metadata": {"type": "project"}}
        ]
        self.mock_chroma_manager.list_collections.return_value = mock_collections
        
        collections = self.query_manager.get_available_collections()
        assert collections == ["collection1", "collection2"]
        self.mock_chroma_manager.list_collections.assert_called_once()
    
    def test_process_batch_queries(self):
        """Test batch query processing."""
        batch_queries = [
            {"query_embedding": [0.1, 0.2], "collections": ["col1"]},
            {"query_embedding": [0.2, 0.3], "collections": ["col2"]}
        ]
        
        # Mock the batch processor
        mock_batch_result = {
            "total_queries": 2,
            "successful_queries": 2,
            "failed_queries": 0,
            "results": [
                QueryResult(total_results=5),
                QueryResult(total_results=3)
            ]
        }
        
        with patch.object(self.query_manager.batch_processor, 'process_batch', return_value=mock_batch_result):
            result = self.query_manager.process_batch_queries(
                batch_queries=batch_queries,
                strategy="sequential"
            )
        
        assert result["total_queries"] == 2
        assert result["successful_queries"] == 2
        assert result["failed_queries"] == 0
    
    def test_configure_cache(self):
        """Test cache configuration."""
        cache_config = {
            "max_size": 500,
            "ttl_seconds": 7200
        }
        
        self.query_manager.configure_cache(cache_config)
        
        # Verify cache was reconfigured
        assert self.query_manager.cache.max_size == 500
        assert self.query_manager.cache.ttl_seconds == 7200
    
    def test_get_cache_statistics(self):
        """Test getting cache statistics."""
        # Add some items to cache first
        for i in range(3):
            key = f"test_key_{i}"
            result = QueryResult(total_results=i)
            self.query_manager.cache.put(key, result)
        
        stats = self.query_manager.get_cache_statistics()
        
        assert stats["cache_size"] == 3
        assert stats["max_cache_size"] == self.query_manager.cache.max_size
        assert "hit_rate" in stats
        assert "miss_rate" in stats
    
    def test_cleanup_cache(self):
        """Test cache cleanup functionality."""
        # Add some items to cache
        for i in range(5):
            key = f"test_key_{i}"
            result = QueryResult(total_results=i)
            self.query_manager.cache.put(key, result)
        
        assert self.query_manager.cache.size == 5
        
        # Cleanup cache
        self.query_manager.cleanup_cache(aggressive=True)
        
        # Cache should be smaller or empty
        assert self.query_manager.cache.size < 5
    
    def test_invalidate_cache(self):
        """Test cache invalidation."""
        # Add some items to cache
        for i in range(3):
            key = f"test_key_{i}"
            result = QueryResult(total_results=i)
            self.query_manager.cache.put(key, result)
        
        assert self.query_manager.cache.size == 3
        
        # Invalidate all cache
        self.query_manager.invalidate_cache(strategy="all")
        
        assert self.query_manager.cache.size == 0


class TestQueryManagerIntegration:
    """Integration tests for QueryManager with various scenarios."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_chroma_manager = Mock(spec=ChromaDBManager)
        self.query_manager = QueryManager(chroma_manager=self.mock_chroma_manager)
    
    def test_complex_query_workflow(self):
        """Test complex query workflow with filters, pagination, and caching."""
        # Setup mock collection with rich data
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]],
            "metadatas": [[
                {"source": "file1.md", "section": "intro", "tags": ["important"]},
                {"source": "file2.md", "section": "body", "tags": ["detail"]},
                {"source": "file1.md", "section": "conclusion", "tags": ["important"]},
                {"source": "file3.md", "section": "intro", "tags": ["overview"]},
                {"source": "file2.md", "section": "intro", "tags": ["important"]}
            ]],
            "distances": [[0.1, 0.2, 0.15, 0.3, 0.25]],
            "ids": [["id1", "id2", "id3", "id4", "id5"]]
        }
        self.mock_chroma_manager.get_collection.return_value = mock_collection
        
        # Create complex query configuration
        config = QueryConfig(
            max_results=100,
            similarity_threshold=0.7,
            enable_caching=True,
            enable_vector_optimization=True
        )
        
        filter_config = FilterConfig(
            logic_operator="AND",
            filters=[
                {"field": "tags", "operator": "contains", "value": "important"}
            ]
        )
        
        pagination_config = PaginationConfig(
            type="offset_limit",
            page_size=2,
            current_page=1
        )
        
        # Execute complex query
        result = self.query_manager.similarity_search(
            query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            collections=["comprehensive_collection"],
            config=config,
            metadata_filter=filter_config,
            pagination=pagination_config,
            enable_performance_monitoring=True,
            analyze_performance=True
        )
        
        # Verify comprehensive result
        assert isinstance(result, QueryResult)
        assert result.filter_applied == True
        assert result.pagination_info is not None
        assert result.pagination_info.page_size == 2
        assert isinstance(result.performance_metrics, PerformanceMetrics)
        assert isinstance(result.optimization_recommendations, list)
        assert result.collections_searched == ["comprehensive_collection"]
    
    def test_multi_collection_search_with_merging(self):
        """Test searching across multiple collections with result merging."""
        # Setup mock collections with different results
        mock_collection1 = Mock()
        mock_collection1.query.return_value = {
            "documents": [["Collection 1 Doc"]],
            "metadatas": [[{"source": "col1.md", "relevance": "high"}]],
            "distances": [[0.1]],
            "ids": [["col1_id1"]]
        }
        
        mock_collection2 = Mock()
        mock_collection2.query.return_value = {
            "documents": [["Collection 2 Doc"]],
            "metadatas": [[{"source": "col2.md", "relevance": "medium"}]],
            "distances": [[0.15]],
            "ids": [["col2_id1"]]
        }
        
        self.mock_chroma_manager.get_collection.side_effect = [mock_collection1, mock_collection2]
        
        # Execute multi-collection search
        result = self.query_manager.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            collections=["collection1", "collection2"],
            merge_strategy="ranked"
        )
        
        # Verify merged results
        assert len(result.results) == 2
        assert result.collections_searched == ["collection1", "collection2"]
        assert result.merge_strategy == "ranked"
        # Results should be merged and ranked by similarity score
        assert result.similarity_scores[0] >= result.similarity_scores[1]
    
    def test_query_optimization_and_strategy_selection(self):
        """Test query optimization and automatic strategy selection."""
        # Setup mock collection
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["Optimized content"]],
            "metadatas": [[{"source": "test.md"}]],
            "distances": [[0.05]],
            "ids": [["opt_id1"]]
        }
        self.mock_chroma_manager.get_collection.return_value = mock_collection
        
        # Test different query sizes for different strategies
        test_cases = [
            (10, "precise"),    # Small query -> precise strategy
            (200, "balanced"),  # Medium query -> balanced strategy
            (800, "fast")       # Large query -> fast strategy
        ]
        
        for max_results, expected_strategy in test_cases:
            config = QueryConfig(max_results=max_results)
            
            result = self.query_manager.similarity_search(
                query_embedding=[0.1, 0.2, 0.3],
                collections=["test_collection"],
                config=config,
                analyze_performance=True
            )
            
            assert result.execution_strategy == expected_strategy
            assert isinstance(result.optimization_recommendations, list)
    
    def test_error_recovery_and_graceful_degradation(self):
        """Test error recovery and graceful degradation scenarios."""
        # Setup mixed success/failure scenario
        mock_collection1 = Mock()
        mock_collection1.query.side_effect = Exception("Network timeout")
        
        mock_collection2 = Mock()
        mock_collection2.query.return_value = {
            "documents": [["Recovered content"]],
            "metadatas": [[{"source": "backup.md"}]],
            "distances": [[0.1]],
            "ids": [["backup_id1"]]
        }
        
        mock_collection3 = Mock()
        mock_collection3.query.side_effect = Exception("Collection not found")
        
        self.mock_chroma_manager.get_collection.side_effect = [
            mock_collection1, mock_collection2, mock_collection3
        ]
        
        # Execute query with graceful degradation
        result = self.query_manager.similarity_search(
            query_embedding=[0.1, 0.2, 0.3],
            collections=["failing_col1", "working_col", "failing_col2"],
            enable_graceful_degradation=True
        )
        
        # Should succeed with partial results and warnings
        assert len(result.results) == 1
        assert len(result.warnings) >= 2  # Warnings for failed collections
        assert "failing_col1" in str(result.warnings)
        assert "failing_col2" in str(result.warnings)
        assert result.collections_searched == ["failing_col1", "working_col", "failing_col2"] 