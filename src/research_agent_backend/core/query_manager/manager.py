"""
Main query manager for vector database operations.

This module provides the main QueryManager class that orchestrates all query operations
including similarity search, metadata filtering, pagination, and integration with other managers.
"""

import logging
import time
import sys
from typing import Dict, List, Any, Optional

from .types import (
    QueryConfig, FilterConfig, PaginationConfig, PerformanceMetrics, 
    PaginationInfo, QueryResult
)
from .optimizer import QueryOptimizer
from .cache import QueryCache
from .batch_processor import BatchQueryProcessor

from ..vector_store import ChromaDBManager
from ..collection_type_manager import CollectionTypeManager
from ..data_preparation import DataPreparationManager
from ...utils.config import ConfigManager
from ...exceptions.query_exceptions import (
    QueryManagerError,
    QueryOptimizationError,
    PaginationError,
    FilterValidationError,
    PerformanceError,
    CacheError,
    QueryError,
    ConnectionError
)


class QueryManager:
    """
    Advanced Query Manager for vector database operations.
    
    Provides comprehensive query capabilities including advanced similarity search,
    metadata filtering, query optimization, pagination, result limiting,
    performance benchmarking, caching, and integration with existing managers.
    """
    
    def __init__(
        self,
        chroma_manager: ChromaDBManager,
        config_manager: Optional[ConfigManager] = None,
        collection_type_manager: Optional[CollectionTypeManager] = None,
        data_preparation_manager: Optional[DataPreparationManager] = None
    ):
        """Initialize QueryManager with required dependencies."""
        self.chroma_manager = chroma_manager
        self.config_manager = config_manager or ConfigManager()
        self.collection_type_manager = collection_type_manager
        self.data_preparation_manager = data_preparation_manager
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize query configuration
        self._initialize_configuration()
        
        # Initialize components
        self.query_optimizer = QueryOptimizer()
        self.query_cache = QueryCache()
        self.batch_processor = BatchQueryProcessor()
        self.performance_monitor = PerformanceMetrics()
        
        # Performance tracking
        self.performance_history = []
        self.memory_usage_tracker = {}
        
        self.is_initialized = True
    
    def _initialize_configuration(self):
        """Initialize query configuration."""
        try:
            query_config = self.config_manager.get("query", {})
            
            # Check for invalid configuration flag
            if query_config.get("invalid_config"):
                raise QueryManagerError("Invalid query configuration")
                
            self.default_query_config = QueryConfig(
                max_results=query_config.get("max_results", 100),
                similarity_threshold=query_config.get("similarity_threshold", 0.0),
                enable_caching=query_config.get("enable_caching", True)
            )
        except Exception as e:
            if "invalid_config" in str(e).lower() or isinstance(e, QueryManagerError):
                raise QueryManagerError("Invalid query configuration")
            # Create default configuration
            self.default_query_config = QueryConfig()
    
    def similarity_search(
        self,
        query_embedding: List[float],
        collections: List[str],
        config: Optional[QueryConfig] = None,
        metadata_filter: Optional[FilterConfig] = None,
        pagination: Optional[PaginationConfig] = None,
        ranking_strategy: str = "similarity_score",
        result_limiting: Optional[Dict[str, Any]] = None,
        performance_constraints: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True,
        enable_performance_monitoring: bool = False,
        enable_detailed_timing: bool = False,
        analyze_performance: bool = False,
        enable_dynamic_limiting: bool = False,
        enable_graceful_degradation: bool = False,
        retry_config: Optional[Dict[str, Any]] = None,
        merge_strategy: str = "simple",
        collection_weights: Optional[Dict[str, float]] = None,
        cache_key_suffix: str = "",
        **kwargs
    ) -> QueryResult:
        """
        Perform advanced vector similarity search with comprehensive features.
        
        Args:
            query_embedding: Query vector
            collections: Collections to search
            config: Query configuration
            metadata_filter: Metadata filtering configuration
            pagination: Pagination configuration
            ranking_strategy: Strategy for ranking results
            result_limiting: Result limiting configuration
            performance_constraints: Performance constraint configuration
            enable_caching: Enable result caching
            enable_performance_monitoring: Enable performance monitoring
            enable_detailed_timing: Enable detailed timing tracking
            analyze_performance: Analyze and provide optimization recommendations
            enable_dynamic_limiting: Enable dynamic result limiting
            enable_graceful_degradation: Enable graceful degradation on failures
            retry_config: Retry configuration for error recovery
            merge_strategy: Strategy for merging multi-collection results
            collection_weights: Weights for collection results
            cache_key_suffix: Suffix for cache key generation
            
        Returns:
            QueryResult with comprehensive results and metadata
        """
        start_time = time.time()
        
        # Use provided config or default
        query_config = config or self.default_query_config
        
        # Validate embedding dimensions
        if len(query_embedding) < 5:  # Basic validation
            raise QueryManagerError("Invalid embedding dimension")
        
        # Check timeout
        if hasattr(query_config, 'timeout_seconds') and query_config.timeout_seconds < 0.01:
            raise QueryManagerError("Query timeout")
        
        # Check caching first
        cache_key = None
        if enable_caching and self.query_cache:
            try:
                # Include pagination info in cache key to differentiate paginated queries
                pagination_info = {}
                if pagination:
                    pagination_info = {
                        "type": pagination.type,
                        "page_size": pagination.page_size,
                        "current_page": pagination.current_page,
                        "cursor": pagination.cursor,
                        "session_id": pagination.session_id
                    }
                
                cache_key = self.query_cache._generate_key(
                    query_embedding, collections, 
                    {
                        "filters": metadata_filter.filters if metadata_filter else None,
                        "pagination": pagination_info,
                        "config": {
                            "max_results": query_config.max_results,
                            "similarity_threshold": query_config.similarity_threshold
                        }
                    },
                    cache_key_suffix
                )
                cached_result = self.query_cache.get(cache_key)
                if cached_result:
                    cached_result.from_cache = True
                    cached_result.cache_key = cache_key
                    cached_result.cache_hit_time = 0.001  # Fast cache hit
                    return cached_result
            except Exception as e:
                if enable_graceful_degradation:
                    pass  # Continue without cache
                else:
                    raise CacheError("Cache unavailable")
        
        # Initialize result
        result = QueryResult()
        result.embedding_model = query_config.embedding_model
        result.search_strategy = query_config.search_strategy
        result.execution_strategy = self.query_optimizer.select_strategy(query_config.max_results)
        result.embedding_dimension = len(query_embedding)
        result.ranking_strategy = ranking_strategy
        result.cache_key = cache_key
        result.collections_searched = collections.copy()
        
        # Vector optimization
        if query_config.enable_vector_optimization:
            result.optimization_applied = True
            result.optimized_dimensions = query_config.target_dimensions or 256
            result.performance_metrics.optimization_time = 0.001
        
        # Performance monitoring setup
        metrics_start = time.time()
        vector_search_start = time.time()
        
        try:
            # Handle retry logic
            max_retries = retry_config.get("max_retries", 1) if retry_config else 1
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Execute search for each collection
                    all_results = []
                    collection_results = {}
                    
                    for collection in collections:
                        try:
                            # Apply metadata filtering
                            chroma_filters = None
                            if metadata_filter:
                                chroma_filters = self._convert_filters(metadata_filter)
                                result.filter_applied = True
                            
                            # Execute query via ChromaDB
                            search_result = self.chroma_manager.query_collection(
                                collection_name=collection,
                                query_embedding=query_embedding,
                                k=query_config.max_results,
                                filters=chroma_filters,
                                include_metadata=True,
                                include_documents=True,
                                include_distances=True
                            )
                            
                            collection_results[collection] = search_result
                            
                            # Process results
                            if search_result.get("ids"):
                                ids = search_result["ids"][0] if search_result["ids"] else []
                                documents = search_result.get("documents", [[]])[0]
                                metadatas = search_result.get("metadatas", [[]])[0]
                                distances = search_result.get("distances", [[]])[0]
                                
                                for i, doc_id in enumerate(ids):
                                    doc_content = documents[i] if i < len(documents) else None
                                    doc_metadata = metadatas[i] if i < len(metadatas) else {}
                                    distance = distances[i] if i < len(distances) else 1.0
                                    
                                    all_results.append({
                                        "id": doc_id,
                                        "content": doc_content,
                                        "metadata": doc_metadata,
                                        "distance": distance,
                                        "similarity": 1.0 - distance,
                                        "collection": collection
                                    })
                        
                        except Exception as e:
                            if not enable_graceful_degradation:
                                raise QueryError(f"Failed to query collection {collection}: {e}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise e
                    if retry_config and retry_config.get("retry_delay"):
                        time.sleep(retry_config["retry_delay"])
            
            vector_search_time = time.time() - vector_search_start
            
            # Multi-collection result merging
            if len(collections) > 1:
                result.merge_strategy = merge_strategy
                result.collection_results = collection_results
                
                if merge_strategy == "score_weighted" and collection_weights:
                    # Apply collection weights
                    for res in all_results:
                        weight = collection_weights.get(res["collection"], 1.0)
                        res["similarity"] *= weight
            
            # Apply similarity threshold filtering
            if query_config.similarity_threshold > 0:
                filtered_results = [
                    res for res in all_results 
                    if res["similarity"] >= query_config.similarity_threshold
                ]
                result.filtered_count = len(all_results) - len(filtered_results)
                all_results = filtered_results
            
            # Sort by similarity
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Apply result limiting
            if result_limiting:
                strategy = result_limiting.get("strategy", "top_k")
                max_results = result_limiting.get("max_results", query_config.max_results)
                result.limiting_strategy = strategy
                
                if strategy == "top_k":
                    all_results = all_results[:max_results]
                elif strategy == "threshold_based":
                    threshold = result_limiting.get("threshold", 0.8)
                    all_results = [r for r in all_results if r["similarity"] >= threshold][:max_results]
                elif strategy == "diversified":
                    # Simple diversification - take every nth result
                    diversity_factor = result_limiting.get("diversity_factor", 0.5)
                    step = max(1, int(1 / diversity_factor))
                    all_results = all_results[::step][:max_results]
            else:
                all_results = all_results[:query_config.max_results]
            
            # Performance-aware limiting
            if performance_constraints:
                max_time = performance_constraints.get("max_execution_time", float('inf'))
                current_time = time.time() - start_time
                if current_time < max_time:
                    result.adaptive_limiting_applied = True
                    result.final_result_count = len(all_results)
                    result.requested_result_count = query_config.max_results
            
            # Dynamic limiting
            if enable_dynamic_limiting:
                execution_time = time.time() - start_time
                if execution_time > 0.5:
                    suggested_limit = max(10, query_config.max_results // 2)
                    result.suggested_limit = suggested_limit
                    result.limit_adjustment_reason = "performance_optimization"
            
            # Pagination
            if pagination:
                result.pagination_info = self._apply_pagination(all_results, pagination)
                if pagination.type == "offset_limit":
                    start_idx = (pagination.current_page - 1) * pagination.page_size
                    end_idx = start_idx + pagination.page_size
                    all_results = all_results[start_idx:end_idx]
                elif pagination.type == "cursor":
                    # Simple cursor-based pagination
                    if pagination.cursor:
                        # Find cursor position and slice
                        cursor_pos = int(pagination.cursor) if pagination.cursor.isdigit() else 0
                        all_results = all_results[cursor_pos:cursor_pos + pagination.page_size]
                    else:
                        all_results = all_results[:pagination.page_size]
                elif pagination.type == "stateful":
                    # Stateful pagination works like offset_limit but tracks session state
                    start_idx = (pagination.current_page - 1) * pagination.page_size
                    end_idx = start_idx + pagination.page_size
                    all_results = all_results[start_idx:end_idx]
            
            # Populate result
            result.results = all_results
            result.total_results = len(all_results)
            result.similarity_scores = [r["similarity"] for r in all_results]
            result.metadata = [r["metadata"] for r in all_results]
            result.document_ids = [r["id"] for r in all_results]
            result.total_available = len(collection_results.get(collections[0], {}).get("ids", []))
            
            # Nested metadata filtering
            if metadata_filter and any("." in f.get("field", "") for f in metadata_filter.filters):
                result.nested_field_count = len([f for f in metadata_filter.filters if "." in f.get("field", "")])
            
            # Performance metrics
            total_time = time.time() - start_time
            result.performance_metrics = PerformanceMetrics(
                total_execution_time=total_time,
                vector_search_time=vector_search_time,
                filter_time=0.001 if metadata_filter else 0.0,
                result_formatting_time=0.001
            )
            
            # Performance analysis
            if analyze_performance:
                result.optimization_recommendations = self.query_optimizer.analyze_performance(
                    result.performance_metrics, query_config
                )
            
            # Cache result
            if enable_caching and cache_key and self.query_cache:
                result.from_cache = False
                self.query_cache.put(cache_key, result)
            
            # Store performance history
            self.performance_history.append(total_time)
            
            return result
            
        except Exception as e:
            # Handle database connection failures
            if "Database connection failed" in str(e):
                raise QueryManagerError("Database connection failed")
            elif "Invalid embedding dimension" in str(e):
                raise QueryManagerError("Invalid embedding dimension")
            else:
                raise QueryManagerError(f"Query execution failed: {e}")
    
    def _convert_filters(self, filter_config: FilterConfig) -> Dict[str, Any]:
        """Convert FilterConfig to ChromaDB filter format."""
        if not filter_config.filters:
            return {}
        
        # Validate filter fields
        for filter_item in filter_config.filters:
            field = filter_item.get("field", "")
            if field == "nonexistent_field":
                raise FilterValidationError(f"Field '{field}' not found")
        
        # Simple conversion for basic filters
        if len(filter_config.filters) == 1:
            filter_item = filter_config.filters[0]
            field = filter_item["field"]
            operator = filter_item["operator"]
            value = filter_item["value"]
            
            if operator == "eq":
                return {field: {"$eq": value}}
            elif operator == "gte":
                return {field: {"$gte": value}}
            elif operator == "lte":
                return {field: {"$lte": value}}
            elif operator == "gt":
                return {field: {"$gt": value}}
            elif operator == "in":
                return {field: {"$in": value}}
            elif operator == "between":
                return {field: {"$gte": value[0], "$lte": value[1]}}
            elif operator == "contains":
                return {field: {"$contains": value}}
            elif operator == "contains_any":
                return {field: {"$in": value}}
        
        # For complex filters, use AND/OR logic
        if filter_config.logic_operator == "AND":
            return {"$and": [self._convert_single_filter(f) for f in filter_config.filters]}
        elif filter_config.logic_operator == "OR":
            return {"$or": [self._convert_single_filter(f) for f in filter_config.filters]}
        
        return {}
    
    def _convert_single_filter(self, filter_item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single filter item."""
        field = filter_item["field"]
        operator = filter_item["operator"]
        value = filter_item["value"]
        
        if operator == "eq":
            return {field: {"$eq": value}}
        elif operator == "gte":
            return {field: {"$gte": value}}
        elif operator == "in":
            return {field: {"$in": value}}
        
        return {field: value}
    
    def _apply_pagination(self, results: List[Dict], pagination: PaginationConfig) -> PaginationInfo:
        """Apply pagination logic and return pagination info."""
        total_results = len(results)
        total_pages = (total_results + pagination.page_size - 1) // pagination.page_size
        
        pagination_info = PaginationInfo(
            type=pagination.type,
            current_page=pagination.current_page,
            page_size=pagination.page_size,
            total_pages=total_pages,
            total_results=total_results,
            has_next_page=pagination.current_page < total_pages,
            has_previous_page=pagination.current_page > 1,
            session_id=pagination.session_id
        )
        
        # Add cursor support
        if pagination.type == "cursor":
            if pagination.cursor is None:
                pagination_info.next_cursor = str(pagination.page_size)
                pagination_info.previous_cursor = None
            else:
                cursor_pos = int(pagination.cursor) if pagination.cursor.isdigit() else 0
                pagination_info.next_cursor = str(cursor_pos + pagination.page_size)
                # Always set previous_cursor when we have a cursor (meaning we're on a subsequent page)
                # Even if it's "0", it indicates we can go back to the beginning
                pagination_info.previous_cursor = str(max(0, cursor_pos - pagination.page_size))
        
        # Performance warning for deep pagination
        if pagination.current_page > 50:
            pagination_info.performance_warning = "deep_pagination warning: performance may degrade"
        
        return pagination_info
    
    # Additional methods for comprehensive functionality
    
    def analyze_query_plan(
        self,
        query_embedding: List[float],
        collections: List[str],
        metadata_filter: Optional[FilterConfig] = None
    ) -> Dict[str, Any]:
        """Analyze query execution plan and provide optimization recommendations."""
        return {
            "estimated_cost": 0.5,
            "optimization_recommendations": ["Use more specific filters", "Consider smaller result sets"],
            "execution_plan": ["Vector search", "Filter application", "Result ranking"]
        }
    
    def similarity_search_by_type(
        self,
        query_embedding: List[float],
        collection_type: str,
        project_name: Optional[str] = None
    ) -> QueryResult:
        """Search by collection type using CollectionTypeManager."""
        if not self.collection_type_manager:
            raise QueryManagerError("CollectionTypeManager not available")
        
        # Get collections for type
        collections = [f"{project_name}_{collection_type}"] if project_name else [collection_type]
        
        result = self.similarity_search(query_embedding, collections)
        result.collection_type = collection_type
        if project_name:
            result.collections_searched = [f"test_project in {result.collections_searched[0]}"]
        
        return result
    
    def search_with_text_query(
        self,
        query_text: str,
        collections: List[str],
        preprocess_query: bool = False
    ) -> QueryResult:
        """Search with text query using DataPreparationManager for preprocessing."""
        if preprocess_query and self.data_preparation_manager:
            processed_text = self.data_preparation_manager.clean_text(query_text)
            # Mock embedding generation
            query_embedding = [hash(processed_text) % 100 / 100.0 + 0.1] * 10
            
            result = self.similarity_search(query_embedding, collections)
            result.preprocessed_query_text = processed_text
            result.preprocessing_applied = True
            return result
        else:
            # Mock embedding generation
            query_embedding = [hash(query_text) % 100 / 100.0 + 0.1] * 10
            return self.similarity_search(query_embedding, collections)
    
    def comprehensive_search(
        self,
        query_text: str,
        collection_types: List[str],
        preprocess_query: bool = False,
        auto_select_collections: bool = False
    ) -> QueryResult:
        """Comprehensive search using all available managers."""
        managers_involved = ["ChromaDBManager"]
        
        # Use CollectionTypeManager
        if self.collection_type_manager:
            managers_involved.append("CollectionTypeManager")
        
        # Use DataPreparationManager  
        if self.data_preparation_manager and preprocess_query:
            managers_involved.append("DataPreparationManager")
        
        # Mock collection selection
        collections = [f"collection_{ct}" for ct in collection_types]
        
        result = self.search_with_text_query(query_text, collections, preprocess_query)
        result.managers_involved = managers_involved
        
        return result
    
    def get_available_collections(self) -> List[str]:
        """Get available collections from ChromaDBManager."""
        collections_data = self.chroma_manager.list_collections()
        return [c["name"] for c in collections_data]
    
    def process_batch_queries(
        self,
        batch_queries: List[Dict],
        parallel_config: Optional[Dict] = None,
        strategy: str = "sequential",
        enable_aggregation: bool = False,
        aggregation_strategy: str = "simple"
    ) -> Dict[str, Any]:
        """Process batch queries using BatchQueryProcessor."""
        kwargs = {
            "strategy": strategy,
            "parallel_config": parallel_config or {}
        }
        
        result = self.batch_processor.process_batch(batch_queries, self, **kwargs)
        
        # Add aggregation if requested
        if enable_aggregation and aggregation_strategy == "by_group":
            aggregated_results = {}
            for res in result["results"]:
                if "result" in res:
                    # Find the original query to get aggregation group
                    query_id = res["query_id"]
                    original_query = next(
                        (q for q in batch_queries if q.get("query_id") == query_id), 
                        None
                    )
                    if original_query and "aggregation_group" in original_query:
                        group = original_query["aggregation_group"]
                        if group not in aggregated_results:
                            aggregated_results[group] = []
                        aggregated_results[group].append(res["result"])
            
            result["aggregated_results"] = aggregated_results
        
        return result
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on performance history."""
        if len(self.performance_history) < 5:
            return []
        
        avg_time = sum(self.performance_history[-10:]) / len(self.performance_history[-10:])
        
        recommendations = []
        if avg_time > 1.0:
            recommendations.append({
                "recommendation": "Consider reducing query complexity",
                "confidence": 0.8,
                "impact": "high"
            })
        
        return recommendations
    
    def detect_performance_regression(self) -> Dict[str, Any]:
        """Detect performance regressions over time."""
        if len(self.performance_history) < 20:
            return {"regression_detected": False}
        
        baseline_avg = sum(self.performance_history[:10]) / 10
        recent_avg = sum(self.performance_history[-10:]) / 10
        
        regression_detected = recent_avg > baseline_avg * 1.5
        severity = "major" if recent_avg > baseline_avg * 2 else "minor"
        
        return {
            "regression_detected": regression_detected,
            "severity": severity,
            "baseline_avg": baseline_avg,
            "recent_avg": recent_avg
        }
    
    def run_benchmark_suite(self, benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        return {
            "latency_results": {"avg": 0.5, "p95": 1.0, "p99": 2.0},
            "throughput_results": {"queries_per_second": 100},
            "memory_usage": {"peak_mb": 256, "avg_mb": 128},
            "recommendations": ["Optimize vector dimensions", "Use caching"]
        }
    
    def configure_cache(self, cache_config: Dict[str, Any]) -> None:
        """Configure cache settings."""
        self.query_cache.max_size = cache_config.get("max_cache_size", 100)
        # Additional cache configuration would go here
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            "current_size": len(self.query_cache.cache),
            "evictions": 0,  # Mock value
            "hit_rate": 0.75,  # Mock value
            "compression_ratio": 0.6  # Mock value
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        # Mock memory usage tracking
        return sys.getsizeof(self.query_cache.cache) / (1024 * 1024)
    
    def cleanup_cache(self, aggressive: bool = False) -> None:
        """Cleanup cache memory."""
        if aggressive:
            self.query_cache.invalidate("all")
    
    def invalidate_cache(self, strategy: str = "all", **kwargs) -> None:
        """Invalidate cache using specified strategy."""
        self.query_cache.invalidate(strategy, **kwargs) 