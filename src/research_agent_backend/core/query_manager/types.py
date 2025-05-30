"""
Types and data structures for query management.

This module contains all dataclasses and type definitions used by the query management system,
including configuration classes, result structures, and performance metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union


@dataclass
class QueryConfig:
    """Configuration for query operations."""
    max_results: int = 100
    similarity_threshold: float = 0.0
    embedding_model: str = "default"
    search_strategy: str = "precise"
    enable_caching: bool = True
    enable_vector_optimization: bool = False
    optimization_strategy: str = "none"
    target_dimensions: Optional[int] = None
    timeout_seconds: float = 30.0


@dataclass
class FilterConfig:
    """Configuration for metadata filtering."""
    logic_operator: str = "AND"
    filters: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PaginationConfig:
    """Configuration for pagination."""
    type: str = "offset_limit"  # offset_limit, cursor, stateful
    page_size: int = 10
    current_page: int = 1
    cursor: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for query execution."""
    total_execution_time: float = 0.0
    vector_search_time: float = 0.0
    filter_time: float = 0.0
    result_formatting_time: float = 0.0
    optimization_time: float = 0.0


@dataclass
class PaginationInfo:
    """Pagination information for query results."""
    type: str = "none"
    current_page: Optional[int] = None
    page_size: Optional[int] = None
    total_pages: Optional[int] = None
    has_next_page: Optional[bool] = None
    has_previous_page: Optional[bool] = None
    next_cursor: Optional[str] = None
    previous_cursor: Optional[str] = None
    total_results: Optional[int] = None
    performance_warning: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class QueryResult:
    """Result of a query operation."""
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    similarity_scores: List[float] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Advanced search attributes
    embedding_model: str = "default"
    search_strategy: str = "precise"
    execution_strategy: str = "precise"
    collection_results: Dict[str, Any] = field(default_factory=dict)
    merge_strategy: str = "simple"
    embedding_dimension: int = 0
    optimization_applied: bool = False
    optimized_dimensions: Optional[int] = None
    
    # Filtering attributes
    filter_applied: bool = False
    filtered_count: int = 0
    total_available: int = 0
    nested_field_count: int = 0
    
    # Pagination attributes
    pagination_info: Optional[PaginationInfo] = None
    
    # Result limiting attributes
    limiting_strategy: str = "top_k"
    ranking_strategy: str = "similarity_score"
    adaptive_limiting_applied: bool = False
    final_result_count: int = 0
    requested_result_count: int = 0
    suggested_limit: Optional[int] = None
    limit_adjustment_reason: Optional[str] = None
    
    # Performance attributes
    optimization_recommendations: List[str] = field(default_factory=list)
    
    # Caching attributes
    from_cache: bool = False
    cache_key: Optional[str] = None
    cache_hit_time: float = 0.0
    
    # Integration attributes
    collection_type: Optional[str] = None
    collections_searched: List[str] = field(default_factory=list)
    preprocessed_query_text: Optional[str] = None
    preprocessing_applied: bool = False
    managers_involved: List[str] = field(default_factory=list)
    
    # Error handling attributes
    warnings: List[str] = field(default_factory=list) 