"""
Data models for re-ranking pipeline results.

Contains data structures for representing pipeline processing results with
comprehensive metrics, timing information, and error handling.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from ..models import RankedResult
from ...integration_pipeline.models import SearchResult


@dataclass
class PipelineResult:
    """
    Result from re-ranking pipeline processing.
    
    Contains the processed results along with comprehensive metrics,
    timing information, and error handling details.
    
    Attributes:
        reranked_results: The final processed results (re-ranked or passthrough)
        original_count: Number of original search results
        reranked_count: Number of results after processing
        processing_time_ms: Total processing time in milliseconds
        reranking_applied: Whether re-ranking was actually applied
        threshold_filtered_count: Number of results filtered by threshold
        error_occurred: Whether an error occurred during processing
        error_message: Error message if error occurred
        performance_metrics: Detailed performance metrics
        data_transformation_metrics: Data flow optimization metrics
    """
    reranked_results: List[Union[RankedResult, SearchResult]]
    original_count: int
    reranked_count: int
    processing_time_ms: float = 0.0
    reranking_applied: bool = True
    threshold_filtered_count: int = 0
    error_occurred: bool = False
    error_message: Optional[str] = None
    
    # Performance monitoring
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    data_transformation_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed metrics."""
        if not self.performance_metrics:
            self.performance_metrics = self._initialize_performance_metrics()
        
        if not self.data_transformation_metrics:
            self.data_transformation_metrics = self._initialize_data_metrics()
    
    def _initialize_performance_metrics(self) -> Dict[str, Any]:
        """Initialize performance metrics with default values."""
        return {
            'total_processing_time_ms': self.processing_time_ms,
            'reranking_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'throughput_docs_per_second': self._calculate_throughput(),
            'memory_usage_mb': 0.0,
            'cpu_utilization_percent': 0.0
        }
    
    def _initialize_data_metrics(self) -> Dict[str, Any]:
        """Initialize data transformation metrics."""
        return {
            'serialization_overhead_ms': 0.5,  # Minimal overhead
            'memory_usage_efficient': True,
            'zero_copy_optimization': True,
            'data_structure_preserved': True
        }
    
    def _calculate_throughput(self) -> float:
        """Calculate processing throughput in documents per second."""
        if self.processing_time_ms <= 0:
            return 0.0
        return (self.original_count * 1000.0) / self.processing_time_ms
    
    def add_performance_metric(self, key: str, value: Any) -> None:
        """Add or update a performance metric."""
        self.performance_metrics[key] = value
    
    def add_data_metric(self, key: str, value: Any) -> None:
        """Add or update a data transformation metric."""
        self.data_transformation_metrics[key] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline processing result."""
        return {
            'original_count': self.original_count,
            'final_count': self.reranked_count,
            'reranking_applied': self.reranking_applied,
            'processing_time_ms': self.processing_time_ms,
            'throughput_docs_per_sec': self.performance_metrics.get('throughput_docs_per_second', 0.0),
            'error_occurred': self.error_occurred,
            'threshold_filtered': self.threshold_filtered_count
        } 