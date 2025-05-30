"""
Re-ranking pipeline processor for search result enhancement.

Implements pipeline processing logic for intercepting search results and 
applying re-ranking with comprehensive configuration, logging, and monitoring.

Implements FR-RQ-005, FR-RQ-008: Core query processing pipeline with re-ranking.
"""

import time
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union, Callable
from functools import wraps
from dataclasses import asdict

from .config import PipelineConfig
from .models import PipelineResult
from ..service import RerankerService
from ..config import RerankerConfig
from ..models import RankedResult
from ...integration_pipeline.models import SearchResult
from ....utils.config import ConfigManager


logger = logging.getLogger(__name__)


def monitor_performance(func: Callable) -> Callable:
    """Decorator for automatic performance monitoring of pipeline operations."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        operation_name = func.__name__
        
        try:
            result = func(self, *args, **kwargs)
            processing_time = (time.time() - start_time) * 1000
            
            if self.config.enable_monitoring:
                self._record_operation_metric(operation_name, processing_time, True)
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            if self.config.enable_monitoring:
                self._record_operation_metric(operation_name, processing_time, False, str(e))
            
            raise
    
    return wrapper


class RerankerPipelineProcessor:
    """
    Advanced pipeline processor for integrating re-ranking with search results.
    
    Intercepts search results from retrieval pipeline and applies re-ranking
    logic with comprehensive configuration, logging, monitoring, and optimization.
    
    Implements Task 6.4: Integration with Retrieval Pipeline
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        pipeline_config: Optional[PipelineConfig] = None,
        reranker_config: Optional[RerankerConfig] = None
    ):
        """
        Initialize advanced pipeline processor.
        
        Args:
            config_manager: Configuration manager for loading settings
            pipeline_config: Direct pipeline configuration
            reranker_config: Direct reranker configuration
        """
        # Initialize pipeline configuration
        if pipeline_config is not None:
            self.config = pipeline_config
        elif config_manager is not None:
            self.config = self._load_pipeline_config_from_manager(config_manager)
        else:
            self.config = PipelineConfig()
        
        # Initialize reranker service
        if reranker_config is not None:
            self.reranker = RerankerService(config=reranker_config)
        elif config_manager is not None:
            self.reranker = RerankerService(config_manager=config_manager)
        else:
            self.reranker = RerankerService()
        
        # Initialize advanced monitoring
        self._operation_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._global_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_processing_time_ms': 0.0,
            'average_throughput_docs_per_sec': 0.0
        }
        
        # Initialize configuration callbacks for dynamic updates
        self._config_update_callbacks: List[Callable[[PipelineConfig], None]] = []
        
        # Initialize threshold filtering tracking
        self._last_threshold_filtered_count = 0
        
        if self.config.enable_logging:
            logger.info(
                f"Advanced RerankerPipelineProcessor initialized",
                extra={
                    'config': asdict(self.config),
                    'reranker_model': getattr(self.reranker.config, 'model_name', 'unknown'),
                    'monitoring_enabled': self.config.enable_monitoring
                }
            )
    
    def _load_pipeline_config_from_manager(self, config_manager: ConfigManager) -> PipelineConfig:
        """Load pipeline configuration from ConfigManager with validation."""
        try:
            config_data = {
                'enable_reranking': config_manager.get('pipeline.enable_reranking', True),
                'rerank_top_k': config_manager.get('pipeline.rerank_top_k', 20),
                'rerank_top_n': config_manager.get('pipeline.rerank_top_n', 5),
                'rerank_threshold': config_manager.get('pipeline.rerank_threshold', 0.1),
                'enable_logging': config_manager.get('pipeline.enable_logging', False),
                'enable_monitoring': config_manager.get('pipeline.enable_monitoring', False),
                'preserve_original_order': config_manager.get('pipeline.preserve_original_order', True),
                'data_optimization': config_manager.get('pipeline.data_optimization', {})
            }
            
            return PipelineConfig(**config_data)
            
        except Exception as e:
            logger.warning(f"Failed to load pipeline config from manager: {e}, using defaults")
            return PipelineConfig()
    
    def _record_operation_metric(
        self, 
        operation: str, 
        duration_ms: float, 
        success: bool, 
        error_msg: Optional[str] = None
    ) -> None:
        """Record operation metrics for monitoring and analysis."""
        if not self.config.enable_monitoring:
            return
        
        metric = {
            'timestamp': time.time(),
            'duration_ms': duration_ms,
            'success': success,
            'error_message': error_msg
        }
        
        if operation not in self._operation_metrics:
            self._operation_metrics[operation] = []
        
        self._operation_metrics[operation].append(metric)
        
        # Update global stats
        self._global_stats['total_operations'] += 1
        self._global_stats['total_processing_time_ms'] += duration_ms
        
        if success:
            self._global_stats['successful_operations'] += 1
        else:
            self._global_stats['failed_operations'] += 1
        
        # Calculate rolling average throughput
        if self._global_stats['total_processing_time_ms'] > 0:
            total_docs_processed = sum(
                len(metrics) for metrics in self._operation_metrics.values()
            )
            self._global_stats['average_throughput_docs_per_sec'] = (
                total_docs_processed * 1000.0 / self._global_stats['total_processing_time_ms']
            )
    
    @monitor_performance
    def process_search_results(
        self,
        query: str,
        search_results: List[SearchResult],
        preserve_original_order: Optional[bool] = None
    ) -> PipelineResult:
        """
        Process search results through the advanced re-ranking pipeline.
        
        Args:
            query: Original search query
            search_results: List of search results from retrieval pipeline
            preserve_original_order: Override for preserving original order
            
        Returns:
            PipelineResult with processed results and comprehensive metrics
        """
        start_time = time.time()
        original_count = len(search_results)
        
        if self.config.enable_logging:
            logger.info("Initiating advanced pipeline processing", extra={
                'query_length': len(query),
                'candidate_count': original_count,
                'reranking_enabled': self.config.enable_reranking
            })
        
        try:
            # Early exit for empty results
            if not search_results:
                return self._create_empty_result(start_time)
            
            # Check if re-ranking is enabled
            if not self.config.enable_reranking:
                return self._create_passthrough_result(search_results, start_time)
            
            # Intelligent pre-processing and filtering
            working_results = self._apply_intelligent_pre_filtering(search_results)
            
            # Enhanced re-ranking with coordination
            reranked_results = self._apply_enhanced_reranking(query, working_results)
            
            # Advanced post-processing
            final_results = self._apply_advanced_post_processing(reranked_results)
            
            # Create comprehensive result
            return self._create_comprehensive_result(
                original_results=search_results,
                final_results=final_results,
                start_time=start_time
            )
            
        except Exception as e:
            return self._create_error_result(search_results, start_time, e)
    
    def _apply_intelligent_pre_filtering(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply intelligent pre-filtering based on configuration and heuristics."""
        if self.config.enable_logging:
            logger.debug("Applying intelligent pre-filtering")
        
        # Apply top_k limiting
        working_results = results[:self.config.rerank_top_k]
        
        # Additional intelligent filtering could be added here:
        # - Relevance score pre-filtering
        # - Diversity filtering to avoid redundant documents
        # - Metadata-based filtering
        
        if len(working_results) < len(results) and self.config.enable_logging:
            logger.debug(f"Pre-filtered {len(results)} -> {len(working_results)} candidates")
        
        return working_results
    
    def _apply_enhanced_reranking(self, query: str, candidates: List[SearchResult]) -> List[RankedResult]:
        """Apply enhanced re-ranking with coordination and optimization."""
        if self.config.enable_logging:
            logger.debug("Initiating enhanced re-ranking", extra={
                'candidate_count': len(candidates),
                'target_output_count': self.config.rerank_top_n
            })
        
        return self.reranker.rerank_results(
            query=query,
            candidates=candidates,
            top_n=self.config.rerank_top_n,
            collect_metrics=self.config.enable_monitoring
        )
    
    def _apply_advanced_post_processing(self, reranked_results: List[RankedResult]) -> List[RankedResult]:
        """Apply advanced post-processing including threshold filtering and optimization."""
        if self.config.rerank_threshold <= 0:
            return reranked_results
        
        # Apply threshold filtering with intelligent fallback
        filtered_results = []
        filtered_count = 0
        
        for result in reranked_results:
            if result.rerank_score >= self.config.rerank_threshold:
                filtered_results.append(result)
            else:
                filtered_count += 1
        
        # Intelligent fallback: if too many filtered, lower threshold temporarily
        if not filtered_results and reranked_results:
            if self.config.enable_logging:
                logger.warning("Threshold filtering removed all results, applying fallback")
            
            # Return top result even if below threshold
            filtered_results = reranked_results[:1]
            filtered_count = len(reranked_results) - 1
        
        if self.config.enable_logging and filtered_count > 0:
            logger.debug(f"Post-processing threshold filtering: {filtered_count} results filtered")
        
        # Store the filtered count for use in result creation
        self._last_threshold_filtered_count = filtered_count
        
        return filtered_results
    
    def _create_empty_result(self, start_time: float) -> PipelineResult:
        """Create result for empty input."""
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            reranked_results=[],
            original_count=0,
            reranked_count=0,
            processing_time_ms=processing_time,
            reranking_applied=False
        )
    
    def _create_passthrough_result(self, results: List[SearchResult], start_time: float) -> PipelineResult:
        """Create result for passthrough (re-ranking disabled)."""
        processing_time = (time.time() - start_time) * 1000
        
        if self.config.enable_logging:
            logger.debug("Re-ranking disabled, passing through original results")
        
        return PipelineResult(
            reranked_results=results,
            original_count=len(results),
            reranked_count=len(results),
            processing_time_ms=processing_time,
            reranking_applied=False
        )
    
    def _create_comprehensive_result(
        self, 
        original_results: List[SearchResult],
        final_results: List[RankedResult], 
        start_time: float
    ) -> PipelineResult:
        """Create comprehensive result with full metrics."""
        processing_time = (time.time() - start_time) * 1000
        
        result = PipelineResult(
            reranked_results=final_results,
            original_count=len(original_results),
            reranked_count=len(final_results),
            processing_time_ms=processing_time,
            reranking_applied=True,
            threshold_filtered_count=self._last_threshold_filtered_count
        )
        
        # Add comprehensive performance metrics
        if self.config.enable_monitoring:
            reranker_metrics = self.reranker.get_last_operation_metrics()
            
            result.add_performance_metric('reranking_time_ms', reranker_metrics.get('processing_time_ms', 0.0))
            result.add_performance_metric('cache_hit_rate', reranker_metrics.get('cache_hit_rate', 0.0))
            result.add_performance_metric('candidates_processed', reranker_metrics.get('candidates_processed', 0))
            result.add_performance_metric('model_inference_time_ms', reranker_metrics.get('model_inference_time_ms', 0.0))
            result.add_performance_metric('throughput_docs_per_second', self._global_stats['average_throughput_docs_per_sec'])
            
            # Advanced metrics
            result.add_performance_metric('pipeline_overhead_ms', processing_time - reranker_metrics.get('processing_time_ms', 0.0))
            result.add_performance_metric('score_improvement_ratio', self._calculate_score_improvement_ratio(final_results))
        
        if self.config.enable_logging:
            logger.info("Advanced pipeline processing completed", extra={
                'final_result_count': len(final_results),
                'processing_time_ms': processing_time,
                'reranking_applied': True
            })
        
        return result
    
    def _create_error_result(self, original_results: List[SearchResult], start_time: float, error: Exception) -> PipelineResult:
        """Create error result with fallback to original results."""
        processing_time = (time.time() - start_time) * 1000
        error_message = f"Advanced pipeline processing failed: {str(error)}"
        
        if self.config.enable_logging:
            logger.error(error_message, extra={'error_type': type(error).__name__})
        
        return PipelineResult(
            reranked_results=original_results,
            original_count=len(original_results),
            reranked_count=len(original_results),
            processing_time_ms=processing_time,
            reranking_applied=False,
            error_occurred=True,
            error_message=error_message
        )
    
    def _calculate_score_improvement_ratio(self, results: List[RankedResult]) -> float:
        """Calculate ratio of re-ranking score improvement over original scores."""
        if not results:
            return 0.0
        
        improvements = []
        for result in results:
            if result.original_score > 0:
                improvement = (result.rerank_score - result.original_score) / result.original_score
                improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def update_config(self, new_config: PipelineConfig) -> None:
        """Update pipeline configuration dynamically with callback execution."""
        old_config = self.config
        self.config = new_config
        
        # Execute configuration update callbacks
        for callback in self._config_update_callbacks:
            try:
                callback(new_config)
            except Exception as e:
                logger.warning(f"Configuration update callback failed: {e}")
        
        if self.config.enable_logging:
            logger.info("Advanced pipeline configuration updated", extra={
                'old_config': asdict(old_config),
                'new_config': asdict(new_config)
            })
    
    def register_config_update_callback(self, callback: Callable[[PipelineConfig], None]) -> None:
        """Register a callback to be executed when configuration is updated."""
        self._config_update_callbacks.append(callback)
    
    def get_current_config(self) -> PipelineConfig:
        """Get current pipeline configuration."""
        return self.config
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including operation history and global stats."""
        reranker_metrics = self.reranker.get_overall_metrics()
        
        return {
            'global_stats': self._global_stats.copy(),
            'operation_metrics': {
                op: {
                    'total_calls': len(metrics),
                    'average_duration_ms': sum(m['duration_ms'] for m in metrics) / len(metrics),
                    'success_rate': sum(1 for m in metrics if m['success']) / len(metrics),
                    'recent_errors': [m['error_message'] for m in metrics[-5:] if not m['success']]
                }
                for op, metrics in self._operation_metrics.items()
            },
            'reranker_metrics': reranker_metrics
        }
    
    def reset_comprehensive_metrics(self) -> None:
        """Reset all metrics including operation history."""
        self._operation_metrics.clear()
        self._global_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_processing_time_ms': 0.0,
            'average_throughput_docs_per_sec': 0.0
        }
        self.reranker.reset_metrics()
        
        if self.config.enable_logging:
            logger.info("Comprehensive pipeline metrics reset")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get pipeline health status and diagnostics."""
        if not self._global_stats['total_operations']:
            return {'status': 'ready', 'message': 'Pipeline ready, no operations yet'}
        
        success_rate = (
            self._global_stats['successful_operations'] / self._global_stats['total_operations']
        )
        
        if success_rate >= 0.95:
            status = 'healthy'
        elif success_rate >= 0.85:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'success_rate': success_rate,
            'total_operations': self._global_stats['total_operations'],
            'average_throughput': self._global_stats['average_throughput_docs_per_sec'],
            'reranker_health': self.reranker.get_last_operation_metrics()
        }
    
    def warmup_cache(self, queries: List[str], documents: List[str]) -> None:
        """Warm up the reranker cache with common query-document pairs."""
        if self.config.enable_logging:
            logger.info(f"Warming up advanced pipeline cache", extra={
                'query_count': len(queries),
                'document_count': len(documents)
            })
        
        self.reranker.warmup_cache(queries, documents)
    
    def clear_cache(self) -> None:
        """Clear the reranker cache."""
        self.reranker.clear_cache()
        
        if self.config.enable_logging:
            logger.info("Advanced pipeline cache cleared") 