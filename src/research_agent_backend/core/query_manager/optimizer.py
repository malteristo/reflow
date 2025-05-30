"""
Query optimization engine for performance and strategy selection.

This module provides query optimization capabilities including execution strategy selection,
performance analysis, and optimization recommendations.
"""

from typing import List

from .types import PerformanceMetrics, QueryConfig


class QueryOptimizer:
    """Query optimization engine."""
    
    def __init__(self):
        self.performance_history = []
        self.optimization_strategies = {
            "precise": {"weight": 1.0, "timeout": 30.0},
            "fast": {"weight": 0.8, "timeout": 10.0},
            "balanced": {"weight": 0.9, "timeout": 20.0}
        }
    
    def select_strategy(self, max_results: int) -> str:
        """Select execution strategy based on query characteristics."""
        if max_results <= 50:
            return "precise"
        elif max_results >= 500:
            return "fast"
        else:
            return "balanced"
    
    def analyze_performance(self, metrics: PerformanceMetrics, config: QueryConfig) -> List[str]:
        """Analyze performance and provide optimization recommendations."""
        recommendations = []
        
        if metrics.total_execution_time > 5.0:
            recommendations.append("Consider reducing max_results for better performance")
        
        if config.similarity_threshold < 0.5:
            recommendations.append("Increase similarity_threshold to filter low-relevance results")
        
        if not config.enable_vector_optimization:
            recommendations.append("Enable vector_optimization for improved search performance")
        
        return recommendations 