"""
Query Manager for Research Agent - Backward Compatibility Layer.

This module provides backward compatibility for existing imports while the actual
implementation has been refactored into a modular package structure.

New code should import directly from the query_manager package:
    from research_agent_backend.core.query_manager import QueryManager, QueryConfig

This file maintains compatibility for existing code that imports from query_manager.py:
    from research_agent_backend.core.query_manager import QueryManager  # Works
    
Refactored Architecture:
- query_manager/types.py: Data structures and configuration classes
- query_manager/optimizer.py: Query optimization and strategy selection  
- query_manager/cache.py: Result caching with thread-safe operations
- query_manager/batch_processor.py: Batch query processing with parallel support
- query_manager/manager.py: Main QueryManager class and core functionality

Implements FR-RQ-005, FR-RQ-008: Query processing and re-ranking pipeline.
"""

# Import all public classes from the modular package for backward compatibility
from .query_manager import (
    # Core types and configuration
    QueryConfig,
    FilterConfig,
    PaginationConfig,
    PerformanceMetrics,
    PaginationInfo,
    QueryResult,
    
    # Component classes
    QueryOptimizer,
    QueryCache,
    BatchQueryProcessor,
    
    # Main manager
    QueryManager
)

# Maintain the same public API
__all__ = [
    # Core types
    "QueryConfig",
    "FilterConfig", 
    "PaginationConfig",
    "PerformanceMetrics",
    "PaginationInfo",
    "QueryResult",
    
    # Component classes
    "QueryOptimizer",
    "QueryCache",
    "BatchQueryProcessor",
    
    # Main manager
    "QueryManager"
] 