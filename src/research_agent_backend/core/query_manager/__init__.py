"""
Query Management Package for Research Agent.

This package provides comprehensive query capabilities for vector database operations,
organized into focused modules for maintainability and testability.

Modules:
    types: Data structures and configuration classes
    optimizer: Query optimization and strategy selection
    cache: Result caching with thread-safe operations
    batch_processor: Batch query processing with parallel support
    manager: Main QueryManager class and core functionality

Usage:
    from research_agent_backend.core.query_manager import QueryManager, QueryConfig
    
    # Or import specific components
    from research_agent_backend.core.query_manager.optimizer import QueryOptimizer
    from research_agent_backend.core.query_manager.cache import QueryCache

Backward compatibility is maintained for all existing imports.
"""

# Core types and configuration
from .types import (
    QueryConfig,
    FilterConfig,
    PaginationConfig,
    PerformanceMetrics,
    PaginationInfo,
    QueryResult
)

# Query optimization
from .optimizer import QueryOptimizer

# Caching system
from .cache import QueryCache

# Batch processing
from .batch_processor import BatchQueryProcessor

# Main query manager
from .manager import QueryManager

# Public API - maintains 100% backward compatibility
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

# Package metadata
__version__ = "1.0.0"
__author__ = "Research Agent Team"
__description__ = "Advanced query management for vector database operations" 