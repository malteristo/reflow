"""
Query-related exceptions for Research Agent backend.

This module defines exceptions specific to query operations,
optimization, pagination, filtering, and performance monitoring.
"""


class QueryManagerError(Exception):
    """Base exception for QueryManager operations."""
    pass


class QueryOptimizationError(QueryManagerError):
    """Exception raised when query optimization fails."""
    pass


class PaginationError(QueryManagerError):
    """Exception raised during pagination operations."""
    pass


class FilterValidationError(QueryManagerError):
    """Exception raised when metadata filter validation fails."""
    pass


class PerformanceError(QueryManagerError):
    """Exception raised when performance constraints are violated."""
    pass


class CacheError(QueryManagerError):
    """Exception raised during cache operations."""
    pass


class QueryError(QueryManagerError):
    """Exception raised during general query execution."""
    pass


class ConnectionError(QueryManagerError):
    """Exception raised when database connection fails."""
    pass 