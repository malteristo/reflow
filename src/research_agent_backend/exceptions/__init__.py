"""
Exceptions package for Research Agent.

This package contains custom exception classes for various error scenarios
in the Research Agent system.
"""

from .config_exceptions import (
    ConfigurationError,
    ConfigurationFileNotFoundError,
    ConfigurationValidationError,
    EnvironmentVariableError,
    ConfigurationSchemaError,
    ConfigurationMergeError,
)

from .vector_store_exceptions import (
    VectorStoreError,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    DocumentInsertionError,
    QueryError,
    ConnectionError,
    DatabaseInitializationError,
    MetadataValidationError,
    EmbeddingDimensionError,
)

from .query_exceptions import (
    QueryManagerError,
    QueryOptimizationError,
    PaginationError,
    FilterValidationError,
    PerformanceError,
    CacheError,
)

from .project_exceptions import (
    ProjectError,
    ProjectNotFoundError,
    ProjectAlreadyExistsError,
    CollectionAlreadyLinkedError,
    CollectionNotLinkedError,
    ProjectContextError,
    ProjectMetadataError,
)

# Import new comprehensive system exceptions
from .system_exceptions import (
    ResearchAgentError,
    ConfigurationSystemError,
    DatabaseSystemError,
    ModelSystemError,
    FileSystemError,
    NetworkSystemError,
    ErrorSeverity,
    ErrorContext,
    ErrorRecoveryAction,
)

__all__ = [
    # Legacy configuration exceptions
    "ConfigurationError",
    "ConfigurationFileNotFoundError", 
    "ConfigurationValidationError",
    "EnvironmentVariableError",
    "ConfigurationSchemaError",
    "ConfigurationMergeError",
    # Legacy vector store exceptions
    "VectorStoreError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "DocumentInsertionError",
    "QueryError",
    "ConnectionError",
    "DatabaseInitializationError",
    "MetadataValidationError",
    "EmbeddingDimensionError",
    # Legacy query exceptions
    "QueryManagerError",
    "QueryOptimizationError",
    "PaginationError",
    "FilterValidationError",
    "PerformanceError",
    "CacheError",
    # Legacy project exceptions
    "ProjectError",
    "ProjectNotFoundError",
    "ProjectAlreadyExistsError",
    "CollectionAlreadyLinkedError",
    "CollectionNotLinkedError",
    "ProjectContextError",
    "ProjectMetadataError",
    # New comprehensive system exceptions
    "ResearchAgentError",
    "ConfigurationSystemError",
    "DatabaseSystemError",
    "ModelSystemError",
    "FileSystemError",
    "NetworkSystemError",
    "ErrorSeverity",
    "ErrorContext",
    "ErrorRecoveryAction",
] 