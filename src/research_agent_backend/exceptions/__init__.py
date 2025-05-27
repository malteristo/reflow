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

__all__ = [
    # Configuration exceptions
    "ConfigurationError",
    "ConfigurationFileNotFoundError", 
    "ConfigurationValidationError",
    "EnvironmentVariableError",
    "ConfigurationSchemaError",
    "ConfigurationMergeError",
    # Vector store exceptions
    "VectorStoreError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "DocumentInsertionError",
    "QueryError",
    "ConnectionError",
    "DatabaseInitializationError",
    "MetadataValidationError",
    "EmbeddingDimensionError",
] 