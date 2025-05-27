"""
Vector Store exceptions for Research Agent.

This module defines custom exception classes for vector database operations
including ChromaDB interactions, collection management, and query processing.
"""


class VectorStoreError(Exception):
    """
    Base exception for vector store operations.
    
    This is the parent class for all vector store related errors.
    """
    pass


class CollectionNotFoundError(VectorStoreError):
    """
    Raised when trying to access a non-existent collection.
    
    This exception is raised when operations are attempted on collections
    that do not exist in the vector database.
    """
    pass


class CollectionAlreadyExistsError(VectorStoreError):
    """
    Raised when trying to create a collection that already exists.
    
    This exception is raised when attempting to create a collection
    with a name that is already in use.
    """
    pass


class DocumentInsertionError(VectorStoreError):
    """
    Raised when document insertion fails.
    
    This exception covers failures during document addition to collections,
    including validation errors, embedding mismatches, and storage failures.
    """
    pass


class QueryError(VectorStoreError):
    """
    Raised when query execution fails.
    
    This exception covers failures during vector similarity search,
    metadata filtering issues, and result formatting problems.
    """
    pass


class ConnectionError(VectorStoreError):
    """
    Raised when database connection fails.
    
    This exception is raised when the vector database cannot be reached
    or connection establishment fails.
    """
    pass


class DatabaseInitializationError(VectorStoreError):
    """
    Raised when database initialization fails.
    
    This exception covers failures during database setup, configuration
    problems, and persistence layer issues.
    """
    pass


class MetadataValidationError(VectorStoreError):
    """
    Raised when metadata validation fails.
    
    This exception is raised when document metadata does not conform
    to the expected schema or contains invalid values.
    """
    pass


class EmbeddingDimensionError(VectorStoreError):
    """
    Raised when embedding dimensions are incompatible.
    
    This exception is raised when embedding vectors have incorrect
    dimensions for the target collection.
    """
    pass 