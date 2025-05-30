"""
ChromaDB Vector Database Integration for Research Agent.

This module provides backward compatibility for the ChromaDBManager class
and related functionality. The implementation has been refactored into a
modular package structure for better maintainability.

Implements FR-ST-002: Vector database operations with metadata support.

For new code, import directly from the vector_store package:
    from .vector_store import ChromaDBManager, CollectionManager, etc.
"""

# Import everything from the modular package to maintain backward compatibility
from .vector_store import (
    # Main manager classes
    ChromaDBManager,
    ChromaDBClient,
    CollectionManager,
    DocumentManager,
    SearchManager,
    
    # Type definitions
    VectorStoreConfig,
    CollectionInfo,
    SearchResult,
    BatchResult,
    HealthStatus,
    CollectionStats,
    FilterDict,
    MetadataDict,
    EmbeddingVector,
    DocumentId,
    
    # Exception re-exports
    VectorStoreError,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    DocumentInsertionError,
    QueryError,
    ConnectionError,
    DatabaseInitializationError,
    MetadataValidationError,
    EmbeddingDimensionError,
    
    # Related types
    CollectionType,
    CollectionMetadata,
    CollectionTypeManager,
    
    # Factory functions
    create_chroma_manager,
    get_default_collection_types,
)

# Re-export all public components for backward compatibility
__all__ = [
    # Main manager classes
    'ChromaDBManager',
    'ChromaDBClient',
    'CollectionManager', 
    'DocumentManager',
    'SearchManager',
    
    # Type definitions
    'VectorStoreConfig',
    'CollectionInfo',
    'SearchResult',
    'BatchResult',
    'HealthStatus',
    'CollectionStats',
    'FilterDict',
    'MetadataDict',
    'EmbeddingVector',
    'DocumentId',
    
    # Exceptions
    'VectorStoreError',
    'CollectionNotFoundError',
    'CollectionAlreadyExistsError',
    'DocumentInsertionError',
    'QueryError',
    'ConnectionError',
    'DatabaseInitializationError',
    'MetadataValidationError',
    'EmbeddingDimensionError',
    
    # Related types
    'CollectionType',
    'CollectionMetadata', 
    'CollectionTypeManager',
    
    # Factory functions
    'create_chroma_manager',
    'get_default_collection_types',
] 