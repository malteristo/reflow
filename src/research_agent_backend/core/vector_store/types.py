"""
Vector Store Configuration and Type Definitions.

This module provides configuration classes and type definitions for the
ChromaDB vector store integration.

Implements FR-ST-002: Vector database operations with metadata support.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Re-export exceptions for convenience
from ...exceptions.vector_store_exceptions import (
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

from ...models.metadata_schema import CollectionType, CollectionMetadata


@dataclass
class VectorStoreConfig:
    """Configuration for vector store operations."""
    persist_directory: Optional[str] = None
    in_memory: bool = False
    collection_metadata: Optional[Dict[str, Any]] = None
    metadata_fields: Optional[List[str]] = None


@dataclass
class CollectionInfo:
    """Information about a collection."""
    name: str
    id: str
    metadata: Dict[str, Any]
    count: int
    created_at: Optional[str] = None
    owner_id: Optional[str] = None
    team_id: Optional[str] = None


@dataclass
class SearchResult:
    """Results from a vector similarity search."""
    collection: str
    ids: List[str]
    query_embedding_dimension: int
    results_count: int
    metadatas: Optional[List[Dict[str, Any]]] = None
    documents: Optional[List[str]] = None
    distances: Optional[List[float]] = None


@dataclass
class BatchResult:
    """Results from batch operations."""
    success_count: int
    error_count: int
    total_count: int
    errors: List[str]
    success_ids: List[str]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_count == 0:
            return 100.0
        return (self.success_count / self.total_count) * 100.0


@dataclass
class HealthStatus:
    """Database health check status."""
    status: str  # 'healthy', 'unhealthy', 'unknown'
    connected: bool
    persist_directory: Optional[str]
    collections_count: int
    collections: List[str]
    timestamp: str
    errors: List[str]


@dataclass
class CollectionStats:
    """Detailed collection statistics."""
    name: str
    id: str
    document_count: int
    metadata: Dict[str, Any]
    timestamp: str
    storage_size_bytes: int = 0
    last_modified: Optional[datetime] = None
    collection_type: Optional[str] = None
    owner_id: Optional[str] = None
    team_id: Optional[str] = None


# Type aliases for common use cases
FilterDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
EmbeddingVector = List[float]
DocumentId = str 