"""
Vector Store Package - ChromaDB Integration.

This package provides modular ChromaDB vector database operations including:
- Client connection management (client.py)
- Collection lifecycle operations (collections.py)  
- Document CRUD operations (documents.py)
- Search and query functionality (search.py)
- Configuration and type definitions (types.py)

Implements FR-ST-002: Vector database operations with metadata support.
"""

# Import core managers
from .client import ChromaDBClient
from .collections import CollectionManager
from .documents import DocumentManager
from .search import SearchManager

# Import type definitions and configuration
from .types import (
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
)

# Import related types from other modules
from ...models.metadata_schema import CollectionType, CollectionMetadata
from ..collection_type_manager import CollectionTypeManager, create_collection_type_manager
from ...utils.config import ConfigManager

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
import chromadb


class ChromaDBManager:
    """
    Main ChromaDB Vector Database Manager.
    
    Provides unified interface for all vector database operations by coordinating
    the specialized managers for client, collections, documents, and search.
    
    This is the primary entry point for vector database operations.
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        persist_directory: Optional[str] = None,
        in_memory: bool = False,
        collection_type_manager: Optional[CollectionTypeManager] = None
    ) -> None:
        """
        Initialize ChromaDB Manager.
        
        Args:
            config_manager: Configuration manager instance (creates new if None)
            persist_directory: Custom persist directory (overrides config)
            in_memory: Use in-memory database (for testing)
            collection_type_manager: Collection type manager (creates new if None)
        """
        self.config_manager = config_manager or ConfigManager()
        self.collection_type_manager = collection_type_manager or create_collection_type_manager(self.config_manager)
        self.logger = logging.getLogger(__name__)
        
        # Initialize client manager
        self.client_manager = ChromaDBClient(
            config_manager=self.config_manager,
            persist_directory=persist_directory,
            in_memory=in_memory
        )
        
        # Initialize specialized managers
        self.collections = CollectionManager(
            client=self.client_manager.client,
            collection_type_manager=self.collection_type_manager,
            collection_metadata=self.config_manager.get('vector_store.collection_metadata', {})
        )
        
        self.documents = DocumentManager(
            client=self.client_manager.client,
            collection_type_manager=self.collection_type_manager
        )
        
        self.search = SearchManager(
            client=self.client_manager.client
        )
    
    # Client operations
    @property
    def client(self) -> chromadb.ClientAPI:
        """Get ChromaDB client, initializing if necessary."""
        return self.client_manager.client
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected and healthy."""
        return self.client_manager.is_connected
    
    def initialize_database(self, db_path: Optional[str] = None) -> None:
        """Initialize ChromaDB database connection."""
        return self.client_manager.initialize_database(db_path)
    
    def health_check(self) -> HealthStatus:
        """Perform comprehensive health check."""
        return self.client_manager.health_check()
    
    def reset_database(self) -> None:
        """Reset the entire database (delete all collections)."""
        return self.client_manager.reset_database()
    
    def close(self) -> None:
        """Close database connection and cleanup resources."""
        return self.client_manager.close()
    
    # Collection operations (delegate to CollectionManager)
    def create_collection(
        self,
        name: str,
        collection_type: Union[CollectionType, str, None] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None,
        force_recreate: bool = False,
        owner_id: str = "",
        team_id: Optional[str] = None
    ) -> chromadb.Collection:
        """Create a new collection with type-specific configuration."""
        return self.collections.create_collection(
            name=name,
            collection_type=collection_type,
            metadata=metadata,
            embedding_function=embedding_function,
            force_recreate=force_recreate,
            owner_id=owner_id,
            team_id=team_id
        )
    
    def get_collection(self, name: str) -> chromadb.Collection:
        """Get an existing collection."""
        return self.collections.get_collection(name)
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        return self.collections.delete_collection(name)
    
    def list_collections(self) -> List[CollectionInfo]:
        """List all collections with their metadata."""
        return self.collections.list_collections()
    
    def get_collection_stats(self, collection_name: str) -> CollectionStats:
        """Get detailed statistics for a collection."""
        return self.collections.get_collection_stats(collection_name)
    
    def validate_collection_type(
        self,
        collection_name: str,
        expected_type: Union[CollectionType, str]
    ) -> Tuple[bool, List[str]]:
        """Validate that a collection matches the expected type configuration."""
        return self.collections.validate_collection_type(collection_name, expected_type)
    
    def get_collections_by_type(self, collection_type: Union[CollectionType, str]) -> List[CollectionInfo]:
        """Get all collections of a specific type."""
        return self.collections.get_collections_by_type(collection_type)
    
    def get_collection_type_summary(self) -> Dict[str, Any]:
        """Get summary of collections organized by type."""
        return self.collections.get_collection_type_summary()
    
    def create_typed_collection(
        self,
        collection_type: Union[CollectionType, str],
        project_name: Optional[str] = None,
        suffix: Optional[str] = None,
        owner_id: str = "",
        team_id: Optional[str] = None,
        force_recreate: bool = False,
        **kwargs
    ) -> Tuple[chromadb.Collection, CollectionMetadata]:
        """Create a collection with auto-generated name and full type integration."""
        return self.collections.create_typed_collection(
            collection_type=collection_type,
            project_name=project_name,
            suffix=suffix,
            owner_id=owner_id,
            team_id=team_id,
            force_recreate=force_recreate,
            **kwargs
        )
    
    # Document operations (delegate to DocumentManager)
    def add_documents(
        self,
        collection_name: str,
        chunks: List[str],
        embeddings: List[EmbeddingVector],
        metadata: Optional[List[MetadataDict]] = None,
        ids: Optional[List[DocumentId]] = None
    ) -> BatchResult:
        """Add documents to a collection."""
        return self.documents.add_documents(
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
            ids=ids
        )
    
    def update_documents(
        self,
        collection_name: str,
        ids: List[DocumentId],
        chunks: Optional[List[str]] = None,
        embeddings: Optional[List[EmbeddingVector]] = None,
        metadata: Optional[List[MetadataDict]] = None
    ) -> BatchResult:
        """Update existing documents in a collection."""
        return self.documents.update_documents(
            collection_name=collection_name,
            ids=ids,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )
    
    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[DocumentId]] = None,
        where: Optional[MetadataDict] = None
    ) -> BatchResult:
        """Delete documents from a collection."""
        return self.documents.delete_documents(
            collection_name=collection_name,
            ids=ids,
            where=where
        )
    
    def get_documents(
        self,
        collection_name: str,
        ids: Optional[List[DocumentId]] = None,
        where: Optional[MetadataDict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Retrieve documents from a collection."""
        return self.documents.get_documents(
            collection_name=collection_name,
            ids=ids,
            where=where,
            limit=limit,
            offset=offset,
            include=include
        )
    
    def count_documents(
        self,
        collection_name: str,
        where: Optional[MetadataDict] = None
    ) -> int:
        """Count documents in a collection."""
        return self.documents.count_documents(collection_name, where)
    
    def determine_collection_for_document(
        self,
        document_metadata: MetadataDict,
        chunk_metadata: Optional[MetadataDict] = None
    ) -> str:
        """Determine the appropriate collection for a document based on its metadata."""
        return self.documents.determine_collection_for_document(
            document_metadata=document_metadata,
            chunk_metadata=chunk_metadata
        )
    
    # Search operations (delegate to SearchManager)
    def query_collection(
        self,
        collection_name: str,
        query_embedding: EmbeddingVector,
        k: int = 10,
        filters: Optional[FilterDict] = None,
        include_metadata: bool = True,
        include_documents: bool = True,
        include_distances: bool = True
    ) -> SearchResult:
        """Query a collection for similar documents."""
        return self.search.query_collection(
            collection_name=collection_name,
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            include_metadata=include_metadata,
            include_documents=include_documents,
            include_distances=include_distances
        )
    
    def multi_collection_query(
        self,
        collection_names: List[str],
        query_embedding: EmbeddingVector,
        k: int = 10,
        filters: Optional[FilterDict] = None,
        include_metadata: bool = True,
        include_documents: bool = True,
        include_distances: bool = True
    ) -> List[SearchResult]:
        """Query multiple collections for similar documents."""
        return self.search.multi_collection_query(
            collection_names=collection_names,
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            include_metadata=include_metadata,
            include_documents=include_documents,
            include_distances=include_distances
        )
    
    def get_similar_documents_by_id(
        self,
        collection_name: str,
        document_id: str,
        k: int = 10,
        filters: Optional[FilterDict] = None
    ) -> Optional[SearchResult]:
        """Find documents similar to a specific document in the collection."""
        return self.search.get_similar_documents_by_id(
            collection_name=collection_name,
            document_id=document_id,
            k=k,
            filters=filters
        )


# Convenience functions for common operations

def create_chroma_manager(
    config_file: Optional[str] = None,
    persist_directory: Optional[str] = None,
    in_memory: bool = False,
    collection_type_manager: Optional[CollectionTypeManager] = None
) -> ChromaDBManager:
    """
    Factory function to create ChromaDBManager instance.
    
    Args:
        config_file: Path to configuration file
        persist_directory: Custom persist directory
        in_memory: Use in-memory database (for testing)
        collection_type_manager: Optional collection type manager
        
    Returns:
        Initialized ChromaDBManager instance
    """
    config_manager = ConfigManager(config_file=config_file) if config_file else ConfigManager()
    
    return ChromaDBManager(
        config_manager=config_manager,
        persist_directory=persist_directory,
        in_memory=in_memory,
        collection_type_manager=collection_type_manager
    )


def get_default_collection_types() -> Dict[str, Dict[str, Any]]:
    """
    Get default collection type configurations.
    
    Returns:
        Dictionary mapping collection type names to their configurations
    """
    return {
        'fundamental': {
            'description': 'Core knowledge and fundamental concepts',
            'metadata': {
                'type': 'fundamental',
                'searchable': True,
                'priority': 'high'
            }
        },
        'project-specific': {
            'description': 'Project-specific knowledge and documentation',
            'metadata': {
                'type': 'project-specific',
                'searchable': True,
                'priority': 'medium'
            }
        },
        'general': {
            'description': 'General knowledge and miscellaneous content',
            'metadata': {
                'type': 'general',
                'searchable': True,
                'priority': 'low'
            }
        }
    }


# Export all public components
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