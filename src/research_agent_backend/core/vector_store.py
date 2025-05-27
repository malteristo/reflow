"""
ChromaDB Vector Database Integration for Research Agent.

This module provides the ChromaDBManager class for interacting with ChromaDB
as the default vector database with proper schema and metadata handling.

Implements FR-ST-002: Vector database operations with metadata support.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..utils.config import ConfigManager
from ..exceptions.config_exceptions import ConfigurationError
from ..exceptions.vector_store_exceptions import (
    VectorStoreError,
    CollectionNotFoundError,
    DocumentInsertionError,
    QueryError,
)


class ChromaDBManager:
    """
    ChromaDB Vector Database Manager.
    
    Provides high-level interface for ChromaDB operations including:
    - Database initialization and connection management
    - Collection creation and management
    - Document insertion with metadata
    - Vector similarity search with filtering
    - Collection lifecycle management
    
    Implements FR-ST-002: Vector database integration with metadata support.
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        persist_directory: Optional[str] = None,
        in_memory: bool = False
    ) -> None:
        """
        Initialize ChromaDB Manager.
        
        Args:
            config_manager: Configuration manager instance (creates new if None)
            persist_directory: Custom persist directory (overrides config)
            in_memory: Use in-memory database (for testing)
        """
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Get vector store configuration
        vector_store_config = self.config_manager.get('vector_store', {})
        
        # Determine persist directory
        if in_memory:
            self.persist_directory = None
            self.logger.info("Using in-memory ChromaDB instance")
        else:
            self.persist_directory = persist_directory or vector_store_config.get('persist_directory', './data/chroma_db')
            # Resolve relative paths
            if self.persist_directory and not os.path.isabs(self.persist_directory):
                self.persist_directory = os.path.join(self.config_manager.project_root, self.persist_directory)
            self.logger.info(f"Using persistent ChromaDB at: {self.persist_directory}")
        
        # Get collection metadata configuration
        self.collection_metadata = vector_store_config.get('collection_metadata', {})
        self.metadata_fields = self.config_manager.get('collections.metadata_fields', [])
        
        # Initialize client
        self._client: Optional[chromadb.ClientAPI] = None
        self._collections_cache: Dict[str, chromadb.Collection] = {}
        
        # Connection status
        self._connected = False
    
    @property
    def client(self) -> chromadb.ClientAPI:
        """Get ChromaDB client, initializing if necessary."""
        if self._client is None:
            self.initialize_database()
        return self._client
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected and healthy."""
        return self._connected and self._client is not None
    
    def initialize_database(self, db_path: Optional[str] = None) -> None:
        """
        Initialize ChromaDB database connection.
        
        Args:
            db_path: Custom database path (overrides instance setting)
            
        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            # Use provided path or instance setting
            persist_dir = db_path or self.persist_directory
            
            if persist_dir is None:
                # In-memory database
                self.logger.info("Initializing in-memory ChromaDB database")
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                self._client = chromadb.Client(settings)
            else:
                # Persistent database
                persist_path = Path(persist_dir)
                persist_path.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Initializing persistent ChromaDB database at: {persist_path}")
                
                # Configure ChromaDB settings
                settings = Settings(
                    persist_directory=str(persist_path),
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
                
                self._client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=settings
                )
            
            # Test connection
            self._test_connection()
            self._connected = True
            
            self.logger.info("ChromaDB database initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB database: {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def _test_connection(self) -> None:
        """Test database connection and basic operations."""
        try:
            # Test basic operations
            self._client.heartbeat()
            collections = self._client.list_collections()
            self.logger.debug(f"Database connection test successful. Found {len(collections)} collections.")
        except Exception as e:
            raise VectorStoreError(f"Database connection test failed: {e}") from e
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status dictionary with detailed information
        """
        health_status = {
            'status': 'unknown',
            'connected': False,
            'persist_directory': self.persist_directory,
            'collections_count': 0,
            'collections': [],
            'timestamp': datetime.utcnow().isoformat(),
            'errors': []
        }
        
        try:
            if not self.is_connected:
                self.initialize_database()
            
            # Test heartbeat
            self._client.heartbeat()
            health_status['connected'] = True
            
            # Get collections info
            collections = self.list_collections()
            health_status['collections_count'] = len(collections)
            health_status['collections'] = collections
            
            health_status['status'] = 'healthy'
            self.logger.info("Database health check passed")
            
        except Exception as e:
            error_msg = f"Health check failed: {e}"
            health_status['errors'].append(error_msg)
            health_status['status'] = 'unhealthy'
            self.logger.error(error_msg)
        
        return health_status
    
    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None,
        force_recreate: bool = False
    ) -> chromadb.Collection:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            metadata: Additional collection metadata
            embedding_function: Custom embedding function (uses default if None)
            force_recreate: Delete existing collection if it exists
            
        Returns:
            Created collection instance
            
        Raises:
            VectorStoreError: If collection creation fails
        """
        try:
            self.logger.info(f"Creating collection: {name}")
            
            # Prepare metadata
            collection_metadata = self.collection_metadata.copy()
            if metadata:
                collection_metadata.update(metadata)
            
            # Add creation timestamp
            collection_metadata['created_at'] = datetime.utcnow().isoformat()
            
            # Handle existing collection
            if force_recreate:
                try:
                    self.delete_collection(name)
                    self.logger.info(f"Deleted existing collection: {name}")
                except CollectionNotFoundError:
                    pass  # Collection didn't exist, which is fine
            
            # Create collection
            collection = self.client.create_collection(
                name=name,
                metadata=collection_metadata,
                embedding_function=embedding_function
            )
            
            # Cache collection
            self._collections_cache[name] = collection
            
            self.logger.info(f"Successfully created collection: {name}")
            return collection
            
        except Exception as e:
            error_msg = f"Failed to create collection '{name}': {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def get_collection(self, name: str) -> chromadb.Collection:
        """
        Get an existing collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection instance
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        # Check cache first
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        try:
            collection = self.client.get_collection(name)
            self._collections_cache[name] = collection
            return collection
        except Exception as e:
            error_msg = f"Collection '{name}' not found: {e}"
            self.logger.error(error_msg)
            raise CollectionNotFoundError(error_msg) from e
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.
        
        Args:
            name: Collection name to delete
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        try:
            self.client.delete_collection(name)
            
            # Remove from cache
            self._collections_cache.pop(name, None)
            
            self.logger.info(f"Successfully deleted collection: {name}")
            
        except Exception as e:
            error_msg = f"Failed to delete collection '{name}': {e}"
            self.logger.error(error_msg)
            raise CollectionNotFoundError(error_msg) from e
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections with their metadata.
        
        Returns:
            List of collection information dictionaries
        """
        try:
            collections = self.client.list_collections()
            
            collection_info = []
            for collection in collections:
                info = {
                    'name': collection.name,
                    'id': collection.id,
                    'metadata': collection.metadata or {},
                    'count': collection.count()
                }
                collection_info.append(info)
            
            self.logger.debug(f"Listed {len(collection_info)} collections")
            return collection_info
            
        except Exception as e:
            error_msg = f"Failed to list collections: {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def add_documents(
        self,
        collection_name: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Target collection name
            chunks: List of text chunks to add
            embeddings: List of embedding vectors for each chunk
            metadata: List of metadata dictionaries for each chunk
            ids: List of document IDs (generates UUIDs if None)
            
        Raises:
            DocumentInsertionError: If insertion fails
            CollectionNotFoundError: If collection doesn't exist
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Validate input lengths
            if len(chunks) != len(embeddings):
                raise DocumentInsertionError("Number of chunks must match number of embeddings")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in chunks]
            elif len(ids) != len(chunks):
                raise DocumentInsertionError("Number of IDs must match number of chunks")
            
            # Prepare metadata
            if metadata is None:
                metadata = [{} for _ in chunks]
            elif len(metadata) != len(chunks):
                raise DocumentInsertionError("Number of metadata entries must match number of chunks")
            
            # Add standard metadata fields
            timestamp = datetime.utcnow().isoformat()
            for i, meta in enumerate(metadata):
                meta['created_at'] = timestamp
                meta['updated_at'] = timestamp
                
                # Ensure all configured metadata fields are present
                # ChromaDB only accepts str, int, float, or bool values - not None
                for field in self.metadata_fields:
                    if field not in meta:
                        meta[field] = ""  # Use empty string instead of None
            
            # Add documents to collection
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
                ids=ids
            )
            
            self.logger.info(f"Successfully added {len(chunks)} documents to collection '{collection_name}'")
            
        except CollectionNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to add documents to collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise DocumentInsertionError(error_msg) from e
    
    def query_collection(
        self,
        collection_name: str,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_documents: bool = True,
        include_distances: bool = True
    ) -> Dict[str, Any]:
        """
        Query a collection for similar documents.
        
        Args:
            collection_name: Collection to query
            query_embedding: Query vector
            k: Number of results to return
            filters: Metadata filters to apply
            include_metadata: Include document metadata in results
            include_documents: Include document text in results
            include_distances: Include similarity distances in results
            
        Returns:
            Query results dictionary
            
        Raises:
            QueryError: If query fails
            CollectionNotFoundError: If collection doesn't exist
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Prepare include list
            include = []
            if include_metadata:
                include.append("metadatas")
            if include_documents:
                include.append("documents")
            if include_distances:
                include.append("distances")
            
            # Execute query
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filters,
                include=include
            )
            
            # Format results
            formatted_results = {
                'ids': results.get('ids', [[]])[0],
                'distances': results.get('distances', [[]])[0] if include_distances else None,
                'metadatas': results.get('metadatas', [[]])[0] if include_metadata else None,
                'documents': results.get('documents', [[]])[0] if include_documents else None,
                'collection': collection_name,
                'query_params': {
                    'k': k,
                    'filters': filters,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            self.logger.debug(f"Query returned {len(formatted_results['ids'])} results from collection '{collection_name}'")
            return formatted_results
            
        except CollectionNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to query collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise QueryError(error_msg) from e
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Collection statistics
        """
        try:
            collection = self.get_collection(collection_name)
            
            stats = {
                'name': collection.name,
                'id': collection.id,
                'document_count': collection.count(),
                'metadata': collection.metadata or {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            error_msg = f"Failed to get stats for collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def reset_database(self) -> None:
        """
        Reset the entire database (delete all collections).
        
        WARNING: This will permanently delete all data!
        """
        try:
            self.logger.warning("Resetting database - all data will be lost!")
            self.client.reset()
            self._collections_cache.clear()
            self.logger.info("Database reset completed")
            
        except Exception as e:
            error_msg = f"Failed to reset database: {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def close(self) -> None:
        """Close database connection and cleanup resources."""
        try:
            self._collections_cache.clear()
            self._connected = False
            self._client = None
            self.logger.info("ChromaDB connection closed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")


# Convenience functions for common operations

def create_chroma_manager(
    config_file: Optional[str] = None,
    persist_directory: Optional[str] = None,
    in_memory: bool = False
) -> ChromaDBManager:
    """
    Factory function to create a ChromaDBManager instance.
    
    Args:
        config_file: Path to configuration file
        persist_directory: Custom persist directory
        in_memory: Use in-memory database
        
    Returns:
        Configured ChromaDBManager instance
    """
    config_manager = ConfigManager(config_file=config_file)
    return ChromaDBManager(
        config_manager=config_manager,
        persist_directory=persist_directory,
        in_memory=in_memory
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