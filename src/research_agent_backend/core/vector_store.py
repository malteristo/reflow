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
from ..models.metadata_schema import CollectionType, CollectionMetadata
from .collection_type_manager import CollectionTypeManager, create_collection_type_manager


class ChromaDBManager:
    """
    ChromaDB Vector Database Manager.
    
    Provides high-level interface for ChromaDB operations including:
    - Database initialization and connection management
    - Collection creation and management with type-specific configurations
    - Document insertion with metadata
    - Vector similarity search with filtering
    - Collection lifecycle management
    
    Implements FR-ST-002: Vector database integration with metadata support.
    Implements FR-KB-005: Collection type management and data organization.
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
        collection_type: Union[CollectionType, str, None] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None,
        force_recreate: bool = False,
        owner_id: str = "",
        team_id: Optional[str] = None
    ) -> chromadb.Collection:
        """
        Create a new collection with type-specific configuration.
        
        Args:
            name: Collection name
            collection_type: Collection type (determines configuration)
            metadata: Additional collection metadata
            embedding_function: Custom embedding function (uses default if None)
            force_recreate: Delete existing collection if it exists
            owner_id: Owner user ID for permissions
            team_id: Team ID for team-based collections
            
        Returns:
            Created collection instance
            
        Raises:
            VectorStoreError: If collection creation fails
        """
        try:
            self.logger.info(f"Creating collection: {name} (type: {collection_type})")
            
            # Determine collection type if not provided
            if collection_type is None:
                collection_type = CollectionType.GENERAL
                self.logger.info(f"Using default collection type: {collection_type}")
            elif isinstance(collection_type, str):
                try:
                    collection_type = CollectionType(collection_type)
                except ValueError:
                    raise VectorStoreError(f"Unknown collection type: {collection_type}")
            
            # Get type-specific configuration
            type_config = self.collection_type_manager.get_collection_config(collection_type)
            
            # Prepare ChromaDB metadata using type configuration
            collection_metadata = type_config.to_chromadb_metadata()
            
            # Add base configuration metadata
            base_metadata = self.collection_metadata.copy()
            collection_metadata.update(base_metadata)
            
            # Add additional metadata if provided
            if metadata:
                collection_metadata.update(metadata)
            
            # Add creation and ownership metadata
            collection_metadata.update({
                'created_at': datetime.utcnow().isoformat(),
                'owner_id': owner_id,
                'team_id': team_id or "",
                'collection_name': name
            })
            
            # Handle existing collection
            if force_recreate:
                try:
                    self.delete_collection(name)
                    self.logger.info(f"Deleted existing collection: {name}")
                except CollectionNotFoundError:
                    pass  # Collection didn't exist, which is fine
            
            # Create collection with type-specific configuration
            collection = self.client.create_collection(
                name=name,
                metadata=collection_metadata,
                embedding_function=embedding_function
            )
            
            # Cache collection
            self._collections_cache[name] = collection
            
            self.logger.info(f"Successfully created collection: {name} with type: {collection_type}")
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
    
    def validate_collection_type(
        self,
        collection_name: str,
        expected_type: Union[CollectionType, str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a collection matches the expected type configuration.
        
        Args:
            collection_name: Name of collection to validate
            expected_type: Expected collection type
            
        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        try:
            collection = self.get_collection(collection_name)
            collection_metadata = collection.metadata or {}
            
            # Extract collection type from metadata
            actual_type_str = collection_metadata.get('collection_type')
            if not actual_type_str:
                return False, ["Collection metadata missing collection_type"]
            
            try:
                actual_type = CollectionType(actual_type_str)
            except ValueError:
                return False, [f"Invalid collection type in metadata: {actual_type_str}"]
            
            # Convert expected type to enum if needed
            if isinstance(expected_type, str):
                try:
                    expected_type = CollectionType(expected_type)
                except ValueError:
                    return False, [f"Invalid expected collection type: {expected_type}"]
            
            # Check if types match
            if actual_type != expected_type:
                return False, [f"Collection type mismatch: expected {expected_type}, got {actual_type}"]
            
            # Get type configuration and validate against it
            type_config = self.collection_type_manager.get_collection_config(expected_type)
            
            # Validate HNSW parameters
            errors = []
            hnsw_space = collection_metadata.get('hnsw:space')
            if hnsw_space != type_config.distance_metric:
                errors.append(f"Distance metric mismatch: expected {type_config.distance_metric}, got {hnsw_space}")
            
            hnsw_ef = collection_metadata.get('hnsw:construction_ef')
            if hnsw_ef != type_config.hnsw_construction_ef:
                errors.append(f"HNSW construction_ef mismatch: expected {type_config.hnsw_construction_ef}, got {hnsw_ef}")
            
            hnsw_m = collection_metadata.get('hnsw:M')
            if hnsw_m != type_config.hnsw_m:
                errors.append(f"HNSW M parameter mismatch: expected {type_config.hnsw_m}, got {hnsw_m}")
            
            return len(errors) == 0, errors
            
        except CollectionNotFoundError:
            return False, [f"Collection '{collection_name}' not found"]
        except Exception as e:
            return False, [f"Validation error: {e}"]
    
    def determine_collection_for_document(
        self,
        document_metadata: Dict[str, Any],
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Determine the appropriate collection for a document based on its metadata.
        
        Args:
            document_metadata: Document-level metadata
            chunk_metadata: Optional chunk-level metadata
            
        Returns:
            Collection name to use for this document
        """
        # Determine collection type
        collection_type = self.collection_type_manager.determine_collection_type(
            document_metadata=document_metadata,
            chunk_metadata=chunk_metadata
        )
        
        # Extract project information for naming
        project_name = None
        if collection_type == CollectionType.PROJECT_SPECIFIC:
            project_name = document_metadata.get('project_name') or document_metadata.get('team_id')
        
        # Generate collection name
        collection_name = self.collection_type_manager.create_collection_name(
            collection_type=collection_type,
            project_name=project_name
        )
        
        # Check if collection exists, create if it doesn't
        try:
            self.get_collection(collection_name)
            self.logger.debug(f"Using existing collection: {collection_name}")
        except CollectionNotFoundError:
            # Create the collection
            owner_id = document_metadata.get('user_id', '')
            team_id = document_metadata.get('team_id')
            
            self.logger.info(f"Auto-creating collection: {collection_name} (type: {collection_type})")
            self.create_collection(
                name=collection_name,
                collection_type=collection_type,
                owner_id=owner_id,
                team_id=team_id
            )
        
        return collection_name
    
    def get_collections_by_type(self, collection_type: Union[CollectionType, str]) -> List[Dict[str, Any]]:
        """
        Get all collections of a specific type.
        
        Args:
            collection_type: Collection type to filter by
            
        Returns:
            List of collection information for collections of the specified type
        """
        if isinstance(collection_type, str):
            try:
                collection_type = CollectionType(collection_type)
            except ValueError:
                raise ValueError(f"Unknown collection type: {collection_type}")
        
        all_collections = self.list_collections()
        type_collections = []
        
        for collection_info in all_collections:
            metadata = collection_info.get('metadata', {})
            coll_type_str = metadata.get('collection_type')
            
            if coll_type_str:
                try:
                    coll_type = CollectionType(coll_type_str)
                    if coll_type == collection_type:
                        type_collections.append(collection_info)
                except ValueError:
                    continue  # Skip collections with invalid type metadata
        
        return type_collections
    
    def get_collection_type_summary(self) -> Dict[str, Any]:
        """
        Get summary of collections organized by type.
        
        Returns:
            Summary of collections grouped by type with statistics
        """
        all_collections = self.list_collections()
        type_summary = {
            'total_collections': len(all_collections),
            'by_type': {},
            'untyped': []
        }
        
        # Group collections by type
        for collection_info in all_collections:
            metadata = collection_info.get('metadata', {})
            coll_type_str = metadata.get('collection_type')
            
            if coll_type_str:
                try:
                    coll_type = CollectionType(coll_type_str)
                    type_key = str(coll_type)
                    
                    if type_key not in type_summary['by_type']:
                        type_summary['by_type'][type_key] = {
                            'count': 0,
                            'total_documents': 0,
                            'collections': []
                        }
                    
                    type_summary['by_type'][type_key]['count'] += 1
                    type_summary['by_type'][type_key]['total_documents'] += collection_info.get('count', 0)
                    type_summary['by_type'][type_key]['collections'].append({
                        'name': collection_info['name'],
                        'document_count': collection_info.get('count', 0),
                        'created_at': metadata.get('created_at', 'unknown')
                    })
                    
                except ValueError:
                    # Invalid collection type
                    type_summary['untyped'].append(collection_info['name'])
            else:
                # No collection type metadata
                type_summary['untyped'].append(collection_info['name'])
        
        return type_summary


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
        """
        Create a collection with auto-generated name and full type integration.
        
        Args:
            collection_type: Collection type for configuration
            project_name: Project name for naming (used with PROJECT_SPECIFIC)
            suffix: Additional suffix for uniqueness
            owner_id: Owner user ID
            team_id: Team ID for permissions
            force_recreate: Delete existing collection if it exists
            **kwargs: Additional metadata
            
        Returns:
            Tuple of (ChromaDB collection, CollectionMetadata instance)
        """
        try:
            # Generate standardized collection name
            collection_name = self.collection_type_manager.create_collection_name(
                collection_type=collection_type,
                project_name=project_name,
                suffix=suffix
            )
            
            # Get type configuration
            type_config = self.collection_type_manager.get_collection_config(collection_type)
            
            # Create collection metadata
            collection_metadata = type_config.create_collection_metadata(
                collection_name=collection_name,
                owner_id=owner_id,
                team_id=team_id,
                **kwargs
            )
            
            # Create ChromaDB collection
            chroma_collection = self.create_collection(
                name=collection_name,
                collection_type=collection_type,
                metadata=collection_metadata.to_dict(),
                force_recreate=force_recreate,
                owner_id=owner_id,
                team_id=team_id
            )
            
            self.logger.info(f"Successfully created typed collection: {collection_name}")
            return chroma_collection, collection_metadata
            
        except Exception as e:
            error_msg = f"Failed to create typed collection: {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e 