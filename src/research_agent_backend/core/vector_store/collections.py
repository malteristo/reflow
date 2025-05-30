"""
Collection Management Operations.

This module handles ChromaDB collection lifecycle operations including
creation, deletion, listing, and type-specific configuration management.

Implements FR-ST-002: Collection management with metadata support.
Implements FR-KB-005: Collection type management and data organization.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb

from ...models.metadata_schema import CollectionType, CollectionMetadata
from ..collection_type_manager import CollectionTypeManager
from .types import (
    CollectionInfo,
    CollectionStats,
    FilterDict,
    MetadataDict,
    VectorStoreError,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
)


class CollectionManager:
    """
    Collection Management for ChromaDB.
    
    Handles collection lifecycle operations, type validation, and metadata management.
    """
    
    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_type_manager: CollectionTypeManager,
        collection_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize Collection Manager.
        
        Args:
            client: ChromaDB client instance
            collection_type_manager: Collection type manager
            collection_metadata: Base collection metadata configuration
        """
        self.client = client
        self.collection_type_manager = collection_type_manager
        self.collection_metadata = collection_metadata or {}
        self.logger = logging.getLogger(__name__)
        self._collections_cache: Dict[str, chromadb.Collection] = {}
    
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
            CollectionAlreadyExistsError: If collection exists and force_recreate=False
            VectorStoreError: If collection creation fails
        """
        try:
            self.logger.info(f"Creating collection: {name} (type: {collection_type})")
            
            # Check for existing collection when force_recreate=False
            if not force_recreate:
                try:
                    existing_collection = self.client.get_collection(name)
                    if existing_collection:
                        error_msg = f"Collection '{name}' already exists"
                        self.logger.error(error_msg)
                        raise CollectionAlreadyExistsError(error_msg)
                except:
                    # Collection doesn't exist, continue with creation
                    pass
            
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
            
        except CollectionAlreadyExistsError:
            # Re-raise our specific exceptions
            raise
        except Exception as e:
            # Check if this is a ChromaDB collection already exists error
            if "already exists" in str(e).lower():
                error_msg = f"Collection '{name}' already exists"
                self.logger.error(error_msg)
                raise CollectionAlreadyExistsError(error_msg) from e
            else:
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
    
    def list_collections(self) -> List[CollectionInfo]:
        """
        List all collections with their metadata.
        
        Returns:
            List of collection information
        """
        try:
            collections = self.client.list_collections()
            
            collection_info = []
            for collection in collections:
                metadata = collection.metadata or {}
                info = CollectionInfo(
                    name=collection.name,
                    id=collection.id,
                    metadata=metadata,
                    count=collection.count(),
                    created_at=metadata.get('created_at'),
                    owner_id=metadata.get('owner_id'),
                    team_id=metadata.get('team_id')
                )
                collection_info.append(info)
            
            self.logger.debug(f"Listed {len(collection_info)} collections")
            return collection_info
            
        except Exception as e:
            error_msg = f"Failed to list collections: {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def get_collection_stats(self, collection_name: str) -> CollectionStats:
        """
        Get detailed statistics for a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Collection statistics
        """
        try:
            collection = self.get_collection(collection_name)
            metadata = collection.metadata or {}
            
            stats = CollectionStats(
                name=collection.name,
                id=collection.id,
                document_count=collection.count(),
                metadata=metadata,
                timestamp=datetime.utcnow().isoformat(),
                collection_type=metadata.get('collection_type'),
                owner_id=metadata.get('owner_id'),
                team_id=metadata.get('team_id')
            )
            
            return stats
            
        except Exception as e:
            error_msg = f"Failed to get stats for collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
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
    
    def get_collections_by_type(self, collection_type: Union[CollectionType, str]) -> List[CollectionInfo]:
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
            coll_type_str = collection_info.metadata.get('collection_type')
            
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
            coll_type_str = collection_info.metadata.get('collection_type')
            
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
                    type_summary['by_type'][type_key]['total_documents'] += collection_info.count
                    type_summary['by_type'][type_key]['collections'].append({
                        'name': collection_info.name,
                        'document_count': collection_info.count,
                        'created_at': collection_info.created_at or 'unknown'
                    })
                    
                except ValueError:
                    # Invalid collection type
                    type_summary['untyped'].append(collection_info.name)
            else:
                # No collection type metadata
                type_summary['untyped'].append(collection_info.name)
        
        return type_summary
    
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